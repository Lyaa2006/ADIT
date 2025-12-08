# FIXED: 修复 DynamicLoRALayer einsum/维度错误 & DynamicEvaluationModel 动态绑定逻辑
"""
ADIT: Adaptive Dynamic Instruction Tuning with HyperNetwork
完全动态生成LoRA，训练HyperNetwork根据上下文生成特异性LoRA权重
对外接口保持不变：apply_ADIT_to_model
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HUGGINGFACE_CO_URL_HOME"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import math
import re
import json
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Union, Any
from contextlib import contextmanager
from util import nethook

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .ADIT_hparams import ADITHyperParams
from .compute_z import find_fact_lookup_idx

# ================================
# Data Structures and Config
# ================================

@dataclass
class EditBatchItem:
    prompt_template: str
    prompt_formatted: str
    subject: str
    target_true: str
    target_new: str
    subject_position: int = -1
    edit_region_start: int = -1
    edit_region_end: int = -1
    paraphrase_prompt:List[str]=None
    locality_prompts: List[str] = None
    neighbor_prompts: List[str] = None
    
    def __post_init__(self):
        if self.locality_prompts is None:
            self.locality_prompts = []
        if self.neighbor_prompts is None:
            self.neighbor_prompts = []

# ================================
# Core Components - 简化版本
# ================================

def get_parent_and_attr(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

class DynamicLoRALayer(nn.Module):
    """
    动态 LoRA 层 wrapper，用于替换 Conv1D / Linear（weight shape: [out, in]）
    runtime_A: A matrix with shape [out, rank]
    runtime_B: B matrix with shape [rank, in]
    """
    def __init__(self, original_layer):
        super().__init__()
        self.original_layer = original_layer
        self.bias = hasattr(original_layer, 'bias') and original_layer.bias is not None
        
        # Conv1D的特殊处理（判定）
        self.is_conv1d = 'Conv1D' in str(type(original_layer))
        
        # 获取正确的维度（原始权重 shape: [out, in]）
        self.out_features = int(original_layer.weight.shape[0])
        self.in_features = int(original_layer.weight.shape[1])
        
        # 运行时 LoRA 权重（默认 None）
        self.runtime_A = None  # expect [out, rank]
        self.runtime_B = None  # expect [rank, in]
        
        # 默认 scale（可以由 hparams.alpha 覆盖，如果需要）
        self.default_alpha = 1.0
        
        print(f"[DEBUG] Created DynamicLoRALayer: in={self.in_features}, out={self.out_features}, is_conv1d={self.is_conv1d}")
    
    def bind_runtime_weights(self, A: torch.Tensor, B: torch.Tensor, alpha: Optional[float] = None, gate: Optional[torch.Tensor] = None):
        """
        绑定运行时生成的LoRA权重，并可绑定 gate(logit)：
        A: [out, rank]  或 [batch, out, rank]
        B: [rank, in]   或 [batch, rank, in]
        gate: scalar logit 或 [1] 或 [batch] （可为 None）
        """
        dev = self.original_layer.weight.device
        if A is not None:
            self.runtime_A = A.to(dev) if A.device != dev else A
        else:
            self.runtime_A = None
        if B is not None:
            self.runtime_B = B.to(dev) if B.device != dev else B
        else:
            self.runtime_B = None

        if gate is not None:
            g = gate.to(dev) if getattr(gate, "device", None) is not None else torch.tensor(gate, device=dev)
            self.runtime_gate = g
        else:
            # 保留已有 runtime_gate 或 None
            if not hasattr(self, "runtime_gate"):
                self.runtime_gate = None

        if alpha is not None:
            self.default_alpha = float(alpha)
    
    def clear_runtime_weights(self):
        """清理运行时权重"""
        self.runtime_A = None
        self.runtime_B = None
        self.default_alpha = 1.0
    
    def forward(self, x: torch.Tensor):
        """
        前向传播： 输出 = base + sigmoid(gate) * scale * lora_output
        如果没有 runtime_gate，则使用 default_alpha / rank 缩放（和以前兼容）
        """
        weight = self.original_layer.weight
        bias = self.original_layer.bias if self.bias else None
        base_output = F.linear(x, weight.t(), bias)

        if (self.runtime_A is not None) and (self.runtime_B is not None):
            A = self.runtime_A
            B = self.runtime_B

            if A.dim() == 3 or B.dim() == 3:
                if A.shape[0] != 1 or B.shape[0] != 1:
                    raise RuntimeError("DynamicLoRALayer currently only supports runtime batched size 1")
                A = A[0]
                B = B[0]

            if A.dim() != 2 or B.dim() != 2:
                raise RuntimeError(f"Expected A [out, r], B [r, in]; got A.dim={A.dim()}, B.dim={B.dim()}")

            rank = A.shape[1]
            if B.shape[0] != rank:
                raise RuntimeError(f"Rank mismatch between A and B: A.shape={A.shape}, B.shape={B.shape}")

            if A.shape[0] != self.out_features:
                raise RuntimeError(f"A dimension mismatch: expected out={self.out_features}, got {A.shape[0]}")
            if B.shape[1] != self.in_features:
                raise RuntimeError(f"B dimension mismatch: expected in={self.in_features}, got {B.shape[1]}")

            scale = getattr(self, "alpha", self.default_alpha) / max(1, rank)

            lora_weight = torch.matmul(A, B)  # [out, in]
            lora_output = F.linear(x, lora_weight.t(), bias=None)

            # gating: 使用 sigmoid(runtime_gate) 作为额外缩放器
            gate_val = None
            if hasattr(self, "runtime_gate") and self.runtime_gate is not None:
                g = self.runtime_gate
                # g 可能是张量标量或 1-d
                try:
                    gate_val = torch.sigmoid(g.float()).item() if g.numel() == 1 else torch.sigmoid(g.float()).view(-1)
                except Exception:
                    gate_val = float(torch.sigmoid(torch.tensor(g)).item())

            if gate_val is None:
                gated = scale * lora_output
            else:
                # gate_val may be scalar
                gated = (scale * lora_output) * float(gate_val)

            return base_output + gated

        return base_output

def replace_layers_with_dynamic_lora(model: nn.Module, target_layers: List[str]):
    """将目标层替换为DynamicLoRALayer"""
    dynamic_layers = {}
    
    print(f"[ADIT] Looking for layers: {target_layers}")
    
    for full_name, module in model.named_modules():
        if full_name in target_layers:
            print(f"[ADIT] Found target layer: {full_name}, type: {type(module)}")
            
            # 检查是否为可替换的线性层 (weight shape == 2)
            if not (hasattr(module, 'weight') and len(module.weight.shape) == 2):
                print(f"[WARN] Layer {full_name} is not a 2D-weight linear/conv1d-like layer, skipping")
                continue
            
            # 替换层
            parent, attr = get_parent_and_attr(model, full_name)
            dynamic_layer = DynamicLoRALayer(module)
            dynamic_layer.to(module.weight.device, dtype=module.weight.dtype)
            setattr(parent, attr, dynamic_layer)
            dynamic_layers[full_name] = dynamic_layer
    
    print(f"[ADIT] Replaced {len(dynamic_layers)} layers with DynamicLoRALayer")
    return dynamic_layers

class ContextExtractor(nn.Module):
    """从输入中提取上下文：优先使用中间层 hidden_states + attention-pool，fallback 为 embedding 均值"""
    def __init__(self, model, tokenizer, ctx_dim: int, preferred_layer: Optional[int] = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.ctx_dim = ctx_dim
        self.preferred_layer = preferred_layer  # 如果 None 则取中间层 len//2

        # 发现模型 hidden/state 接口
        self.has_hidden_states = hasattr(model.config, "n_layer") or getattr(model, "config", None) is not None

        # fallback embedding layer detection (和之前代码一致)
        if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            self.embedding_layer = model.transformer.wte
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            self.embedding_layer = model.model.embed_tokens
        else:
            self.embedding_layer = None

        # 若需要投影到 ctx_dim
        self._proj_initialized = False

    def attention_pool(self, hidden, attn_mask=None):
        """
        attention-pooled vector: 使用 attention mask 做加权平均（更鲁棒）
        hidden: [batch, seq, dim]
        attn_mask: [batch, seq]
        """
        if attn_mask is None:
            # 简单均值
            return hidden.mean(dim=1)
        mask = attn_mask.unsqueeze(-1).float()  # [batch, seq, 1]
        weighted = hidden * mask
        denom = mask.sum(dim=1).clamp_min(1.0)  # avoid div0
        return weighted.sum(dim=1) / denom

    def extract_context(self, prompt: str, target: str, device):
        """
        优先流程：
        1) 用 tokenizer 编码 prompt+target
        2) 用 model(..., output_hidden_states=True) 获取 hidden_states
        3) 选取 preferred_layer（或中间层），对 hidden 做 attention_pool
        4) 若维度不匹配，learnable linear 投影到 ctx_dim
        回退流程：embedding.mean 或 随机向量（仅测试）
        """
        try:
            text = prompt + target
            enc = self.tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)

            # 尝试用 hidden_states（多数 transformer 支持）
            with torch.no_grad():
                # some huggingface models accept output_hidden_states flag in forward
                outputs = None
                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
                    hstates = outputs.hidden_states  # tuple length = n_layers+1
                except Exception:
                    # fallback: run model with config to get hidden_states via hooks if not supported
                    outputs = self.model(input_ids=input_ids, attention_mask=attn)
                    hstates = getattr(outputs, "hidden_states", None)

            # 如果拿到了 hidden states
            if hstates is not None:
                # choose preferred layer
                n = len(hstates)
                if self.preferred_layer is None:
                    layer_idx = max(1, n // 2)  # avoid embedding layer at 0
                else:
                    layer_idx = min(max(1, self.preferred_layer), n - 1)

                hidden = hstates[layer_idx]  # [batch, seq, dim]
                ctx = self.attention_pool(hidden, attn)  # [batch, dim]

                # 若需要，投影
                if ctx.shape[1] != self.ctx_dim:
                    if not self._proj_initialized:
                        self.proj_layer = nn.Linear(ctx.shape[1], self.ctx_dim).to(device)
                        self._proj_initialized = True
                    ctx = self.proj_layer(ctx)

                return ctx

            # fallback: use token embeddings mean
            if self.embedding_layer is not None:
                with torch.no_grad():
                    embeddings = self.embedding_layer(input_ids)
                    ctx = embeddings.mean(dim=1)
                    if ctx.shape[1] != self.ctx_dim:
                        if not self._proj_initialized:
                            self.proj_layer = nn.Linear(ctx.shape[1], self.ctx_dim).to(device)
                            self._proj_initialized = True
                        ctx = self.proj_layer(ctx)
                    return ctx

            # last resort: random
            return torch.randn(1, self.ctx_dim, device=device)
        except Exception as e:
            print(f"[WARN] Context extraction failed (enhanced): {e}")
            return torch.randn(1, self.ctx_dim, device=device)


class HyperNetwork(nn.Module):
    """超网络：根据上下文生成LoRA权重并输出 per-layer gate_logit"""
    def __init__(self, dynamic_layers: Dict[str, DynamicLoRALayer],
                 ctx_dim: int, rank: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.rank = rank
        self.layer_names = list(dynamic_layers.keys())

        self.layer_dims = {}
        for name, layer in dynamic_layers.items():
            orig_layer = layer.original_layer
            out_features = int(orig_layer.weight.shape[0])
            in_features = int(orig_layer.weight.shape[1])
            self.layer_dims[name] = (out_features, in_features)

        self.generators = nn.ModuleDict()
        for name in self.layer_names:
            out_dim, in_dim = self.layer_dims[name]
            total_params = out_dim * rank + rank * in_dim
            # 现在输出: mean/var for params (2*total_params) + gate_logit_mean + gate_logit_logvar (2 scalars)
            generator = nn.Sequential(
                nn.Linear(ctx_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, total_params * 2 + 2),
            )
            safe_name = name.replace(".", "_")
            self.generators[safe_name] = generator

    def forward(self, ctx: torch.Tensor):
        batch_size = ctx.shape[0]
        device = ctx.device
        weights_dict = {}

        for name in self.layer_names:
            safe_name = name.replace(".", "_")
            generator = self.generators[safe_name]
            out_dim, in_dim = self.layer_dims[name]
            total_params = out_dim * self.rank + self.rank * in_dim

            params = generator(ctx)  # [batch, total_params*2 + 2]

            mean = params[:, :total_params]
            log_var = params[:, total_params: 2 * total_params]

            # gate logits mean/logvar
            gate_mean = params[:, 2 * total_params]
            gate_logvar = params[:, 2 * total_params + 1]

            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sampled = mean + eps * std  # [batch, total_params]

            # sample gate logit
            gate_std = torch.exp(0.5 * gate_logvar)
            gate_eps = torch.randn_like(gate_std)
            gate_sample = gate_mean + gate_eps * gate_std  # [batch]

            A_size = out_dim * self.rank
            B_size = self.rank * in_dim

            A_flat = sampled[:, :A_size]
            B_flat = sampled[:, A_size:A_size + B_size]

            if batch_size == 1:
                A = A_flat.view(out_dim, self.rank)
                B = B_flat.view(self.rank, in_dim)
                gate = gate_sample.view(1)  # scalar in batch dim
            else:
                A = A_flat.view(batch_size, out_dim, self.rank)
                B = B_flat.view(batch_size, self.rank, in_dim)
                gate = gate_sample.view(batch_size)

            # 返回三元组 (A,B, gate_logit)
            weights_dict[name] = (A, B, gate)

        return weights_dict


class TokenHelper:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def encode_label_for_target(self, prompt: str, target: str, device):
        full = prompt + target
        enc = self.tok(full, return_tensors="pt")
        ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(ids))
        pl = len(self.tok(prompt)["input_ids"])
        labels = ids.clone()
        labels[:, :pl] = -100
        return ids.to(device), attn.to(device), labels.to(device)

    def encode(self, text: str, device):
        enc = self.tok(text, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(ids)
        attn = attn.to(device)
        return ids, attn

def find_subject_position(tokenizer, prompt: str, subject: str) -> int:
    """找到subject在prompt中的起始token位置"""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    subject_tokens = tokenizer.encode(subject, add_special_tokens=False)
    
    for i in range(len(prompt_tokens) - len(subject_tokens) + 1):
        if prompt_tokens[i:i+len(subject_tokens)] == subject_tokens:
            return i
    return -1

def lm_ce_loss(model, ids, attn, labels):
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    return out.loss

def chunks(lst: list, bs: int) -> Iterable[list]:
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# ==========================
# ADIT Editor - 动态生成版本
# ==========================

class ADITEditor:
    def __init__(self, model, tokenizer, target_layers: List[str], hparams):
        self.base_model = model.to(hparams.device)
        self.tokenizer = tokenizer
        self.hparams = hparams
        
        # 替换目标层为动态LoRA层
        self.dynamic_layers = replace_layers_with_dynamic_lora(
            self.base_model, target_layers
        )
        
        # 创建上下文提取器
        self.context_extractor = ContextExtractor(
            self.base_model, tokenizer, hparams.ctx_dim
        )
        
        # 创建两个超网络：LF（遗忘）和 LE（编辑）
        self.hyper_lf = HyperNetwork(
            self.dynamic_layers, 
            ctx_dim=hparams.ctx_dim, 
            rank=hparams.lf_rank
        ).to(hparams.device)
        
        self.hyper_le = HyperNetwork(
            self.dynamic_layers,
            ctx_dim=hparams.ctx_dim,
            rank=hparams.le_rank
        ).to(hparams.device)
        
        # 冻结基础模型
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False
        
        # 优化器
        self.opt_lf = torch.optim.AdamW(self.hyper_lf.parameters(), lr=hparams.lr_lf)
        self.opt_le = torch.optim.AdamW(self.hyper_le.parameters(), lr=hparams.lr_le)
        
        # Token helper
        self.tok_helper = TokenHelper(tokenizer)
        from .compute_ks import compute_ks
        from .compute_z import compute_z, find_fact_lookup_idx
        
        self.compute_ks = compute_ks
        self.compute_z = compute_z
        self.find_fact_lookup_idx = find_fact_lookup_idx
        
        # 存储预计算的向量
        self.vector_cache = {}
    
    def _apply_lora_weights(self, weights_dict, clear_first=True, alpha: Optional[float]=None):
       """将LoRA权重应用到动态层。兼容 (A,B) 和 (A,B,gate)"""
       if clear_first:
        for layer in self.dynamic_layers.values():
            layer.clear_runtime_weights()

       for layer_name, vals in weights_dict.items():
        # 支持两种结构： (A,B) 或 (A,B,gate)
        if isinstance(vals, tuple) and (len(vals) == 3):
            A, B, gate = vals
        elif isinstance(vals, tuple) and (len(vals) == 2):
            A, B = vals
            gate = None
        else:
            # 如果外部传入 dict key->(A,B) 仍兼容
            try:
                A, B = vals
                gate = None
            except Exception:
                A = B = gate = None

        if layer_name in self.dynamic_layers:
            self.dynamic_layers[layer_name].bind_runtime_weights(A, B, alpha=alpha, gate=gate)
        else:
            try:
                parent, attr = get_parent_and_attr(self.base_model, layer_name)
                layer_obj = getattr(parent, attr)
                if isinstance(layer_obj, DynamicLoRALayer):
                    layer_obj.bind_runtime_weights(A, B, alpha=alpha, gate=gate)
            except Exception:
                print(f"[WARN] Could not bind weights for {layer_name}")

    
    def _clear_all_lora(self):
        """清除所有LoRA权重"""
        for layer in self.dynamic_layers.values():
            layer.clear_runtime_weights()
            
    def precompute_guidance_vectors(self, requests: List[Dict]):
        """预计算所有请求的精准向量用于监督"""
        print("[ADIT] Precomputing guidance vectors for direct supervision...")
        
        for request in requests:
            subject = request.get('subject', '')
            if not subject:
                continue
                
            # 为每个目标层计算向量
            for layer in self.hparams.layers:
                try:
                    # 1. 计算subject在该层的输入表示
                    ks_vector = self.compute_ks(
                        self.base_model,
                        self.tokenizer,
                        [request],
                        self.hparams,
                        layer,
                        templates=[request.get('prompt', '')]
                    )
                    
                    # 2. 计算target在该层的目标表示
                    z_vector = self.compute_z(
                        self.base_model,
                        self.tokenizer,
                        request,
                        self.hparams,
                        layer,
                        templates=[request.get('prompt', '')]
                    )
                    
                    # 缓存向量
                    cache_key = f"{subject}_layer{layer}"
                    self.vector_cache[cache_key] = {
                        'subject_vector': ks_vector.squeeze(0).detach().cpu(),
                        'target_vector': z_vector.detach().cpu()
                    }
                    
                except Exception as e:
                    print(f"[WARN] Failed to compute vectors for {subject} at layer {layer}: {e}")
                    # 使用零向量占位
                    hidden_size = self.base_model.config.hidden_size
                    self.vector_cache[cache_key] = {
                        'subject_vector': torch.zeros(hidden_size),
                        'target_vector': torch.zeros(hidden_size)
                    }
    
    def compute_hyper_regularization(self, weights_dict):
      """
    计算 hyper network 的正则项：
      - orthogonality loss for A: ||A^T A - I||_F
      - L2 norm for A and B
    weights_dict: { layer_name: (A,B,gate) }
    返回：dict { 'orth':.., 'l2':.., 'total':.. }
    """
      orth_loss = torch.tensor(0.0, device=self.hparams.device)
      l2_loss = torch.tensor(0.0, device=self.hparams.device)
      cnt = 0

      for name, vals in weights_dict.items():
        if vals is None:
            continue
        if isinstance(vals, tuple) and (len(vals) >= 2):
            A, B = vals[0], vals[1]
        else:
            continue

        # 若 batched (batch, out, rank)
        if A.dim() == 3:
            A_mat = A[0]
        else:
            A_mat = A
        if B.dim() == 3:
            B_mat = B[0]
        else:
            B_mat = B

        # orth loss on A: A^T A ≈ I (rank x rank)
        try:
            AtA = torch.matmul(A_mat.t(), A_mat)  # [r, r]
            r = AtA.shape[0]
            I = torch.eye(r, device=AtA.device, dtype=AtA.dtype)
            orth_loss = orth_loss + F.mse_loss(AtA, I)
        except Exception:
            pass

        # l2 norms
        l2_loss = l2_loss + (A_mat.norm() + B_mat.norm())
        cnt += 1

      if cnt > 0:
        orth_loss = orth_loss / cnt
        l2_loss = l2_loss / cnt

      total = getattr(self.hparams, "lambda_orth", 0.05) * orth_loss + getattr(self.hparams, "lambda_spec", 0.5) * l2_loss
      return {"orth": orth_loss, "l2": l2_loss, "total": total}

    
    def step_forget_batch(self, items: List[EditBatchItem]):
      """遗忘步骤：训练 hyper_lf，增加正则约束"""
      self.opt_lf.zero_grad()
      total_loss = torch.tensor(0.0, device=self.hparams.device)

      for bi in items:
        ctx = self.context_extractor.extract_context(bi.prompt_formatted, bi.target_true, self.hparams.device)
        lf_weights = self.hyper_lf(ctx)

        # 计算 CE on true target (we want forgetting to avoid producing target)
        self._apply_lora_weights(lf_weights)
        ids, attn, labels = self.tok_helper.encode_label_for_target(bi.prompt_formatted, bi.target_true, self.hparams.device)
        ce_true = lm_ce_loss(self.base_model, ids, attn, labels)

        # locality KL on locality_prompts
        kl_loc = torch.tensor(0.0, device=self.hparams.device)
        for loc_prompt in bi.locality_prompts[:2]:
            ids_loc, attn_loc = self.tok_helper.encode(loc_prompt, self.hparams.device)
            with torch.no_grad():
                self._clear_all_lora()
                outputs_orig = self.base_model(input_ids=ids_loc, attention_mask=attn_loc)
                P = F.softmax(outputs_orig.logits, dim=-1)
            self._apply_lora_weights(lf_weights, clear_first=False)
            outputs_lf = self.base_model(input_ids=ids_loc, attention_mask=attn_loc)
            Q_log = F.log_softmax(outputs_lf.logits, dim=-1)
            kl_batch = F.kl_div(Q_log, P, reduction="batchmean")
            kl_loc = kl_loc + kl_batch

        # hyper regularization
        reg = self.compute_hyper_regularization(lf_weights)
        loss_i = -ce_true + self.hparams.lambda_loc * kl_loc + reg["total"]
        total_loss = total_loss + loss_i

        self._clear_all_lora()

      total_loss = total_loss / max(1, len(items))
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(self.hyper_lf.parameters(), 1.0)
      self.opt_lf.step()
      return {"forget_loss": total_loss.item()}
    
    def step_edit_batch(self, edit_items, edit_requests):
        """
        完整且兼容 paraphrase_prompts 为 list 的 step_edit_batch 实现。
        - 签名与调用处一致：step_edit_batch(edit_items, edit_requests)
        - 仅支持 paraphrase_prompts 为 list（若不是 list 则跳过 paraphrase 部分）
        """

        hparams = self.hparams
        device = hparams.device

        # reset optimizer gradients
        try:
            self.opt_le.zero_grad()
        except Exception:
            pass

        total_loss = torch.tensor(0.0, device=device)

        # logging accumulators
        log_ce = 0.0
        log_para_ce = 0.0
        log_para_kl = 0.0
        log_neighbor_kl = 0.0
        log_vec = 0.0

        # iterate paired items & requests
        for bi, request in zip(edit_items, edit_requests):

            # ------------------------------------------------------
            # 0) build context & generate LoRA via hyper_le
            # ------------------------------------------------------
            ctx = self.build_guided_context(bi, request)
            le_weights = self.hyper_le(ctx)

            # ------------------------------------------------------
            # 1) apply LoRA (rewrite/paraphrase CE uses edited model)
            # ------------------------------------------------------
            self._apply_lora_weights(le_weights, clear_first=True)

            # ------------------------------------------------------
            # 2) Rewrite CE (main editing loss)
            # ------------------------------------------------------
            tgt_new = request["target_new"]["str"]
            ids, attn, labels = self.tok_helper.encode_label_for_target(
                bi.prompt_formatted, tgt_new, device
            )
            ce_loss = lm_ce_loss(self.base_model, ids, attn, labels)
            log_ce += float(ce_loss.detach().cpu())

            # ------------------------------------------------------
            # 3) Paraphrase CE & KL (paraphrase_prompts as list)
            # ------------------------------------------------------
            para_ce_total = torch.tensor(0.0, device=device)
            para_kl_total = torch.tensor(0.0, device=device)

            paraphrase_prompts = request.get("paraphrase_prompts", None)
            
            if paraphrase_prompts and isinstance(paraphrase_prompts, list):
                for pp in paraphrase_prompts:

                    # 3a) paraphrase CE
                    para_ids, para_attn, para_labels = self.tok_helper.encode_label_for_target(
                        pp, tgt_new, device
                    )
                    para_ce = lm_ce_loss(self.base_model, para_ids, para_attn, para_labels)
                    para_ce_total = para_ce_total + para_ce
                    log_para_ce += float(para_ce.detach().cpu())

                    # 3b) paraphrase KL:
                    #     P = original model w/o LoRA
                    with torch.no_grad():
                        self._clear_all_lora()
                        ref_logits = self.base_model(
                            input_ids=para_ids,
                            attention_mask=para_attn
                        ).logits
                        P_para = F.softmax(ref_logits, dim=-1)

                    #     Q = edited model
                    self._apply_lora_weights(le_weights, clear_first=True)
                    edited_logits = self.base_model(
                        input_ids=para_ids,
                        attention_mask=para_attn
                    ).logits
                    Q_log = F.log_softmax(edited_logits, dim=-1)

                    para_kl = F.kl_div(Q_log, P_para, reduction="batchmean")
                    para_kl_total = para_kl_total + para_kl
                    log_para_kl += float(para_kl.detach().cpu())

            # ------------------------------------------------------
            # 4) Neighbor Stability KL
            # ------------------------------------------------------
            neighbor_kl = torch.tensor(0.0, device=device)
            neighbor_prompts = request.get("neighbor_prompts", []) or []

            if len(neighbor_prompts) > 0:
                max_neighbors = min(
                    len(neighbor_prompts),
                    getattr(hparams, "max_neighbors_eval", 5)
                )

                for nprompt in neighbor_prompts[:max_neighbors]:

                    n_ids, n_attn = self.tok_helper.encode(nprompt, device)

                    # P = original model
                    with torch.no_grad():
                        self._clear_all_lora()
                        ref_logits = self.base_model(
                            input_ids=n_ids,
                            attention_mask=n_attn
                        ).logits
                        P_n = F.softmax(ref_logits, dim=-1)

                    # Q = edited model
                    self._apply_lora_weights(le_weights, clear_first=True)
                    edited_logits = self.base_model(
                        input_ids=n_ids,
                        attention_mask=n_attn
                    ).logits
                    Q_log_n = F.log_softmax(edited_logits, dim=-1)

                    kl_n = F.kl_div(Q_log_n, P_n, reduction="batchmean")
                    neighbor_kl = neighbor_kl + kl_n

                neighbor_kl = neighbor_kl / max_neighbors
                log_neighbor_kl += float(neighbor_kl.detach().cpu())

            lambda_neighbor = getattr(
                hparams,
                "lambda_neighbor",
                hparams.lambda_loc if hasattr(hparams, "lambda_loc") else 1.0
            )

            # ------------------------------------------------------
            # 5) vector guidance
            # ------------------------------------------------------
            vector_loss = torch.tensor(0.0, device=device)
            if getattr(hparams, "use_vector_guidance", False):
                vector_loss = self.compute_direct_supervision_loss(bi, request, le_weights)
                log_vec += float(vector_loss.detach().cpu())

            # ------------------------------------------------------
            # 6) total loss
            # ------------------------------------------------------
            loss_i = (
                ce_loss
                + 0.6 * para_ce_total
                + 0.2 * para_kl_total
                + lambda_neighbor * neighbor_kl
                + hparams.vector_alignment_weight * vector_loss
            )

            total_loss = total_loss + loss_i

        # ------------------------------------------------------
        # backward + optimizer
        # ------------------------------------------------------
        total_loss = total_loss / max(1, len(edit_items))
        total_loss.backward()

        try:
            torch.nn.utils.clip_grad_norm_(self.hyper_le.parameters(), 1.0)
        except Exception:
            pass

        try:
            self.opt_le.step()
        except Exception:
            pass

        # logging output
        return {
            "edit_loss": total_loss.item(),
            "edit/ce_new": log_ce / max(1, len(edit_items)),
            "edit/paraphrase_ce": log_para_ce / max(1, len(edit_items)),
            "edit/paraphrase_kl": log_para_kl / max(1, len(edit_items)),
            "edit/neighbor_kl": log_neighbor_kl / max(1, len(edit_items)),
            "edit/vector_loss": log_vec / max(1, len(edit_items)),
        }



    
    def compute_vector_alignment_loss(self, batch: EditBatchItem, request: Dict, lookup_idx: int):
        """计算向量对齐损失 - 简化版本"""
        if not self.hparams.use_vector_guidance:
            return torch.tensor(0.0, device=self.hparams.device)
        
        subject = batch.subject
        if not subject:
            return torch.tensor(0.0, device=self.hparams.device)
        
        # 只使用第一个层
        layer = self.hparams.layers[0]
        cache_key = f"{subject}_layer{layer}"
        
        if cache_key in self.vector_cache:
            try:
                # 获取目标向量
                target_vector = self.vector_cache[cache_key]['target_vector'].to(self.hparams.device)
                
                # 准备输入
                enc = self.tokenizer(batch.prompt_formatted + batch.target_new, 
                                    return_tensors="pt", truncation=True).to(self.hparams.device)
                
                # 获取最后一层的输出
                layer_name = self.hparams.rewrite_module_tmp.format(layer)
                with nethook.Trace(self.base_model, layername=layer_name) as tr:
                    _ = self.base_model(input_ids=enc["input_ids"], 
                                      attention_mask=enc.get("attention_mask", None))
                
                # 获取最后一个token的隐藏状态
                hidden = tr.output[:, -1, :]
                
                # 计算对齐损失
                loss = F.mse_loss(hidden, target_vector.unsqueeze(0))
                return loss
                
            except Exception as e:
                print(f"[WARN] vector alignment failed: {e}")
        
        return torch.tensor(0.0, device=self.hparams.device)
    
    def _get_base_context(self, batch: EditBatchItem) -> torch.Tensor:
        """获取基础上下文（原始方法）"""
        try:
            with torch.no_grad():
                ids, _ = self.tok_helper.encode(batch.prompt_formatted + batch.target_new, self.hparams.device)
                if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
                    emb = self.base_model.transformer.wte(ids)
                elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
                    emb = self.base_model.model.embed_tokens(ids)
                else:
                    emb = torch.randn(1, ids.size(1), self.hparams.ctx_dim, device=self.hparams.device)
                return emb.mean(dim=1)
        except:
            # 回退
            return torch.randn(1, self.hparams.ctx_dim, device=self.hparams.device)

    
    def build_guided_context(self, batch: EditBatchItem, request: Dict) -> torch.Tensor:
      """
    修复 Train-Test Context Mismatch 的版本：
    - 训练与推理均使用同一类型的 context：仅基于 prompt_formatted
    - 不再将 target_new 注入 HyperNetwork 输入（防止泄漏导致 rewrite 失败）
    - 用 subject_vector（ks_vector）增强可控性
    """

      try:
        device = self.hparams.device

        # ================================================================
        # 1) 基础上下文：只使用 prompt_formatted（无 target_new）
        # ---------------------------------------------------------------
        # 不再使用 prompt + target_new 的 embedding.mean —— 这是导致 rewrite 全/崩的核心问题
        # ================================================================

        ids, _ = self.tok_helper.encode(batch.prompt_formatted, device)

        # 用 embedding 得到 prompt 表达
        if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
            emb = self.base_model.transformer.wte(ids)
        elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            emb = self.base_model.model.embed_tokens(ids)
        else:
            # fallback
            emb = torch.randn(1, ids.size(1), self.hparams.ctx_dim, device=device)

        base_ctx = emb.mean(dim=1)  # [1, hidden]

        # ================================================================
        # 2) subject_vector guidance（如果有）
        # ---------------------------------------------------------------
        # 使用你预计算的 ks_vector 强化 subject-specific rewrite
        # ================================================================
        if self.hparams.use_vector_guidance and batch.subject:
            layer = self.hparams.layers[0]
            key = f"{batch.subject}_layer{layer}"

            if key in self.vector_cache:
                ks_vector = self.vector_cache[key]["subject_vector"].to(device)

                if ks_vector.dim() == 1:
                    ks_vector = ks_vector.unsqueeze(0)

                guided_ctx = base_ctx + self.hparams.vector_guidance_weight * ks_vector
                return guided_ctx

        # 无 subject 或无向量 —— 仅用 prompt context
        return base_ctx

      except Exception as e:
        print(f"[ERROR] build_guided_context failed (patched version): {e}")
        return torch.randn(1, self.hparams.ctx_dim, device=self.hparams.device)


    
    def compute_direct_supervision_loss(self, batch: EditBatchItem, request: Dict, le_weights):
        """
        综合直接监督：
          - CE(prompt -> target_new)
          - paraphrase CE (如果提供 paraphrase_prompt)
          - representation alignment using compute_z (如果 enabled)
        """
        total_loss = torch.tensor(0.0, device=self.hparams.device)

        # (1) main CE
        ids, attn, labels = self.tok_helper.encode_label_for_target(
            batch.prompt_formatted, batch.target_new, self.hparams.device
        )
        ce_loss = lm_ce_loss(self.base_model, ids, attn, labels)
        total_loss = total_loss + ce_loss * 1.0

        # (2) paraphrase CE if provided
        paraphrase_prompt = request.get("paraphrase_prompt", None)
        if paraphrase_prompt:
            para_ids, para_attn, para_labels = self.tok_helper.encode_label_for_target(
                paraphrase_prompt, batch.target_new, self.hparams.device
            )
            para_ce = lm_ce_loss(self.base_model, para_ids, para_attn, para_labels)
            total_loss = total_loss + 0.8 * para_ce

        # (3) z-vector guidance (representation alignment)
        if self.hparams.use_vector_guidance:
            layer = self.hparams.layers[0]
            key = f"{batch.subject}_layer{layer}"
            if key in self.vector_cache:
                target_vec = self.vector_cache[key]['target_vector'].to(self.hparams.device)
                # try to fetch last token hidden from that layer
                try:
                    enc = self.tokenizer(batch.prompt_formatted + batch.target_new, 
                                        return_tensors="pt", truncation=True).to(self.hparams.device)
                    with nethook.Trace(self.base_model, 
                                      layername=self.hparams.rewrite_module_tmp.format(layer)) as tr:
                        _ = self.base_model(input_ids=enc["input_ids"], 
                                          attention_mask=enc.get("attention_mask", None))
                    hidden = tr.output[:, -1, :]
                    rep_loss = F.mse_loss(hidden, target_vec.unsqueeze(0))
                    total_loss = total_loss + 0.5 * rep_loss
                except Exception as e:
                    # best-effort, don't crash training
                    print("[WARN] vector alignment failed:", e)

        return total_loss

    
    
    
    def train_multiedit(self, items: list, requests: List[Dict], 
                       epochs: int = 1, bs_forget: int = 4, 
                       bs_edit: int = 4, edit_per_forget: int = 3, 
                       shuffle: bool = True):
        """训练循环"""
        print(f"[ADIT] Starting training on {len(items)} requests...")
        
        for ep in range(epochs):
            if shuffle:
                combined = list(zip(items, requests))
                random.shuffle(combined)
                items, requests = zip(*combined)
                items, requests = list(items), list(requests)

            f_iter = list(chunks(list(zip(items, requests)), bs_forget))
            e_iter = list(chunks(list(zip(items, requests)), bs_edit))
            e_idx = 0

            for fi, f_batch in enumerate(f_iter):
                f_items, f_requests = zip(*f_batch)
                #log_f = self.step_forget_batch(f_items)
                
                for _ in range(edit_per_forget):
                    if e_idx >= len(e_iter):
                        e_idx = 0
                        if shuffle: 
                            combined = list(zip(items, requests))
                            random.shuffle(combined)
                            items, requests = zip(*combined)
                            items, requests = list(items), list(requests)
                            e_iter = list(chunks(list(zip(items, requests)), bs_edit))
                    
                    e_batch = e_iter[e_idx]
                    e_items, e_requests = zip(*e_batch)
                    
                    log_e = self.step_edit_batch(e_items, e_requests)
                    e_idx += 1

                print(log_e)
    
    def get_model_for_evaluation(self):
      """返回用于评估的动态模型，保证评估时使用 build_guided_context 生成 LoRA"""

      editor = self

      class DynamicEvaluationModel:
        def __init__(self, base_model, hyper_le, context_extractor, hparams, tokenizer, editor):
            self.base_model = base_model
            self.hyper_le = hyper_le
            self.context_extractor = context_extractor
            self.hparams = hparams
            self.tokenizer = tokenizer
            self.editor = editor

            self.config = base_model.config
            self.device = hparams.device
            self.dtype = base_model.dtype

            # 保存当前 request (必需)
            self.current_request = None
            # 保存当前 prefix（由 wrapper 设置）
            self.current_prefix = None
            # 保存当前 subject
            self.current_subject = None

            # 缓存已生成的 LoRA 权重
            self.lora_cache = {}

        # -------------------------------------------------------
        # 必需的新接口：设置 prefix + subject + request
        # -------------------------------------------------------
        def set_edit_context(self, prefix: str, subject: str, request: dict):
            self.current_prefix = prefix
            self.current_subject = subject
            self.current_request = request

            # ------------------------------------------------------
            # FIX: 按训练时的 EditBatchItem 完整字段构造 batch
            # ------------------------------------------------------
            batch = EditBatchItem(
                prompt_template=prefix,
                prompt_formatted=prefix,
                subject=subject,
                target_true=request["target_true"]["str"] if "target_true" in request else "",
                target_new=request["target_new"]["str"] if "target_new" in request else "",
                
            )
            # ------------------------------------------------------

            # cache key
            cache_key = f"{prefix}::{subject}"

            if cache_key not in self.lora_cache:
                ctx = self.editor.build_guided_context(batch, request)
                with torch.no_grad():
                    le_weights = self.hyper_le(ctx)
                self.lora_cache[cache_key] = le_weights

            self._apply_lora(self.lora_cache[cache_key])


        # -------------------------------------------------------
        def _apply_lora(self, weights):
            for layer_name, vals in weights.items():
                A, B = vals[0], vals[1]
                gate = vals[2] if len(vals) > 2 else None

                if layer_name in self.editor.dynamic_layers:
                    self.editor.dynamic_layers[layer_name].bind_runtime_weights(
                        A, B, alpha=self.hparams.alpha, gate=gate
                    )

        def _clear_lora(self):
            for layer in self.editor.dynamic_layers.values():
                layer.clear_runtime_weights()

        # -------------------------------------------------------
        # Forward：只负责推理（LoRA 已在外部绑定好）
        # -------------------------------------------------------
        def __call__(self, input_ids=None, attention_mask=None, **kwargs):
            with torch.no_grad():
                return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

      return DynamicEvaluationModel(
        self.base_model, self.hyper_le, self.context_extractor,
        self.hparams, self.tokenizer, self
    )


# ================================
# Main Interface Function (保持不变)
# ================================

def apply_ADIT_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: Any,
    **kwargs
):
    """
    ADIT主接口函数 - 完全兼容现有接口
    """
    # 配置处理
    if not hasattr(hparams, 'device'):
        from .ADIT_hparams import ADITConfig
        cfg = ADITConfig(
            lf_rank=getattr(hparams, 'lf_rank', 8),
            le_rank=getattr(hparams, 'le_rank', 16),
            alpha=getattr(hparams, 'alpha', 16.0),
            ctx_dim=getattr(hparams, 'ctx_dim', 1600),
            lr_lf=getattr(hparams, 'lr_lf', 5e-4),
            lr_le=getattr(hparams, 'lr_le', 5e-4),
            lambda_loc=getattr(hparams, 'lambda_loc', 1.0),
            lambda_kl=getattr(hparams, 'lambda_kl', 1.0),
            lambda_spec=getattr(hparams, 'lambda_spec', 0.5),
            lambda_orth=getattr(hparams, 'lambda_orth', 0.05),
            device="cuda" if torch.cuda.is_available() else "cpu",
            v_num_grad_steps=getattr(hparams, 'v_num_grad_steps', 20),
            batch_size_forget=getattr(hparams, 'batch_size_forget', 3),
            batch_size_edit=getattr(hparams, 'batch_size_edit', 1),
            edit_per_forget=getattr(hparams, 'edit_per_forget', 5),
            fact_token=getattr(hparams, 'fact_token', 'subject_first'),
            rewrite_module_tmp=getattr(hparams, 'rewrite_module_tmp', 'transformer.h.{}.mlp.c_proj'),
            layers=getattr(hparams, 'layers', [17])
        )
    else:
        cfg = hparams
    
    # 确保模型在正确设备和精度
    if model.dtype != torch.float32:
        print(f"[ADIT] Converting model from {model.dtype} to float32 for compatibility")
        model = model.float()
    
    model.to(cfg.device)
    
    # 检查是否已经存在编辑器
    if hasattr(model, '_adit_editor'):
        print("[ADIT] Reusing existing editor, training on new requests")
        editor = model._adit_editor
        
        # 转换requests为EditBatchItem
        batch_items = []
        for request in requests:
            prompt = request['prompt']
            subject = request.get('subject', '')
            
            if subject:
                prompt_formatted = prompt.format(subject)
                subject_position = find_subject_position(tok, prompt_formatted, subject)
            else:
                prompt_formatted = prompt
                subject_position = -1
            
            target_true = request.get('target_true', '')
            if isinstance(target_true, dict):
                target_true = target_true.get('str', '')
                
            target_new = request['target_new']
            if isinstance(target_new, dict):
                target_new = target_new.get('str', '')
            
            def _norm_target(s: str):
                s = s or ""
                s = s.rstrip("\n")
                return s if s.startswith(" ") else " " + s
            
            batch_items.append(EditBatchItem(
                prompt_template=request['prompt'],
                paraphrase_prompt=request.get('paraphrase_prompts',None),
                prompt_formatted=prompt_formatted,
                subject=subject,
                target_true=_norm_target(target_true),
                target_new=_norm_target(target_new),
                subject_position=subject_position,
                edit_region_start=subject_position,
                edit_region_end=subject_position + 1,
                locality_prompts=request.get('locality_prompts', []) or [],
                neighbor_prompts=request.get('neighbor_prompts', []) or [],
            ))
        
        # 训练
        editor.train_multiedit(
            batch_items,
            requests,
            epochs=getattr(cfg, 'v_num_grad_steps', 20),
            bs_forget=getattr(cfg, 'batch_size_forget', 3),
            bs_edit=getattr(cfg, 'batch_size_edit', 1),
            edit_per_forget=getattr(cfg, 'edit_per_forget', 5),
            shuffle=True
        )
        
        return model, editor
    
    # 第一次运行：初始化编辑器
    batch_items = []
    for request in requests:
        prompt = request['prompt']
        subject = request.get('subject', '')
        
        if subject:
            prompt_formatted = prompt.format(subject)
            subject_position = find_subject_position(tok, prompt_formatted, subject)
        else:
            prompt_formatted = prompt
            subject_position = -1
        
        target_true = request.get('target_true', '')
        if isinstance(target_true, dict):
            target_true = target_true.get('str', '')
            
        target_new = request['target_new']
        if isinstance(target_new, dict):
            target_new = target_new.get('str', '')
        
        def _norm_target(s: str):
            s = s or ""
            s = s.rstrip("\n")
            return s if s.startswith(" ") else " " + s
        
        batch_items.append(EditBatchItem(
            prompt_template=request['prompt'],
            paraphrase_prompt=request.get('paraphrase_prompts',None),
            prompt_formatted=prompt_formatted,
            subject=subject,
            target_true=_norm_target(target_true),
            target_new=_norm_target(target_new),
            subject_position=subject_position,
            edit_region_start=subject_position,
            edit_region_end=subject_position + 1,
            locality_prompts=request.get('locality_prompts', []) or [],
            neighbor_prompts=request.get('neighbor_prompts', []) or [],
        ))

    # 构建目标层列表
    target_layer_names = []
    for layer_id in getattr(cfg, 'layers', [17]):
        layer_name = cfg.rewrite_module_tmp.format(layer_id)
        target_layer_names.append(layer_name)
    
    print(f"[ADIT] Target layers: {target_layer_names}")

    # 初始化ADIT编辑器
    editor = ADITEditor(model, tok, target_layer_names, cfg)
    
    # 训练
    print(f"[ADIT] Starting training on {len(batch_items)} requests...")
    editor.train_multiedit(
        batch_items,
        requests,
        epochs=getattr(cfg, 'v_num_grad_steps', 20),
        bs_forget=getattr(cfg, 'batch_size_forget', 3),
        bs_edit=getattr(cfg, 'batch_size_edit', 1),
        edit_per_forget=getattr(cfg, 'edit_per_forget', 5),
        shuffle=True
    )

    # 保存编辑器引用
    model._adit_editor = editor
    
    print("[ADIT] Training completed! Returning model and editor for evaluation.")
    return model, editor