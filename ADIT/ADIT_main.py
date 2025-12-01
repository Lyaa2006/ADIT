
"""
ADIT with Vector Guidance: 使用compute_z和compute_z计算精准向量来指导LoRA生成
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

# 导入AlphaEdit的核心方法
from .compute_ks import compute_ks
from .compute_z import compute_z, find_fact_lookup_idx

# ================================
# Data Structures and Config
# ================================

@dataclass
class BatchItem:
    prompt_template: str  # ✅ 新增：存储模板，如 "The mother tongue of {} is"
    prompt_formatted: str  # ✅ 新增：存储格式化后的字符串
    subject: str          # ✅ 新增：存储subject
    target_true: str
    target_new: str
    locality_prompts: List[str]
    neighbor_prompts: List[str]

@dataclass 
class ADITConfig:
    lf_rank: int = 8
    le_rank: int = 16
    alpha: float = 16.0
    ctx_dim: int = 1600
    lr_lf: float = 5e-4
    lr_le: float = 5e-4
    lambda_loc: float = 1.0
    lambda_kl: float = 1.0
    lambda_spec: float = 0.5
    lambda_orth: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Training parameters
    v_num_grad_steps: int = 20
    batch_size_forget: int = 3
    batch_size_edit: int = 1
    edit_per_forget: int = 5
    # Precision editing parameters
    fact_token: str = "subject_first"
    rewrite_module_tmp: str = "transformer.h.{}.mlp.c_proj"  # 固定c_proj
    layers: List[int] = None
    # Vector guidance parameters
    use_vector_guidance: bool = True  # 是否使用向量指导
    vector_guidance_weight: float = 0.3  # 向量指导权重
    vector_alignment_weight: float = 0.1  # 向量对齐损失权重

    def __post_init__(self):
        if self.layers is None:
            self.layers = [17]  # Default to layer 17 for GPT2

# ================================
# Core Components (保持不变)
# ================================

class LoRALinear(nn.Module):
    def __init__(self, module):
        super().__init__()
        
        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
            self.in_features = module.weight.shape[0]
            self.out_features = module.weight.shape[1]
            self.is_conv1d = True
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")
        
        self.original_module = module
        self.bias = hasattr(module, 'bias') and module.bias is not None
        self.adapters = {}
        self.active = []
        self.runtime = {}
        self.runtime_mask = {}
        
    def bind_runtime(self, name: str, A: torch.Tensor, B: torch.Tensor):
        self.runtime[name] = (A, B)

    def clear_runtime(self):
        self.runtime.clear()
    
    def bind_token_mask(self, name: str, mask: torch.Tensor):
        self.runtime_mask[name] = mask

    def clear_token_masks(self):
        self.runtime_mask.clear()

    def add_adapter(self, name: str, rank: int, alpha: float = 8.0, trainable: bool = True):
        assert name not in self.adapters, f"Adapter {name} already exists"
        dev, dt = self.original_module.weight.device, self.original_module.weight.dtype
        
        if trainable:
            A = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dt))
            B = nn.Parameter(torch.zeros(rank, self.in_features, device=dev, dtype=dt))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))
        else:
            A = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dt), requires_grad=False)
            B = nn.Parameter(torch.zeros(rank, self.in_features, device=dev, dtype=dt), requires_grad=False)

        self.adapters[name] = {"A": A, "B": B, "alpha": alpha, "rank": rank, "trainable": trainable}
        self.register_parameter(f"{name}_A", A)
        self.register_parameter(f"{name}_B", B)
        
    def set_adapter_weights(self, name: str, A: torch.Tensor, B: torch.Tensor, alpha: float = None):
        assert name in self.adapters, f"Adapter {name} not found"
        assert not self.adapters[name]["trainable"], "Use optimizer to update trainable adapters"
        A_dst = self.adapters[name]["A"]
        B_dst = self.adapters[name]["B"]
        assert A.shape == A_dst.shape and B.shape == B_dst.shape
        with torch.no_grad():
            A_dst.copy_(A.to(A_dst.device, dtype=A_dst.dtype))
            B_dst.copy_(B.to(B_dst.device, dtype=B_dst.dtype))
            if alpha is not None:
                self.adapters[name]["alpha"] = float(alpha)
        
    def disable_all(self):
        self.active = []
        
    def enable_adapters(self, names):
        self.active = list(names)

    def forward(self, x: torch.Tensor):
        weight = self.original_module.weight.t()
        base_output = F.linear(x, weight, 
                         self.original_module.bias if self.bias else None)

        for name in self.active:
            ad = self.adapters[name]
            
            A_param = ad["A"]; B_param = ad["B"]
            A_rt, B_rt = self.runtime.get(name, (None, None))
            A_use = A_rt if A_rt is not None else A_param
            B_use = B_rt if B_rt is not None else B_param
        
            lora_output = (ad["alpha"] / ad["rank"]) * F.linear(F.linear(x, B_use), A_use)
            base_output = base_output + lora_output

        return base_output

def get_parent_and_attr(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def replace_linear_with_lora(model: nn.Module, target_layers: List[str],
                                   lf_rank=8, le_rank=8, alpha=8.0):
    lora_hosts = {}
    
    print(f"[DEBUG] Looking for layers: {target_layers}")
    for full_name, module in model.named_modules():
        
        if full_name in target_layers:
            print(f"[DEBUG] Found target layer: {full_name}, type: {type(module)}")
            
            if isinstance(module, LoRALinear):
                print(f"[DEBUG] Layer {full_name} is already LoRALinear, reusing")
                lora_hosts[full_name] = module
            elif hasattr(module, 'weight') and len(module.weight.shape) == 2:
                print(f"[DEBUG] Replacing original layer: {full_name}")
                parent, attr = get_parent_and_attr(model, full_name)
                host = LoRALinear(module)
                host.add_adapter("LF", rank=lf_rank, alpha=alpha, trainable=True)
                host.add_adapter("LE", rank=le_rank, alpha=alpha, trainable=False)
                host.to(module.weight.device, dtype=module.weight.dtype)
                setattr(parent, attr, host)
                lora_hosts[full_name] = host
            else:
                print(f"[DEBUG] Layer {full_name} is not a replaceable module")
    
    print(f"[ADIT] Replaced/Found {len(lora_hosts)} layers with LoRA")
    return lora_hosts

class PerLayerGate(nn.Module):
    def __init__(self, ctx_dim: int, rank: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * rank),
        )

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.net(ctx)
        r2 = g.shape[-1] // 2
        gA = torch.tanh(g[..., :r2])
        gB = torch.tanh(g[..., r2:])
        return gA, gB

class HyperNetwork(nn.Module):
    def __init__(self, lora_hosts: Dict[str, LoRALinear], ctx_dim: int, rank: int = 8, hidden: int = 256):
        super().__init__()
        self.rank = rank
        self.ctx_dim = ctx_dim
        self.layer_names = list(lora_hosts.keys())
        self.base_A = nn.ParameterDict()
        self.base_B = nn.ParameterDict()
        self.heads = nn.ModuleDict()

        for name, host in lora_hosts.items():
            assert host.adapters["LE"]["rank"] == rank, f"Rank mismatch at {name}"
            A0 = nn.Parameter(torch.empty(host.out_features, rank))
            B0 = nn.Parameter(torch.empty(rank, host.in_features))
            nn.init.kaiming_uniform_(A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B0, a=math.sqrt(5))
            key = name.replace(".", "_")
            self.base_A[key] = A0
            self.base_B[key] = B0
            self.heads[key] = PerLayerGate(ctx_dim, rank, hidden)

    def forward(self, ctx: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in self.layer_names:
            key = name.replace(".", "_")
            gA, gB = self.heads[key](ctx)
            A0 = self.base_A[key]
            B0 = self.base_B[key]
            gA1 = gA[0].view(1, -1)
            gB1 = gB[0].view(-1, 1)
            A = A0 * gA1
            B = gB1 * B0
            out[name] = (A, B)
        return out

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

def build_target_token_mask(tokenizer, prompt: str, target: str, device) -> torch.Tensor:
    full_ids = tokenizer(prompt + target, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    T = full_ids.size(0)
    P = prompt_ids.size(0)
    mask = torch.zeros(1, T, 1, dtype=torch.float32, device=device)
    mask[:, P:, :] = 1.0
    return mask

def lm_ce_loss(model, ids, attn, labels):
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    return out.loss

def chunks(lst: list, bs: int) -> Iterable[list]:
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# ==========================
# Enhanced ADIT Editor with Vector Guidance
# ==========================

class ADITEditor:
    """
    ADIT Editor with vector guidance from compute_ks/compute_z
    """
    def __init__(self, model, tokenizer, target_layers: List[str], hparams):
        self.base_model = model.to(hparams.device)
        self.tokenizer = tokenizer
        self.tok = TokenHelper(tokenizer)
        self.hparams = hparams

        # Replace target layers with LoRA
        self.lora_hosts = replace_linear_with_lora(
            self.base_model, target_layers, lf_rank=hparams.lf_rank, le_rank=hparams.le_rank, alpha=hparams.alpha
        )

        # Initialize hypernetwork with enhanced context dimension if using vector guidance
        ctx_dim = hparams.ctx_dim
        
            
        
        self.hyper = HyperNetwork(self.lora_hosts, ctx_dim=ctx_dim, rank=hparams.le_rank).to(hparams.device)

        # Freeze base model
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Enable LF parameters for training
        for host in self.lora_hosts.values():
            host.adapters["LF"]["A"].requires_grad = True
            host.adapters["LF"]["B"].requires_grad = True

        # Optimizers
        lf_params = []
        for host in self.lora_hosts.values():
            lf_params += [host.adapters["LF"]["A"], host.adapters["LF"]["B"]]
        
        self.opt_lf = torch.optim.AdamW(lf_params, lr=self.hparams.lr_lf)
        self.opt_le = torch.optim.AdamW(self.hyper.parameters(), lr=hparams.lr_le)
        
        # Vector guidance storage
        self.vector_cache = {}  # 缓存精准向量

    def _activate(self, adapters, *_, **__):
        for host in self.lora_hosts.values():
            host.enable_adapters(adapters)

    def _deactivate_all(self, *_, **__):
        for host in self.lora_hosts.values():
            host.disable_all()

    def precompute_guidance_vectors(self, requests: List[Dict]):
        """预计算所有请求的精准向量"""
        if not self.hparams.use_vector_guidance:
            print("[ADIT] Vector guidance is disabled, skipping precomputation")
            return
        
        print("[ADIT] Precomputing guidance vectors...")
        
        for req_idx, request in enumerate(requests):
            subject = request.get('subject', '')
            if not subject:
                continue
                
            # 为每个编辑层计算向量
            for layer in self.hparams.layers:
                # 获取上下文模板
                templates = self.get_specific_context_templates(request)
                print(templates)
                try:
                    # 1. 计算subject在该层的输入表示 (compute_ks)
                    ks_vector = compute_ks(
                        self.base_model,
                        self.tokenizer,
                        [request],  # 单个请求
                        self.hparams,
                        layer,
                        templates
                    )
                    
                    # 2. 计算target在该层的目标表示 (compute_z)
                    z_vector = compute_z(
                        self.base_model,
                        self.tokenizer,
                        request,
                        self.hparams,
                        layer,
                        templates
                    )
                    
                    # 缓存向量
                    cache_key = f"{subject}_layer{layer}"
                    self.vector_cache[cache_key] = {
                        'ks': ks_vector.squeeze(0).detach().cpu(),  # [hidden_size]
                        'z': z_vector.detach().cpu()
                    }
                    
                    if req_idx == 0:  # 只打印第一个请求的调试信息
                        print(f"  Layer {layer}: KS shape {ks_vector.shape}, Z shape {z_vector.shape}")
                        
                except Exception as e:
                    print(f"[WARN] Failed to compute vectors for {subject} at layer {layer}: {e}")
                    # 使用零向量作为占位符
                    cache_key = f"{subject}_layer{layer}"
                    hidden_size = self.base_model.config.hidden_size
                    self.vector_cache[cache_key] = {
                        'ks': torch.zeros(hidden_size),
                        'z': torch.zeros(hidden_size)
                    }
        
        print(f"[ADIT] Cached vectors for {len(requests)} requests, {len(self.vector_cache)} entries")

    def get_specific_context_templates(self, request):
      """为特定事实类型生成相关模板"""
      subject = request.get('subject', '')
    
    # 基于request内容生成模板，不重复原始模板
      original_prompt = request['prompt']  # 如：'The mother tongue of {} is'
    
    # 提取核心部分用于生成相关模板
      if "mother tongue" in original_prompt.lower() or "language" in original_prompt.lower():
        # 不要包含原始模板，只生成相关变体
        templates = [
            "{}'s native language is", 
            "The primary language of {} is",
            "What language does {} speak?",
            "{} speaks",
        ]
      elif "capital" in original_prompt.lower():
         templates = [
            "{}'s capital city is",
            "The capital city of {} is", 
            "What is the capital of {}?",
        ]
      elif "born" in original_prompt.lower() or "birth" in original_prompt.lower():
        templates = [
            "{} was born in",
            "The birthplace of {} is",
            "{} originated from",
            "Where was {} born?",
        ]
      else:
        # 通用模板变体
        templates = [
            "{} is known for",
            "About {}:",
            "Facts about {}:",
        ]
    
      return templates  # ✅ 返回单层列表

    def build_guided_context(self, batch: BatchItem, request: Dict) -> torch.Tensor:
        """构建带向量指导的上下文"""
        try:
            # 获取基础上下文
            base_ctx = self._get_base_context(batch)
            
            if not self.hparams.use_vector_guidance:
                return base_ctx
            
            # 尝试获取向量指导
            subject = batch.subject
            if subject:
                # 使用第一个层的向量（假设所有层向量相似）
                layer = self.hparams.layers[0]
                cache_key = f"{subject}_layer{layer}"
                
                if cache_key in self.vector_cache:
                    ks_vector = self.vector_cache[cache_key]['ks'].to(self.hparams.device)
                    # 将subject表示融入上下文（加权融合）
                    alpha = self.hparams.vector_guidance_weight
                    guided_ctx = base_ctx + alpha * ks_vector.unsqueeze(0)
                    return guided_ctx
            
            return base_ctx
            
        except Exception as e:
            print(f"[ERROR] Error in build_guided_context: {e}")
            return self._get_base_context(batch)

    def _get_base_context(self, batch: BatchItem) -> torch.Tensor:
        """获取基础上下文（原始方法）"""
        try:
            with torch.no_grad():
                ids, _ = self.tok.encode(batch.prompt_formatted + batch.target_new, self.hparams.device)
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

    def step_forget_batch(self, items: list, clip_norm: float = 1.0):
        """遗忘步骤（保持不变）"""
        self.opt_lf.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_neg_ce, log_kl = 0.0, 0.0

        for bi in items:
            self._deactivate_all()
            self._activate(["LF"])
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt_template, bi.target_true, self.hparams.device)
            ce_true = lm_ce_loss(self.base_model, ids, attn, labels)

            kl_loc = torch.tensor(0.0, device=self.hparams.device)
            for t in bi.locality_prompts[:2]:
                ids_loc, attn_loc = self.tok.encode(t, self.hparams.device)
                self._deactivate_all()
                with torch.no_grad():
                    lp = self.base_model(input_ids=ids_loc, attention_mask=attn_loc).logits
                    P = F.log_softmax(lp, dim=-1).exp()
                self._activate(["LF"])
                lq = self.base_model(input_ids=ids_loc, attention_mask=attn_loc).logits
                Q = F.log_softmax(lq, dim=-1)
                kl_batch = F.kl_div(Q, P, reduction="batchmean", log_target=False)
                kl_loc = kl_loc + kl_batch

            loss_i = -ce_true + self.hparams.lambda_loc * kl_loc
            total_loss = total_loss + loss_i

            log_neg_ce += float((-ce_true).detach().cpu())
            log_kl     += float(kl_loc.detach().cpu())

        total_loss = total_loss / max(1, len(items))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [host.adapters["LF"]["A"] for host in self.lora_hosts.values()] + 
            [host.adapters["LF"]["B"] for host in self.lora_hosts.values()],
            clip_norm
        )
        self.opt_lf.step()
        self._deactivate_all()

        return {
            "forget/neg_ce_true": log_neg_ce / max(1, len(items)),
            "forget/kl_loc": log_kl / max(1, len(items)),
        }

    def step_edit_batch_with_guidance(self, items: list, requests: List[Dict], clip_norm: float = 1.0):
        """使用向量指导的编辑步骤"""
        self.opt_le.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_ce = 0.0
        log_vector = 0.0  # 向量对齐损失

        for i, (bi, request) in enumerate(zip(items, requests)):
            # 使用向量指导的上下文
            ctx = self.build_guided_context(bi, request)
            le_weights = self.hyper(ctx)
            
            for name, host in self.lora_hosts.items():
                A, B = le_weights[name]
                host.bind_runtime("LE", A, B)

            # 使用精准位置信息
            subject = bi.subject
            lookup_idx = find_fact_lookup_idx(
                bi.prompt_template,
                subject,
                self.tokenizer,
                self.hparams.fact_token,
                verbose=False
            )
            
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt_formatted, bi.target_new, self.hparams.device)
            
            # 创建精准mask，只在事实位置应用编辑
            tok_mask = torch.zeros_like(ids, dtype=torch.float32, device=self.hparams.device).unsqueeze(-1)
            if lookup_idx < tok_mask.shape[1]:
                tok_mask[:, lookup_idx:, :] = 1.0
                
            for host in self.lora_hosts.values():
                host.bind_token_mask("LE", tok_mask)

            self._deactivate_all()
            self._activate(["LE"])
            
            # 基础损失：语言模型损失
            ce_new = lm_ce_loss(self.base_model, ids, attn, labels)
            
            # 向量对齐损失（如果启用）
            vector_loss = self.compute_vector_alignment_loss(bi, request, lookup_idx) if self.hparams.use_vector_guidance else 0.0
            
            # 总损失
            loss_i = ce_new + self.hparams.vector_alignment_weight * vector_loss
            total_loss = total_loss + loss_i
            
            log_ce += float(ce_new.detach().cpu())
            log_vector += float(vector_loss.detach().cpu()) if isinstance(vector_loss, torch.Tensor) else 0.0

            # 清理
            for host in self.lora_hosts.values():
                host.clear_runtime()
                host.clear_token_masks()
            self._deactivate_all()

        total_loss = total_loss / max(1, len(items))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), clip_norm)
        self.opt_le.step()

        metrics = {"edit/ce_new": log_ce / max(1, len(items))}
        if self.hparams.use_vector_guidance:
            metrics["edit/vector_loss"] = log_vector / max(1, len(items))
        return metrics

    def compute_vector_alignment_loss(self, batch: BatchItem, request: Dict, lookup_idx: int):
        """计算向量对齐损失"""
        if not self.hparams.use_vector_guidance:
            return torch.tensor(0.0, device=self.hparams.device)
        
        subject = batch.subject
        if not subject:
            return torch.tensor(0.0, device=self.hparams.device)
        
        total_loss = 0.0
        layer_count = 0
        
        for layer in self.hparams.layers:
            cache_key = f"{subject}_layer{layer}"
            if cache_key not in self.vector_cache:
                continue
                
            # 获取目标向量
            target_vector = self.vector_cache[cache_key]['z'].to(self.hparams.device)
            
            # 追踪该层的实际输出
            layer_name = f"transformer.h.{layer}.mlp.c_proj"
            
            # 准备输入（只包含prompt，用于计算实际输出）
            prompt_text = batch.prompt_formatted
            input_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.hparams.device)
            
            # 临时激活LE以获取编辑后的输出
            self._activate(["LE"])
            with torch.no_grad():
                # 注意：这里需要重新绑定权重
                ctx = self.build_guided_context(batch, request)
                le_weights = self.hyper(ctx)
                if layer_name in self.lora_hosts:
                    A, B = le_weights[layer_name]
                    self.lora_hosts[layer_name].bind_runtime("LE", A, B)
                
                # 追踪输出
                with nethook.TraceDict(
                    self.base_model,
                    layers=[layer_name],
                    retain_output=True
                ) as tr:
                    _ = self.base_model(input_ids=input_ids)
                    
                    if layer_name in tr:
                        layer_output = tr[layer_name].output
                        if isinstance(layer_output, tuple):
                            layer_output = layer_output[0]
                        
                        # 获取subject位置的实际输出
                        if lookup_idx < layer_output.shape[1]:
                            actual_vector = layer_output[0, lookup_idx, :]
                            
                            # 计算对齐损失
                            loss = F.mse_loss(actual_vector, target_vector)
                            total_loss += loss
                            layer_count += 1
            
            # 清理
            if layer_name in self.lora_hosts:
                self.lora_hosts[layer_name].clear_runtime()
            self._deactivate_all()
        
        if layer_count > 0:
            return total_loss / layer_count
        return torch.tensor(0.0, device=self.hparams.device)

    def train_multiedit_with_guidance(self, items: list, requests: List[Dict], epochs: int = 1,
                                    bs_forget: int = 4, bs_edit: int = 4,
                                    edit_per_forget: int = 3, shuffle: bool = True):
        """使用向量指导的训练循环"""
        # 预计算精准向量
        self.precompute_guidance_vectors(requests)
        
        print(f"[ADIT] Starting training with vector guidance: {self.hparams.use_vector_guidance}")
        print(f"[ADIT] Vector guidance weight: {self.hparams.vector_guidance_weight}")
        print(f"[ADIT] Vector alignment weight: {self.hparams.vector_alignment_weight}")
        
        for ep in range(epochs):
            if shuffle:
                # 同步打乱items和requests
                combined = list(zip(items, requests))
                random.shuffle(combined)
                items, requests = zip(*combined)
                items, requests = list(items), list(requests)

            f_iter = list(chunks(list(zip(items, requests)), bs_forget))
            e_iter = list(chunks(list(zip(items, requests)), bs_edit))
            e_idx = 0

            for fi, f_batch in enumerate(f_iter):
                f_items, f_requests = zip(*f_batch)
                log_f = self.step_forget_batch(f_items)
                
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
                    log_e = self.step_edit_batch_with_guidance(e_items, e_requests)
                    e_idx += 1

                print(f"[ep {ep+1}] [forget {fi+1}/{len(f_iter)}] {log_f}  |  last edit {log_e}")

    # 保持其他方法不变...
    def _generate_text(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0):
        tok = self.tokenizer
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(self.hparams.device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.hparams.device)

        gen = self.base_model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=bool(temperature and temperature > 0),
            temperature=max(temperature, 1e-5) if temperature else 1.0,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        text = tok.decode(gen[0], skip_special_tokens=True)
        return text

    def _prepare_LE_for_batch(self, batch: BatchItem):
        ctx = self.build_guided_context(batch, {})
        le_weights = self.hyper(ctx)
        for name, host in self.lora_hosts.items():
            A, B = le_weights[name]
            host.set_adapter_weights("LE", A, B)

    @contextmanager
    def _adapters(self, names):
        prev = {k: list(h.active) for k, h in self.lora_hosts.items()}
        try:
            for h in self.lora_hosts.values():
                h.enable_adapters(names)
            yield
        finally:
            for k, h in self.lora_hosts.items():
                h.enable_adapters(prev[k])

    def preview_batch(self, batch: BatchItem, prompts, max_new_tokens: int = 15):
        self._prepare_LE_for_batch(batch)

        lines = []
        for p in prompts:
            self._deactivate_all()
            base = self._generate_text(p, max_new_tokens=max_new_tokens)

            with self._adapters(["LF"]):
                lf = self._generate_text(p, max_new_tokens=max_new_tokens)

            with self._adapters(["LE"]):
                le = self._generate_text(p, max_new_tokens=max_new_tokens)
            
            with self._adapters(["LF", "LE"]):
                lelf = self._generate_text(p, max_new_tokens=max_new_tokens)

            lines.append(
                "\n".join([
                    f"—— Prompt: {p!r}",
                    f"[BASE] {base[len(p):] if base.startswith(p) else base}",
                    f"[ LF ] {lf[len(p):]  if lf.startswith(p)  else lf}",
                    f"[ LE ] {le[len(p):]  if le.startswith(p)  else le}",
                    f"[L+E] {lelf[len(p):] if lelf.startswith(p) else lelf}",
                    "-"*60
                ])
            )
        print("\n".join(lines))

# ================================
# Main Interface Function
# ================================

def apply_ADIT_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: Any,
    **kwargs
) -> Tuple[AutoModelForCausalLM, ADITEditor]:
    """
    ADIT主接口函数 - 支持向量指导
    """
    # 配置处理
    if not hasattr(hparams, 'device'):
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
            layers=getattr(hparams, 'layers', [17]),
            use_vector_guidance=getattr(hparams, 'use_vector_guidance', True),
            vector_guidance_weight=getattr(hparams, 'vector_guidance_weight', 0.3),
            vector_alignment_weight=getattr(hparams, 'vector_alignment_weight', 0.1)
        )
    else:
        cfg = hparams
    if model.dtype != torch.float32:
        print(f"[ADIT] Converting model from {model.dtype} to float32 for LoRA compatibility")
        model = model.float()
    else:
        print(f"[ADIT] Model already in float32, no conversion needed")

    # 检查是否已经存在编辑器
    if hasattr(model, '_adit_editor'):
        print("[ADIT] Reusing existing editor, training on new requests")
        editor = model._adit_editor
        
        # 转换 requests 为 BatchItem
        batch_items = []
        for request in requests:
            prompt = request['prompt']
            if 'subject' in request:
                prompt = prompt.format(request['subject'])
            
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
            
            batch_items.append(BatchItem(
    prompt_template=request['prompt'],  # 原始模板
    prompt_formatted=prompt,            # 格式化后的
    subject=request.get('subject', ''), # subject
    target_true=_norm_target(target_true),
    target_new=_norm_target(target_new),
    locality_prompts=request.get('locality_prompts', []) or [],
    neighbor_prompts=request.get('neighbor_prompts', []) or [],
))
        
        # 使用向量指导训练
        editor.train_multiedit_with_guidance(
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
        if 'subject' in request:
            prompt = prompt.format(request['subject'])
        
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
        
        batch_items.append(BatchItem(
    prompt_template=request['prompt'],  # 原始模板
    prompt_formatted=prompt,            # 格式化后的
    subject=request.get('subject', ''), # subject
    target_true=_norm_target(target_true),
    target_new=_norm_target(target_new),
    locality_prompts=request.get('locality_prompts', []) or [],
    neighbor_prompts=request.get('neighbor_prompts', []) or [],
))

    # Build target layers
    target_layer_names = []
    for layer_id in getattr(cfg, 'layers', [17]):
        layer_name = cfg.rewrite_module_tmp.format(layer_id)
        target_layer_names.append(layer_name)
    
    print(f"[ADIT] Target layers: {target_layer_names}")
    print(f"[ADIT] Using vector guidance: {cfg.use_vector_guidance}")

    # Initialize ADIT editor
    editor = ADITEditor(model, tok, target_layer_names, cfg)

    # Execute training with vector guidance
    print(f"[ADIT] Starting training with vector guidance on {len(batch_items)} requests...")
    editor.train_multiedit_with_guidance(
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