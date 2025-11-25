"""
M&M-AdvEdit: MACE-Forget × MEMIT-Localized LoRA + Adversarial Training
ADIT Main Implementation with Apply-Evaluate Separation
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
from typing import Dict, List, Tuple, Optional, Iterable, Union
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .ADIT_hparams import ADITHyperParams
from .compute_z import compute_z
from .compute_ks import compute_ks
from util.generate import generate_fast
from .compute_z import find_fact_lookup_idx

# ================================
# Core Components
# ================================

CONTEXT_TEMPLATES_CACHE = None

class LoRALinear(nn.Module):
    def __init__(self, module):
        super().__init__()
        
        # 处理 Conv1D 模块
        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
            # Conv1D 权重形状是 [in_features, out_features]
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
            # 对于 Conv1D，A: [out_features, rank], B: [rank, in_features]
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
    # print(f"[DEBUG] Input shape: {x.shape}")
     #print(f"[DEBUG] Original weight shape: {self.original_module.weight.shape}")
    
     weight = self.original_module.weight.t()
    # print(f"[DEBUG] Transposed weight shape: {weight.shape}")
    
     base_output = F.linear(x, weight, 
                     self.original_module.bias if self.bias else None)
     #print(f"[DEBUG] Base output shape: {base_output.shape}")

     for name in self.active:
        ad = self.adapters[name]
        #print(f"[DEBUG] Adapter {name}: A shape {ad['A'].shape}, B shape {ad['B'].shape}")
        
        A_param = ad["A"]; B_param = ad["B"]
        A_rt, B_rt = self.runtime.get(name, (None, None))
        A_use = A_rt if A_rt is not None else A_param
        B_use = B_rt if B_rt is not None else B_param
    
        lora_output = (ad["alpha"] / ad["rank"]) * F.linear(F.linear(x, B_use), A_use)
       # print(f"[DEBUG] LoRA output shape: {lora_output.shape}")
        
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
            
            # 如果已经是 LoRALinear，直接使用
            if isinstance(module, LoRALinear):
                print(f"[DEBUG] Layer {full_name} is already LoRALinear, reusing")
                lora_hosts[full_name] = module
            # 如果是原始的 Conv1D/Linear 层，进行替换
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

@dataclass
class BatchItem:
    prompt_template: str  # 带{}的原始模板，如 "The mother tongue of {} is"
    prompt_formatted: str  # 格式化后的prompt，如 "The mother tongue of Danielle Darrieux is"
    subject: str
    target_true: str
    target_new: str
    locality_prompts: List[str]
    neighbor_prompts: List[str]

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
    

def build_target_token_mask(tokenizer, prompt_template: str, prompt_formatted: str, target: str, subject: str, fact_token_strategy: str, device) -> torch.Tensor:
    """
    构建目标 token mask - 修复版：
    - 使用原始模板查找关键位置
    - 使用格式化后的prompt构建完整输入
    """
    # 使用格式化后的prompt构建完整输入文本
    full_text = prompt_formatted + target
    full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    full_ids = full_enc["input_ids"][0]

    T = full_ids.size(0)
    mask = torch.zeros(1, T, 1, dtype=torch.float32, device=device)

    # 关键修改：使用原始模板（带{}）来查找关键位置
    try:
        key_indices = find_fact_lookup_idx(prompt_template, subject, tokenizer, fact_token_strategy)  # ✅ 使用模板
    except Exception as e:
        # 如果查找过程中出错，则 fallback 到 prompt 后部分
        prompt_ids = tokenizer(prompt_formatted, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        P = prompt_ids.size(0)
        mask[:, P:, :] = 1.0
        return mask

    # 正常化 key_indices 为列表
    if key_indices is None:
        key_indices_list = []
    elif isinstance(key_indices, int):
        key_indices_list = [key_indices]
    else:
        # 假设是可迭代的索引集合
        key_indices_list = list(key_indices)

    # 处理并填充 mask
    filled = False
    for idx in key_indices_list:
        # 可能出现负索引（-1 表示最后一个 token）
        if idx is None:
            continue
        if idx < 0:
            idx = T + idx  # -1 -> T-1
        if 0 <= idx < T:
            mask[:, idx, :] = 1.0
            filled = True
        else:
            # 忽略越界索引
            continue

    if not filled:
        # 回退到原始逻辑：从 prompt 后开始全部 mask
        prompt_ids = tokenizer(prompt_formatted, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        P = prompt_ids.size(0)
        # 如果 prompt 长度超出 total length, 保护性处理
        P = min(P, T)
        mask[:, P:, :] = 1.0

    return mask

def lm_ce_loss(model, ids, attn, labels):
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    return out.loss

def chunks(lst: list, bs: int) -> Iterable[list]:
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

# ==========================
# ADIT Editor Core Class
# ==========================

@dataclass
class ADITConfig:
    lf_rank: int = 8
    le_rank: int = 16
    alpha: float = 16.0
    ctx_dim: int = 768
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

class ADITEditor:
    """
    ADIT Editor that can be returned from apply function and used in evaluate function
    """
    def __init__(self, model, tokenizer, target_layers: List[str], cfg: ADITConfig):
        self.base_model = model.to(cfg.device)
        self.tokenizer = tokenizer
        self.tok = TokenHelper(tokenizer)
        self.cfg = cfg

        # Replace target layers with LoRA
        self.lora_hosts = replace_linear_with_lora(
            self.base_model, target_layers, lf_rank=cfg.lf_rank, le_rank=cfg.le_rank, alpha=cfg.alpha
        )

        # Initialize hypernetwork
        self.hyper = HyperNetwork(self.lora_hosts, ctx_dim=cfg.ctx_dim, rank=cfg.le_rank).to(cfg.device)

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
        
        self.opt_lf = torch.optim.AdamW(lf_params, lr=self.cfg.lr_lf)
        self.opt_le = torch.optim.AdamW(self.hyper.parameters(), lr=cfg.lr_le)
        self.context_templates = self.get_context_templates()

    def _activate(self, adapters, *_, **__):
        for host in self.lora_hosts.values():
            host.enable_adapters(adapters)

    def _deactivate_all(self, *_, **__):
        for host in self.lora_hosts.values():
            host.disable_all()
            
    def compute_layer_contributions(self, batch_items: List[BatchItem]):
        """使用compute_ks分析各层对编辑的贡献度"""
        print("Computing layer contributions with compute_ks...")
    
    # 创建requests格式 - 关键修改：使用带{}的模板
        requests = []
        for bi in batch_items:
        # 这里需要原始带{}的模板，但我们现在只有格式化后的
        # 临时解决方案：使用默认模板
            requests.append({
            "prompt": bi.prompt_template,  # 使用默认模板
            "subject": bi.subject,
            "target_new": {"str": bi.target_new}
        })
    
        layer_contributions = {}
        for layer_name in self.lora_hosts.keys():
        # 提取层ID
            import re
            m = re.search(r"\.h\.(\d+)\.", layer_name)
            if not m:
                raise ValueError(f"[ADIT] Cannot parse layer id from layer_name: {layer_name}")
            layer_id = int(m.group(1))
        
        # 计算该层的键向量
            try:
                keys = compute_ks(
                self.base_model,
                self.tokenizer,
                requests,
                self.cfg,
                layer_id,
                self.context_templates,
            )
                contribution = keys.norm(dim=1).mean().item()
                layer_contributions[layer_name] = contribution
                print(f"Layer {layer_name} contribution: {contribution:.4f}")
            except Exception as e:
                print(f"Error computing keys for layer {layer_name}: {e}")
                layer_contributions[layer_name] = 1.0
    
        return layer_contributions
    
    def get_context_templates(self):
        """获取上下文模板，用于compute_z和compute_ks"""
        global CONTEXT_TEMPLATES_CACHE
        
        if CONTEXT_TEMPLATES_CACHE is None:
            CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        self.base_model,
                        self.tokenizer,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=5,
                        max_out_len=10,
                    )
                ]
            ]
        return CONTEXT_TEMPLATES_CACHE

    def build_context(self, batch: BatchItem) -> torch.Tensor:
     """构建上下文，使用原始模板"""
    
    # 使用更详细的缓存键，包含所有相关信息
     cache_key = f"{batch.prompt_template}_{batch.subject}_{batch.target_new}_{id(batch)}"
    
     if not hasattr(self, '_context_cache'):
        self._context_cache = {}
    
     if cache_key in self._context_cache:
        print(f"[ADIT] 使用缓存的上下文向量")  # 添加调试
        return self._context_cache[cache_key]
    
    # 使用原始模板（带{}）
     temp_request = {
        "prompt": batch.prompt_template,  # 使用原始模板
        "subject": batch.subject,
        "target_new": {"str": batch.target_new}
    }
    
     print(f"[ADIT] 计算新的上下文向量，使用模板: {repr(batch.prompt_template)}")  # 调试
    
     with torch.no_grad():
        z_layer = len(self.lora_hosts) // 2
        target_z = compute_z(
            self.base_model,
            self.tokenizer,
            temp_request,
            self.cfg,
            z_layer,
            self.context_templates,
        )
        
        ctx = target_z.unsqueeze(0)
        self._context_cache[cache_key] = ctx
        
        # 限制缓存大小，避免内存泄漏
        if len(self._context_cache) > 10:  # 减少缓存大小
            oldest_key = next(iter(self._context_cache))
            del self._context_cache[oldest_key]
            print(f"[ADIT] 清理缓存，当前大小: {len(self._context_cache)}")
    
     return ctx

    def step_forget_batch(self, items: list, clip_norm: float = 1.0):
        self.opt_lf.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_neg_ce, log_kl = 0.0, 0.0

        for bi in items:
            self._deactivate_all()
            self._activate(["LF"])
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt_formatted, bi.target_true, self.cfg.device)
            ce_true = lm_ce_loss(self.base_model, ids, attn, labels)

            kl_loc = torch.tensor(0.0, device=self.cfg.device)
            for t in bi.locality_prompts[:2]:
                ids_loc, attn_loc = self.tok.encode(t, self.cfg.device)
                self._deactivate_all()
                with torch.no_grad():
                    lp = self.base_model(input_ids=ids_loc, attention_mask=attn_loc).logits
                    P = F.log_softmax(lp, dim=-1).exp()
                self._activate(["LF"])
                lq = self.base_model(input_ids=ids_loc, attention_mask=attn_loc).logits
                Q = F.log_softmax(lq, dim=-1)
                kl_batch = F.kl_div(Q, P, reduction="batchmean", log_target=False)
                kl_loc = kl_loc + kl_batch

            loss_i = -ce_true + self.cfg.lambda_loc * kl_loc
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

    def step_edit_batch(self, items: list, clip_norm: float = 1.0):
        """
    增强版的编辑步骤，使用compute_z和compute_ks
    """
        self.opt_le.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_ce = 0.0

    # 只在第一次或必要时计算层贡献
        if not hasattr(self, '_cached_layer_contributions'):
            print("[ADIT] 首次计算层贡献度...")
            self._cached_layer_contributions = self.compute_layer_contributions(items)
    
        layer_contributions = self._cached_layer_contributions
        print(f"[ADIT] 使用缓存的层贡献度，共 {len(layer_contributions)} 层")
    
        for bi in items:
        # 使用compute_z构建更精确的上下文
            ctx = self.build_context(bi)
        
        # 根据层贡献度调整超网络输出
            le_weights = self.hyper(ctx)
            for name, host in self.lora_hosts.items():
                A, B = le_weights[name]
            
            # 根据层贡献度调整权重
                contribution = layer_contributions.get(name, 1.0)
                A = A * contribution
                B = B * contribution
            
                host.bind_runtime("LE", A, B)

        # 原有训练逻辑保持不变
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt_formatted, bi.target_new, self.cfg.device)
            tok_mask = build_target_token_mask(self.tokenizer, bi.prompt_template,bi.prompt_formatted, bi.target_new, bi.subject, self.cfg.fact_token, self.cfg.device)
        
            for host in self.lora_hosts.values():
                host.bind_token_mask("LE", tok_mask)

            self._deactivate_all()
            self._activate(["LE"])
            ce_new = lm_ce_loss(self.base_model, ids, attn, labels)

            total_loss = total_loss + ce_new
            log_ce += float(ce_new.detach().cpu())

        # 清理
            for host in self.lora_hosts.values():
                host.clear_runtime()
                host.clear_token_masks()
            self._deactivate_all()

        total_loss = total_loss / max(1, len(items))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), clip_norm)
        self.opt_le.step()

        return {"edit/ce_new": log_ce / max(1, len(items))}

    def train_multiedit(self, items: list, epochs: int = 1,
                    bs_forget: int = 4, bs_edit: int = 4,
                    edit_per_forget: int = 3, shuffle: bool = True):
        for ep in range(epochs):
            if shuffle:
                random.shuffle(items)

            f_iter = list(chunks(items, bs_forget))
            e_iter = list(chunks(items, bs_edit))
            e_idx = 0

            for fi, f_batch in enumerate(f_iter):
                log_f = self.step_forget_batch(f_batch)
                for _ in range(edit_per_forget):
                    if e_idx >= len(e_iter):
                        e_idx = 0
                        if shuffle: 
                            random.shuffle(items)
                            e_iter = list(chunks(items, bs_edit))
                    log_e = self.step_edit_batch(e_iter[e_idx])
                    e_idx += 1

                print(f"[ep {ep+1}] [forget {fi+1}/{len(f_iter)}] {log_f}  |  last edit {log_e}")

    # ====== Evaluation Methods ======
    
    def _generate_text(self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.0):
        tok = self.tokenizer
        enc = tok(prompt, return_tensors="pt")
        ids = enc["input_ids"].to(self.cfg.device)
        attn = enc.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(self.cfg.device)

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
        """Prepare LE weights for a specific batch (for evaluation)"""
        ctx = self.build_context(batch)
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
        """
        Evaluation method: Preview model outputs for different adapter configurations
        """
        self._prepare_LE_for_batch(batch)

        lines = []
        for p in prompts:
            # Baseline
            self._deactivate_all()
            base = self._generate_text(p, max_new_tokens=max_new_tokens)

            # LF only
            with self._adapters(["LF"]):
                lf = self._generate_text(p, max_new_tokens=max_new_tokens)

            # LE only
            with self._adapters(["LE"]):
                le = self._generate_text(p, max_new_tokens=max_new_tokens)
            
            # LF + LE
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
# ADIT Main Interface Function
# ================================

def apply_ADIT_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ADITHyperParams,
    **kwargs
) -> Tuple[AutoModelForCausalLM, ADITEditor]:
    """
    ADIT main interface function that returns both model and editor
    """
    # 检查是否已经存在编辑器（用于连续编辑）
    if hasattr(model, '_adit_editor'):
        print("[ADIT] Reusing existing editor, training on new requests")
        editor = model._adit_editor
        
        # 直接在这里转换 requests，避免重复函数
        batch_items = []
        for request in requests:
    # 保存原始模板
            prompt_template = request['prompt']  # 原始模板，带{}
    
    # 格式化prompt用于训练
            prompt_formatted = request['prompt']
            if 'subject' in request:
                prompt_formatted = prompt_formatted.format(request['subject'])
    
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
        prompt_template=prompt_template,      # 原始模板（带{}）
        prompt_formatted=prompt_formatted,    # 格式化后的
        subject=request.get('subject',''),
        target_true=_norm_target(target_true),
        target_new=_norm_target(target_new),
        locality_prompts=request.get('locality_prompts', []) or [],
        neighbor_prompts=request.get('neighbor_prompts', []) or [],
    ))
        
        # 在新数据上继续训练
        editor.train_multiedit(
            batch_items,
            epochs=getattr(hparams, 'v_num_grad_steps', 20),
            bs_forget=getattr(hparams, 'batch_size_forget', 3),
            bs_edit=getattr(hparams, 'batch_size_edit', 1),
            edit_per_forget=getattr(hparams, 'edit_per_forget', 5),
            shuffle=True
        )
        return model, editor
    
    # 第一次运行：初始化编辑器（这里也使用相同的转换逻辑）
    batch_items = []
    for request in requests:
    # 保存原始模板
        prompt_template = request['prompt']  # 原始模板，带{}
    
    # 格式化prompt用于训练
        prompt_formatted = request['prompt']
        if 'subject' in request:
            prompt_formatted = prompt_formatted.format(request['subject'])
    
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
        prompt_template=prompt_template,      # 原始模板（带{}）
        prompt_formatted=prompt_formatted,    # 格式化后的
        subject=request.get('subject',''),
        target_true=_norm_target(target_true),
        target_new=_norm_target(target_new),
        locality_prompts=request.get('locality_prompts', []) or [],
        neighbor_prompts=request.get('neighbor_prompts', []) or [],
    ))

    # Build target layers
    target_layer_names = []
    for layer_id in getattr(hparams, 'layers', [3, 4, 5, 6, 7, 8]):
        layer_name = getattr(hparams, 'rewrite_module_tmp', 'transformer.h.{}.mlp.c_proj').format(layer_id)
        target_layer_names.append(layer_name)
    
    print(f"[ADIT] Target layers: {target_layer_names}")

    # Initialize ADIT editor
    editor = ADITEditor(model, tok, target_layer_names, hparams)

    # Execute training
    print(f"[ADIT] Training on {len(batch_items)} requests...")
    editor.train_multiedit(
        batch_items,
        epochs=getattr(hparams, 'v_num_grad_steps', 20),
        bs_forget=getattr(hparams, 'batch_size_forget', 3),
        bs_edit=getattr(hparams, 'batch_size_edit', 1),
        edit_per_forget=getattr(hparams, 'edit_per_forget', 5),
        shuffle=True
    )

    # 保存编辑器引用以便后续使用
    model._adit_editor = editor
    
    print("[ADIT] Editing completed! Returning model and editor for evaluation.")
    return model, editor