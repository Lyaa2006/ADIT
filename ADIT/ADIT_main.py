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
    paraphrase_prompts:List[str]

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
    lambda_neighbor: float=0.2
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

# 在文件顶部，删除LoRALinear类，替换为：

# ---------- REPLACE Conv1DLoRA.__init__ AND Conv1DLoRA.forward START ----------
class Conv1DLoRA(nn.Module):
    """
    专门适配GPT-2 Conv1D层的LoRA实现，简洁版、去掉大量调试输出
    """
    def __init__(self, conv1d_module):
        super().__init__()

        if not (hasattr(conv1d_module, 'weight') and len(conv1d_module.weight.shape) == 2):
            raise ValueError(f"Module must be Conv1D-like with 2D weight, got {type(conv1d_module)}")

        # 保存原始Conv1D模块的引用
        self.original_module = conv1d_module

        # 获取维度信息
        self.in_features = conv1d_module.weight.shape[0]
        self.out_features = conv1d_module.weight.shape[1]

        # 检查是否有偏置
        self.has_bias = hasattr(conv1d_module, 'bias') and conv1d_module.bias is not None

        # LoRA适配器存储（字典结构）
        self.adapters = {}
        self.active = []
        self.runtime_weights = {}
        # token_masks keyed by adapter name; values are tensors
        self.token_masks = {}

    def add_adapter(self, name: str, rank: int, alpha: float = 8.0, trainable: bool = True):
        """添加一个LoRA适配器"""
        assert name not in self.adapters, f"Adapter {name} already exists"

        device = self.original_module.weight.device
        dtype = self.original_module.weight.dtype

        if trainable:
            A = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
            B = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B, a=math.sqrt(5))
        else:
            A = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype),
                           requires_grad=False)
            B = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype),
                           requires_grad=False)

        self.adapters[name] = {
            "A": A,
            "B": B,
            "alpha": alpha,
            "rank": rank,
            "trainable": trainable
        }

        # 注册参数（即便是不可训练的，我们也注册以便保存/加载）
        self.register_parameter(f"{name}_A", A)
        self.register_parameter(f"{name}_B", B)

    def bind_runtime(self, name: str, A: torch.Tensor, B: torch.Tensor):
        """绑定运行时权重（用于LE适配器）"""
        self.runtime_weights[name] = (A, B)

    def bind_token_mask(self, name: str, mask: torch.Tensor):
        """绑定token级别的mask，存储 clone 以避免后续外部修改影响内部"""
        # 保证 mask 是 float32 在正确设备上
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32, device=next(self.parameters()).device)
        mask = mask.to(dtype=torch.float32, device=next(self.parameters()).device)
        self.token_masks[name] = mask.clone()

    def clear_runtime(self):
        """清除运行时权重"""
        self.runtime_weights.clear()

    def clear_token_masks(self):
        """清除token masks"""
        self.token_masks.clear()

    def set_adapter_weights(self, name: str, A: torch.Tensor, B: torch.Tensor, alpha: float = None):
        """直接设置适配器权重（用于持久化）"""
        assert name in self.adapters, f"Adapter {name} not found"
        assert not self.adapters[name]["trainable"], "不能直接设置可训练适配器的权重"

        A_dst = self.adapters[name]["A"]
        B_dst = self.adapters[name]["B"]

        assert A.shape == A_dst.shape, f"A shape mismatch: {A.shape} vs {A_dst.shape}"
        assert B.shape == B_dst.shape, f"B shape mismatch: {B.shape} vs {B_dst.shape}"

        with torch.no_grad():
            A_dst.copy_(A.to(A_dst.device, dtype=A_dst.dtype))
            B_dst.copy_(B.to(B_dst.device, dtype=B_dst.dtype))
            if alpha is not None:
                self.adapters[name]["alpha"] = float(alpha)

    def enable_adapters(self, names):
        """激活指定的适配器"""
        self.active = list(names)

    def disable_all(self):
        """禁用所有适配器"""
        self.active = []

    def forward(self, x: torch.Tensor):
        """
        前向传播
        x: [batch_size, seq_len, in_features]
        返回: [batch_size, seq_len, out_features]
        """
        # 基础输出（与原始权重）
        weight = self.original_module.weight  # [out_features, in_features]
        if self.has_bias:
            bias = self.original_module.bias
            base_output = torch.einsum('bsi,io->bso', x, weight) + bias
        else:
            base_output = torch.einsum('bsi,io->bso', x, weight)

        # 应用激活的LoRA适配器
        for adapter_name in self.active:
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]

                # 获取权重（优先使用运行时权重）
                if adapter_name in self.runtime_weights:
                    A, B = self.runtime_weights[adapter_name]
                else:
                    A, B = adapter["A"], adapter["B"]

                # LoRA 计算: Δy = scale * (x @ B.T) @ A.T
                scale = adapter["alpha"] / max(1, adapter["rank"])

                # 第一步：x @ B.T -> [batch, seq, rank]
                intermediate = torch.einsum('bsi,ri->bsr', x, B)

                # 第二步：intermediate @ A.T -> [batch, seq, out]
                lora_output = torch.einsum('bsr,or->bso', intermediate, A)
                lora_output = scale * lora_output

                # 如果该 adapter 有 token mask，按位置应用
                if adapter_name in self.token_masks:
                    mask = self.token_masks[adapter_name]
                    # 标准化维度为 [batch, seq, 1]
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(-1)
                    elif mask.dim() == 3 and mask.shape[-1] != 1:
                        mask = mask[..., :1]
                    # 广播相乘（确保 device/dtype 匹配）
                    lora_output = lora_output * mask.to(dtype=lora_output.dtype, device=lora_output.device)

                base_output = base_output + lora_output

        return base_output
# ---------- REPLACE Conv1DLoRA.__init__ AND Conv1DLoRA.forward END ----------


def get_parent_and_attr(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]

def replace_conv1d_with_lora(model: nn.Module, target_layers: List[str],
                            lf_rank=8, le_rank=8, alpha=8.0):
    """
    将指定的Conv1D层替换为Conv1DLoRA层
    """
    lora_hosts = {}
    
    print(f"[ADIT] Looking for Conv1D layers: {target_layers}")
    
    for full_name, module in model.named_modules():
        if full_name in target_layers:
            print(f"[ADIT] Found target layer: {full_name}, type: {type(module)}")
            print(f"[ADIT] Module weight shape: {module.weight.shape}")
            
            # 检查是否已经是Conv1DLoRA
            if isinstance(module, Conv1DLoRA):
                print(f"[ADIT] Layer {full_name} is already Conv1DLoRA, reusing")
                lora_hosts[full_name] = module
                continue
            
            # 检查是否是Conv1D层（GPT-2风格）
            if hasattr(module, 'weight') and len(module.weight.shape) == 2:
                print(f"[ADIT] Replacing Conv1D layer: {full_name}")
                print(module.weight.shape)
               
                # 获取父模块和属性名
                parent, attr = get_parent_and_attr(model, full_name)
                
                # 创建Conv1DLoRA包装器
                host = Conv1DLoRA(module)
                
                # 添加两个适配器：LF（可训练）和 LE（不可训练）
                host.add_adapter("LF", rank=lf_rank, alpha=alpha, trainable=True)
                host.add_adapter("LE", rank=le_rank, alpha=alpha, trainable=False)
                
                # 移动到正确的设备
                host.to(module.weight.device, dtype=module.weight.dtype)
                
                # 替换原模块
                setattr(parent, attr, host)
                lora_hosts[full_name] = host
            else:
                print(f"[WARNING] Layer {full_name} is not a Conv1D layer")
    
    print(f"[ADIT] Successfully replaced {len(lora_hosts)} layers with Conv1DLoRA")
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
    def __init__(self, lora_hosts: Dict[str, Conv1DLoRA], ctx_dim: int, rank: int = 8, hidden: int = 256):
        super().__init__()
        self.rank = rank
        self.ctx_dim = ctx_dim
        self.layer_names = list(lora_hosts.keys())
        self.base_A = nn.ParameterDict()
        self.base_B = nn.ParameterDict()
        self.heads = nn.ModuleDict()
        
        for name, host in lora_hosts.items():
            # 验证适配器存在且秩匹配
            assert "LE" in host.adapters, f"LE adapter not found in layer {name}"
            assert host.adapters["LE"]["rank"] == rank, f"Rank mismatch at {name}: {host.adapters['LE']['rank']} != {rank}"
            
            # 获取维度信息
            out_features = host.out_features
            in_features = host.in_features
            
            # 初始化基础权重
            A0 = nn.Parameter(torch.empty(out_features, rank))
            B0 = nn.Parameter(torch.empty(rank, in_features))
            nn.init.kaiming_uniform_(A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(B0, a=math.sqrt(5))
            
            # 使用安全的关键字
            key = name.replace(".", "_")
            self.base_A[key] = A0
            self.base_B[key] = B0
            
            # 门控网络
            self.heads[key] = PerLayerGate(ctx_dim, rank, hidden)
            
    def forward(self, ctx: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        
        for name in self.layer_names:
            key = name.replace(".", "_")
            
            # 生成门控参数
            gA, gB = self.heads[key](ctx)  # gA: [1, rank], gB: [1, rank]
            
            # 获取基础权重
            A0 = self.base_A[key]  # [out_features, rank]
            B0 = self.base_B[key]  # [rank, in_features]
            
            # 应用门控
            A = A0 * gA.view(1, -1)  # [out_features, rank] * [1, rank]
            B = B0 * gB.view(-1, 1)  # [rank, in_features] * [rank, 1]
            
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
        self.lora_hosts = replace_conv1d_with_lora(
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
        self._ks_proj_map = {}  

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
                if templates is None:
                    templates=[]
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
        
    def _get_or_create_ks_proj(self, ks_dim: int) -> torch.Tensor:
        """
        返回把 ks_dim -> ctx_dim 的投影矩阵（tensor），若不存在则创建并缓存。
        该矩阵是固定的随机矩阵（非可训练参数）。
        """
        ctx_dim = int(self.hparams.ctx_dim)
        device = self.hparams.device if isinstance(self.hparams.device, str) else self.hparams.device

        # 如果已经存在并形状匹配，直接返回
        existing = self._ks_proj_map.get(ks_dim, None)
        if existing is not None:
            # 确保在正确 device
            if existing.device != torch.device(device):
                existing = existing.to(device)
                self._ks_proj_map[ks_dim] = existing
            return existing

        # 否则创建新的随机投影矩阵
        # 初始化标准差 scaled by sqrt(1/ks_dim)
        std = 1.0 / (ks_dim ** 0.5)
        P = torch.randn(ks_dim, ctx_dim, device=device, dtype=torch.float32) * std

        # 存入缓存并返回
        self._ks_proj_map[ks_dim] = P
        return P


    # ---------- REPLACE get_specific_context_templates START ----------
    def get_specific_context_templates(self, request):
        """为特定事实类型生成相关模板"""
        # 尝试直接使用 paraphrase_prompts 字段（counterfact 风格）
        paraphrases = request.get("paraphrase_prompts", None)
        templates = []

        if paraphrases:
            # 统一转换为列表
            if isinstance(paraphrases, str):
                paraphrases = [paraphrases]
            elif isinstance(paraphrases, (list, tuple)):
                paraphrases = [p for p in paraphrases if isinstance(p, str)]
            else:
                paraphrases = []

            subject = request.get("subject", "").strip()
            
            # 处理每个paraphrase模板
            for p in paraphrases:
                p = p.strip()
                if not p:
                    continue
                
                # 检查是否已经包含 {}
                if "{}" in p:
                    templates.append(p)
                    continue
                
                # 如果包含 subject，替换为 {}
                if subject and subject in p:
                    temp = p.replace(subject, "{}")
                    templates.append(temp)
                else:
                    # 如果不包含 subject 也不包含 {}，直接使用（假设用户写的是模板句）
                    templates.append(p)
        
            # 去重
            if templates:
                seen = set()
                unique = []
                for t in templates:
                    if t not in seen:
                        unique.append(t)
                        seen.add(t)
                
                if unique:
                    # 调试输出
                    print(f"[DEBUG] Using paraphrase templates for subject '{subject}':")
                    for i, t in enumerate(unique[:5]):  # 只显示前5个
                        print(f"  Template {i}: '{t}'")
                    if len(unique) > 5:
                        print(f"  ... and {len(unique)-5} more")
                    
                    return unique

        # 如果 paraphrase_prompts 不存在或处理失败，fallback 到原来的手写变体逻辑
        original_prompt = request.get('prompt', '') or ''
        lower = original_prompt.lower()

        if "mother tongue" in lower or "language" in lower:
            templates = [
                "{}'s native language is",
                "The primary language of {} is",
                "What language does {} speak?",
                "{} speaks",
            ]
        elif "capital" in lower:
            templates = [
                "{}'s capital city is",
                "The capital city of {} is",
                "What is the capital of {}?",
            ]
        elif "born" in lower or "birth" in lower:
            templates = [
                "{} was born in",
                "The birthplace of {} is",
                "{} originated from",
                "Where was {} born?",
            ]
        else:
            # 通用回退模板
            templates = [
                "{} is known for",
                "About {}:",
                "Facts about {}:",
                "The thing about {} is",
            ]
        
        # 调试输出
        subject = request.get('subject', '').strip()
        if subject:
            print(f"[DEBUG] Using fallback templates for subject '{subject}':")
            for i, t in enumerate(templates[:3]):
                print(f"  Template {i}: '{t}'")
        
        return templates


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
                ks_vector = self.vector_cache[cache_key]['ks']  # 可能在 CPU 上
                # 确保 ks_vector 在 device 上并为 float32
                ks_vector = ks_vector.to(self.hparams.device).to(dtype=torch.float32)

                # 如果 ks_vector 长度等于 ctx_dim，直接用；否则通过随机投影映射到 ctx_dim
                ks_dim = ks_vector.numel()
                ctx_dim = int(self.hparams.ctx_dim)

                if ks_dim == ctx_dim:
                    ks_mapped = ks_vector
                else:
                    P = self._get_or_create_ks_proj(ks_dim)  # [ks_dim, ctx_dim]
                    # ks_vector: [ks_dim]; mapped: [ctx_dim]
                    ks_mapped = torch.matmul(ks_vector.view(1, ks_dim), P).view(-1)

                # 融合（注意类型/设备一致）
                alpha = float(self.hparams.vector_guidance_weight)
                guided_ctx = base_ctx + alpha * ks_mapped.unsqueeze(0)
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
    def compute_neighbor_loss(self, batch_item: BatchItem, le_weights: Dict) -> torch.Tensor:
        """计算邻居prompts上的稳定性损失"""
        if not batch_item.neighbor_prompts:
            return torch.tensor(0.0, device=self.hparams.device)
        
        total_loss = 0.0
        neighbor_count = 0
        
        # 为每个邻居prompt计算损失
        for neighbor_prompt in batch_item.neighbor_prompts:
            if not neighbor_prompt or not isinstance(neighbor_prompt, str):
                continue
            
            try:
                # 1. 注入LE权重到所有LoRA层
                for layer_name, host in self.lora_hosts.items():
                    if layer_name in le_weights:
                        A, B = le_weights[layer_name]
                        host.bind_runtime("LE", A, B)
                
                # 2. 获取基础模型的输出（无编辑）
                self._deactivate_all()
                with torch.no_grad():
                    ids_base, attn_base = self.tok.encode(neighbor_prompt, self.hparams.device)
                    if attn_base is None:
                        attn_base = torch.ones_like(ids_base)
                    logits_base = self.base_model(input_ids=ids_base, attention_mask=attn_base).logits
                
                # 3. 获取编辑后的输出（激活LE）
                self._activate(["LE"])
                logits_edit = self.base_model(input_ids=ids_base, attention_mask=attn_base).logits
                
                # 4. 计算KL散度损失
                P = F.softmax(logits_base, dim=-1)  # 基础分布
                Q_log = F.log_softmax(logits_edit, dim=-1)  # 编辑后log分布
                loss = F.kl_div(Q_log, P, reduction="batchmean", log_target=False)
                
                total_loss += loss
                neighbor_count += 1
                
            except Exception as e:
                print(f"[WARN] Failed to compute neighbor loss for prompt '{neighbor_prompt[:50]}...': {e}")
                continue
            
            finally:
                # 清理运行时权重
                for host in self.lora_hosts.values():
                    host.clear_runtime()
                self._deactivate_all()
        
        # 返回平均损失
        if neighbor_count > 0:
            return total_loss / neighbor_count
        return torch.tensor(0.0, device=self.hparams.device)

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

    # ---------- REPLACE entire step_edit_batch_with_guidance body START ----------
    def step_edit_batch_with_guidance(self, items: list, requests: List[Dict], clip_norm: float = 1.0):
        """使用向量指导的编辑步骤（包含邻居损失）"""
        self.opt_le.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_ce = 0.0
        log_vector = 0.0
        log_neighbor = 0.0  # 新增：邻居损失日志

        for i, (bi, request) in enumerate(zip(items, requests)):
            # 使用向量指导的上下文（hyper -> LE 权重）
            ctx = self.build_guided_context(bi, request)
            le_weights = self.hyper(ctx)

            # 将 LE 权重注入每个 host
            for name, host in self.lora_hosts.items():
                A, B = le_weights[name]
                host.bind_runtime("LE", A, B)

            # 编码用于 LM 训练 / 生成的 ids
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt_formatted, bi.target_new, self.hparams.device)

            # 激活 LE 并计算语言模型损失
            self._deactivate_all()
            self._activate(["LE"])
            ce_new = lm_ce_loss(self.base_model, ids, attn, labels)

            # 向量对齐损失
            vector_loss = self.compute_vector_alignment_loss(bi, request) if self.hparams.use_vector_guidance else 0.0

            # 新增：邻居损失
            neighbor_loss = self.compute_neighbor_loss(bi, le_weights)

            # 总损失 = CE损失 + 向量对齐损失 + 邻居损失
            loss_i = (
                ce_new + 
                self.hparams.vector_alignment_weight * vector_loss + 
                self.hparams.lambda_neighbor * neighbor_loss
            )
            
            total_loss = total_loss + loss_i

            log_ce += float(ce_new.detach().cpu())
            log_vector += float(vector_loss.detach().cpu()) if isinstance(vector_loss, torch.Tensor) else 0.0
            log_neighbor += float(neighbor_loss.detach().cpu())  # 记录邻居损失

            # 清理运行时权重
            for host in self.lora_hosts.values():
                host.clear_runtime()
                host.clear_token_masks()
            self._deactivate_all()

        total_loss = total_loss / max(1, len(items))
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), clip_norm)
        self.opt_le.step()

        # 返回包含邻居损失的指标
        metrics = {
            "edit/ce_new": log_ce / max(1, len(items)),
            "edit/neighbor_loss": log_neighbor / max(1, len(items))  # 新增指标
        }
        if self.hparams.use_vector_guidance:
            metrics["edit/vector_loss"] = log_vector / max(1, len(items))
        return metrics


    # ---------- REPLACE compute_vector_alignment_loss START ----------
    def compute_vector_alignment_loss(self, batch: BatchItem, request: Dict):
        """计算向量对齐损失（ROME-style，基于 subject 的实际 token 位置）"""
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

            # 目标向量（来自 compute_z）
            target_vector = self.vector_cache[cache_key]['z'].to(self.hparams.device)

            # 要监控的层名称
            layer_name = f"transformer.h.{layer}.mlp.c_proj"

            # 准备输入：使用 batch.prompt_formatted（已经 format 了 subject）
            prompt_text = batch.prompt_formatted
            enc = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.hparams.device)
            attn_mask = enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(self.hparams.device)

            # 找到 subject 在 prompt_text 中的 token 索引（使用你的 find_fact_lookup_idx 工具）
            # 这里我们调用 find_fact_lookup_idx，但传入 prompt_formatted，确保定位的是实际输入位置
            try:
                lookup_idx = find_fact_lookup_idx(
                    batch.prompt_template,  # 传入 template 以保持原有策略（某些实现可能需要）
                    subject,
                    self.tokenizer,
                    self.hparams.fact_token,
                    verbose=False
                )
            except Exception:
                # 若 find_fact_lookup_idx 不是对 formatted prompt 工作良好，则退回到在 input_ids 中查找 token span:
                lookup_idx = None

            # 如果 find_fact_lookup_idx 未能给出稳健位置，我们尝试在 tokenizer 的 input_ids 里寻找 subject 的 token span
            if lookup_idx is None:
                # tokenise subject and search as subsequence
                subj_ids = self.tokenizer(subject, add_special_tokens=False)["input_ids"]
                subj_len = len(subj_ids)
                seq = input_ids[0].tolist()
                found = -1
                for i in range(0, len(seq) - subj_len + 1):
                    if seq[i:i + subj_len] == subj_ids:
                        found = i
                        break
                if found >= 0:
                    lookup_idx = found
                else:
                    # fallback: use last token
                    lookup_idx = input_ids.shape[1] - 1

            # 清理 runtime 状态，确保我们测量的是在只注入 LE 权重时的输出
            for host in self.lora_hosts.values():
                host.clear_runtime()
                host.clear_token_masks()

            # 临时激活 LE：将 hyper 网络在当前 ctx 上生成的权重注入（注意这里不用 mask）
            self._activate(["LE"])
            with torch.no_grad():
                ctx = self.build_guided_context(batch, request)
                le_weights = self.hyper(ctx)
                # Bind runtime weights only for this measurement
                if layer_name in self.lora_hosts:
                    A, B = le_weights[layer_name]
                    self.lora_hosts[layer_name].bind_runtime("LE", A, B)

                # Trace layer output
                with nethook.TraceDict(self.base_model, layers=[layer_name], retain_output=True) as tr:
                    _ = self.base_model(input_ids=input_ids, attention_mask=attn_mask)
                    if layer_name in tr:
                        layer_output = tr[layer_name].output
                        if isinstance(layer_output, tuple):
                            layer_output = layer_output[0]

                        # layer_output: [batch, seq_len, hidden]
                        batch_size, seq_len, hidden_size = layer_output.shape

                        # bounds check for lookup_idx
                        if lookup_idx >= seq_len:
                            actual_idx = seq_len - 1
                        else:
                            actual_idx = lookup_idx

                        actual_vector = layer_output[0, actual_idx, :]

                        loss = F.mse_loss(actual_vector, target_vector)
                        total_loss += loss
                        layer_count += 1

            # 清理 runtime injection for this layer
            if layer_name in self.lora_hosts:
                self.lora_hosts[layer_name].clear_runtime()
                self.lora_hosts[layer_name].clear_token_masks()

            self._deactivate_all()

        if layer_count > 0:
            return total_loss / layer_count
        return torch.tensor(0.0, device=self.hparams.device)
# ---------- REPLACE compute_vector_alignment_loss END ----------


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
    
    print(requests)
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
    paraphrase_prompts=request.get('paraphrase_prompts',[]) or [],
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
    paraphrase_prompts=request.get('paraphrase_prompts',[]) or [],
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
