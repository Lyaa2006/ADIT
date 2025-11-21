
"""
M&M-AdvEdit: MACE-Forget × MEMIT-Localized LoRA + Adversarial Training (Demo)
Author: ChatGPT (demo scaffold)
Notes:
- This is a minimal end‑to‑end scaffold showing the overall flow.
- It targets HuggingFace causal LMs (e.g., LLaMA/Qwen/GPT-J/GPT-2), but uses generic PyTorch hooks.
- Replace the regex/name filters in `MEMITLocator` with model‑specific layer names for production.
- The hypernetwork generates LE (edit) LoRA weights via low‑rank gating over per‑layer dictionaries.
- LF (forget) LoRA is trainable (standard LoRA parameters). LE LoRA is per‑batch generated.
- The training loop alternates: Forget step (maximize PPL for old object + locality control),
  then Edit step (minimize PPL for new object + KL locality/specificity + optional orthogonality).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from .ADIT_hparams import ADITHyperParams  # 如果你已经创建了
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["HUGGINGFACE_CO_URL_HOME"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ================================
# LoRA wrapper (multi-adapter host)
# ================================

class LoRALinear(nn.Module):
    def __init__(self, module):
        super().__init__()
        
        assert hasattr(module, 'weight'), "Module must have weight attribute"
        weight_shape = module.weight.shape
    # GPT-2 Conv1D 权重是 [in_features, out_features]
    # 但 F.linear 需要 [out_features, in_features]
        if len(weight_shape) == 2:
            self.in_features = weight_shape[0]  # 6400
            self.out_features = weight_shape[1] # 1600
        else:
            raise ValueError(f"Unsupported weight shape: {weight_shape}")
        if len(weight_shape) == 2:
            # GPT-2 Conv1D 权重形状是 [in_features, out_features]
            self.in_features = weight_shape[0]  # 6400
            self.out_features = weight_shape[1] # 1600
        else:
            raise ValueError(f"Unsupported weight shape: {weight_shape}")
        
        self.bias = hasattr(module, 'bias') and module.bias is not None
        self.original_module = module
        self.adapters = {}
        self.active = []
        self.runtime = {}
        self.runtime_mask = {}
        
    def bind_runtime(self, name: str, A: torch.Tensor, B: torch.Tensor):
            self.runtime[name] = (A, B)

    def clear_runtime(self):
            self.runtime.clear()
    
    def bind_token_mask(self, name: str, mask: torch.Tensor):
        """
        绑定一个运行期 mask（只在 mask==1 的位置施加该适配器的增量）
        mask 形状与 Linear 的广播一致：
        - GPT-J 的 MLP linear 输入是 [B, T, H]，这里用 [B, T, 1] 即可自动广播
        """
        self.runtime_mask[name] = mask

    def clear_token_masks(self):
        self.runtime_mask.clear()

    def add_adapter(self, name: str, rank: int, alpha: float = 8.0, trainable: bool = True):
        assert name not in self.adapters, f"Adapter {name} already exists"
        dev, dt = self.original_module.weight.device, self.original_module.weight.dtype
        
        if trainable:
            # 修正：A: [out_features, rank], B: [rank, in_features]
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
    # 转置权重：从 [6400, 1600] 变为 [1600, 6400]
        weight = self.original_module.weight.t()  # [1600, 6400]
    
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
    

# ============================
# Utility: replace target lines
# ============================

import math

def get_parent_and_attr(model: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():      # 支持 transformer.h.0.mlp.fc_out 这样的路径
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def replace_linear_with_lora(model: nn.Module, target_layers: List[str],
                                   lf_rank=8, le_rank=8, alpha=8.0):
    lora_hosts = {}
    
    for full_name, module in model.named_modules():
        if full_name in target_layers and hasattr(module, 'weight') and len(module.weight.shape) == 2:
            parent, attr = get_parent_and_attr(model, full_name)
            host = LoRALinear(module)
            host.add_adapter("LF", rank=lf_rank, alpha=alpha, trainable=True)
            host.add_adapter("LE", rank=le_rank, alpha=alpha, trainable=False)
            host.to(module.weight.device, dtype=module.weight.dtype)
            setattr(parent, attr, host)
            lora_hosts[full_name] = host
    
    print(f"[Locator] matched {len(lora_hosts)} layers")
    return lora_hosts
# ========================
# MEMIT-like layer locator
# ========================

class MEMITLocator:
    """
    Very light "locator": you should replace the regex below for your backbone.
    For LLaMA/Qwen: target MLP down_proj or out_proj is common.
    For GPT2/GPT-J: target mlp.c_proj or fc_out is common.
    """
    def __init__(self, pattern: str = r"(mlp\.(down_proj|c_proj))|(feed_forward\.(o|out)_proj)"):
        self.pattern = re.compile(pattern)

    @staticmethod
    def from_layer_ids(ids, fmt: str):
        """
        ids: 例如 [10,12,14,16,18,20,22]
        fmt: 例如 "transformer.h.{i}.mlp.fc_out" / "transformer.h.{i}.mlp.c_proj" /
             "model.layers.{i}.mlp.down_proj"
        """
        import re
    # 修复：使用 {} 格式而不是 {i} 格式
        pats = [re.escape(fmt.format(i)) for i in ids]
        full = r"(" + r"|".join(pats) + r")$"
        print(f"[DEBUG] Generated layer pattern for {len(ids)} layers: {full}")
        return MEMITLocator(full)

    def select(self, model: nn.Module) -> re.Pattern:
        """
        Return a compiled regex that matches target Linear modules.
        You can also implement a true causal-tracing selector here.
        """
        return self.pattern

# =====================
# HyperNetwork (for LE)
# =====================

class PerLayerGate(nn.Module):
    """Per-layer gating head: maps context -> rank gates (gA, gB)."""
    def __init__(self, ctx_dim: int, rank: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * rank),  # gA || gB
        )

    def forward(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.net(ctx)  # [B, 2r]
        r2 = g.shape[-1] // 2
        gA = torch.tanh(g[..., :r2])   # [-1, r]
        gB = torch.tanh(g[..., r2:])   # [-1, r]
        return gA, gB

class HyperNetwork(nn.Module):
    """
    Generates LE LoRA weights via gating over per-layer base dictionaries.
    For each selected layer l, we maintain base_A[l] (out x r) and base_B[l] (r x in).
    Given context c, we produce gates gA_l, gB_l in R^r, forming:
      A_l(c) = base_A[l] * diag(gA_l),  B_l(c) = diag(gB_l) * base_B[l]
    """
    def __init__(self, lora_hosts: Dict[str, LoRALinear], ctx_dim: int, rank: int = 8, hidden: int = 256):
        super().__init__()
        self.rank = rank
        self.ctx_dim = ctx_dim

        # Register per-layer bases
        self.layer_names = list(lora_hosts.keys())
        self.base_A = nn.ParameterDict()
        self.base_B = nn.ParameterDict()
        self.heads = nn.ModuleDict()

        for name, host in lora_hosts.items():
            # Ensure LE rank matches
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
        """
        Returns a dict: {layer_name: (A, B)} for LE.
        ctx: [B, ctx_dim]
        For simplicity, we support B=1 (per-sample personalization). Batch>1 is also ok;
        caller must loop/adapt to set weights per-sample if needed.
        """
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name in self.layer_names:
            key = name.replace(".", "_")
            gA, gB = self.heads[key](ctx)     # [B, r], [B, r]
            A0 = self.base_A[key]             # [out, r]
            B0 = self.base_B[key]             # [r, in]
            # Produce weights per batch item; here we use the first item (B==1)
            gA1 = gA[0].view(1, -1)           # [1, r]
            gB1 = gB[0].view(-1, 1)           # [r, 1]
            A = A0 * gA1                       # [out, r]
            B = gB1 * B0                       # [r, in]
            out[name] = (A, B)
        return out

# ===============================
# Interface to backbone and tokens
# ===============================
import json, csv, random
from typing import Iterable

def _norm_target(s: str) -> str:
    s = s or ""
    s = s.rstrip("\n")
    return s if s.startswith(" ") else " " + s  # 目标前导空格（tokenization 友好）

def load_batchitems_from_jsonl(path: str) -> list:
    """
    JSONL 每行一个对象：
    {
      "prompt": "The capital of France is",
      "target_true": " Paris",
      "target_new": " Lyon",
      "locality_prompts": ["...","..."],
      "neighbor_prompts": ["...","..."]
    }
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            items.append(BatchItem(
                prompt=o["prompt"],
                target_true=_norm_target(o.get("target_true","")),
                target_new=_norm_target(o.get("target_new","")),
                locality_prompts=o.get("locality_prompts", []) or [],
                neighbor_prompts=o.get("neighbor_prompts", []) or [],
            ))
    return items

def load_batchitems_from_csv(path: str) -> list:
    """
    CSV 需包含列：prompt,target_true,target_new
    可选：locality_prompts,neighbor_prompts（用 || 分隔多个）
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            def _split(s): 
                return [x for x in (s or "").split("||") if x.strip()]
            items.append(BatchItem(
                prompt=r["prompt"],
                target_true=_norm_target(r.get("target_true","")),
                target_new=_norm_target(r.get("target_new","")),
                locality_prompts=_split(r.get("locality_prompts","")),
                neighbor_prompts=_split(r.get("neighbor_prompts","")),
            ))
    return items

def chunks(lst: list, bs: int) -> Iterable[list]:
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]


@dataclass
class BatchItem:
    prompt: str
    target_true: str       # original object (to be forgotten by LF)
    target_new: str        # new object (to be installed by LE)
    locality_prompts: List[str]  # unrelated sentences for drawdown
    neighbor_prompts: List[str]  # similar subjects for specificity

def build_target_token_mask(tokenizer, prompt: str, target: str, device) -> torch.Tensor:
    """
    返回一个形如 [1, T, 1] 的 mask，T 为 prompt+target 的长度；
    只有 target 的 token 位置为 1，其它为 0。
    """
    # 注意：为了和 encode_label_for_target 对齐，这里用相同的拼接与分词。
    full_ids = tokenizer(prompt + target, return_tensors="pt")["input_ids"][0]
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    T = full_ids.size(0)
    P = prompt_ids.size(0)
    mask = torch.zeros(1, T, 1, dtype=torch.float32, device=device)
    mask[:, P:, :] = 1.0
    return mask


class TokenHelper:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def encode_label_for_target(self, prompt: str, target: str, device):
        """
        Build inputs so that loss is computed ONLY on target tokens.
        """
        full = prompt + target
        enc = self.tok(full, return_tensors="pt")
        ids = enc["input_ids"]
        attn = enc.get("attention_mask", torch.ones_like(ids))
        # Compute prompt length
        pl = len(self.tok(prompt)["input_ids"])
        labels = ids.clone()
        labels[:, :pl] = -100  # ignore prompt tokens
        return ids.to(device), attn.to(device), labels.to(device)

    def encode(self, text: str, device):
        enc = self.tok(text, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", None)
        if attn is None:
            attn = torch.ones_like(ids)
        attn = attn.to(device)              # ✅ 关键：mask 也迁到同一 device
        return ids, attn
# =====================
# Losses / divergences
# =====================

def lm_ce_loss(model, ids, attn, labels):
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    return out.loss

def kl_divergence_on_prompt(model_p, model_q, ids, attn=None, temp: float = 1.0):
    # ✅ 保险：无论上游如何，这里统一迁到 model_q 的 device
    dev = next(model_q.parameters()).device
    ids = ids.to(dev)
    if attn is not None:
        attn = attn.to(dev)

    with torch.no_grad():
        lp = model_p(input_ids=ids, attention_mask=attn).logits / temp
        P = F.log_softmax(lp, dim=-1)
        Pp = P.exp()
    lq = model_q(input_ids=ids, attention_mask=attn).logits / temp
    Q = F.log_softmax(lq, dim=-1)
    kl = F.kl_div(Q, Pp, reduction="batchmean", log_target=False)
    return kl

def orthogonality_loss(lora_hosts: Dict[str, LoRALinear]):
    """
    Encourage LF and LE subspaces to be orthogonal: ||A_LF^T A_LE||_F^2 + ||B_LF^T B_LE||_F^2.
    """
    loss = 0.0
    for host in lora_hosts.values():
        A_lf = host.adapters["LF"]["A"]
        B_lf = host.adapters["LF"]["B"]
        A_le = host.adapters["LE"]["A"]
        B_le = host.adapters["LE"]["B"]
        loss = loss + (A_lf.T @ A_le).pow(2).sum() + (B_lf @ B_le.T).pow(2).sum()
    return loss

# ==========================
# High-level training runner
# ==========================

@dataclass
class AdvEditConfig:
    lf_rank: int = 8
    le_rank: int = 16
    alpha: float = 16.0
    ctx_dim: int = 768        # match your backbone hidden size or pooled emb size
    lr_lf: float = 5e-4
    lr_le: float = 5e-4
    lambda_loc: float = 1.0
    lambda_kl: float = 1.0
    lambda_spec: float = 0.5
    lambda_orth: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class AdvEditor:
    def __init__(self, model, tokenizer, locator: MEMITLocator, cfg: AdvEditConfig):
        self.base_model = model.to(cfg.device)
        self.tokenizer = tokenizer
        self.tok = TokenHelper(tokenizer)
        self.cfg = cfg

        # 1) Locate MLP projection layers to patch
        target_layer_names = []
        if hasattr(cfg, 'layers') and hasattr(cfg, 'rewrite_module_tmp'):
            for layer_id in cfg.layers:
                layer_name = cfg.rewrite_module_tmp.format(layer_id)
                target_layer_names.append(layer_name)
        
        print(f"[DEBUG] Direct target layers: {target_layer_names}")
        
        # 使用直接替换函数
        self.lora_hosts = replace_linear_with_lora(
            self.base_model, target_layer_names, lf_rank=cfg.lf_rank, le_rank=cfg.le_rank, alpha=cfg.alpha
        )

        # 2) Hypernetwork for LE
        self.hyper = HyperNetwork(self.lora_hosts, ctx_dim=cfg.ctx_dim, rank=cfg.le_rank).to(cfg.device)

        # 3) Separate copies for KL baselines (frozen)
        self.base_model.eval()
        # for p in self.base_model.parameters(): p.requires_grad = False
        self.model_lf = self.base_model # same module with LF adapters trainable
        self.model_le = self.base_model # same physical module; we toggle adapters at run-time

        # # Collect trainable params
        # self.opt_lf = torch.optim.AdamW(
        #     [p for n,p in self.model_lf.named_parameters() if n.endswith("_A") or n.endswith("_B") and "LF" in n] , lr=cfg.lr_lf
        # )
        # 1) 冻结整座模型（包含 LoRA 在内的所有参数先关掉）
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 2) 仅重新打开 LF 的 A/B（LE 是 buffer，不训练）
        for host in self.lora_hosts.values():
            host.adapters["LF"]["A"].requires_grad = True
            host.adapters["LF"]["B"].requires_grad = True
            # 可选：确保基座权重不训练
            if hasattr(host, "weight"):
                host.weight.requires_grad = False
            if getattr(host, "bias_param", None) is not None:
                host.bias_param.requires_grad = False

        # 3) 优化器直接拿 LF 的 A/B
        lf_params = []
        for host in self.lora_hosts.values():
            lf_params += [host.adapters["LF"]["A"], host.adapters["LF"]["B"]]
        if not lf_params:
            raise RuntimeError("Found LoRA hosts but no LF params; check adapter creation.")
        self.opt_lf = torch.optim.AdamW(lf_params, lr=self.cfg.lr_lf)
        self.opt_le = torch.optim.AdamW(self.hyper.parameters(), lr=cfg.lr_le)

    # ---------- Adapter toggles ----------

    # def _activate(self, adapters: List[str]):
    #     for host in self.lora_hosts.values():
    #         host.enable_adapters(adapters)

    # 原: def _activate(self, model, adapters: List[str]):
    def _activate(self, adapters, *_, **__):
        """开启一组 LoRA 适配器名（如 ['LF'] 或 ['LE']），忽略多余位置/关键字参数。"""
        for host in self.lora_hosts.values():
            host.enable_adapters(adapters)

    # 原: def _deactivate_all(self, model):
    def _deactivate_all(self, *_, **__):
        """关闭所有 LoRA 适配器，忽略多余位置/关键字参数。"""
        for host in self.lora_hosts.values():
            host.disable_all()


    # ---------- Context builder (toy) ----------

    def build_context(self, batch: BatchItem) -> torch.Tensor:
        """
        A simple context: CLS-like pooled embedding of prompt + target_new (avg of token embeddings).
        In practice you may use the S-token key-vector or hidden state at subject position.
        """
        with torch.no_grad():
            ids, _ = self.tok.encode(batch.prompt + batch.target_new, self.cfg.device)
            # Use embedding matrix if available
            if hasattr(self.base_model, "transformer") and hasattr(self.base_model.transformer, "wte"):
                emb = self.base_model.transformer.wte(ids)  # GPT-2 style
            elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
                emb = self.base_model.model.embed_tokens(ids)  # LLaMA/Qwen style
            else:
                # Fallback: random context
                emb = torch.randn(1, ids.size(1), self.cfg.ctx_dim, device=self.cfg.device)
            ctx = emb.mean(dim=1)  # [1, hidden]
        return ctx

    # ---------- Step: Forget (maximize PPL for old object) ----------

    def step_forget(self, batch: BatchItem):
        self._deactivate_all(self.model_lf)
        # Activate LF only
        self._activate(self.model_lf, ["LF"])
        self.model_lf.train()

        # Edit CE on true object: we want to INCREASE it => minimize (-CE)
        ids, attn, labels = self.tok.encode_label_for_target(batch.prompt, batch.target_true, self.cfg.device)
        ce_true = lm_ce_loss(self.model_lf, ids, attn, labels)

        # Locality KL: baseline (no adapters) vs LF on unrelated prompts (keep small)
        self._deactivate_all(self.model_lf)  # baseline off
        kl_sum = 0.0
        for t in batch.locality_prompts[:2]:  # keep small for demo
            ids_loc, attn_loc = self.tok.encode(t, self.cfg.device)
            # P (baseline): run with no adapters; Q (LF): run with LF
            self._deactivate_all(self.model_lf)
            P_model = self.model_lf
            self._activate(["LF"])
            Q_model = self.model_lf
            kl_sum = kl_sum + kl_divergence_on_prompt(P_model, Q_model, ids_loc, attn_loc)
        loss = -ce_true + self.cfg.lambda_loc * kl_sum
        reg = sum((h.adapters["LF"]["A"]**2).sum()+(h.adapters["LF"]["B"]**2).sum() for h in self.lora_hosts.values())
        loss = loss + 1e-4 * reg    # λ_l2 ~ 1e-4

        self.opt_lf.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_lf.step()

        # Cleanup
        self._deactivate_all(self.model_lf)
        return {"forget/neg_ce_true": float((-ce_true).detach().cpu()), "forget/kl_loc": float(kl_sum.detach().cpu())}

    # ---------- Step: Edit (minimize PPL for new object) ----------
    def step_edit(self, batch: BatchItem):
        # 1) 生成 LE（可微）
        ctx = self.build_context(batch)            # 仍可用 S-token 隐状态（与 MEMIT compute k* 一致的思路）
        le_weights = self.hyper(ctx)
        for name, host in self.lora_hosts.items():
            A, B = le_weights[name]
            host.bind_runtime("LE", A, B)

        # 2) 只在 “编辑对象 token” 时间步上启用 LE：构造掩码并绑定
        ids, attn, labels = self.tok.encode_label_for_target(batch.prompt, batch.target_new, self.cfg.device)
        tok_mask = build_target_token_mask(self.tokenizer, batch.prompt, batch.target_new, self.cfg.device)  # [1,T,1]
        for host in self.lora_hosts.values():
            host.bind_token_mask("LE", tok_mask)

        # 3) 前向计算：仅用目标 tokens 的 CE（labels 已屏蔽非目标位）
        self._deactivate_all()
        self._activate(["LE"])
        ce_new = lm_ce_loss(self.model_le, ids, attn, labels)

        self.opt_le.zero_grad(set_to_none=True)
        ce_new.backward()
        torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), 1.0)
        self.opt_le.step()

        # 4) 清理运行期绑定
        for host in self.lora_hosts.values():
            host.clear_runtime()
            host.clear_token_masks()
        self._deactivate_all()

        return {"edit/ce_new": float(ce_new.detach().cpu())}

    # def step_edit(self, batch: BatchItem):
    #     self._deactivate_all(self.model_le)

    #     # Generate LE weights from hypernetwork
    #     ctx = self.build_context(batch)  # [1, ctx_dim]
    #     le_weights = self.hyper(ctx)     # dict: name -> (A,B)
    #     # for name, host in self.lora_hosts.items():
    #     #     A, B = le_weights[name]
    #     #     host.set_adapter_weights("LE", A, B)
    #     for name, host in self.lora_hosts.items():
    #         A, B = le_weights[name]
    #         host.bind_runtime("LE", A, B) # ✅ 可微绑定
    #     # Activate LE only
    #     self._activate(self.model_le, ["LE"])
    #     self.model_le.train()

    #     # CE on new target (we want to minimize)
    #     ids, attn, labels = self.tok.encode_label_for_target(batch.prompt, batch.target_new, self.cfg.device)
    #     ce_new = lm_ce_loss(self.model_le, ids, attn, labels)

    #     # Locality KL (baseline vs LE) on unrelated prompts
    #     kl_loc = 0.0
    #     for t in batch.locality_prompts[:2]:
    #         ids_loc, attn_loc = self.tok.encode(t, self.cfg.device)
    #         self._deactivate_all(self.model_le)
    #         P_model = self.model_le
    #         self._activate(["LE"])
    #         Q_model = self.model_le
    #         kl_loc = kl_loc + kl_divergence_on_prompt(P_model, Q_model, ids_loc, attn_loc)

    #     # Specificity KL on neighbors (keep same as baseline)
    #     kl_spec = 0.0
    #     for t in batch.neighbor_prompts[:2]:
    #         ids_n, attn_n = self.tok.encode(t, self.cfg.device)
    #         self._deactivate_all(self.model_le)
    #         P_model = self.model_le
    #         self._activate(["LE"])
    #         Q_model = self.model_le
    #         kl_spec = kl_spec + kl_divergence_on_prompt(P_model, Q_model, ids_n, attn_n)

    #     # Orthogonality penalty between LF and LE subspaces
    #     orth = orthogonality_loss(self.lora_hosts)

    #     loss = ce_new + self.cfg.lambda_kl * kl_loc + self.cfg.lambda_spec * kl_spec + self.cfg.lambda_orth * orth

    #     self.opt_le.zero_grad(set_to_none=True)
    #     loss.backward()
    #     self.opt_le.step()

    #     for host in self.lora_hosts.values():
    #         host.clear_runtime()          # ✅ 清理绑定
    #     self._deactivate_all(self.model_le)
    #     return {
    #         "edit/ce_new": float(ce_new.detach().cpu()),
    #         "edit/kl_loc": float(kl_loc.detach().cpu()),
    #         "edit/kl_spec": float(kl_spec.detach().cpu()),
    #         "edit/orth": float(orth.detach().cpu()),
    #     }
    
    # ====== 生成辅助 ======
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
        """用超网络为当前 batch 生成并写入 LE 权重"""
        ctx = self.build_context(batch)       # [1, ctx_dim]
        le_weights = self.hyper(ctx)
        for name, host in self.lora_hosts.items():
            A, B = le_weights[name]
            host.set_adapter_weights("LE", A, B)

    @contextmanager
    def _adapters(self, names):
        """临时开启某些适配器，退出时恢复关闭"""
        # 记录旧状态
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
        打印 Baseline / LF-only / LE-only 的对比输出
        """
        # 先准备 LE 权重（按当前 batch 的 target_new 生成）
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
    
    def step_forget_batch(self, items: list, clip_norm: float = 1.0):
        self.opt_lf.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_neg_ce, log_kl = 0.0, 0.0

        for bi in items:
            self._deactivate_all()
            self._activate(["LF"])
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt, bi.target_true, self.cfg.device)
            ce_true = lm_ce_loss(self.model_lf, ids, attn, labels)

        # locality KL - 重新初始化 kl_loc 为 tensor
            kl_loc = torch.tensor(0.0, device=self.cfg.device)
            for t in bi.locality_prompts[:2]:
                ids_loc, attn_loc = self.tok.encode(t, self.cfg.device)
                self._deactivate_all()
                with torch.no_grad():
                    lp = self.model_lf(input_ids=ids_loc, attention_mask=attn_loc).logits
                    P = F.log_softmax(lp, dim=-1).exp()
                self._activate(["LF"])
                lq = self.model_lf(input_ids=ids_loc, attention_mask=attn_loc).logits
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
        "forget/kl_loc":      log_kl     / max(1, len(items)),
    }
    # def step_edit_batch(self, items: list, clip_norm: float = 1.0):
    #     """
    #     对一批 BatchItem 批量进行编辑步（超网络训练）：每条 item 绑定自己的 LE 运行期权重，累加 loss。
    #     返回均值日志。
    #     """
    #     self.opt_le.zero_grad(set_to_none=True)
    #     total_loss = 0.0
    #     log_ce_new, log_kl_loc, log_kl_spec, log_orth = 0.0, 0.0, 0.0, 0.0

    #     for bi in items:
    #         # 为该样本生成 LE（可微）
    #         ctx = self.build_context(bi)                    # [1, ctx_dim]
    #         le_weights = self.hyper(ctx)                    # dict: name -> (A,B)
    #         for name, host in self.lora_hosts.items():
    #             A, B = le_weights[name]
    #             host.bind_runtime("LE", A, B)

    #         # —— 计算损失（与 step_edit 相同逻辑）——
    #         self._deactivate_all()
    #         self._activate(["LE"])

    #         ids, attn, labels = self.tok.encode_label_for_target(bi.prompt, bi.target_new, self.cfg.device)
    #         ce_new = lm_ce_loss(self.model_le, ids, attn, labels)

    #         kl_loc = 0.0
    #         for t in bi.locality_prompts[:2]:
    #             ids_loc, attn_loc = self.tok.encode(t, self.cfg.device)
    #             # P: baseline
    #             self._deactivate_all()
    #             with torch.no_grad():
    #                 lp = self.model_le(input_ids=ids_loc, attention_mask=attn_loc).logits
    #                 P = F.log_softmax(lp, dim=-1).exp()
    #             # Q: LE
    #             self._activate(["LE"])
    #             lq = self.model_le(input_ids=ids_loc, attention_mask=attn_loc).logits
    #             Q = F.log_softmax(lq, dim=-1)
    #             kl_loc = kl_loc + F.kl_div(Q, P, reduction="batchmean", log_target=False)

    #         kl_spec = 0.0
    #         for t in bi.neighbor_prompts[:2]:
    #             ids_n, attn_n = self.tok.encode(t, self.cfg.device)
    #             # P: baseline
    #             self._deactivate_all()
    #             with torch.no_grad():
    #                 lp = self.model_le(input_ids=ids_n, attention_mask=attn_n).logits
    #                 P = F.log_softmax(lp, dim=-1).exp()
    #             # Q: LE
    #             self._activate(["LE"])
    #             lq = self.model_le(input_ids=ids_n, attention_mask=attn_n).logits
    #             Q = F.log_softmax(lq, dim=-1)
    #             kl_spec = kl_spec + F.kl_div(Q, P, reduction="batchmean", log_target=False)

    #         # 正交约束（LF vs LE，注意此处 LE 是 runtime 权重，正交项只对持久参数起作用；可保留或关闭）
    #         orth = orthogonality_loss(self.lora_hosts)

    #         loss_i = ce_new + self.cfg.lambda_kl   * kl_loc \
    #                         + self.cfg.lambda_spec * kl_spec \
    #                         + self.cfg.lambda_orth * orth
    #         total_loss = total_loss + loss_i

    #         log_ce_new  += float(ce_new.detach().cpu())
    #         log_kl_loc  += float(kl_loc.detach().cpu())
    #         log_kl_spec += float(kl_spec.detach().cpu())
    #         log_orth    += float(orth.detach().cpu())

    #         # 清理该样本的 runtime 绑定，准备下一个样本
    #         for host in self.lora_hosts.values():
    #             host.clear_runtime()
    #         self._deactivate_all()

    #     total_loss = total_loss / max(1, len(items))
    #     total_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.hyper.parameters(), clip_norm)
    #     self.opt_le.step()
    #     self._deactivate_all()

    #     return {
    #         "edit/ce_new":  log_ce_new  / max(1, len(items)),
    #         "edit/kl_loc":  log_kl_loc  / max(1, len(items)),
    #         "edit/kl_spec": log_kl_spec / max(1, len(items)),
    #         "edit/orth":    log_orth    / max(1, len(items)),
    #     }

    def step_edit_batch(self, items: list, clip_norm: float = 1.0):
        self.opt_le.zero_grad(set_to_none=True)
        total_loss = 0.0
        log_ce = 0.0

        for bi in items:
            # 生成 LE（可微）并绑定
            ctx = self.build_context(bi)
            le_weights = self.hyper(ctx)
            for name, host in self.lora_hosts.items():
                A, B = le_weights[name]
                host.bind_runtime("LE", A, B)

            # 只在目标 tokens 位置生效
            ids, attn, labels = self.tok.encode_label_for_target(bi.prompt, bi.target_new, self.cfg.device)
            tok_mask = build_target_token_mask(self.tokenizer, bi.prompt, bi.target_new, self.cfg.device)
            for host in self.lora_hosts.values():
                host.bind_token_mask("LE", tok_mask)

            self._deactivate_all()
            self._activate(["LE"])
            ce_new = lm_ce_loss(self.model_le, ids, attn, labels)

            total_loss = total_loss + ce_new
            log_ce += float(ce_new.detach().cpu())

            # 清理，为下一条样本准备
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
        """
        多内容编辑的主训练循环：
        - 每个 epoch：按小批执行 1 次遗忘 + K 次编辑
        - items：一组 BatchItem（可来自 JSONL/CSV）
        """
        for ep in range(epochs):
            if shuffle:
                random.shuffle(items)

            # 分块迭代
            f_iter = list(chunks(items, bs_forget))
            e_iter = list(chunks(items, bs_edit))
            e_idx  = 0

            for fi, f_batch in enumerate(f_iter):
                log_f = self.step_forget_batch(f_batch)
                # 连续做 K 次编辑步
                for _ in range(edit_per_forget):
                    if e_idx >= len(e_iter):
                        e_idx = 0
                        if shuffle: random.shuffle(items)
                        e_iter = list(chunks(items, bs_edit))
                    log_e = self.step_edit_batch(e_iter[e_idx])
                    e_idx += 1

                print(f"[ep {ep+1}] [forget {fi+1}/{len(f_iter)}] {log_f}  |  last edit {log_e}")

ADITEditor = AdvEditor

# ================================
# ADIT Main Interface Function
# ================================

def apply_ADIT_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ADITHyperParams,
    **kwargs
) -> AutoModelForCausalLM:
    """
    ADIT main interface function following AlphaEdit architecture
    """
    # Convert AlphaEdit-style requests to ADIT BatchItems
            
    batch_items = []
    for request in requests:
        # Handle different request formats
        prompt = request['prompt']
        if 'subject' in request:
            prompt = prompt.format(request['subject'])
        
        # Extract target strings
        target_true = request.get('target_true', '')
        if isinstance(target_true, dict):
            target_true = target_true.get('str', '')
            
        target_new = request['target_new']
        if isinstance(target_new, dict):
            target_new = target_new.get('str', '')
        
        # Normalize targets (add space if needed)
        def _norm_target(s: str):
            s = s or ""
            s = s.rstrip("\n")
            return s if s.startswith(" ") else " " + s
        
        batch_items.append(BatchItem(
            prompt=prompt,
            target_true=_norm_target(target_true),
            target_new=_norm_target(target_new),
            locality_prompts=request.get('locality_prompts', []) or [],
            neighbor_prompts=request.get('neighbor_prompts', []) or [],
        ))

    # Print sample requests
    for request in requests[:3]:
        subject = request.get('subject', '')
        prompt_text = request['prompt'].format(subject) if 'subject' in request else request['prompt']
        target_new_text = request['target_new']['str'] if isinstance(request['target_new'], dict) else request['target_new']
        print(f"ADIT request: [{prompt_text}] -> [{target_new_text}]")

    # 直接构建目标层名称列表，跳过 MEMITLocator
    target_layer_names = []
    for layer_id in hparams.layers:
        try:
            # 使用 rewrite_module_tmp 格式构建完整的层名
            layer_name = hparams.rewrite_module_tmp.format(layer_id)
            target_layer_names.append(layer_name)
        except (IndexError, KeyError) as e:
            print(f"Error formatting layer {layer_id} with format '{hparams.rewrite_module_tmp}': {e}")
            # 如果格式化失败，尝试直接拼接
            if "{}" in hparams.rewrite_module_tmp:
                layer_name = hparams.rewrite_module_tmp.replace("{}", str(layer_id))
            elif "{i}" in hparams.rewrite_module_tmp:
                layer_name = hparams.rewrite_module_tmp.replace("{i}", str(layer_id))
            else:
                layer_name = f"{hparams.rewrite_module_tmp}.{layer_id}"
            target_layer_names.append(layer_name)
    
    print(f"[DEBUG] Target layers: {target_layer_names}")

    # 创建简单的目标层匹配模式
    import re
    # 转义所有特殊字符并创建精确匹配模式
    escaped_layers = [re.escape(layer_name) for layer_name in target_layer_names]
    layer_pattern = r"(" + "|".join(escaped_layers) + r")$"
    name_filter = re.compile(layer_pattern)
    
    print(f"[DEBUG] Layer pattern: {layer_pattern}")

    # 修改 ADITEditor 的初始化，直接传入层名列表或模式
    class SimpleLocator:
        def __init__(self, pattern):
            self.pattern = pattern
        
        def select(self, model: nn.Module):
            return self.pattern

    locator = SimpleLocator(name_filter)

    # Initialize ADIT editor
    editor = ADITEditor(model, tok, locator, hparams)

    # Execute training
    print(f"Executing ADIT on {len(batch_items)} requests...")
    editor.train_multiedit(
        batch_items,
        epochs=hparams.v_num_grad_steps,
        bs_forget=hparams.batch_size_forget,
        bs_edit=hparams.batch_size_edit,
        edit_per_forget=hparams.edit_per_forget,
        shuffle=True
    )

    print("ADIT editing completed!")
    return model,None