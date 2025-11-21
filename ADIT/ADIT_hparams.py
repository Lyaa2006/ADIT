from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any
import torch
from util.hparams import HyperParams


@dataclass
class ADITHyperParams(HyperParams):
    # ====== 模型架构配置 ======
    model_name: str = "gpt-j-6b"
    layers: List[int] = None  # 目标层ID列表，例如 [3,4,5,6,7,8]
    
    # 设备配置 - 添加这一行
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 层选择策略
    layer_selection: Literal["all", "random", "specified"] = "specified"
    
    # 事实token定位策略
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ] = "last"
    
    # ====== LoRA 配置 ======
    lf_rank: int = 8           # 遗忘LoRA的秩
    le_rank: int = 16          # 编辑LoRA的秩  
    alpha: float = 16.0        # LoRA缩放系数
    
    # ====== 训练参数 ======
    # 遗忘步参数
    v_num_grad_steps: int = 10     # 梯度步数
    v_lr: float = 5e-4            # 遗忘学习率
    v_loss_layer: int = -1        # 损失层
    v_weight_decay: float = 1e-4  # 权重衰减
    
    lr_lf: float = 5e-4   
    # 编辑步参数  
    lr_le: float = 2e-4           # 编辑学习率
    le_num_grad_steps: int = 5    # 编辑梯度步数
    
    # ====== 损失权重 ======
    lambda_loc: float = 0.1       # 局部性KL损失权重
    lambda_kl: float = 0.2        # KL散度权重
    lambda_spec: float = 0.1      # 特异性损失权重  
    lambda_orth: float = 0.005    # 正交性损失权重
    kl_factor: float = 1.0        # KL因子
    
    # ====== 训练约束 ======
    clamp_norm_factor: float = 1.0    # 梯度裁剪范数因子
    L2: float = 1e-4                  # L2正则化
    
    # ====== 动量调整 ======
    mom2_adjustment: bool = True          # 是否使用动量调整
    mom2_update_weight: float = 0.1       # 动量更新权重
    
    # ====== 模块模板 ======
    rewrite_module_tmp: str = "transformer.h.{}.mlp.fc_out"  # 重写模块模板
    layer_module_tmp: str = "transformer.h.{}"              # 层模块模板
    mlp_module_tmp: str = "transformer.h.{}.mlp"            # MLP模块模板
    attn_module_tmp: str = "transformer.h.{}.attn"          # 注意力模块模板
    ln_f_module: str = "transformer.ln_f"                   # LayerNorm模块
    lm_head_module: str = "lm_head"                         # 输出头模块
    
    # ====== 统计配置 ======
    mom2_dataset: str = "wikitext"          # 动量数据集
    mom2_n_samples: int = 10000             # 动量样本数
    mom2_dtype: str = "float32"             # 动量数据类型
    
    # ====== 超网络配置 ======
    ctx_dim: int = 4096                     # 上下文维度（匹配模型隐藏层）
    hyper_hidden_dim: int = 256             # 超网络隐藏层维度
    
    # ====== 批次训练 ======
    batch_size_forget: int = 4              # 遗忘批次大小
    batch_size_edit: int = 4                # 编辑批次大小
    edit_per_forget: int = 3                # 每次遗忘对应的编辑次数
    
    # ====== 空空间阈值 ======
    nullspace_threshold: float = 1e-5       # 空空间阈值
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [3, 4, 5, 6, 7, 8]  # 默认层