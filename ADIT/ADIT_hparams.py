from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Any
import torch
from util.hparams import HyperParams


@dataclass
class ADITHyperParams(HyperParams):
    # ====== 基础配置 ======
    model_name: str = "gpt2-large"
    device: str =  "cpu"
    
    # ====== 目标层配置 ======
    layers: List[int] = None  # 目标层ID列表
    rewrite_module_tmp: str = "transformer.h.{}.mlp.c_proj"  # GPT2的目标模块
    
    # ====== LoRA 配置 ======
    lf_rank: int = 8           # 遗忘LoRA的秩
    le_rank: int = 16          # 编辑LoRA的秩  
    alpha: float = 16.0        # LoRA缩放系数
    
    # ====== 训练参数 ======
    v_num_grad_steps: int = 20     # 训练epoch数
    lr_lf: float = 5e-4           # LF超网络学习率
    lr_le: float = 5e-4           # LE超网络学习率
    
    # 批次大小
    batch_size_forget: int = 3
    batch_size_edit: int = 1
    edit_per_forget: int = 5
    
    # ====== 损失权重 ======
    lambda_loc: float = 1.0       # 局部性损失权重
    lambda_kl: float = 1.0        # KL散度权重
    lambda_spec: float = 0.5      # 特异性损失权重  
    lambda_orth: float = 0.05     # 正交性损失权重
    
    # ====== 直接监督参数 ======
    use_direct_supervision: bool = True  # 是否使用直接监督
    prediction_weight: float = 1.0       # 预测损失权重
    representation_weight: float = 0.5   # 表示对齐权重
    invariance_weight: float = 0.2       # 不变性损失权重
    specificity_weight: float = 0.3      # 特异性损失权重
    
    # ====== 上下文生成参数 ======
    ctx_dim: int = 1600                     # 上下文维度（匹配模型隐藏层）
    context_generation: str = "embedding_mean"  # "embedding_mean", "learned", "fixed"
    learned_context_dim: int = 512
    use_position_context: bool = True       # 是否使用位置信息
    
    # ====== 事实token定位策略 ======
    fact_token: str = "subject_first"
    
    # ====== 向量指导参数（向后兼容） ======
    use_vector_guidance: bool = True       # 向后兼容
    vector_guidance_weight: float = 0.3
    vector_alignment_weight: float = 0.1
    
    # ====== 其他参数 ======
    clamp_norm_factor: float = 1.0         # 梯度裁剪
    
    def __post_init__(self):
        if self.layers is None:
            self.layers = [17]  # GPT2-large默认层
        
        # 确保向后兼容
        if hasattr(self, 'use_vector_guidance') and self.use_vector_guidance:
            # 如果启用了向量指导，自动启用直接监督
            self.use_direct_supervision = True
        
        # 自动调整ctx_dim为模型隐藏层大小（如果在训练时检测到）
        if hasattr(self, 'model_hidden_size'):
            self.ctx_dim = self.model_hidden_size