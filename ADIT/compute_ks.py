from typing import Dict, List
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from .ADIT_hparams import ADITHyperParams
from .compute_z import get_module_input_output_at_words


def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ADITHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    ADIT版本：计算键向量 - 修复维度问题
    """

    print(f"ADIT: Computing key vectors for layer {layer}")

    # 计算总处理量
    total_contexts = sum(len(context_type) for context_type in context_templates)
    total_processing = len(requests) * total_contexts

    # 构建上下文模板和词语列表
    context_list = []
    words_list = []
    
    for req_idx, request in enumerate(requests):
        for ctx_type_idx, context_type in enumerate(context_templates):
            for ctx_idx, context in enumerate(context_type):
                context_list.append(context.format(request["prompt"]))
                words_list.append(request["subject"])

    # 使用统一的函数获取键向量 - 获取输入向量
    print(f"  [进度] 调用 get_module_input_output_at_words...")
    input_vectors, output_vectors = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_list,
        words=words_list,
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )
    
    print(f"  [进度] 获取到输入向量形状: {input_vectors.shape}")
    print(f"  [进度] 获取到输出向量形状: {output_vectors.shape}")
    
    # 使用输入向量作为键向量（维度应该是1600）
    layer_ks = output_vectors
    print(f"  [进度] 使用输入向量作为键向量: {layer_ks.shape}")

    # 平均处理
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    final_keys = []
    total_requests = layer_ks.size(0) // context_len if context_len > 0 else 0

    for i in range(0, layer_ks.size(0), context_len):
        request_idx = i // context_len
        progress = (request_idx + 1) / total_requests * 100
        
        request_keys = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            template_vectors = layer_ks[i + start : i + end]
            template_avg = template_vectors.mean(0)
            request_keys.append(template_avg)
        
        request_avg = torch.stack(request_keys, 0).mean(0)
        final_keys.append(request_avg)

    result = torch.stack(final_keys, dim=0)
    print(f"ADIT: 计算完成！得到 {result.shape[0]} 个键向量，维度: {result.shape}")
    print(f"  [进度] 层 {layer} 的键向量计算 100% 完成")
    
    return result