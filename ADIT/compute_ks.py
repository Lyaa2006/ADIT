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
    ADIT版本：计算键向量 - 带百分比进度
    """

    print(f"ADIT: Computing key vectors for layer {layer}")
    print(f"  [进度] 开始处理，共有 {len(requests)} 个请求")

    # 计算总处理量
    total_contexts = sum(len(context_type) for context_type in context_templates)
    total_processing = len(requests) * total_contexts
    print(f"  [进度] 每个请求使用 {total_contexts} 个上下文模板")
    print(f"  [进度] 总共需要处理 {total_processing} 个组合")

    # 构建上下文模板和词语列表
    print(f"  [进度] 构建上下文模板列表...")
    context_list = []
    words_list = []
    
    for req_idx, request in enumerate(requests):
        for ctx_type_idx, context_type in enumerate(context_templates):
            for ctx_idx, context in enumerate(context_type):
                context_list.append(context.format(request["prompt"]))
                words_list.append(request["subject"])
        
        progress = (req_idx + 1) / len(requests) * 100
        print(f"  [进度] 构建列表: {progress:.1f}% ({req_idx + 1}/{len(requests)})")

    print(f"  [进度] 构建完成: {len(context_list)} 个上下文, {len(words_list)} 个词语")

    # 使用统一的函数获取键向量
    print(f"  [进度] 调用 get_module_input_output_at_words...")
    layer_ks, _ = get_module_input_output_at_words(
        model=model,
        tok=tok,
        layer=layer,
        context_templates=context_list,
        words=words_list,
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )
    print(f"  [进度] get_module_input_output_at_words 完成")
    print(f"  [进度] 获取到键向量形状: {layer_ks.shape}")

    # 平均处理
    print(f"  [进度] 开始平均处理...")
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    final_keys = []
    total_requests = layer_ks.size(0) // context_len if context_len > 0 else 0

    for i in range(0, layer_ks.size(0), context_len):
        request_idx = i // context_len
        progress = (request_idx + 1) / total_requests * 100
        
        print(f"  [进度] 处理请求: {progress:.1f}% ({request_idx + 1}/{total_requests})")
        
        request_keys = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            template_vectors = layer_ks[i + start : i + end]
            template_avg = template_vectors.mean(0)
            request_keys.append(template_avg)
        
        request_avg = torch.stack(request_keys, 0).mean(0)
        final_keys.append(request_avg)

    result = torch.stack(final_keys, dim=0)
    print(f"ADIT: 计算完成！得到 {result.shape[0]} 个键向量")
    print(f"  [进度] 层 {layer} 的键向量计算 100% 完成")
    
    return result