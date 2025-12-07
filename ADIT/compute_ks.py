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
    context_templates: List[str],  # 注意：这里是字符串列表，不是嵌套列表！
) -> torch.Tensor:
    """
    ADIT版本：计算键向量 - 修复模板处理问题
    """
    print(f"ADIT: Computing key vectors for layer {layer}")
    
    # 🔥 调试：查看传入的模板
    print(f"  [DEBUG] Received {len(context_templates)} templates")
    for i, template in enumerate(context_templates[:3]):  # 只显示前3个
        print(f"    Template[{i}]: {repr(template)}")
    if len(context_templates) > 3:
        print(f"    ... and {len(context_templates)-3} more")

    # 计算总处理量
    total_contexts = len(context_templates)
    total_processing = len(requests) * total_contexts
    print(f"  Total contexts: {total_contexts}, requests: {len(requests)}")

    # 构建上下文模板和词语列表
    context_list = []
    words_list = []
    
    for req_idx, request in enumerate(requests):
        subject = request.get("subject", "")
        if not subject:
            print(f"  [WARN] Request {req_idx} has no subject, skipping")
            continue
            
        # 🔥 修复：直接遍历模板列表，不要嵌套循环！
        for template_idx, template in enumerate(context_templates):
            # 确保template是字符串
            if not isinstance(template, str):
                print(f"  [ERROR] Template {template_idx} is not string: {type(template)}")
                continue
                
            try:
                # 用 subject 替换模板中的 {}
                formatted_context = template.format(subject)
                context_list.append(formatted_context)
                words_list.append(subject)
                
                # 调试输出（只显示第一个样本的第一个模板）
                if req_idx == 0 and template_idx == 0:
                    print(f"  [DEBUG] First template formatting:")
                    print(f"    Raw template: {repr(template)}")
                    print(f"    Subject: {repr(subject)}")
                    print(f"    Formatted: {repr(formatted_context)}")
                    
            except Exception as e:
                print(f"  [ERROR] Failed to format template {template_idx}:")
                print(f"    Template: {repr(template)}")
                print(f"    Subject: {repr(subject)}")
                print(f"    Error: {e}")
                # 如果格式化失败，直接使用原始模板（不带subject）
                context_list.append(template)
                words_list.append(subject)

    if not context_list:
        print("  [ERROR] No valid contexts generated!")
        # 返回零向量
        hidden_size = model.config.hidden_size
        return torch.zeros(len(requests), hidden_size, device=model.device)

    print(f"  Generated {len(context_list)} context strings")
    if context_list:
        print(f"  Sample context: {repr(context_list[0][:50])}...")

    # 使用统一的函数获取键向量
    print(f"  [进度] 调用 get_module_input_output_at_words...")
    try:
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
        
        # 使用输出向量作为键向量（MLP层的输出）
        layer_ks = output_vectors
        print(f"  [进度] 使用输出向量作为键向量: {layer_ks.shape}")

    except Exception as e:
        print(f"  [ERROR] Failed in get_module_input_output_at_words: {e}")
        # 返回零向量
        hidden_size = model.config.hidden_size
        return torch.zeros(len(requests), hidden_size, device=model.device)

    # 平均处理
    # 🔥 注意：现在只有一个模板组，所以直接平均
    final_keys = []
    
    for i in range(0, layer_ks.size(0), len(context_templates)):
        request_idx = i // len(context_templates)
        if request_idx < len(requests):
            # 获取该请求的所有模板向量
            template_vectors = layer_ks[i:i+len(context_templates)]
            # 平均所有模板
            request_avg = template_vectors.mean(0)
            final_keys.append(request_avg)

    if final_keys:
        result = torch.stack(final_keys, dim=0)
        print(f"ADIT: 计算完成！得到 {result.shape[0]} 个键向量，维度: {result.shape}")
    else:
        print("  [ERROR] No final keys generated!")
        hidden_size = model.config.hidden_size
        result = torch.zeros(len(requests), hidden_size, device=model.device)
    
    return result