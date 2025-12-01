from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from .ADIT_hparams import ADITHyperParams
from . import repr_tools


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ADITHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    ADIT版本：计算目标值向量 - 修复维度问题
    """

    print("ADIT: Computing target representation")
    print(f"[DEBUG] Request prompt: {repr(request['prompt'])}")
    print(f"[DEBUG] Request subject: {repr(request['subject'])}")

    # Tokenize target
    target_ids = tok(request["target_new"]["str"], return_tensors="pt")["input_ids"][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    # 构建编辑提示
    editing_templates = []
    input_texts = []
    
    for context_types in context_templates:
        for context in context_types:
            formatted_template = context.format(request["subject"])
            full_template = formatted_template + tok.decode(target_ids[:-1])
            editing_templates.append(full_template)
        
            formatted_text = full_template  # 已经格式化，不需要再次格式化
            input_texts.append(formatted_text)

    # 为ADIT准备输入
    input_tok = tok(
        input_texts,
        return_tensors="pt",
        padding=True,
    )

    # 找到关键token位置
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,
            request["subject"], 
            tok, 
            hparams.fact_token, 
            verbose=(i == 0)
        )
        for i, prompt in enumerate(editing_templates)
    ]

    # 获取目标层的输出表示
    with nethook.TraceDict(
        module=model,
        layers=[hparams.rewrite_module_tmp.format(layer)],
        retain_output=True,
    ) as tr:
        _ = model(**input_tok)
        raw_output = tr[hparams.rewrite_module_tmp.format(layer)].output

        if isinstance(raw_output, tuple):
            layer_output = raw_output[0]
        else:
            layer_output = raw_output

    print(f"[DEBUG] Layer output shape: {layer_output.shape}")

    # 提取关键位置的表示作为目标z - 输出向量（维度应该是6400）
    z_list = []
    for i, idx in enumerate(lookup_idxs):
        if idx < layer_output[i].shape[0]:
            z_vector = layer_output[i, idx, :].detach()
            z_list.append(z_vector)
            print(f"[DEBUG] Sample {i} z vector shape: {z_vector.shape}")

    # 平均所有提示的目标表示
    if z_list:
        target_z = torch.stack(z_list).mean(dim=0)
    else:
        target_z = layer_output[:, -1, :].mean(dim=0)

    print(f"ADIT: Computed target z vector with norm {target_z.norm()}, shape: {target_z.shape}")
    return target_z


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ADIT版本：获取指定层在关键token位置的输入和输出表示 - 修复版
    """

    print(f"ADIT: Getting module input/output at words for layer {layer}")

    # 准备输入文本
    input_texts = []
    for context, word in zip(context_templates, words):
        input_texts.append(context.format(word))

    # Tokenize
    input_tok = tok(
        input_texts,
        return_tensors="pt",
        padding=True,
    )

    # 找到每个输入的关键token位置
    lookup_indices = []
    for context, word in zip(context_templates, words):
        idx = find_fact_lookup_idx(context, word, tok, fact_token_strategy, verbose=False)
        lookup_indices.append(idx)

    # 跟踪指定层的输入和输出
    with nethook.TraceDict(
        module=model,
        layers=[module_template.format(layer)],
        retain_input=True,
        retain_output=True,
    ) as tr:
        _ = model(**input_tok)
        
        # 获取输入和输出
        layer_module = tr[module_template.format(layer)]
        
        # 修复：处理可能的维度问题
        input_repr = layer_module.input
        output_repr = layer_module.output
        
        # 统一处理输入：可能是元组或单个张量
        if isinstance(input_repr, tuple):
            input_repr = input_repr[0]
        if input_repr.dim() == 2:
            input_repr = input_repr.unsqueeze(0)
            
        # 统一处理输出：可能是元组或单个张量  
        if isinstance(output_repr, tuple):
            output_repr = output_repr[0]
        if output_repr.dim() == 2:
            output_repr = output_repr.unsqueeze(0)

    print(f"[DEBUG] Input repr shape: {input_repr.shape}")
    print(f"[DEBUG] Output repr shape: {output_repr.shape}")

    # 提取关键位置的输入和输出表示
    input_vectors = []
    output_vectors = []
    
    for i, idx in enumerate(lookup_indices):
        if idx < input_repr[i].shape[0]:
            input_vec = input_repr[i, idx, :].detach()
            output_vec = output_repr[i, idx, :].detach()
        else:
            input_vec = input_repr[i, -1, :].detach()
            output_vec = output_repr[i, -1, :].detach()
        
        input_vectors.append(input_vec)
        output_vectors.append(output_vec)

    input_result = torch.stack(input_vectors, dim=0)
    output_result = torch.stack(output_vectors, dim=0)

    print(f"ADIT: Got input shape {input_result.shape}, output shape {output_result.shape}")
    
    return input_result, output_result


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose: bool = True,
) -> int:
    """
    ADIT查找关键Token位置 — 改进版，解决tokenization上下文依赖问题
    """
    
    if verbose:
        print("\n[DEBUG] find_fact_lookup_idx")
        print("raw: ",prompt)
        print(f"  prompt: {repr(prompt)}")
        print(f"  subject: {repr(subject)}")
        print(f"  strategy: {fact_token_strategy}")

    # 直接使用我们改进的repr_tools函数
    if fact_token_strategy == "last":
        # 最后一个token策略
        result = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[""],  # 空字符串会触发默认的最后一个token逻辑
            subtoken="last",  # 明确指定策略
        )[0][0]
        
    elif fact_token_strategy.startswith("subject_"):
        # subject相关策略
        subtoken = fact_token_strategy[len("subject_"):]
        
        result = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=subtoken,
        )[0][0]
        
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    # 验证和输出结果
    if verbose:
        # 构建完整文本用于验证
        if "{}" in prompt:
            full_text = prompt.format(subject)
        else:
            full_text = prompt + " " + subject if subject else prompt
        
        try:
            # 获取tokenization结果
            encoding = tok(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
            tokens = encoding["input_ids"]
            
            if 0 <= result < len(tokens):
                token_at_pos = tok.decode([tokens[result]])
                print(f"  → 最终位置: {result}, 对应token: '{token_at_pos}'")
            else:
                print(f"  → 最终位置: {result} (超出范围, tokens长度: {len(tokens)})")
                
        except:
            # 如果offset_mapping失败，使用简单方法
            tokens = tok.encode(full_text, add_special_tokens=False)
            if 0 <= result < len(tokens):
                token_at_pos = tok.decode([tokens[result]])
                print(f"  → 最终位置: {result}, 对应token: '{token_at_pos}'")
            else:
                print(f"  → 最终位置: {result} (超出范围)")
    
    return result