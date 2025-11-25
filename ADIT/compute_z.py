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
    ADIT版本：计算目标值向量 - 彻底修复版
    """

    print("ADIT: Computing target representation")
    print(f"[DEBUG] Request prompt: {repr(request['prompt'])}")  # 添加调试
    print(f"[DEBUG] Request subject: {repr(request['subject'])}")  # 添加调试

    # Tokenize target
    target_ids = tok(request["target_new"]["str"], return_tensors="pt")["input_ids"][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    # 构建编辑提示 - 彻底修复：分离模板和格式化文本
    editing_templates = []  # 带{}的模板，用于find_fact_lookup_idx
    input_texts = []        # 格式化后的文本，用于模型输入
    
    for context_types in context_templates:
        for context in context_types:
            # 构建完整的prompt模板（带{}）
            prompt_template = context.format(request["prompt"])  # 如 "The mother tongue of {} is"
            full_template = prompt_template + tok.decode(target_ids[:-1])  # 加上目标词
            editing_templates.append(full_template)  # 保持带{}
            
            # 构建格式化后的输入文本
            formatted_text = full_template.format(request["subject"])  # 格式化
            input_texts.append(formatted_text)
            
            # 添加调试信息
            print(f"[DEBUG] Template: {repr(full_template)}")  # 应该带{}
            print(f"[DEBUG] Formatted: {repr(formatted_text)}")  # 应该格式化

    # 为ADIT准备输入
    input_tok = tok(
        input_texts,  # 使用已经格式化的文本
        return_tensors="pt",
        padding=True,
    )

    # 找到关键token位置 - 使用带{}的模板
    print(f"[DEBUG] Calling find_fact_lookup_idx with templates:")
    for i, prompt in enumerate(editing_templates):
        print(f"[DEBUG] Template {i}: {repr(prompt)}")  # 检查传给find_fact_lookup_idx的内容
    
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,  # 这是带{}的模板，如 "The mother tongue of {} is English"
            request["subject"], 
            tok, 
            hparams.fact_token, 
            verbose=(i == 0)
        )
        for i, prompt in enumerate(editing_templates)  # 使用editing_templates（带{}）
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

    # 提取关键位置的表示作为目标z
    z_list = []
    for i, idx in enumerate(lookup_idxs):
        if idx < layer_output[i].shape[0]:
            z_list.append(layer_output[i, idx, :].detach())

    # 平均所有提示的目标表示
    if z_list:
        target_z = torch.stack(z_list).mean(dim=0)
    else:
        # Fallback: 使用最后一个token的表示
        target_z = layer_output[:, -1, :].mean(dim=0)

    print(f"ADIT: Computed target z vector with norm {target_z.norm()}")
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
        # 确保是3D张量 [batch, seq_len, hidden]
        if input_repr.dim() == 2:
            input_repr = input_repr.unsqueeze(0)  # [seq_len, hidden] -> [1, seq_len, hidden]
            
        # 统一处理输出：可能是元组或单个张量  
        if isinstance(output_repr, tuple):
            output_repr = output_repr[0]
        if output_repr.dim() == 2:
            output_repr = output_repr.unsqueeze(0)  # [seq_len, hidden] -> [1, seq_len, hidden]

    # 提取关键位置的输入和输出表示
    input_vectors = []
    output_vectors = []
    
    for i, idx in enumerate(lookup_indices):
        # 检查索引是否在有效范围内
        if idx < input_repr[i].shape[0]:
            input_vec = input_repr[i, idx, :].detach()
            output_vec = output_repr[i, idx, :].detach()
        else:
            # 使用最后一个token作为fallback
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
    ADIT查找关键Token位置 — 使用repr_tools版本
    """
    
    if verbose:
        print("\n[DEBUG] find_fact_lookup_idx")
        print("  prompt:", repr(prompt))
        print("  subject:", repr(subject))
        print("  strategy:", fact_token_strategy)

    ret = None
    
    if fact_token_strategy.startswith("subject_"):
        # 使用 repr_tools 查找subject位置
        subtoken = fact_token_strategy[len("subject_"):]
        try:
            # 调用 repr_tools 的查找函数
            result = repr_tools.get_words_idxs_in_templates(
                tok=tok,
                context_templates=[prompt],
                words=[subject],
                subtoken=subtoken,
            )
            ret = result[0][0]  # 获取第一个结果
        except Exception as e:
            if verbose:
                print(f"  [WARN] repr_tools查找失败: {e}")
            # Fallback: 使用最后一个token
            prompt_enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
            ret = len(prompt_enc["input_ids"][0]) - 1
            
    elif fact_token_strategy == "last":
        # 直接返回最后一个token位置
        prompt_enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        ret = len(prompt_enc["input_ids"][0]) - 1
    else:
        if verbose:
            print(f"  [WARN] 未知策略: {fact_token_strategy}")
        prompt_enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        ret = len(prompt_enc["input_ids"][0]) - 1

    if verbose:
        print(f"  → 最终选择token位置: {ret}")
        prompt_enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_tokens = tok.convert_ids_to_tokens(prompt_enc["input_ids"][0])
        if 0 <= ret < len(prompt_tokens):
            print(f"  → 对应token: '{prompt_tokens[ret]}'")
        print("  [END DEBUG]\n")

    return ret

