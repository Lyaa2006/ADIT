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
    context_templates: List[str],  # 仍然保留接口，但在 ROME-style 中不再使用
) -> torch.Tensor:
    """
    ROME-style compute_z:
    --------------------------------------
    提取目标 object 的内部表征，作为编辑目标向量 z。
    
    不依赖模板，不依赖 subject，不依赖 lookup_idx。
    只依赖 target_new 的 token 序列，获得其在指定层的 MLP 输出表示。
    """

    # 1. 获取 target_new 的 string
    target_str = request.get("target_new", {}).get("str", "")
    if not target_str:
        hidden = model.config.hidden_size
        return torch.zeros(hidden, device=model.device)

    # 2. Tokenize target object
    enc = tok(target_str, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(model.device)

    # 3. 目标模块名称
    module_name = hparams.rewrite_module_tmp.format(layer)

    # 4. 前向并截取目标层输出
    with nethook.TraceDict(
        model,
        layers=[module_name],
        retain_output=True,
    ) as tr:
        _ = model(input_ids=input_ids, attention_mask=attn_mask)
        raw = tr[module_name].output

    # 5. 统一输出维度：GPT2 的 Conv1D 可能返回 tuple 或 [seq, hidden]
    if isinstance(raw, tuple):
        raw = raw[0]
    if raw.dim() == 2:
        raw = raw.unsqueeze(0)

    # raw: [1, seq_len, hidden]
    _, seq_len, hidden = raw.shape

    # 6. ROME-style：取 target object 最后一个 token 的表示作为 z
    # （这是 ROME 稳定且标准的做法）
    z = raw[0, -1, :].detach()

    return z



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
    ADIT版本：获取指定层在关键token位置的输入和输出表示 - 修复GPT-2 Conv1D兼容性
    """
    print(f"ADIT: Getting module input/output at words for layer {layer}")

    # 准备输入文本
    input_texts = []
    for context, word in zip(context_templates, words):
        input_texts.append(context.format(word) if "{}" in context else context)

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
        
        # 关键修复：处理GPT-2 Conv1D的特殊输入输出格式
        input_repr = layer_module.input
        output_repr = layer_module.output
        
        # 统一处理：如果是元组，取第一个元素
        if isinstance(input_repr, tuple):
            input_repr = input_repr[0]
        if isinstance(output_repr, tuple):
            output_repr = output_repr[0]
        
        print(f"[DEBUG] Raw input shape: {input_repr.shape}")
        print(f"[DEBUG] Raw output shape: {output_repr.shape}")
        
        # 🔥 关键修复：确保维度正确，处理GPT-2可能的维度转置
        # 对于GPT-2 Conv1D层，维度应该是 [batch_size, seq_len, hidden_size]
        if input_repr.dim() == 2:
            # [seq_len, hidden_size] -> [1, seq_len, hidden_size]
            input_repr = input_repr.unsqueeze(0)
        elif input_repr.dim() == 3:
            # 检查是否是转置的维度 [seq_len, batch, hidden]
            if input_repr.shape[0] != len(input_texts):
                # 尝试转置到正确的维度
                if input_repr.shape[1] == len(input_texts):
                    print(f"[DEBUG] Fixing input dimension: transposing {input_repr.shape} -> [{len(input_texts)}, {input_repr.shape[0]}, {input_repr.shape[2]}]")
                    input_repr = input_repr.transpose(0, 1)
        
        if output_repr.dim() == 2:
            output_repr = output_repr.unsqueeze(0)
        elif output_repr.dim() == 3:
            if output_repr.shape[0] != len(input_texts):
                if output_repr.shape[1] == len(input_texts):
                    print(f"[DEBUG] Fixing output dimension: transposing {output_repr.shape} -> [{len(input_texts)}, {output_repr.shape[0]}, {output_repr.shape[2]}]")
                    output_repr = output_repr.transpose(0, 1)

    print(f"[DEBUG] Processed input shape: {input_repr.shape}")
    print(f"[DEBUG] Processed output shape: {output_repr.shape}")

    # 提取关键位置的输入和输出表示
    input_vectors = []
    output_vectors = []
    
    batch_size, seq_len, hidden_size = input_repr.shape
    
    for i, idx in enumerate(lookup_indices):
        # 确保索引在范围内
        if idx >= seq_len:
            idx = seq_len - 1
        elif idx < 0:
            idx = 0
        
        # 提取指定位置的向量
        input_vec = input_repr[i, idx, :].detach()
        output_vec = output_repr[i, idx, :].detach()
        
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
    
    '''if verbose:
        print("\n[DEBUG] find_fact_lookup_idx")
        print("raw: ",prompt)
        print(f"  prompt: {repr(prompt)}")
        print(f"  subject: {repr(subject)}")
        print(f"  strategy: {fact_token_strategy}")'''

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
            
            '''if 0 <= result < len(tokens):
                token_at_pos = tok.decode([tokens[result]])
                print(f"  → 最终位置: {result}, 对应token: '{token_at_pos}'")
            else:
                print(f"  → 最终位置: {result} (超出范围, tokens长度: {len(tokens)})")'''
                
        except:
            # 如果offset_mapping失败，使用简单方法
            tokens = tok.encode(full_text, add_special_tokens=False)
            '''if 0 <= result < len(tokens):
                token_at_pos = tok.decode([tokens[result]])
                print(f"  → 最终位置: {result}, 对应token: '{token_at_pos}'")
            else:
                print(f"  → 最终位置: {result} (超出范围)")'''
    
    return result