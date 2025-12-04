"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

from util import nethook

def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i] for i in range(len(words))],  # 直接使用原上下文，不格式化
        idxs,
        layer,
        module_template,
        track,
    )

def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: List[str], words: List[str], subtoken: str
) -> List[List[int]]:
    """
    改进版：通过字符偏移映射精确查找token位置
    解决tokenization上下文依赖问题
    """

    idxs = []
    
    for i, context in enumerate(context_templates):
        word = words[i]
        
        # 1. 构建完整文本
        if "{}" in context:
            # 处理带占位符的模板
            full_text = context.format(word)
        else:
            # 处理不带占位符的模板（直接在末尾添加）
            full_text = context + " " + word if word else context
        
        # 2. 尝试使用offset_mapping进行精确映射
        try:
            # 获取tokenization的字符偏移映射
            encoding = tok(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False  # 不添加特殊token以便精确定位
            )
            tokens = encoding["input_ids"]
            offsets = encoding["offset_mapping"]  # [(start_char, end_char), ...]
            
            # 3. 计算word在完整文本中的字符位置
            if "{}" in context:
                # 找到占位符位置
                placeholder_idx = context.index("{}")
                context_before = context[:placeholder_idx]
                word_char_start = len(context_before)
            else:
                # word在前缀后的位置
                context_before = context
                word_char_start = len(context_before)
                # 检查是否有空格
                if word_char_start < len(full_text) and full_text[word_char_start] == ' ':
                    word_char_start += 1  # 跳过空格
            
            word_char_end = word_char_start + len(word) - 1
            
            # 4. 根据策略找到对应的token位置
            if subtoken == "first":
                # 找到覆盖word起始字符的token
                token_idx = None
                for idx, (start, end) in enumerate(offsets):
                    if start <= word_char_start < end:
                        token_idx = idx
                        break
                if token_idx is None:
                    # 回退：找到第一个结束位置大于word_char_start的token
                    for idx, (start, end) in enumerate(offsets):
                        if end > word_char_start:
                            token_idx = idx
                            break
                
            elif subtoken == "last":
                # 找到覆盖word结束字符的token
                token_idx = None
                for idx, (start, end) in enumerate(offsets):
                    if start <= word_char_end < end:
                        token_idx = idx
                
                if token_idx is None:
                    # 回退：找到最后一个起始位置小于等于word_char_end的token
                    last_valid_idx = None
                    for idx, (start, end) in enumerate(offsets):
                        if start <= word_char_end:
                            last_valid_idx = idx
                    token_idx = last_valid_idx
                    
            elif subtoken == "first_after_last":
                # 先找到word的最后一个token
                last_token_idx = None
                for idx, (start, end) in enumerate(offsets):
                    if start <= word_char_end < end:
                        last_token_idx = idx
                
                if last_token_idx is not None and last_token_idx + 1 < len(offsets):
                    token_idx = last_token_idx + 1
                else:
                    # 回退到最后一个token
                    token_idx = len(offsets) - 1
                    
            else:
                # 默认使用最后一个token
                token_idx = len(offsets) - 1
            
            # 5. 验证和调整
            if token_idx is None or token_idx >= len(tokens):
                token_idx = len(tokens) - 1
            
            idxs.append([token_idx])
            
            # 调试信息
            '''print(f"\n[DEBUG] get_words_idxs_in_templates - Sample {i}")
            print(f"  Context template: {repr(context)}")
            print(f"  Word: {repr(word)}")
            print(f"  Full text: {repr(full_text)}")
            print(f"  Word char range: [{word_char_start}, {word_char_end}]")
            print(f"  Strategy: {subtoken}")
            print(f"  Selected token index: {token_idx}")
            
            if 0 <= token_idx < len(tokens):
                token_text = tok.decode([tokens[token_idx]])
                print(f"  Selected token: {repr(token_text)}")
                print(f"  Token ID: {tokens[token_idx]}")'''
            
        except Exception as e:
            # 如果offset_mapping失败，使用回退方法
            print(f"[WARN] Offset mapping failed: {e}, using fallback method")
            
            # 回退到原始方法（有局限性，但作为备份）
            if "{}" in context:
                formatted_context = context.format(word)
            else:
                formatted_context = context + " " + word if word else context
            
            context_tokens = tok.encode(formatted_context)
            word_tokens = tok.encode(word)
            
            # 移除word tokens中可能的特殊token
            clean_word_tokens = []
            for token in word_tokens:
                if token not in [tok.bos_token_id, tok.eos_token_id, tok.unk_token_id]:
                    clean_word_tokens.append(token)
            
            if not clean_word_tokens:
                idxs.append([len(context_tokens) - 1])
                continue
                
            # 在context tokens中查找word tokens
            found_positions = []
            for j in range(len(context_tokens) - len(clean_word_tokens) + 1):
                if context_tokens[j:j+len(clean_word_tokens)] == clean_word_tokens:
                    found_positions.append(j)
            
            if not found_positions:
                idxs.append([len(context_tokens) - 1])
                continue
                
            # 根据策略选择位置
            match_start = found_positions[0]
            match_end = match_start + len(clean_word_tokens) - 1
            
            if subtoken == "first":
                idx = match_start
            elif subtoken == "last":
                idx = match_end
            elif subtoken == "first_after_last":
                idx = match_end + 1
                if idx >= len(context_tokens):
                    idx = len(context_tokens) - 1
            else:
                idx = match_end
                
            idxs.append([idx])
            
            '''print(f"\n[DEBUG] Fallback method - Sample {i}")
            print(f"  Context: '{formatted_context}'")
            print(f"  Word: '{word}' -> tokens: {clean_word_tokens}")
            print(f"  Found at positions: {found_positions}")
            print(f"  Using strategy '{subtoken}' -> index: {idx}")
            print(f"  Token at index: '{tok.decode([context_tokens[idx]])}'")'''
    
    return idxs


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        if cur_repr.shape[0]!=len(batch_idxs):
            cur_repr=cur_repr.transpose(0,1)
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            if isinstance(model, GPTJForCausalLM) and module_name == 'transformer.h.8':
                with torch.no_grad():
                    with nethook.Trace(
                        module=model,
                        layer=module_name + '.ln_1',
                        retain_input=tin,
                        retain_output=tout,
                    ) as tr2:
                        model(**contexts_tok)
                tr.input = tr2.input

            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]