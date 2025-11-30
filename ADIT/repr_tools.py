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
    ADIT改进版：直接在上下文中查找词语位置，支持大小写不敏感匹配
    """

    idxs = []
    
    for i, context in enumerate(context_templates):
        word = words[i]
        
        # 大小写不敏感匹配
        context_lower = context.lower()
        word_lower = word.lower()
        
        if word_lower in context_lower:
            # 找到word的起始位置
            start_idx = context_lower.index(word_lower)
            # 编码context到该位置的部分
            prefix = context[:start_idx]
            prefix_tokens = tok.encode(prefix)
            prefix_len = len(prefix_tokens)
            
            # 编码整个word来获取长度
            word_tokens = tok.encode(word)
            word_len = len(word_tokens)
            
            # 根据subtoken策略返回位置
            if subtoken == "last":
                # subject的最后一个token
                idx = prefix_len + word_len - 1
            elif subtoken == "first":
                # subject的第一个token  
                idx = prefix_len
            elif subtoken == "first_after_last":
                # subject后面的第一个token
                idx = prefix_len + word_len
            else:
                # 默认使用最后一个token
                idx = prefix_len + word_len - 1
                
            idxs.append([idx])
        else:
            # fallback: 使用最后一个token
            context_tokens = tok.encode(context)
            idxs.append([len(context_tokens) - 1])
    
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