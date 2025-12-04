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
    ROME-style compute_ks:
    ----------------------
    为每个 request 生成若干带 subject 的上下文（如果没有 templates，回退到 prompt），
    在指定 layer 上提取 module 输入表示（pre-MLP），并对同一 request 的多个模板取平均，
    返回 shape [len(requests), hidden_size] 的 ks 矩阵（float tensor，device 与 model 相同）。
    """

    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size

    # Build per-request formatted contexts and track counts per request
    all_contexts = []   # flattened list of context strings (fully formatted or raw; get_module_input_output_at_words handles both)
    all_words = []      # corresponding subject words (one per context)
    counts = []         # number of contexts per request, to regroup later

    # If context_templates is None or empty, we'll fallback to each request's prompt
    templates_provided = bool(context_templates)

    for req in requests:
        subj = req.get("subject", "") or ""
        # Determine templates for this request
        if templates_provided:
            # use provided templates list; allow non-string filtering
            this_templates = [t for t in context_templates if isinstance(t, str) and t.strip()]
            if not this_templates:
                # fallback to prompt
                this_templates = []
        else:
            this_templates = []

        if not this_templates:
            # fallback: use request['prompt'] if available, otherwise a generic template that includes subject
            raw_prompt = req.get("prompt", "") or ""
            if raw_prompt:
                this_templates = [raw_prompt]
            else:
                # last resort: simple template
                this_templates = ["{}"]

        # For each template, prefer to keep it as-is (get_module_input_output_at_words will format
        # if "{}" present); but to be robust, if template does not contain "{}" and does not contain subject,
        # we append subject to ensure subject appears at inference time.
        safe_templates = []
        for t in this_templates:
            try:
                if "{}" in t:
                    safe_templates.append(t)
                else:
                    # if subject already in template, keep; else append subject to make sure it's present
                    if subj and subj in t:
                        safe_templates.append(t)
                    else:
                        # append a space + subject so subject will appear in the string
                        safe_templates.append(t + " " + subj if subj else t)
            except Exception:
                # on any format-related error, fallback to "{}"
                safe_templates.append("{}")

        # record
        for templ in safe_templates:
            all_contexts.append(templ)
            all_words.append(subj)
        counts.append(len(safe_templates))

    if not all_contexts:
        # nothing to do
        return torch.zeros((len(requests), hidden_size), device=device, dtype=torch.float32)

    # Call helper to get module input/output at the word positions
    try:
        input_vecs, output_vecs = get_module_input_output_at_words(
            model=model,
            tok=tok,
            layer=layer,
            context_templates=all_contexts,  # can be formatted strings or templates with {}
            words=all_words,
            module_template=hparams.rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )
        # Ensure tensors on model device and float32
        input_vecs = input_vecs.to(device)
    except Exception as e:
        # On failure, return zero keys
        # Keep function robust: catch and return zeros
        # (caller should handle zero ks appropriately)
        # Optionally print minimal error for debugging
        print(f"[compute_ks] error in get_module_input_output_at_words: {e}")
        return torch.zeros((len(requests), hidden_size), device=device, dtype=torch.float32)

    # input_vecs shape: [sum_counts, hidden_size]
    # Now aggregate per original request using counts
    ks_list = []
    offset = 0
    for c in counts:
        if c <= 0:
            ks_list.append(torch.zeros(hidden_size, device=device))
            continue
        slice_vecs = input_vecs[offset: offset + c]  # shape [c, hidden]
        # mean across templates for this request
        ks_mean = slice_vecs.mean(dim=0)
        ks_list.append(ks_mean)
        offset += c

    # If for any reason offset != input_vecs.size(0), we handle gracefully (pad zeros)
    if len(ks_list) != len(requests):
        # fallback: pad or truncate
        new_list = []
        for i in range(len(requests)):
            if i < len(ks_list):
                new_list.append(ks_list[i])
            else:
                new_list.append(torch.zeros(hidden_size, device=device))
        ks_list = new_list

    ks = torch.stack(ks_list, dim=0)  # [len(requests), hidden_size]
    return ks
