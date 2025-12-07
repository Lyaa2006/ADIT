"""
Contains evaluation utilities for pytorch-based rewriting methods.
ADIT兼容版本：支持动态LoRA模型的评估
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity
import torch.nn.functional as F


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    ADIT兼容版本的评估函数：支持动态LoRA模型
    接口与原始版本完全一致
    """
    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]
    
    # 检查是否为ADIT动态模型
    is_adit_dynamic = check_if_adit_model(model)
    
    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    
    # Flatten all the evaluated prefixes into one list.
    flat_prefixes = list(chain(*prob_prompts))
    flat_which_correct = list(chain(*which_correct))
    
    # 根据模型类型选择评估策略
    if is_adit_dynamic:
        # ADIT动态模型：逐个prompt处理
        probs, targets_correct = test_batch_prediction_adit(
            model,
            tok,
            flat_prefixes,
            flat_which_correct,
            target_new["str"],
            target_true["str"],
        )
    else:
        # 原始静态模型：批量处理
        probs, targets_correct = test_batch_prediction_original(
            model,
            tok,
            flat_prefixes,
            flat_which_correct,
            target_new["str"],
            target_true["str"],
        )
    
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    
    # Structure the results as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }
    
    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        
        # 根据模型类型选择生成测试策略
        if is_adit_dynamic:
            gen_stats = test_generation_adit(
                model,
                tok,
                generation_prompts,
                consistency_texts,
                essence_texts,
                vec,
            )
        else:
            gen_stats = test_generation_original(
                model,
                tok,
                generation_prompts,
                consistency_texts,
                essence_texts,
                vec,
            )
        ret.update(gen_stats)
    
    return ret


def check_if_adit_model(model):
    """检查模型是否为ADIT动态模型"""
    # 检查ADIT特定属性
    if hasattr(model, 'is_adit_model'):
        return True
    if hasattr(model, '_adit_editor'):
        return True
    if hasattr(model, 'generate_lora_for_prompt'):
        return True
    if hasattr(model, 'set_current_prefix'):
        return True
    
    # 检查是否有动态层
    try:
        # 遍历模型寻找DynamicLoRALayer实例
        for module in model.modules():
            if 'DynamicLoRALayer' in str(type(module)):
                return True
    except:
        pass
    
    return False


def test_batch_prediction_adit(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: typing.List[int],
    target_new: str,
    target_true: str,
):
    """
    ADIT兼容版本：为每个prefix单独生成LoRA权重
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """
    print(f"[ADIT-EVAL] Using dynamic evaluation for {len(prefixes)} prefixes")
    
    probs = np.zeros((len(prefixes) * 2,), dtype=np.float32)  # 每个prefix有两个目标
    targets_correct = []
    
    # 确保which_correct是列表
    if not isinstance(which_correct, list):
        which_correct = [which_correct] * len(prefixes)
    
    # 逐个处理每个prefix
    for i, prefix in enumerate(prefixes):
        correct_for_this_prefix = which_correct[i]
        
        # 如果模型支持设置当前prefix，则设置
        if hasattr(model, 'set_current_prefix'):
            model.set_current_prefix(prefix)
        
        # 计算target_new的概率
        prob_new = compute_single_probability(
            model, tok, prefix, target_new
        )
        
        # 计算target_true的概率
        prob_true = compute_single_probability(
            model, tok, prefix, target_true
        )
        
        # 存储结果
        probs[i * 2] = prob_new
        probs[i * 2 + 1] = prob_true
        
        # 检查准确性
        if correct_for_this_prefix == 0:
            is_correct = check_single_accuracy(model, tok, prefix, target_new)
        else:
            is_correct = check_single_accuracy(model, tok, prefix, target_true)
        
        targets_correct.append(is_correct)
        
        # 清理当前prefix（如果支持）
        if hasattr(model, 'clear_current_prefix'):
            model.clear_current_prefix()
    
    # 格式化为与原始版本一致的返回格式
    result_list = [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ]
    
    return result_list, targets_correct


def compute_single_probability(model, tok, prefix: str, target: str):
    """计算单个prefix+target的平均负对数概率"""
    # 构建完整文本
    full_text = prefix + target
    
    # 编码
    inputs = tok(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # 获取token IDs
    prefix_tokens = tok(prefix, add_special_tokens=False)["input_ids"]
    target_tokens = tok(f" {target}", add_special_tokens=False)["input_ids"]
    
    # 处理Llama/Mistral的特殊情况
    if 'llama' in str(model.config._name_or_path).lower() or 'mistral' in str(model.config._name_or_path).lower():
        # Llama分词器会给target前面加空格，这里调整
        if len(target_tokens) > 0 and target_tokens[0] == tok.unk_token_id:
            target_tokens = target_tokens[1:]
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 处理Llama的特殊偏移
    if 'llama' in str(model.config._name_or_path).lower() or 'mistral' in str(model.config._name_or_path).lower():
        # Llama模型的logits可能有一个偏移
        logits = logits[:, 1:, :] if logits.shape[1] == input_ids.shape[1] else logits
    
    # 计算target部分的负对数概率
    total_neg_log_prob = 0.0
    seq_len = input_ids.shape[1]
    
    for j in range(len(target_tokens)):
        # 计算token在序列中的位置
        token_idx = len(prefix_tokens) + j
        
        # 确保位置有效
        if token_idx >= seq_len - 1:  # -1 因为logits长度比input_ids少1
            break
        
        current_token = target_tokens[j]
        
        # 获取该位置的logits并计算log softmax
        log_probs = F.log_softmax(logits[0, token_idx, :], dim=0)
        
        # 添加负对数概率
        token_log_prob = log_probs[current_token].item()
        total_neg_log_prob += -token_log_prob
    
    # 返回平均负对数概率（越小越好）
    if len(target_tokens) > 0:
        avg_neg_log_prob = total_neg_log_prob / len(target_tokens)
    else:
        avg_neg_log_prob = float('inf')
    
    return avg_neg_log_prob


def check_single_accuracy(model, tok, prefix: str, target: str):
    """检查模型是否准确生成target"""
    # 构建完整文本
    full_text = prefix + target
    
    # 编码
    inputs = tok(full_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # 获取token IDs
    prefix_tokens = tok(prefix, add_special_tokens=False)["input_ids"]
    target_tokens = tok(f" {target}", add_special_tokens=False)["input_ids"]
    
    # 处理Llama/Mistral的特殊情况
    if 'llama' in str(model.config._name_or_path).lower() or 'mistral' in str(model.config._name_or_path).lower():
        if len(target_tokens) > 0 and target_tokens[0] == tok.unk_token_id:
            target_tokens = target_tokens[1:]
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 处理Llama的特殊偏移
    if 'llama' in str(model.config._name_or_path).lower() or 'mistral' in str(model.config._name_or_path).lower():
        logits = logits[:, 1:, :] if logits.shape[1] == input_ids.shape[1] else logits
    
    # 检查每个target token
    seq_len = input_ids.shape[1]
    
    for j in range(len(target_tokens)):
        token_idx = len(prefix_tokens) + j
        
        if token_idx >= seq_len - 1:
            break
        
        current_token = target_tokens[j]
        predicted_token = logits[0, token_idx, :].argmax().item()
        
        if predicted_token != current_token:
            return False
    
    return True


def test_batch_prediction_original(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: typing.List[int],
    target_new: str,
    target_true: str,
):
    """
    原始版本的批量预测测试（用于静态模型）
    保持与原始代码完全一致
    """
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    )
    
    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    
    if 'llama' in model.config._name_or_path.lower():
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [lengths - 1 for lengths in prefix_lens]
    
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    
    if 'llama' in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]
    
    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []
    
    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        
        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len
        
        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                
                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)
    
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation_adit(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    """ADIT兼容的生成测试"""
    # 对于ADIT模型，我们也需要逐个处理prompt
    gen_texts = []
    
    for prefix in prefixes:
        # 如果模型支持设置当前prefix，则设置
        if hasattr(model, 'set_current_prefix'):
            model.set_current_prefix(prefix)
        
        # 生成文本
        generated = generate_fast(
            model,
            tok,
            [prefix],
            n_gen_per_prompt=1,
            max_out_len=100,
        )
        gen_texts.extend(generated)
        
        # 清理当前prefix
        if hasattr(model, 'clear_current_prefix'):
            model.clear_current_prefix()
    
    # 计算指标
    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )
    
    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }
    
    if len(essence_texts) > 0:
        # 对于ADIT模型，评估本质性时需要特别处理
        if hasattr(model, 'set_current_prefix'):
            # 使用第一个prefix作为参考
            model.set_current_prefix(prefixes[0] if len(prefixes) > 0 else "")
        
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})
        
        if hasattr(model, 'clear_current_prefix'):
            model.clear_current_prefix()
    
    return ret


def test_generation_original(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    """原始版本的生成测试"""
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )
    
    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )
    
    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }
    
    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})
    
    return ret


# 以下函数与原始版本保持完全一致
def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]
    
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]
    
    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        
        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))
    
    entropy_list = np.array(entropy_list) * np.array(weights)
    
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()