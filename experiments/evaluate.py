"""
ADIT Enhanced Evaluation Script
支持向量指导的ADIT版本评估
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union, Dict, List
import numpy as np
import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from util import nethook
from util.globals import *
from ADIT.ADIT_main import apply_ADIT_to_model, ADITEditor
from ADIT.ADIT_hparams import ADITHyperParams

ALG_DICT = {
    "ADIT": (ADITHyperParams, apply_ADIT_to_model), 
}

DS_DICT = {
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
}

def create_batch_item_from_request(request: Dict) -> Dict:
    """从请求字典创建BatchItem所需的数据"""
    prompt = request['prompt']
    subject = request.get('subject', '')
    
    # 格式化prompt
    if subject and "{}" in prompt:
        formatted_prompt = prompt.format(subject)
    else:
        formatted_prompt = prompt
    
    target_new = request['target_new']
    if isinstance(target_new, dict):
        target_new = target_new.get('str', '')
    
    # 规范化target（确保以空格开头）
    def _norm_target(s: str):
        s = s or ""
        s = s.rstrip("\n")
        return s if s.startswith(" ") else " " + s
    
    return {
        'prompt_template': prompt,  # 原始模板
        'prompt_formatted': formatted_prompt,  # 格式化后的prompt
        'subject': subject,
        'target_new': _norm_target(target_new),
        'locality_prompts': request.get('locality_prompts', []) or [],
        'neighbor_prompts': request.get('neighbor_prompts', []) or [],
    }

def evaluate_with_editor(editor: ADITEditor, record: Dict, ds_eval_method, 
                        gen_test_vars: List, generation_test_interval: int) -> Dict:
    """使用ADIT编辑器进行评估"""
    requested_rewrite = record["requested_rewrite"]
    
    # 创建评估用的BatchItem数据
    batch_data = create_batch_item_from_request(requested_rewrite)
    
    # 准备评估（激活LE适配器）
    try:
        # 构建上下文
        from dataclasses import dataclass
        
        @dataclass
        class EvalBatchItem:
            prompt: str
            target_new: str
            subject:str
        
        # 创建临时BatchItem对象
        eval_item = EvalBatchItem(
            prompt=batch_data['prompt_formatted'],
            subject=batch_data['subject'],
            target_new=batch_data['target_new']
        )
        
        # 构建上下文并准备LE权重
        ctx = editor.build_guided_context(eval_item, requested_rewrite)
        le_weights = editor.hyper(ctx)
        
        # 绑定LE权重到所有LoRA主机
        for name, host in editor.lora_hosts.items():
            A, B = le_weights[name]
            host.set_adapter_weights("LE", A, B)
        
        # 只激活LE适配器进行评估
        editor._deactivate_all()
        editor._activate(["LE"])
        
        # 执行评估
        post_results = ds_eval_method(
            editor.base_model,  # 使用基础模型，但已绑定LE权重
            editor.tokenizer,
            record,
            *(
                gen_test_vars
                if record["case_id"] % generation_test_interval == 0
                else [None, None]
            ),
        )
        
    except Exception as e:
        print(f"[ERROR] Evaluation with editor failed: {e}")
        # 回退到使用编辑后的模型
        editor._deactivate_all()
        post_results = ds_eval_method(
            editor.base_model,  # 使用原始模型
            editor.tokenizer,
            record,
            *(
                gen_test_vars
                if record["case_id"] % generation_test_interval == 0
                else [None, None]
            ),
        )
    finally:
        # 清理：停用所有适配器
        editor._deactivate_all()
    
    return post_results

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    use_vector_guidance: bool = True,  # 新增：是否使用向量指导
    vector_guidance_weight: float = 0.3,  # 新增：向量指导权重
    vector_alignment_weight: float = 0.1,  # 新增：向量对齐权重
):
    """
    ADIT主评估函数
    
    Args:
        use_vector_guidance: 是否使用向量指导
        vector_guidance_weight: 向量指导权重（0-1）
        vector_alignment_weight: 向量对齐损失权重（0-1）
    """
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    
    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    
    # 添加向量指导参数到hparams
    hparams.use_vector_guidance = use_vector_guidance
    hparams.vector_guidance_weight = vector_guidance_weight
    hparams.vector_alignment_weight = vector_alignment_weight
    
    # 保存更新后的参数
    if not (run_dir / "params.json").exists():
        with open(run_dir / "params.json", "w") as f:
            json.dump(hparams.__dict__, f, indent=2)
    print(f"Executing {alg_name} with parameters:")
    print(f"  - use_vector_guidance: {hparams.use_vector_guidance}")
    print(f"  - vector_guidance_weight: {hparams.vector_guidance_weight}")
    print(f"  - vector_alignment_weight: {hparams.vector_alignment_weight}")
    print(f"  - Other params: {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if conserve_memory else torch.float32
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # 根据模型自动调整ctx_dim
    model_hidden_size = getattr(model.config, 'hidden_size', 768)
    if hasattr(hparams, 'ctx_dim') and hparams.ctx_dim != model_hidden_size:
        print(f"[INFO] Adjusting ctx_dim from {hparams.ctx_dim} to model hidden_size {model_hidden_size}")
        hparams.ctx_dim = model_hidden_size

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit)

    cnt = 0
    editor = None
    
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        print(f"\n{'='*80}")
        print(f"Edit {cnt+1}: Processing {len(record_chunks)} records")
        print(f"{'='*80}")
        
        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            print(f"Skipping already finished cases")
            continue
        
        # Apply ADIT algorithm and get editor
        start = time()
        
        # 准备请求数据
        requests = []

        for record in record_chunks:
            requested_rewrite = record["requested_rewrite"]
    
    # 从record顶层获取paraphrase_prompts
            paraphrase_prompts = record.get("paraphrase_prompts", [])
            neighbor_prompts=record.get("neighborhood_prompts",[])
            print(neighbor_prompts)
            
            if isinstance(requested_rewrite, list):
                for rewrite in requested_rewrite:
                    requests.append({
                "case_id": record["case_id"],
                **rewrite,
                "paraphrase_prompts": paraphrase_prompts,
                "neighbor_prompts":neighbor_prompts# ✅ 添加
            })
            else:
                requests.append({
            "case_id": record["case_id"],
            **requested_rewrite,
            "paraphrase_prompts": paraphrase_prompts,
           "neighbor_prompts":neighbor_prompts# ✅ 添加
        })
        print(f"Applying ADIT with vector guidance={use_vector_guidance}")
        edited_model, editor = apply_algo(
            model,
            tok,
            requests,
            hparams,
            conserve_memory=conserve_memory,
            use_vector_guidance=use_vector_guidance,
            vector_guidance_weight=vector_guidance_weight,
            vector_alignment_weight=vector_alignment_weight,
        )
        
        exec_time = time() - start
        cnt += 1
        print(f"Execution took {exec_time:.2f} seconds")
        
        # Evaluate using the returned editor
        gen_test_vars = [snips, vec]
        for record in record_chunks:
            out_file = Path(case_result_template.format(num_edits, record["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue
            
            eval_start = time()
            try:
                if editor is not None:
                    # 使用编辑器进行评估
                    print(f"Evaluating case {record['case_id']} with ADIT editor")
                    post_results = evaluate_with_editor(
                        editor, 
                        record, 
                        ds_eval_method,
                        gen_test_vars,
                        generation_test_interval
                    )
                else:
                    # 回退：使用编辑后的模型进行评估
                    print(f"Evaluating case {record['case_id']} with edited model")
                    post_results = ds_eval_method(
                        edited_model,
                        tok,
                        record,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),
                    )
                
                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": [r["case_id"] for r in record_chunks],
                    "num_edits": num_edits,
                    "requested_rewrite": record["requested_rewrite"],
                    "time": exec_time,
                    "post": post_results,
                    "params": {
                        "use_vector_guidance": use_vector_guidance,
                        "vector_guidance_weight": vector_guidance_weight,
                        "vector_alignment_weight": vector_alignment_weight,
                    }
                }
                
                # 保存结果
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                
                eval_time = time() - eval_start
                print(f"Evaluation for case {record['case_id']} took {eval_time:.2f} seconds")
                
                # 打印关键指标
                if "rewrite" in post_results:
                    rewrite_acc = post_results["rewrite"]["precise"]
                    print(f"  Rewrite accuracy: {rewrite_acc:.2%}")
                
            except Exception as e:
                print(f"[ERROR] Failed to evaluate case {record['case_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 清理内存
        if conserve_memory:
            torch.cuda.empty_cache()


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADIT Enhanced Evaluation")
    parser.add_argument(
        "--alg_name",
        choices=["ADIT"],
        default="ADIT",
        help="Editing algorithm to use.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-large",  # 默认使用GPT2-large
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-large.json",
        help="Name of hyperparameters file.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf"],
        default="cf",
        help="Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=100,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Skip slow generation tests.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    # 新增：向量指导参数
    parser.add_argument(
        "--use_vector_guidance",
        dest="use_vector_guidance",
        action="store_true",
        default=True,
        help="Use vector guidance from compute_ks/compute_z",
    )
    parser.add_argument(
        "--no_vector_guidance",
        dest="use_vector_guidance",
        action="store_false",
        help="Disable vector guidance",
    )
    parser.add_argument(
        "--vector_guidance_weight",
        type=float,
        default=0.3,
        help="Weight for vector guidance (0.0-1.0)",
    )
    parser.add_argument(
        "--vector_alignment_weight",
        type=float,
        default=0.1,
        help="Weight for vector alignment loss (0.0-1.0)",
    )
    parser.set_defaults(
        skip_generation_tests=False, 
        conserve_memory=False,
        use_vector_guidance=True
    )
    args = parser.parse_args()

    main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        continue_from_run=args.continue_from_run,
        skip_generation_tests=args.skip_generation_tests,
        generation_test_interval=args.generation_test_interval,
        conserve_memory=args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        use_vector_guidance=args.use_vector_guidance,
        vector_guidance_weight=args.vector_guidance_weight,
        vector_alignment_weight=args.vector_alignment_weight,
    )