# -*- coding: utf-8 -*-
"""
TokenEdit evaluation script following AlphaEdit's evaluation logic
Implements comprehensive metrics: 
1. Efficacy/Generalization (Loose): Probability Comparison (P_new > P_old)
2. Efficacy/Generalization (Strict): Argmax Exact Match (Top-1)
"""
import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from model_config import load_model_optimized
    from tokenedit import TokenEditEditor, TokenEditHyperParams
except ImportError as e:
    print(f"Error: Cannot import required modules")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    sys.exit(1)


def _json_default(o: Any):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def load_hparams_from_json(model_name: str, hparams_dir: str = "hparams/TokenEdit"):
    """Load hyperparameters from JSON file"""
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"Warning: Config file not found {hparams_path}, using default values")
        return TokenEditHyperParams(model_name=model_name)

    print(f"Loading config from {hparams_path}")
    with open(hparams_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 打印部分关键参数确认
    print(f"  Config parameters:")
    print(f"    - target_layers: {config.get('target_layers', 'not set')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")

    return TokenEditHyperParams(**config)


def load_data(num_samples=100, data_dir: str = "data"):
    """
    Load CounterFact dataset
    """
    # Try to load CounterFact dataset
    data_path = Path(data_dir) / "counterfact.json"
    sample_path = Path(data_dir) / "sample_data.json"

    # Auto-download if not exists
    if not data_path.exists():
        print(f"CounterFact dataset not found at {data_path}")
        print("Attempting to download...")
        try:
            import requests
            data_dir = Path(data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
            url = "https://rome.baulab.info/data/dsets/counterfact.json"
            print(f"Downloading from {url}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
            print(f"Downloaded CounterFact dataset: {len(response.json())} samples")
        except Exception as e:
            print(f"Failed to download: {e}")
            print(f"Using sample data from {sample_path}")
            data_path = sample_path

    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {data_path.name}")

    # Convert to request format
    requests = []
    for item in data[:num_samples]:
        req = item['requested_rewrite']
        requests.append({
            'case_id': item.get('case_id', len(requests)),
            'prompt': req['prompt'],
            'subject': req['subject'],
            'relation_id': req.get('relation_id', 'P36'),
            'target_new': req['target_new']['str'],
            'target_new_id': req['target_new']['id'],
            'target_true': req['target_true']['str'],
            'target_true_id': req['target_true']['id'],
            'paraphrase_prompts': item.get('paraphrase_prompts', []),
            'neighborhood_prompts': item.get('neighborhood_prompts', []),
            'generation_prompts': item.get('generation_prompts', []),
        })
    return requests


def test_batch_prediction_multi(
    editor: TokenEditEditor,
    prefixes: List[str],
    targets_new: List[str],
    targets_true: List[str],
    which_correct: List[int] = None,
) -> Tuple[List[Dict], List[bool], List[bool]]:
    """
    ULTRA-FAST batch prediction with BOTH Probability and Argmax checks.
    
    Args:
        editor: TokenEdit editor
        prefixes: List of all test prompts
        targets_new: List of target_new strings
        targets_true: List of target_true strings
        which_correct: List of 0/1 (0=target_new, 1=target_true)

    Returns:
        (probs, prob_corrects, argmax_corrects) tuple
    """
    probs = []
    prob_corrects = []    # 基于概率对比 (宽松)
    argmax_corrects = []  # 基于 Argmax (严苛)

    # 预计算所有目标的第一个 Token ID (用于 Argmax 比较)
    # 注意：这里假设 encode 后的第一个 token 就是我们要预测的词
    target_new_first_ids = [editor.tokenizer.encode(t, add_special_tokens=False)[0] for t in targets_new]
    target_true_first_ids = [editor.tokenizer.encode(t, add_special_tokens=False)[0] for t in targets_true]

    for i, (prefix, target_new, target_true) in enumerate(zip(prefixes, targets_new, targets_true)):
        # === 1. 路由与注入 ===
        # Get prompt embedding for routing
        inputs = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)

        with torch.no_grad():
            outputs = editor.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        # Route
        edit_id = editor.router.route(prefix, prompt_emb)

        # Inject if triggered
        if edit_id is not None:
            req = editor.edits_registry[edit_id]
            subject_positions = editor.utils.find_subject_positions(
                prefix,
                req['subject'],
                verbose=False,
                add_special_tokens=True
            )

            if subject_positions:
                editor.injector.inject(
                    editor.model,
                    edit_id,
                    editor.edit_module,
                    subject_positions
                )

        # === 2. 准备输入 (构造 "Prefix + Target" 用于计算 Perplexity) ===
        all_texts = [f"{prefix} {target_new}", f"{prefix} {target_true}"]
        all_inputs = editor.tokenizer(all_texts, padding=True, return_tensors="pt").to(editor.device)

        # 获取 prefix 长度，用于定位预测位置
        prefix_len = len(editor.tokenizer(prefix, add_special_tokens=True)["input_ids"])

        # Tokenize targets (用于 LogProb 计算)
        target_new_tokens = editor.tokenizer(target_new, add_special_tokens=False)["input_ids"]
        target_true_tokens = editor.tokenizer(target_true, add_special_tokens=False)["input_ids"]

        # === 3. Forward Pass ===
        with torch.no_grad():
            outputs = editor.model(**all_inputs)
            all_logits = outputs.logits

        # 清除注入
        editor.injector.clear()

        # === 4. 计算指标 ===
        logits_new = all_logits[0]  # Prefix + TargetNew
        logits_true = all_logits[1]  # Prefix + TargetTrue

        # --- A. Argmax 判定 (严苛指标) ---
        # 我们看 prefix 最后一个 token 输出的 logits，它预测的是下一个词
        # 只要看 logits_new[prefix_len - 1] 即可
        next_token_logits = logits_new[prefix_len - 1, :]
        pred_token_id = torch.argmax(next_token_logits).item()
        
        # 确定期望的 Target ID
        if which_correct is None or which_correct[i] == 0:
            expected_id = target_new_first_ids[i]  # 期望是新知识
        else:
            expected_id = target_true_first_ids[i] # 期望是旧知识 (Neighborhood)
            
        argmax_corrects.append(pred_token_id == expected_id)

        # --- B. 概率对比判定 (宽松指标) ---
        # 计算 Target New 的平均 LogProb
        log_probs_new = []
        for j, tok in enumerate(target_new_tokens):
            if prefix_len + j - 1 < logits_new.shape[0]:
                log_prob = torch.nn.functional.log_softmax(
                    logits_new[prefix_len + j - 1, :], dim=0
                )[tok].item()
                log_probs_new.append(log_prob)
        prob_new = np.mean(log_probs_new) if len(log_probs_new) > 0 else -999.0

        # 计算 Target True 的平均 LogProb
        log_probs_true = []
        for j, tok in enumerate(target_true_tokens):
            if prefix_len + j - 1 < logits_true.shape[0]:
                log_prob = torch.nn.functional.log_softmax(
                    logits_true[prefix_len + j - 1, :], dim=0
                )[tok].item()
                log_probs_true.append(log_prob)
        prob_true = np.mean(log_probs_true) if len(log_probs_true) > 0 else -999.0

        probs.append({
            "target_new": prob_new,
            "target_true": prob_true
        })

        # 判断概率大小
        if which_correct is None or which_correct[i] == 0:
            is_prob_correct = (prob_new > prob_true)
        else:
            is_prob_correct = (prob_true > prob_new)
        prob_corrects.append(is_prob_correct)

    return probs, prob_corrects, argmax_corrects


def compute_batch_rewrite_quality(
    editor: TokenEditEditor,
    records: List[Dict],
    skip_generation: bool = False,
) -> List[Dict]:
    """
    Compute rewrite quality for a BATCH of records.
    """
    all_prompts = []
    all_targets_new = []
    all_targets_true = []
    all_correct = []
    
    # Organize all prompts from all records
    for record_idx, record in enumerate(records):
        subject = record['subject']
        target_new = record['target_new']
        target_true = record['target_true']

        # Test prompts
        rewrite_prompt = record['prompt'].format(subject)
        paraphrase_prompts = record.get('paraphrase_prompts', [])[:5]
        neighborhood_prompts = record.get('neighborhood_prompts', [])[:5]

        # Handle neighborhood_prompts format
        if neighborhood_prompts and isinstance(neighborhood_prompts[0], dict):
            neighborhood_prompts_list = [nb['prompt'] for nb in neighborhood_prompts]
        else:
            neighborhood_prompts_list = neighborhood_prompts

        # Collect all prompts
        test_prompts = [
            [rewrite_prompt],           # rewrite_prompts
            paraphrase_prompts,         # paraphrase_prompts
            neighborhood_prompts_list   # neighborhood_prompts
        ]

        # Mark correct targets (0=New, 1=True)
        which_correct = [
            [0],  # rewrite: target_new
            [0] * len(paraphrase_prompts),  # paraphrase: target_new
            [1] * len(neighborhood_prompts_list),  # neighborhood: target_true
        ]

        # Flatten
        for i, prompts in enumerate(test_prompts):
            for prompt in prompts:
                all_prompts.append(prompt)
                all_targets_new.append(target_new)
                all_targets_true.append(target_true)
                if i == 0:
                    all_correct.append(0)
                elif i == 1:
                    all_correct.append(0)
                else:
                    all_correct.append(1)

    # Batch compute all metrics
    probs, prob_corrects, argmax_corrects = test_batch_prediction_multi(
        editor,
        all_prompts,
        all_targets_new,
        all_targets_true,
        all_correct
    )

    # Reorganize results by record
    metrics_list = []
    prompt_idx = 0

    for record_idx, record in enumerate(records):
        rewrite_prompt = record['prompt'].format(record['subject'])
        paraphrase_prompts = record.get('paraphrase_prompts', [])[:5]
        neighborhood_prompts = record.get('neighborhood_prompts', [])[:5]

        num_prompts = 1 + len(paraphrase_prompts) + len(neighborhood_prompts)

        # Extract results for this record
        # 1. Prob Corrects (Loose)
        record_prob_correct = prob_corrects[prompt_idx:prompt_idx + num_prompts]
        ret_prob_corrects = [
            record_prob_correct[0:1],
            record_prob_correct[1:1+len(paraphrase_prompts)],
            record_prob_correct[1+len(paraphrase_prompts):num_prompts]
        ]

        # 2. Argmax Corrects (Strict)
        record_argmax_correct = argmax_corrects[prompt_idx:prompt_idx + num_prompts]
        ret_argmax_corrects = [
            record_argmax_correct[0:1],
            record_argmax_correct[1:1+len(paraphrase_prompts)],
            record_argmax_correct[1+len(paraphrase_prompts):num_prompts]
        ]
        
        # Extract Probs
        record_probs = probs[prompt_idx:prompt_idx + num_prompts]
        ret_probs = [
            record_probs[0:1],
            record_probs[1:1+len(paraphrase_prompts)],
            record_probs[1+len(paraphrase_prompts):num_prompts]
        ]

        # Structure results
        ret = {
            "rewrite_prompts_probs": ret_probs[0],
            "paraphrase_prompts_probs": ret_probs[1],
            "neighborhood_prompts_probs": ret_probs[2],
            
            # Loose Metrics
            "efficacy": np.mean(ret_prob_corrects[0]) if len(ret_prob_corrects[0]) > 0 else 0.0,
            "generalization": np.mean(ret_prob_corrects[1]) if len(ret_prob_corrects[1]) > 0 else 0.0,
            "specificity": np.mean(ret_prob_corrects[2]) if len(ret_prob_corrects[2]) > 0 else 0.0,
            
            # Strict Metrics (Argmax)
            "efficacy_strict": np.mean(ret_argmax_corrects[0]) if len(ret_argmax_corrects[0]) > 0 else 0.0,
            "generalization_strict": np.mean(ret_argmax_corrects[1]) if len(ret_argmax_corrects[1]) > 0 else 0.0,
            "specificity_strict": np.mean(ret_argmax_corrects[2]) if len(ret_argmax_corrects[2]) > 0 else 0.0,
        }

        metrics_list.append(ret)
        prompt_idx += num_prompts

    return metrics_list


def evaluate_model(
    model_name: str = "gpt2-xl",
    num_samples: int = 100,
    skip_generation: bool = True,
    num_epochs: int = None,
    eval_batch_size: int = 10
):
    """
    Main evaluation function
    """
    print("=" * 70)
    print(f"TokenEdit Evaluation - {model_name} (Hybrid Metrics)")
    print("=" * 70)

    # [1/4] Load model
    print("\n[1/4] Loading model...")
    model, tokenizer, _ = load_model_optimized(model_name)

    # [2/4] Load data
    print("\n[2/4] Loading data...")
    requests = load_data(num_samples)
    print(f"Loaded {len(requests)} edit samples")

    # [3/4] Create editor
    print("\n[3/4] Creating editor...")
    hparams = load_hparams_from_json(model_name)

    if num_epochs is not None:
        print(f"  Overriding num_epochs: {hparams.num_epochs} -> {num_epochs}")
        hparams.num_epochs = num_epochs

    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.verbose = True

    editor = TokenEditEditor(model, tokenizer, hparams)

    # [4/4] Apply edits
    print("\n[4/4] Applying edits...")
    start_time = time.time()
    try:
        editor.apply_edits(requests)
        edit_time = time.time() - start_time
        print(f"  Edits applied in {edit_time:.2f}s")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("  Out of memory! Try reducing samples or epochs")
            return
        raise

    # Evaluate with BATCHING
    print(f"\n[Evaluating] Computing rewrite quality metrics (batch_size={eval_batch_size})...")

    results = []
    # Loose Lists
    efficacy_list = []
    generalization_list = []
    specificity_list = []
    # Strict Lists
    efficacy_strict_list = []
    generalization_strict_list = []
    specificity_strict_list = []

    # Process in batches
    for batch_start in tqdm(range(0, len(requests), eval_batch_size), desc="Evaluating"):
        batch_end = min(batch_start + eval_batch_size, len(requests))
        batch_requests = requests[batch_start:batch_end]

        # Batch evaluate
        batch_metrics = compute_batch_rewrite_quality(
            editor,
            batch_requests,
            skip_generation=skip_generation
        )

        # Collect results
        for req, metrics in zip(batch_requests, batch_metrics):
            result = {
                "case_id": req['case_id'],
                "requested_rewrite": {
                    "prompt": req['prompt'],
                    "subject": req['subject'],
                    "target_new": req['target_new'],
                    "target_true": req['target_true']
                },
                "metrics": metrics
            }
            results.append(result)

            efficacy_list.append(metrics["efficacy"])
            generalization_list.append(metrics["generalization"])
            specificity_list.append(metrics["specificity"])
            
            efficacy_strict_list.append(metrics["efficacy_strict"])
            generalization_strict_list.append(metrics["generalization_strict"])
            specificity_strict_list.append(metrics["specificity_strict"])

    # Summary statistics
    summary = {
        "num_samples": num_samples,
        "num_epochs": hparams.num_epochs,
        "edit_time": edit_time,
        
        # Loose
        "efficacy_mean": np.mean(efficacy_list),
        "efficacy_std": np.std(efficacy_list),
        "generalization_mean": np.mean(generalization_list),
        "generalization_std": np.std(generalization_list),
        "specificity_mean": np.mean(specificity_list),
        "specificity_std": np.std(specificity_list),
        
        # Strict
        "efficacy_strict_mean": np.mean(efficacy_strict_list),
        "efficacy_strict_std": np.std(efficacy_strict_list),
        "generalization_strict_mean": np.mean(generalization_strict_list),
        "generalization_strict_std": np.std(generalization_strict_list),
        "specificity_strict_mean": np.mean(specificity_strict_list),
        "specificity_strict_std": np.std(specificity_strict_list),
        
        "results": results
    }

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print("LOOSE Metrics (Probability: P_new > P_old)")
    print(f"  Efficacy:       {summary['efficacy_mean']:.2%} ± {summary['efficacy_std']:.2%}")
    print(f"  Generalization: {summary['generalization_mean']:.2%} ± {summary['generalization_std']:.2%}")
    print(f"  Specificity:    {summary['specificity_mean']:.2%} ± {summary['specificity_std']:.2%}")
    print("-" * 70)
    print("STRICT Metrics (Argmax: Top-1 Match)")
    print(f"  Efficacy:       {summary['efficacy_strict_mean']:.2%} ± {summary['efficacy_strict_std']:.2%}")
    print(f"  Generalization: {summary['generalization_strict_mean']:.2%} ± {summary['generalization_strict_std']:.2%}")
    print(f"  Specificity:    {summary['specificity_strict_mean']:.2%} ± {summary['specificity_strict_std']:.2%}")
    print("=" * 70)
    print(f"Edit time:     {edit_time:.2f}s")

    # Save results
    Path("results").mkdir(exist_ok=True)
    results_file = f"results/tokenedit_{model_name.replace('/', '_')}_hybrid.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\nResults saved to: {results_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TokenEdit (Hybrid Metrics)")
    parser.add_argument('--model', type=str, default='gpt2-xl',
                       choices=['gpt2-xl', 'gpt-j-6b', 'llama3-8b'],
                       help='Model name')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of edit samples')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip generation tests (faster)')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Evaluation batch size')

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        num_samples=args.samples,
        skip_generation=args.skip_generation,
        num_epochs=args.epochs,
        eval_batch_size=args.batch_size
    )