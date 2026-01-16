# -*- coding: utf-8 -*-
"""
TokenEdit evaluation script - ARGMAX VERSION (Most Reliable)
Uses ARGMAX to check if predicted token exactly matches target
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
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn.functional as F
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

    print(f"  Config parameters:")
    print(f"    - target_layers: {config.get('target_layers', 'not set')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")
    print(f"    - w_edit: {config.get('w_edit', 1.0)}")
    print(f"    - w_suppress: {config.get('w_suppress', 0.5)}")

    return TokenEditHyperParams(**config)


def load_data(num_samples=100, data_dir: str = "data"):
    """
    Load CounterFact dataset

    Args:
        num_samples: Number of samples to load
        data_dir: Data directory path

    Returns:
        List of edit requests
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


def compute_rewrite_quality_argmax(
    editor: TokenEditEditor,
    record: Dict,
) -> Dict:
    """
    Compute rewrite quality using ARGMAX (most reliable method)
    
    For each prompt:
    1. Get logits with edit injection
    2. Check if argmax(logits) == target_first_token
    
    Args:
        editor: TokenEdit editor instance
        record: Single edit request record

    Returns:
        Dictionary containing evaluation metrics
    """
    subject = record['subject']
    target_new = record['target_new']
    target_true = record['target_true']

    # Test prompts
    rewrite_prompt = record['prompt'].format(subject)
    paraphrase_prompts = record.get('paraphrase_prompts', [])[:5]
    neighborhood_prompts = record.get('neighborhood_prompts', [])[:5]

    # Organize test prompts
    # Handle both string and dict formats for neighborhood_prompts
    neighborhood_prompts_list = []
    if neighborhood_prompts:
        for nb in neighborhood_prompts:
            if isinstance(nb, dict):
                neighborhood_prompts_list.append(nb['prompt'])
            elif isinstance(nb, str):
                neighborhood_prompts_list.append(nb)

    test_prompts = [
        [rewrite_prompt],
        paraphrase_prompts,
        neighborhood_prompts_list
    ]

    # 0 = should predict target_new, 1 = should predict target_true
    which_correct = [
        [0],  # rewrite_prompts
        [0] * len(test_prompts[1]),  # paraphrase_prompts
        [1] * len(test_prompts[2]),  # neighborhood_prompts
    ]

    # Flatten
    all_prompts = [p for prompts in test_prompts for p in prompts]
    all_correct = [c for corrects in which_correct for c in corrects]

    # CRITICAL: Use argmax-based evaluation
    probs, targets_correct, pred_tokens = test_batch_prediction_argmax(
        editor,
        all_prompts,
        all_correct,
        target_new,
        target_true
    )

    # Unflatten results
    cutoffs = [0] + np.cumsum(list(map(len, test_prompts))).tolist()
    ret_probs = [probs[cutoffs[i]:cutoffs[i+1]] for i in range(len(test_prompts))]
    ret_corrects = [targets_correct[cutoffs[i]:cutoffs[i+1]] for i in range(len(test_prompts))]
    ret_pred_tokens = [pred_tokens[cutoffs[i]:cutoffs[i+1]] for i in range(len(test_prompts))]

    # Structure results
    ret = {
        "rewrite_prompts_probs": ret_probs[0],
        "paraphrase_prompts_probs": ret_probs[1],
        "neighborhood_prompts_probs": ret_probs[2],
        "rewrite_prompts_correct": ret_corrects[0],
        "paraphrase_prompts_correct": ret_corrects[1],
        "neighborhood_prompts_correct": ret_corrects[2],
        "rewrite_prompts_pred": ret_pred_tokens[0],
        "paraphrase_prompts_pred": ret_pred_tokens[1],
        "neighborhood_prompts_pred": ret_pred_tokens[2],
    }

    # Compute summary metrics
    ret["efficacy"] = np.mean(ret_corrects[0]) if len(ret_corrects[0]) > 0 else 0.0
    ret["generalization"] = np.mean(ret_corrects[1]) if len(ret_corrects[1]) > 0 else 0.0
    ret["specificity"] = np.mean(ret_corrects[2]) if len(ret_corrects[2]) > 0 else 0.0

    return ret


def test_batch_prediction_argmax(
    editor: TokenEditEditor,
    prefixes: List[str],
    which_correct: List[int],
    target_new: str,
    target_true: str,
) -> tuple:
    """
    Test batch prediction using ARGMAX (most reliable)
    
    Method:
    1. For each prompt, inject edit and get logits
    2. predicted_token = argmax(logits)
    3. Check if predicted_token matches target_first_token
    
    Args:
        editor: TokenEdit editor
        prefixes: List of test prompts
        which_correct: 0 for target_new, 1 for target_true
        target_new: New target string
        target_true: True target string

    Returns:
        (probs, targets_correct, pred_tokens) tuple
    """
    probs = []
    targets_correct = []
    pred_tokens = []
    
    # Tokenize targets to get first token
    target_new_tokens = editor.tokenizer.encode(target_new, add_special_tokens=False)
    target_true_tokens = editor.tokenizer.encode(target_true, add_special_tokens=False)
    
    # Get first token ID (most important for evaluation)
    target_new_first = target_new_tokens[0] if len(target_new_tokens) > 0 else -1
    target_true_first = target_true_tokens[0] if len(target_true_tokens) > 0 else -1

    for i, prefix in enumerate(prefixes):
        # Step 1: Route to find which edit to apply
        inputs = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)
        
        with torch.no_grad():
            outputs = editor.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)
        
        edit_id = editor.router.route(prefix, prompt_emb)
        
        # Step 2: Find subject positions and prepare injection
        subject_positions = None
        if edit_id is not None:
            req = editor.edits_registry.get(edit_id)
            if req:
                subject_positions = editor.utils.find_subject_positions(
                    prefix,
                    req['subject'],
                    verbose=False,
                    add_special_tokens=True
                )
        
        # Step 3: Inject edit
        if edit_id is not None and subject_positions:
            editor.injector.inject(
                editor.model,
                edit_id,
                editor.edit_module,
                subject_positions
            )
        
        try:
            # Step 4: Forward pass to get logits
            inputs = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)
            
            with torch.no_grad():
                outputs = editor.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits (next token prediction)
            
            # Step 5: Get predicted token using ARGMAX
            predicted_token_id = torch.argmax(logits).item()
            predicted_token_str = editor.tokenizer.decode([predicted_token_id])
            
            # Step 6: Get probabilities for both targets
            probs_tensor = F.softmax(logits, dim=-1)
            prob_new = probs_tensor[target_new_first].item() if target_new_first >= 0 else 0.0
            prob_true = probs_tensor[target_true_first].item() if target_true_first >= 0 else 0.0
            
            probs.append({
                "target_new": prob_new,
                "target_true": prob_true,
                "predicted_token": predicted_token_str,
                "predicted_token_id": predicted_token_id
            })
            
            # Step 7: Determine correctness by comparing argmax with expected target
            if which_correct[i] == 0:
                # Should predict target_new
                is_correct = (predicted_token_id == target_new_first)
                expected = target_new
            else:
                # Should predict target_true (for neighborhood prompts)
                is_correct = (predicted_token_id == target_true_first)
                expected = target_true
            
            targets_correct.append(is_correct)
            pred_tokens.append({
                "predicted": predicted_token_str,
                "expected": expected,
                "match": is_correct
            })
            
        finally:
            # Step 8: Clear injection
            if edit_id is not None:
                editor.injector.clear()

    return probs, targets_correct, pred_tokens


def evaluate_model(
    model_name: str = "gpt2-xl",
    num_samples: int = 100,
    num_epochs: int = None,
):
    """
    Main evaluation function

    Args:
        model_name: Model name
        num_samples: Number of samples to evaluate
        num_epochs: Override num_epochs from config
    """
    print("=" * 70)
    print(f"TokenEdit Evaluation - ARGMAX Method (Most Reliable)")
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
        result = editor.apply_edits(requests)
        edit_time = time.time() - start_time
        print(f"  Edits applied in {edit_time:.2f}s")
        
        # Print training stats
        stats = result['stats']
        print(f"\n  Training stats:")
        print(f"    Initial loss: {stats['losses'][0]:.4f}")
        print(f"    Final loss: {stats['losses'][-1]:.4f}")
        print(f"    Loss reduction: {(stats['losses'][0] - stats['losses'][-1]) / stats['losses'][0] * 100:.1f}%")
        
        # Check edit module status
        print(f"\n  Edit module status:")
        print(f"    Number of edits: {editor.edit_module.num_edits}")
        print(f"    Alpha mean: {editor.edit_module.alpha.mean().item():.4f}")
        print(f"    Alpha min/max: {editor.edit_module.alpha.min().item():.4f} / {editor.edit_module.alpha.max().item():.4f}")
        print(f"    v_new norm mean: {torch.norm(editor.edit_module.v_new, dim=-1).mean().item():.4f}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("  Out of memory! Try reducing samples or epochs")
            return
        raise

    # Evaluate
    print(f"\n[Evaluating] Computing rewrite quality metrics (argmax-based)...")
    
    results = []
    efficacy_list = []
    generalization_list = []
    specificity_list = []

    # Process each sample
    for i, req in enumerate(tqdm(requests, desc="Evaluating")):
        metrics = compute_rewrite_quality_argmax(editor, req)
        
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
        
        # Debug output for first few samples
        if i < 5:
            print(f"\nSample {i}:")
            print(f"  Prompt: {req['prompt'].format(req['subject'])}")
            print(f"  Target new: {req['target_new']}, Target true: {req['target_true']}")
            print(f"  Efficacy: {metrics['efficacy']:.2%}")
            if len(metrics['rewrite_prompts_pred']) > 0:
                pred_info = metrics['rewrite_prompts_pred'][0]
                print(f"  Predicted: '{pred_info['predicted']}', Expected: '{pred_info['expected']}', Match: {pred_info['match']}")
            print(f"  Generalization: {metrics['generalization']:.2%}")
            print(f"  Specificity: {metrics['specificity']:.2%}")

    # Summary statistics
    summary = {
        "num_samples": num_samples,
        "num_epochs": hparams.num_epochs,
        "edit_time": edit_time,
        "efficacy_mean": np.mean(efficacy_list),
        "efficacy_std": np.std(efficacy_list),
        "generalization_mean": np.mean(generalization_list),
        "generalization_std": np.std(generalization_list),
        "specificity_mean": np.mean(specificity_list),
        "specificity_std": np.std(specificity_list),
        "results": results
    }

    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary (ARGMAX Method)")
    print("=" * 70)
    print(f"Efficacy:       {summary['efficacy_mean']:.2%} ± {summary['efficacy_std']:.2%}")
    print(f"Generalization: {summary['generalization_mean']:.2%} ± {summary['generalization_std']:.2%}")
    print(f"Specificity:    {summary['specificity_mean']:.2%} ± {summary['specificity_std']:.2%}")
    print(f"Edit time:      {edit_time:.2f}s")

    # Save results
    Path("results").mkdir(exist_ok=True)
    results_file = f"results/tokenedit_{model_name.replace('/', '_')}_argmax.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\nResults saved to: {results_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TokenEdit (ARGMAX Method)")
    parser.add_argument('--model', type=str, default='gpt2-xl',
                       choices=['gpt2-xl', 'gpt-j-6b', 'llama3-8b'],
                       help='Model name')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of edit samples')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: use JSON config)')

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        num_samples=args.samples,
        num_epochs=args.epochs
    )