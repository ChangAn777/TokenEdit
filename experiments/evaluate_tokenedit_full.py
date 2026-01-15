# -*- coding: utf-8 -*-
"""
TokenEdit evaluation script following AlphaEdit's evaluation logic
Implements comprehensive metrics: efficacy, generalization, specificity, generation quality
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
            'prompt': req['prompt'],  # Keep template format, e.g., "The capital of {} is"
            'subject': req['subject'],
            'relation_id': req.get('relation_id', 'P36'),  # Use dataset relation ID
            'target_new': req['target_new']['str'],
            'target_new_id': req['target_new']['id'],
            'target_true': req['target_true']['str'],
            'target_true_id': req['target_true']['id'],
            'paraphrase_prompts': item.get('paraphrase_prompts', []),
            'neighborhood_prompts': item.get('neighborhood_prompts', []),
            'generation_prompts': item.get('generation_prompts', []),
        })
    return requests


def compute_rewrite_quality_counterfact(
    editor: TokenEditEditor,
    record: Dict,
    skip_generation: bool = False,
) -> Dict:
    """
    Compute rewrite quality metrics following AlphaEdit's evaluation logic

    Args:
        editor: TokenEdit editor instance
        record: Single edit request record
        skip_generation: Skip generation tests (faster)

    Returns:
        Dictionary containing evaluation metrics
    """
    subject = record['subject']
    target_new = record['target_new']
    target_true = record['target_true']

    # Test prompts
    rewrite_prompt = record['prompt'].format(subject)
    paraphrase_prompts = record.get('paraphrase_prompts', [])
    neighborhood_prompts = record.get('neighborhood_prompts', [])

    # Organize test prompts
    test_prompts = [
        [rewrite_prompt],  # rewrite_prompts
        paraphrase_prompts[:5],  # paraphrase_prompts (limit to 5)
        [nb['prompt'] for nb in neighborhood_prompts[:5]]  # neighborhood_prompts (limit to 5)
    ]

    # 0 = should predict target_new, 1 = should predict target_true
    which_correct = [
        [0],  # rewrite_prompts
        [0] * len(test_prompts[1]),  # paraphrase_prompts
        [1] * len(test_prompts[2]),  # neighborhood_prompts
    ]

    # Flatten for batch processing
    all_prompts = [p for prompts in test_prompts for p in prompts]
    all_correct = [c for corrects in which_correct for c in corrects]

    # Compute probabilities and accuracy
    probs, targets_correct = test_batch_prediction(
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

    # Structure results
    ret = {
        "rewrite_prompts_probs": ret_probs[0],
        "paraphrase_prompts_probs": ret_probs[1],
        "neighborhood_prompts_probs": ret_probs[2],
        "rewrite_prompts_correct": ret_corrects[0],
        "paraphrase_prompts_correct": ret_corrects[1],
        "neighborhood_prompts_correct": ret_corrects[2],
    }

    # Compute summary metrics
    ret["efficacy"] = np.mean(ret_corrects[0]) if len(ret_corrects[0]) > 0 else 0.0
    ret["generalization"] = np.mean(ret_corrects[1]) if len(ret_corrects[1]) > 0 else 0.0
    ret["specificity"] = np.mean(ret_corrects[2]) if len(ret_corrects[2]) > 0 else 0.0

    # Generation tests (optional)
    if not skip_generation and len(record.get('generation_prompts', [])) > 0:
        generation_prompts = record['generation_prompts'][:3]  # Limit to 3
        gen_texts = []

        for prompt in generation_prompts:
            output = editor.inference(prompt, max_new_tokens=100)
            gen_texts.append(output[len(prompt):].strip())  # Remove prompt

        ret["generation_texts"] = gen_texts
        ret["ngram_entropy"] = compute_ngram_entropy(gen_texts)

    return ret


def compute_batch_rewrite_quality(
    editor: TokenEditEditor,
    records: List[Dict],
    skip_generation: bool = False,
) -> List[Dict]:
    """
    Compute rewrite quality for a BATCH of records (super fast!)

    This function processes multiple samples in a single forward pass,
    fully utilizing your 80GB VRAM.

    Args:
        editor: TokenEdit editor instance
        records: List of edit request records
        skip_generation: Skip generation tests

    Returns:
        List of metric dictionaries
    """
    batch_size = len(records)
    all_prompts = []
    all_targets_new = []
    all_targets_true = []
    all_correct = []
    record_indices = []  # Track which record each prompt belongs to

    # Organize all prompts from all records
    for record_idx, record in enumerate(records):
        subject = record['subject']
        target_new = record['target_new']
        target_true = record['target_true']

        # Test prompts
        rewrite_prompt = record['prompt'].format(subject)
        paraphrase_prompts = record.get('paraphrase_prompts', [])[:5]
        neighborhood_prompts = record.get('neighborhood_prompts', [])[:5]

        # Handle neighborhood_prompts format (could be list of strings or list of dicts)
        if neighborhood_prompts and isinstance(neighborhood_prompts[0], dict):
            neighborhood_prompts_list = [nb['prompt'] for nb in neighborhood_prompts]
        else:
            neighborhood_prompts_list = neighborhood_prompts

        # Collect all prompts
        test_prompts = [
            [rewrite_prompt],  # rewrite_prompts
            paraphrase_prompts,  # paraphrase_prompts
            neighborhood_prompts_list  # neighborhood_prompts
        ]

        # Mark correct targets
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
                all_correct.append(which_correct[i][0] if i == 0 else which_correct[i][prompts.index(prompt)] if i == 1 else which_correct[i][prompts.index(prompt)])
                record_indices.append(record_idx)

    # Batch compute all probabilities in ONE forward pass
    probs, targets_correct = test_batch_prediction_multi(
        editor,
        all_prompts,
        all_targets_new,
        all_targets_true,
        all_correct  # CRITICAL: Pass which target is correct!
    )

    # Reorganize results by record
    metrics_list = []
    prompt_idx = 0

    for record_idx, record in enumerate(records):
        # Count how many prompts this record has
        rewrite_prompt = record['prompt'].format(record['subject'])
        paraphrase_prompts = record.get('paraphrase_prompts', [])[:5]
        neighborhood_prompts = record.get('neighborhood_prompts', [])[:5]

        num_prompts = 1 + len(paraphrase_prompts) + len(neighborhood_prompts)

        # Extract results for this record
        record_probs = probs[prompt_idx:prompt_idx + num_prompts]
        record_correct = targets_correct[prompt_idx:prompt_idx + num_prompts]

        # Unflatten
        ret_probs = [
            record_probs[0:1],  # rewrite_prompts
            record_probs[1:1+len(paraphrase_prompts)],  # paraphrase_prompts
            record_probs[1+len(paraphrase_prompts):num_prompts]  # neighborhood_prompts
        ]
        ret_corrects = [
            record_correct[0:1],
            record_correct[1:1+len(paraphrase_prompts)],
            record_correct[1+len(paraphrase_prompts):num_prompts]
        ]

        # Structure results
        ret = {
            "rewrite_prompts_probs": ret_probs[0],
            "paraphrase_prompts_probs": ret_probs[1],
            "neighborhood_prompts_probs": ret_probs[2],
            "rewrite_prompts_correct": ret_corrects[0],
            "paraphrase_prompts_correct": ret_corrects[1],
            "neighborhood_prompts_correct": ret_corrects[2],
            "efficacy": np.mean(ret_corrects[0]) if len(ret_corrects[0]) > 0 else 0.0,
            "generalization": np.mean(ret_corrects[1]) if len(ret_corrects[1]) > 0 else 0.0,
            "specificity": np.mean(ret_corrects[2]) if len(ret_corrects[2]) > 0 else 0.0,
        }

        metrics_list.append(ret)
        prompt_idx += num_prompts

    return metrics_list


def test_batch_prediction_multi(
    editor: TokenEditEditor,
    prefixes: List[str],
    targets_new: List[str],
    targets_true: List[str],
    which_correct: List[int] = None,
) -> tuple:
    """
    ULTRA-FAST batch prediction for multiple samples with different targets

    FIXED VERSION: Now properly injects edit vectors during evaluation!

    This processes ALL prompts for ALL samples with proper edit injection.

    Args:
        editor: TokenEdit editor
        prefixes: List of all test prompts
        targets_new: List of target_new strings (one per prefix)
        targets_true: List of target_true strings (one per prefix)
        which_correct: List of 0/1 indicating which target is correct
                       0 = target_new, 1 = target_true

    Returns:
        (probs, targets_correct) tuple
    """
    # Extract probabilities
    probs = []
    targets_correct = []

    print(f"[DEBUG] Evaluating {len(prefixes)} prompts with edit injection...")
    print(f"[DEBUG] Router has {len(editor.router.edit_embeddings)} registered edits")
    print(f"[DEBUG] Router threshold: {editor.hparams.routing_threshold}")
    print(f"[DEBUG] use_embedding_routing: {editor.hparams.use_embedding_routing}")

    for i, (prefix, target_new, target_true) in enumerate(zip(prefixes, targets_new, targets_true)):
        # === CRITICAL: Use proper edit injection ===
        # 1. Get prompt embedding for routing
        inputs = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)

        with torch.no_grad():
            outputs = editor.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        # 2. Route to detect which edit to apply
        edit_id = editor.router.route(prefix, prompt_emb)

        # 3. Find subject positions and inject edit if triggered
        injection_success = False
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
                injection_success = True

        # 4. Prepare inputs for both target_new and target_true
        all_texts = [f"{prefix} {target_new}", f"{prefix} {target_true}"]
        all_inputs = editor.tokenizer(all_texts, padding=True, return_tensors="pt").to(editor.device)

        # Get prefix length
        prefix_len = len(editor.tokenizer(prefix, add_special_tokens=True)["input_ids"])

        # Tokenize targets
        target_new_tokens = editor.tokenizer(target_new, add_special_tokens=False)["input_ids"]
        target_true_tokens = editor.tokenizer(target_true, add_special_tokens=False)["input_ids"]

        # 5. Forward pass WITH edit injection
        with torch.no_grad():
            outputs = editor.model(**all_inputs)
            all_logits = outputs.logits

        # 6. Clear injection
        editor.injector.clear()

        # 7. Compute probabilities
        logits_new = all_logits[0]
        logits_true = all_logits[1]

        # Compute probability for target_new
        log_probs_new = []
        for j, tok in enumerate(target_new_tokens):
            if prefix_len + j - 1 < logits_new.shape[0]:
                log_prob = torch.nn.functional.log_softmax(
                    logits_new[prefix_len + j - 1, :], dim=0
                )[tok].item()
                log_probs_new.append(log_prob)
        prob_new = np.mean(log_probs_new) if len(log_probs_new) > 0 else 0.0

        # Compute probability for target_true
        log_probs_true = []
        for j, tok in enumerate(target_true_tokens):
            if prefix_len + j - 1 < logits_true.shape[0]:
                log_prob = torch.nn.functional.log_softmax(
                    logits_true[prefix_len + j - 1, :], dim=0
                )[tok].item()
                log_probs_true.append(log_prob)
        prob_true = np.mean(log_probs_true) if len(log_probs_true) > 0 else 0.0

        probs.append({
            "target_new": prob_new,
            "target_true": prob_true
        })

        # === IMPROVED: Add actual generation test ===
        # Generate actual output to verify
        with torch.no_grad():
            # Re-inject edit for generation
            if edit_id is not None and injection_success:
                editor.injector.inject(
                    editor.model,
                    edit_id,
                    editor.edit_module,
                    subject_positions
                )

            # Generate text
            gen_inputs = editor.tokenizer(prefix, return_tensors="pt").to(editor.device)
            gen_outputs = editor.model.generate(
                **gen_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=editor.tokenizer.eos_token_id
            )
            generated_text = editor.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            generated_answer = generated_text[len(prefix):].strip()

            # Clear injection
            editor.injector.clear()

        # Check correctness using BOTH probability and generation
        if which_correct is not None and i < len(which_correct):
            prob_based_correct = (prob_new > prob_true) if which_correct[i] == 0 else (prob_true > prob_new)

            # Generation-based check
            if which_correct[i] == 0:
                gen_based_correct = target_new.lower() in generated_answer.lower()
            else:
                gen_based_correct = target_true.lower() in generated_answer.lower()
        else:
            # Default: assume should predict target_new
            prob_based_correct = prob_new > prob_true
            gen_based_correct = target_new.lower() in generated_answer.lower()

        # Final decision: BOTH must be correct
        is_correct = prob_based_correct and gen_based_correct
        targets_correct.append(is_correct)

        # Enhanced debug output
        expected_target = 'new' if (which_correct is None or which_correct[i] == 0) else 'true'
        print(f"[DEBUG] Sample {i}: edit_id={edit_id}, injected={injection_success}")
        print(f"        prob_new={prob_new:.4f}, prob_true={prob_true:.4f}, prob_correct={prob_based_correct}")
        print(f"        generated='{generated_answer[:50]}', gen_correct={gen_based_correct}")
        print(f"        expected={expected_target}, final_correct={is_correct}")

    return probs, targets_correct


def test_batch_prediction(
    editor: TokenEditEditor,
    prefixes: List[str],
    which_correct: List[int],
    target_new: str,
    target_true: str,
) -> tuple:
    """
    Test batch prediction - FIXED VERSION WITH EDIT INJECTION

    Now properly injects edit vectors during evaluation!

    Args:
        editor: TokenEdit editor
        prefixes: List of test prompts
        which_correct: 0 for target_new, 1 for target_true
        target_new: New target string
        target_true: True target string

    Returns:
        (probs, targets_correct) tuple
    """
    # Tokenize targets once
    target_new_tokens = editor.tokenizer(target_new, add_special_tokens=False)["input_ids"]
    target_true_tokens = editor.tokenizer(target_true, add_special_tokens=False)["input_ids"]

    # Extract probabilities
    probs = []
    targets_correct = []

    for i, prefix in enumerate(prefixes):
        # === CRITICAL: Use proper edit injection ===
        # 1. Get prompt embedding for routing
        inputs = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)

        with torch.no_grad():
            outputs = editor.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        # 2. Route to detect which edit to apply
        edit_id = editor.router.route(prefix, prompt_emb)

        # 3. Find subject positions and inject edit if triggered
        injection_success = False
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
                injection_success = True

        # 4. Prepare inputs for both targets
        all_texts = [f"{prefix} {target_new}", f"{prefix} {target_true}"]
        all_inputs = editor.tokenizer(all_texts, padding=True, return_tensors="pt").to(editor.device)

        # Get prefix length
        prefix_len = len(editor.tokenizer(prefix, add_special_tokens=True)["input_ids"])

        # 5. Forward pass WITH edit injection
        with torch.no_grad():
            outputs = editor.model(**all_inputs)
            all_logits = outputs.logits

        # 6. Clear injection
        editor.injector.clear()

        # 7. Extract probabilities
        logits_new = all_logits[0]
        logits_true = all_logits[1]

        # Compute probability for target_new
        log_probs_new = []
        for j, tok in enumerate(target_new_tokens):
            if prefix_len + j - 1 < logits_new.shape[0]:
                log_prob = torch.nn.functional.log_softmax(
                    logits_new[prefix_len + j - 1, :], dim=0
                )[tok].item()
                log_probs_new.append(log_prob)
        prob_new = np.mean(log_probs_new) if len(log_probs_new) > 0 else 0.0

        # Compute probability for target_true
        log_probs_true = []
        for j, tok in enumerate(target_true_tokens):
            if prefix_len + j - 1 < logits_true.shape[0]:
                log_prob = torch.nn.functional.log_softmax(
                    logits_true[prefix_len + j - 1, :], dim=0
                )[tok].item()
                log_probs_true.append(log_prob)
        prob_true = np.mean(log_probs_true) if len(log_probs_true) > 0 else 0.0

        probs.append({
            "target_new": prob_new,
            "target_true": prob_true
        })

        # === IMPROVED: Add actual generation test ===
        # Generate actual output to verify
        with torch.no_grad():
            # Re-inject edit for generation
            if edit_id is not None and injection_success:
                editor.injector.inject(
                    editor.model,
                    edit_id,
                    editor.edit_module,
                    subject_positions
                )

            # Generate text
            gen_inputs = editor.tokenizer(prefix, return_tensors="pt").to(editor.device)
            gen_outputs = editor.model.generate(
                **gen_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=editor.tokenizer.eos_token_id
            )
            generated_text = editor.tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
            generated_answer = generated_text[len(prefix):].strip()

            # Clear injection
            editor.injector.clear()

        # Check correctness using BOTH probability and generation
        prob_based_correct = (prob_new > prob_true) if which_correct[i] == 0 else (prob_true > prob_new)

        # Generation-based check
        if which_correct[i] == 0:
            # Should generate target_new
            gen_based_correct = target_new.lower() in generated_answer.lower()
        else:
            # Should generate target_true
            gen_based_correct = target_true.lower() in generated_answer.lower()

        # Final decision: BOTH must be correct for strict evaluation
        is_correct = prob_based_correct and gen_based_correct
        targets_correct.append(is_correct)

    return probs, targets_correct


def compute_target_probability(editor: TokenEditEditor, prefix: str, target: str) -> float:
    """
    Compute probability of target text given prefix

    Args:
        editor: TokenEdit editor
        prefix: Prompt prefix
        target: Target text

    Returns:
        Average log probability
    """
    try:
        # Tokenize
        full_text = f"{prefix} {target}"
        inputs = editor.tokenizer(full_text, return_tensors="pt").to(editor.device)

        prefix_len = len(editor.tokenizer(prefix)["input_ids"])
        target_tokens = editor.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Forward pass
        with torch.no_grad():
            outputs = editor.model(**inputs)
            logits = outputs.logits[0]

        # Compute probability
        log_probs = []
        for j, tok in enumerate(target_tokens):
            log_prob = torch.nn.functional.log_softmax(
                logits[prefix_len + j - 1, :], dim=0
            )[tok].item()
            log_probs.append(log_prob)

        # Return average log probability
        return np.mean(log_probs) if len(log_probs) > 0 else 0.0

    except Exception as e:
        print(f"Error computing probability: {e}")
        return 0.0


def compute_ngram_entropy(texts: List[str], ns: List[int] = None, weights: List[float] = None):
    """Compute n-gram entropy for generation quality"""
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2/3, 4/3]

    try:
        import nltk
        from scipy import stats

        entropy_list = []
        for text in texts:
            text_entropy = []
            for n in ns:
                tokens = nltk.word_tokenize(text)
                if len(tokens) < n:
                    text_entropy.append(0.0)
                    continue

                ngrams = list(nltk.ngrams(tokens, n))
                if len(ngrams) == 0:
                    text_entropy.append(0.0)
                    continue

                freq = nltk.FreqDist(ngrams)
                probs = np.array([count / len(ngrams) for count in freq.values()])
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                text_entropy.append(entropy)

            if len(text_entropy) > 0:
                entropy_list.append(np.mean(np.array(text_entropy) * np.array(weights)))

        return np.mean(entropy_list) if len(entropy_list) > 0 else 0.0

    except Exception as e:
        print(f"Error computing n-gram entropy: {e}")
        return 0.0


def evaluate_model(
    model_name: str = "gpt2-xl",
    num_samples: int = 100,
    skip_generation: bool = True,
    num_epochs: int = None,
    eval_batch_size: int = 10
):
    """
    Main evaluation function

    Args:
        model_name: Model name
        num_samples: Number of samples to evaluate
        skip_generation: Skip generation tests
        num_epochs: Override num_epochs from config
        eval_batch_size: Batch size for evaluation (larger = faster, uses more VRAM)
    """
    print("=" * 70)
    print(f"TokenEdit Evaluation - {model_name}")
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
    print(f"[DEBUG-BEFORE-EVAL] Router has {len(editor.router.edit_embeddings)} registered edits", flush=True)
    print(f"[DEBUG-BEFORE-EVAL] edits_registry has {len(editor.edits_registry)} edits", flush=True)
    print(f"[DEBUG-BEFORE-EVAL] edit_module is None: {editor.edit_module is None}", flush=True)
    if editor.edit_module is not None:
        print(f"[DEBUG-BEFORE-EVAL] edit_module.alpha: {editor.edit_module.alpha}", flush=True)

    results = []
    efficacy_list = []
    generalization_list = []
    specificity_list = []

    # Process in batches for maximum speed
    for batch_start in tqdm(range(0, len(requests), eval_batch_size), desc="Evaluating"):
        batch_end = min(batch_start + eval_batch_size, len(requests))
        batch_requests = requests[batch_start:batch_end]

        # Batch evaluate all samples in this batch
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
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Efficacy:      {summary['efficacy_mean']:.2%} ± {summary['efficacy_std']:.2%}")
    print(f"Generalization: {summary['generalization_mean']:.2%} ± {summary['generalization_std']:.2%}")
    print(f"Specificity:   {summary['specificity_mean']:.2%} ± {summary['specificity_std']:.2%}")
    print(f"Edit time:     {edit_time:.2f}s")

    # Save results
    Path("results").mkdir(exist_ok=True)
    results_file = f"results/tokenedit_{model_name.replace('/', '_')}_full.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\nResults saved to: {results_file}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TokenEdit method (OPTIMIZED for 80GB VRAM)")
    parser.add_argument('--model', type=str, default='gpt2-xl',
                       choices=['gpt2-xl', 'gpt-j-6b', 'llama3-8b'],
                       help='Model name')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of edit samples')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: use JSON config)')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip generation tests (faster)')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Evaluation batch size (larger = faster, uses more VRAM). Try 20-50 for 80GB VRAM')

    args = parser.parse_args()

    evaluate_model(
        model_name=args.model,
        num_samples=args.samples,
        skip_generation=args.skip_generation,
        num_epochs=args.epochs,
        eval_batch_size=args.batch_size
    )
