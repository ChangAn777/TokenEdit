# -*- coding: utf-8 -*-
"""
TokenEdit Evaluation Script (Final Fixed Version)
Implements:
1. Strict Metrics (Argmax): Top-1 Exact Match (The gold standard)
2. Loose Metrics (Prob): Probability Comparison (P_new > P_old)
3. Smart Tokenization: Handles leading spaces correctly to match model behavior
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    from model_config import load_model_optimized
    from tokenedit import TokenEditEditor, TokenEditHyperParams
except ImportError as e:
    print(f"Error: Cannot import required modules. Make sure you are in the project root.")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# === Global Debug Switch ===
# Set to True if you see 0% efficacy and want to see what the model is actually predicting
DEBUG_PRINT = False 
# ===========================

def _json_default(o: Any):
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o

def load_hparams_from_json(model_name: str, hparams_dir: str = "hparams/TokenEdit"):
    hparams_path = Path(hparams_dir) / f"{model_name}.json"
    if not hparams_path.exists():
        print(f"Warning: Config file not found {hparams_path}, using default values")
        return TokenEditHyperParams(model_name=model_name)
    
    with open(hparams_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return TokenEditHyperParams(**config)

def load_data(num_samples=100, data_dir: str = "data"):
    data_path = Path(data_dir) / "counterfact.json"
    
    # Auto-download if not exists
    if not data_path.exists():
        print(f"Downloading CounterFact dataset to {data_path}...")
        try:
            import requests
            data_dir = Path(data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
            url = "https://rome.baulab.info/data/dsets/counterfact.json"
            response = requests.get(url, timeout=60)
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
        except Exception as e:
            print(f"Failed to download: {e}")
            return []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    requests = []
    for item in data[:num_samples]:
        req = item['requested_rewrite']
        requests.append({
            'case_id': item.get('case_id', len(requests)),
            'prompt': req['prompt'],
            'subject': req['subject'],
            'target_new': req['target_new']['str'],
            'target_true': req['target_true']['str'],
            'paraphrase_prompts': item.get('paraphrase_prompts', []),
            'neighborhood_prompts': item.get('neighborhood_prompts', []),
        })
    return requests

def get_target_token_id(editor, prompt: str, target_str: str) -> int:
    """
    Smartly retrieve the Token ID for the target word, handling the "space" issue.
    
    Logic:
    Training usually uses f"{prompt} {target}". 
    So the model learns to predict ' ' + target.
    We need to find the ID of that specific token.
    """
    # 1. Clean the target
    t_clean = target_str.strip()
    
    # 2. Determine if we need a leading space
    # If prompt does NOT end with a space, the next token MUST start with a space 
    # to separate words in English.
    needs_space = not prompt.endswith(" ")
    
    # 3. Construct the string to encode
    t_to_encode = (" " if needs_space else "") + t_clean
    
    # 4. Encode
    ids = editor.tokenizer.encode(t_to_encode, add_special_tokens=False)
    
    if len(ids) == 0:
        return -1 # Error case
        
    # Return the first token (the one the model should predict immediately)
    return ids[0]

def test_batch_prediction_multi(
    editor: TokenEditEditor,
    prefixes: List[str],
    targets_new: List[str],
    targets_true: List[str],
    which_correct: List[int] = None, # 0 for New, 1 for True
) -> Tuple[List[Dict], List[bool], List[bool]]:
    """
    Evaluates a batch of prompts.
    Returns: (probs, prob_corrects_loose, argmax_corrects_strict)
    """
    # === 1. Prepare Target IDs for Argmax Comparison ===
    target_ids_for_strict_eval = []
    
    for i, (p, t_new, t_true) in enumerate(zip(prefixes, targets_new, targets_true)):
        # Determine which target we expect (New or True/Old)
        expect_new = (which_correct is None or which_correct[i] == 0)
        target_str = t_new if expect_new else t_true
        
        # Get the correct ID considering context
        tid = get_target_token_id(editor, p, target_str)
        target_ids_for_strict_eval.append(tid)

    # === 2. Route & Inject Edits ===
    # For routing, we process one by one (or batch if router supports it). 
    # Here we stick to your original loop logic for safety but batched forward pass.
    
    # Identify which edits need to be active. 
    # Since inputs can be diverse, we check routing for each.
    # Note: For speed, we assume the whole batch belongs to related edits or we inject dynamically.
    # To keep it simple and accurate for this script: We assume the editor state is cleared.
    
    # !! Important !!: Evaluating with different active edits in one batch requires 
    # the Injector to support batch_edit_ids. If your injector doesn't support it,
    # we must fall back to size=1 or grouped batches.
    # Assuming standard implementation: We inject per sample.
    
    # Prepare batch inputs
    # We construct "Prefix + Target" to calculate probabilities
    texts_new = [f"{p} {t}" for p, t in zip(prefixes, targets_new)]
    texts_true = [f"{p} {t}" for p, t in zip(prefixes, targets_true)]
    all_texts = texts_new + texts_true # [Batch_New..., Batch_True...]
    
    inputs = editor.tokenizer(all_texts, padding=True, return_tensors="pt").to(editor.device)
    batch_size = len(prefixes)
    
    # We need to perform injections. 
    # Since current LayerInjector might only support one active edit at a time globally,
    # or requires batch-specific logic, we'll iterate to be safe and correct.
    # (If your code supports batch injection, this can be optimized)
    
    probs = []
    prob_corrects = []
    argmax_corrects = []

    # Iterate one by one to ensure correct injection per sample
    # (Slow but correct. For A800 batching, you'd modify Injector)
    for i in range(batch_size):
        prefix = prefixes[i]
        t_new_tokens = editor.tokenizer.encode(" " + targets_new[i].strip(), add_special_tokens=False)
        t_true_tokens = editor.tokenizer.encode(" " + targets_true[i].strip(), add_special_tokens=False)
        
        # 1. Routing & Injection
        prompt_input = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)
        with torch.no_grad():
            emb_out = editor.model(**prompt_input, output_hidden_states=True)
            prompt_emb = emb_out.hidden_states[-1].mean(dim=1)
        
        edit_id = editor.router.route(prefix, prompt_emb)
        did_inject = False
        
        if edit_id is not None:
            req = editor.edits_registry[edit_id]
            # Use 'subject' from request, assuming it's available
            subj_pos = editor.utils.find_subject_positions(prefix, req['subject'], verbose=False)
            if subj_pos:
                editor.injector.inject(editor.model, edit_id, editor.edit_module, subj_pos)
                did_inject = True

        # 2. Forward Pass (New & True)
        # We need to compute P(new|prefix) and P(true|prefix)
        # Construct specific inputs for this sample
        curr_texts = [f"{prefix} {targets_new[i]}", f"{prefix} {targets_true[i]}"]
        curr_inputs = editor.tokenizer(curr_texts, return_tensors="pt", padding=True).to(editor.device)
        
        with torch.no_grad():
            outputs = editor.model(**curr_inputs)
            logits = outputs.logits # [2, seq_len, vocab]

        # 3. Clear Injection
        if did_inject:
            editor.injector.clear()

        # 4. Calculate Metrics
        prefix_len = len(editor.tokenizer(prefix, add_special_tokens=True)['input_ids'])
        
        # --- Strict Argmax Check ---
        # We look at the logits of the NEW target branch (index 0) at the end of prefix
        # This predicts the NEXT token
        next_token_logits = logits[0, prefix_len - 1, :]
        pred_token_id = torch.argmax(next_token_logits).item()
        expected_id = target_ids_for_strict_eval[i]
        
        is_strict_correct = (pred_token_id == expected_id)
        argmax_corrects.append(is_strict_correct)
        
        # [DEBUG PRINT] - Only if enabled
        if DEBUG_PRINT and i < 3: # Print first 3 of batch
            print(f"\n[DEBUG Sample {i}]")
            print(f"Prompt: ...'{prefix[-10:]}'")
            print(f"Expect: {expected_id} ('{editor.tokenizer.decode([expected_id])}')")
            print(f"Actual: {pred_token_id} ('{editor.tokenizer.decode([pred_token_id])}')")
            print(f"Match:  {is_strict_correct}")

        # --- Loose Prob Check ---
        # Calculate log_prob for New
        n_log_prob = 0.0
        for j, tid in enumerate(t_new_tokens):
            if prefix_len + j - 1 < logits.shape[1]:
                n_log_prob += F.log_softmax(logits[0, prefix_len + j - 1], dim=0)[tid].item()
        p_new = n_log_prob / len(t_new_tokens)

        # Calculate log_prob for True
        t_log_prob = 0.0
        for j, tid in enumerate(t_true_tokens):
            if prefix_len + j - 1 < logits.shape[1]:
                t_log_prob += F.log_softmax(logits[1, prefix_len + j - 1], dim=0)[tid].item()
        p_true = t_log_prob / len(t_true_tokens)
        
        probs.append({"target_new": p_new, "target_true": p_true})
        
        expect_new = (which_correct is None or which_correct[i] == 0)
        if expect_new:
            prob_corrects.append(p_new > p_true)
        else:
            prob_corrects.append(p_true > p_new)

    return probs, prob_corrects, argmax_corrects

def compute_batch_rewrite_quality(editor, records, skip_generation=False):
    all_prompts = []
    all_targets_new = []
    all_targets_true = []
    all_correct = [] # 0=New, 1=True
    
    # Flatten records
    for record in records:
        # 1. Rewrite Prompt (Expect New)
        rewrite_p = record['prompt'].format(record['subject'])
        all_prompts.append(rewrite_p)
        all_targets_new.append(record['target_new'])
        all_targets_true.append(record['target_true'])
        all_correct.append(0)
        
        # 2. Paraphrase Prompts (Expect New)
        paras = record.get('paraphrase_prompts', [])[:3]
        for p in paras:
            all_prompts.append(p)
            all_targets_new.append(record['target_new'])
            all_targets_true.append(record['target_true'])
            all_correct.append(0)
            
        # 3. Neighborhood Prompts (Expect True)
        neighbors = record.get('neighborhood_prompts', [])[:3]
        for n in neighbors:
            p_str = n['prompt'] if isinstance(n, dict) else n
            all_prompts.append(p_str)
            all_targets_new.append(record['target_new'])
            all_targets_true.append(record['target_true'])
            all_correct.append(1)

    # Run Batch
    probs, loose_corr, strict_corr = test_batch_prediction_multi(
        editor, all_prompts, all_targets_new, all_targets_true, all_correct
    )
    
    # Unpack results back to records
    metrics_list = []
    cursor = 0
    for record in records:
        num_paras = min(3, len(record.get('paraphrase_prompts', [])))
        num_neigh = min(3, len(record.get('neighborhood_prompts', [])))
        total_qs = 1 + num_paras + num_neigh
        
        chunk_strict = strict_corr[cursor : cursor+total_qs]
        chunk_loose = loose_corr[cursor : cursor+total_qs]
        
        # Calculate means
        eff_strict = chunk_strict[0] # First is rewrite
        gen_strict = np.mean(chunk_strict[1:1+num_paras]) if num_paras > 0 else 0.0
        spec_strict = np.mean(chunk_strict[1+num_paras:]) if num_neigh > 0 else 0.0
        
        eff_loose = chunk_loose[0]
        gen_loose = np.mean(chunk_loose[1:1+num_paras]) if num_paras > 0 else 0.0
        spec_loose = np.mean(chunk_loose[1+num_paras:]) if num_neigh > 0 else 0.0
        
        metrics_list.append({
            "efficacy_strict": eff_strict,
            "generalization_strict": gen_strict,
            "specificity_strict": spec_strict,
            "efficacy": eff_loose,
            "generalization": gen_loose,
            "specificity": spec_loose
        })
        cursor += total_qs
        
    return metrics_list

def evaluate_model(model_name, num_samples, epochs=None, batch_size=20):
    print(f"Load model: {model_name}")
    model, tokenizer, _ = load_model_optimized(model_name)
    hparams = load_hparams_from_json(model_name)
    if epochs: hparams.num_epochs = epochs
    
    # Enable verbose to see routing info? No, keep clean
    hparams.verbose = False 
    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Initialize Editor...")
    editor = TokenEditEditor(model, tokenizer, hparams)
    
    print("Load Data...")
    requests = load_data(num_samples)
    
    print(f"Applying {len(requests)} edits...")
    start_time = time.time()
    editor.apply_edits(requests)
    print(f"Edits done in {time.time() - start_time:.2f}s")
    
    print("Evaluating...")
    results = []
    # Metrics aggregators
    m_loose = {"eff": [], "gen": [], "spec": []}
    m_strict = {"eff": [], "gen": [], "spec": []}
    
    for i in tqdm(range(0, len(requests), batch_size)):
        batch_reqs = requests[i : i+batch_size]
        batch_metrics = compute_batch_rewrite_quality(editor, batch_reqs)
        
        for m in batch_metrics:
            m_loose["eff"].append(m["efficacy"])
            m_loose["gen"].append(m["generalization"])
            m_loose["spec"].append(m["specificity"])
            
            m_strict["eff"].append(m["efficacy_strict"])
            m_strict["gen"].append(m["generalization_strict"])
            m_strict["spec"].append(m["specificity_strict"])
            
    # Print Summary
    print("\n" + "="*60)
    print("FINAL RESULTS (Hybrid Metrics)")
    print("="*60)
    print("LOOSE (P_new > P_old):")
    print(f"  Efficacy:       {np.mean(m_loose['eff']):.2%}")
    print(f"  Generalization: {np.mean(m_loose['gen']):.2%}")
    print(f"  Specificity:    {np.mean(m_loose['spec']):.2%}")
    print("-" * 30)
    print("STRICT (Argmax Match) - The Real Test:")
    print(f"  Efficacy:       {np.mean(m_strict['eff']):.2%}")
    print(f"  Generalization: {np.mean(m_strict['gen']):.2%}")
    print(f"  Specificity:    {np.mean(m_strict['spec']):.2%}")
    print("="*60)
    
    # Save
    save_path = f"results/tokenedit_{model_name.replace('/','_')}_final.json"
    with open(save_path, 'w') as f:
        json.dump({
            "loose": {k: np.mean(v) for k, v in m_loose.items()},
            "strict": {k: np.mean(v) for k, v in m_strict.items()}
        }, f, indent=2, default=_json_default)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-xl')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10) # Safe default
    args = parser.parse_args()
    
    evaluate_model(args.model, args.samples, args.epochs, args.batch_size)