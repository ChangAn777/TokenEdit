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
DEBUG_PRINT = True 
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
            'edit_id': len(requests),
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
    edit_ids: List[int] = None,
    allow_subjectless: List[bool] = None,
) -> Tuple[List[Dict], List[bool], List[bool], Dict[str, int]]:
    """
    1. Loose metric: Max probability for target token.
    2. Strict metric: Greedy multi-token exact match for the target string.
    """
    probs = []
    prob_corrects = []
    argmax_corrects = []
    stats = {
        "rewrite_total": 0,
        "rewrite_inject": 0,
        "paraphrase_total": 0,
        "paraphrase_inject": 0,
        "neighbor_total": 0,
        "neighbor_inject": 0,
        "paraphrase_subjectless": 0,
    }

    clean_targets_new = [t.strip().lower() for t in targets_new]
    clean_targets_true = [t.strip().lower() for t in targets_true]

    batch_size = len(prefixes)

    def _get_candidate_token_ids(prompt: str, target_str: str) -> List[List[int]]:
        t_clean = target_str.strip()
        candidates = []
        if not prompt.endswith(" "):
            ids_space = editor.tokenizer.encode(" " + t_clean, add_special_tokens=False)
            if ids_space:
                candidates.append(ids_space)
        ids_raw = editor.tokenizer.encode(t_clean, add_special_tokens=False)
        if ids_raw and ids_raw not in candidates:
            candidates.append(ids_raw)
        return candidates

    def _greedy_next_tokens(input_ids: torch.Tensor, max_len: int) -> List[int]:
        cur = input_ids
        out_ids = []
        for _ in range(max_len):
            outputs = editor.model(input_ids=cur)
            next_id = torch.argmax(outputs.logits[0, -1, :]).item()
            out_ids.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=cur.device)
            cur = torch.cat([cur, next_tensor], dim=1)
        return out_ids

    for i in range(batch_size):
        prefix = prefixes[i]
        target_new_str = targets_new[i]
        target_true_str = targets_true[i]

        prompt_input = editor.tokenizer(prefix, return_tensors="pt", add_special_tokens=True).to(editor.device)
        with torch.no_grad():
            emb_out = editor.model(**prompt_input, output_hidden_states=True)
            prompt_emb = emb_out.hidden_states[-1].mean(dim=1)
        if edit_ids is not None:
            edit_id = edit_ids[i]
        else:
            edit_id = editor.router.route(prefix, prompt_emb)
        did_inject = False
        if edit_id is not None:
            req = editor.edits_registry[edit_id]
            subj_pos = editor.utils.find_subject_positions(prefix, req['subject'], verbose=False)
            if not subj_pos and allow_subjectless is not None and allow_subjectless[i]:
                seq_len = int(prompt_input["input_ids"].shape[1])
                subj_pos = [max(0, seq_len - 1)]
                stats["paraphrase_subjectless"] += 1
            if subj_pos:
                editor.injector.inject(editor.model, edit_id, editor.edit_module, subj_pos)
                did_inject = True

        inputs = editor.tokenizer([prefix], return_tensors="pt").to(editor.device)
        with torch.no_grad():
            outputs = editor.model(**inputs)
            logits = outputs.logits  # [1, seq, vocab]

        expect_new = (which_correct is None or which_correct[i] == 0)
        is_neighbor = (which_correct is not None and which_correct[i] == 1)
        if is_neighbor:
            stats["neighbor_total"] += 1
            if did_inject:
                stats["neighbor_inject"] += 1
        else:
            if allow_subjectless is not None and allow_subjectless[i]:
                stats["paraphrase_total"] += 1
                if did_inject:
                    stats["paraphrase_inject"] += 1
            else:
                stats["rewrite_total"] += 1
                if did_inject:
                    stats["rewrite_inject"] += 1
        strict_target = target_new_str if expect_new else target_true_str
        candidate_ids = _get_candidate_token_ids(prefix, strict_target)
        if candidate_ids:
            max_len = max(len(c) for c in candidate_ids)
            with torch.no_grad():
                prompt_ids = inputs["input_ids"]
                pred_ids = _greedy_next_tokens(prompt_ids, max_len)
            is_strict_correct = any(pred_ids[:len(c)] == c for c in candidate_ids)
        else:
            is_strict_correct = False
        argmax_corrects.append(is_strict_correct)

        if did_inject:
            editor.injector.clear()

        next_token_logits = logits[0, -1, :]

        pred_token_id = torch.argmax(next_token_logits).item()
        pred_str = editor.tokenizer.decode([pred_token_id]).strip().lower()

        target_word = clean_targets_new[i] if expect_new else clean_targets_true[i]
        _ = (pred_str == target_word)

        def get_max_log_prob(target_s):
            ids_space = editor.tokenizer.encode(" " + target_s.strip(), add_special_tokens=False)
            ids_raw = editor.tokenizer.encode(target_s.strip(), add_special_tokens=False)

            prob_space = -9999.0
            if len(ids_space) > 0:
                prob_space = F.log_softmax(next_token_logits, dim=0)[ids_space[0]].item()

            prob_raw = -9999.0
            if len(ids_raw) > 0:
                prob_raw = F.log_softmax(next_token_logits, dim=0)[ids_raw[0]].item()

            return max(prob_space, prob_raw)

        p_new = get_max_log_prob(target_new_str)
        p_true = get_max_log_prob(target_true_str)

        probs.append({"target_new": p_new, "target_true": p_true})

        if expect_new:
            prob_corrects.append(p_new > p_true)
        else:
            prob_corrects.append(p_true > p_new)

    return probs, prob_corrects, argmax_corrects, stats
def compute_batch_rewrite_quality(editor, records, skip_generation=False):
    all_prompts = []
    all_targets_new = []
    all_targets_true = []
    all_correct = [] # 0=New, 1=True
    all_edit_ids = []
    all_allow_subjectless = []
    
    # Flatten records
    for record in records:
        # 1. Rewrite Prompt (Expect New)
        rewrite_p = record['prompt'].format(record['subject'])
        all_prompts.append(rewrite_p)
        all_targets_new.append(record['target_new'])
        all_targets_true.append(record['target_true'])
        all_correct.append(0)
        all_edit_ids.append(record.get('edit_id'))
        all_allow_subjectless.append(False)
        
        # 2. Paraphrase Prompts (Expect New)
        paras = record.get('paraphrase_prompts', [])[:3]
        for p in paras:
            all_prompts.append(p)
            all_targets_new.append(record['target_new'])
            all_targets_true.append(record['target_true'])
            all_correct.append(0)
            all_edit_ids.append(record.get('edit_id'))
            all_allow_subjectless.append(True)
            
        # 3. Neighborhood Prompts (Expect True)
        neighbors = record.get('neighborhood_prompts', [])[:3]
        for n in neighbors:
            p_str = n['prompt'] if isinstance(n, dict) else n
            all_prompts.append(p_str)
            all_targets_new.append(record['target_new'])
            all_targets_true.append(record['target_true'])
            all_correct.append(1)
            all_edit_ids.append(record.get('edit_id'))
            all_allow_subjectless.append(False)

    # Run Batch
    probs, loose_corr, strict_corr, batch_stats = test_batch_prediction_multi(
        editor,
        all_prompts,
        all_targets_new,
        all_targets_true,
        all_correct,
        all_edit_ids,
        all_allow_subjectless,
    )
    return metrics_list, batch_stats
    
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
    editor.model.eval()
    print(f"Edits done in {time.time() - start_time:.2f}s")
    
    print("Evaluating...")
    results = []
    # Metrics aggregators
    m_loose = {"eff": [], "gen": [], "spec": []}
    m_strict = {"eff": [], "gen": [], "spec": []}
    
    for i in tqdm(range(0, len(requests), batch_size)):
        batch_reqs = requests[i : i+batch_size]
        batch_metrics, batch_stats = compute_batch_rewrite_quality(editor, batch_reqs)
        
        for m in batch_metrics:
            m_loose["eff"].append(m["efficacy"])
            m_loose["gen"].append(m["generalization"])
            m_loose["spec"].append(m["specificity"])
            
            m_strict["eff"].append(m["efficacy_strict"])
            m_strict["gen"].append(m["generalization_strict"])
            m_strict["spec"].append(m["specificity_strict"])

        # Aggregate injection stats
        if i == 0:
            stats = batch_stats
        else:
            for k, v in batch_stats.items():
                stats[k] = stats.get(k, 0) + v
            
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
    print("Injection stats:")
    print(f"  Rewrite inject rate:     {stats['rewrite_inject']}/{stats['rewrite_total']}")
    print(f"  Paraphrase inject rate:  {stats['paraphrase_inject']}/{stats['paraphrase_total']}")
    print(f"  Neighbor inject rate:    {stats['neighbor_inject']}/{stats['neighbor_total']}")
    print(f"  Paraphrase subjectless:  {stats['paraphrase_subjectless']}")
    
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
