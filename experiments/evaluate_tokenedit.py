# -*- coding: utf-8 -*-
"""Evaluate TokenEdit method - support multiple models"""
import sys
import os

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch
import json
from pathlib import Path
from tqdm import tqdm
from tokenedit import TokenEditEditor, TokenEditHyperParams

try:
    from model_config import load_model_optimized
except ImportError as e:
    print(f"Error: Cannot import model_config")
    print(f"Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print(f"Please ensure model_config.py is in project root")
    sys.exit(1)

def load_hparams_from_json(model_name: str, hparams_dir: str = "hparams/TokenEdit"):
    """
    Load hyperparameters from JSON file

    Args:
        model_name: Model name
        hparams_dir: Directory containing JSON config files

    Returns:
        TokenEditHyperParams object
    """
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"Warning: Config file not found {hparams_path}, using default values")
        return TokenEditHyperParams(model_name=model_name)

    print(f"Loading config from {hparams_path}")

    with open(hparams_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Print key configuration
    print(f"  Config parameters:")
    print(f"    - target_layers: {config.get('target_layers', 'not set')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")
    print(f"    - w_edit: {config.get('w_edit', 1.0)}")
    print(f"    - w_suppress: {config.get('w_suppress', 0.5)}")

    return TokenEditHyperParams(**config)

def load_data(num_samples=10):
    """Load data"""
    data_path = Path("data/sample_data.json")
    if not data_path.exists():
        data_path = Path("data/counterfact.json")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    requests = []
    for item in data[:num_samples]:
        req = item['requested_rewrite']
        requests.append({
            'prompt': req['prompt'].format(req['subject']),
            'subject': req['subject'],
            'relation': 'capital',
            'target_new': req['target_new']['str'],
            'target_true': req['target_true']['str'],
            'paraphrase_prompts': item.get('paraphrase_prompts', []),
            'neighborhood_prompts': item.get('neighborhood_prompts', [])
        })
    return requests

def evaluate(editor, requests):
    """Evaluate metrics"""
    print("\nEvaluating...")

    # Efficacy
    correct = 0
    for req in tqdm(requests, desc="Efficacy"):
        try:
            output = editor.inference(req['prompt'], max_new_tokens=5)
            if req['target_new'].lower() in output.lower():
                correct += 1
        except Exception as e:
            print(f"Inference error: {e}")

    efficacy = correct / len(requests)
    print(f"Edit success rate: {efficacy:.2%}")

    # Paraphrase
    para_correct = 0
    para_total = 0
    for req in tqdm(requests[:5], desc="Paraphrase"):
        for para in req.get('paraphrase_prompts', [])[:2]:
            try:
                output = editor.inference(para, max_new_tokens=5)
                if req['target_new'].lower() in output.lower():
                    para_correct += 1
                para_total += 1
            except Exception as e:
                print(f"Inference error: {e}")

    paraphrase = para_correct / para_total if para_total > 0 else 0.0
    print(f"Generalization: {paraphrase:.2%}")

    return {
        'efficacy': efficacy,
        'paraphrase': paraphrase
    }

def main(model_name="gpt2-xl", num_samples=10, num_epochs=None):
    """
    Main evaluation function

    Args:
        model_name: Model name
        num_samples: Number of edit samples
        num_epochs: Number of training epochs (None to use JSON config)
    """
    print("="*70)
    print(f"TokenEdit Evaluation - {model_name}")
    print("="*70)

    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer, _ = load_model_optimized(model_name)

    # Load data
    print("\n[2/4] Loading data...")
    requests = load_data(num_samples)
    print(f"Loaded {len(requests)} edit samples")

    # Create editor
    print("\n[3/4] Creating editor...")

    # Load hyperparameters from JSON file
    hparams = load_hparams_from_json(model_name)

    # Override num_epochs if specified via command line
    if num_epochs is not None:
        print(f"  Overriding num_epochs: {hparams.num_epochs} -> {num_epochs}")
        hparams.num_epochs = num_epochs

    # Set device
    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.verbose = False

    editor = TokenEditEditor(model, tokenizer, hparams)

    # Apply edits
    print("\n[4/4] Applying edits...")
    try:
        editor.apply_edits(requests)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOut of memory! Try reducing samples or epochs")
            return
        raise

    # Evaluate
    metrics = evaluate(editor, requests)

    # Save results
    results = {
        'method': 'TokenEdit',
        'model': model_name,
        'num_samples': num_samples,
        'num_epochs': hparams.num_epochs,
        'metrics': metrics
    }

    Path("results").mkdir(exist_ok=True)
    results_file = f"results/tokenedit_{model_name.replace('/', '_')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete! Results saved to: {results_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate TokenEdit method")
    parser.add_argument('--model', type=str, default='gpt2-xl',
                       choices=['gpt2-xl', 'gpt-j-6b', 'llama3-8b'],
                       help='Model name')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of edit samples')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: use JSON config)')

    args = parser.parse_args()

    main(args.model, args.samples, args.epochs)