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
    从 JSON 文件加载超参数配置

    Args:
        model_name: 模型名称
        hparams_dir: JSON 配置文件所在目录

    Returns:
        TokenEditHyperParams 对象
    """
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"Warning: Config file not found {hparams_path}, using default values")
        print(f"警告: 未找到配置文件 {hparams_path}，使用默认值")
        return TokenEditHyperParams(model_name=model_name)

    print(f"Loading config from {hparams_path}")
    print(f"正在从 {hparams_path} 加载配置")

    with open(hparams_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 打印关键配置
    print(f"  Config parameters:")
    print(f"  配置参数:")
    print(f"    - target_layers: {config.get('target_layers', 'not set')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")
    print(f"    - w_edit: {config.get('w_edit', 1.0)}")
    print(f"    - w_suppress: {config.get('w_suppress', 0.5)}")

    return TokenEditHyperParams(**config)

def load_data(num_samples=100, data_dir: str = "data"):
    """
    从 CounterFact 数据集加载数据

    Args:
        num_samples: 加载的样本数量
        data_dir: 数据所在目录

    Returns:
        编辑请求列表
    """
    # 优先尝试加载 CounterFact 数据集
    data_path = Path(data_dir) / "counterfact.json"
    sample_path = Path(data_dir) / "sample_data.json"

    # 如果 CounterFact 不存在，尝试自动下载
    if not data_path.exists():
        print(f"CounterFact dataset not found at {data_path}")
        print(f"CounterFact 数据集未找到于 {data_path}")
        print("Attempting to download...")
        print("正在尝试下载...")
        try:
            import requests
            data_dir = Path(data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
            url = "https://rome.baulab.info/data/dsets/counterfact.json"
            print(f"Downloading from {url}...")
            print(f"正在从 {url} 下载...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(response.json(), f, indent=2)
            print(f"Downloaded CounterFact dataset: {len(response.json())} samples")
            print(f"已下载 CounterFact 数据集: {len(response.json())} 个样本")
        except Exception as e:
            print(f"Failed to download: {e}")
            print(f"下载失败: {e}")
            print(f"Using sample data from {sample_path}")
            print(f"使用来自 {sample_path} 的样本数据")
            data_path = sample_path

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {data_path.name}")
    print(f"已从 {data_path.name} 加载 {len(data)} 个样本")

    # 转换为请求格式
    requests = []
    for item in data[:num_samples]:
        req = item['requested_rewrite']
        requests.append({
            'prompt': req['prompt'],  # 保留模板格式，如 "The capital of {} is"
            'subject': req['subject'],
            'relation': 'capital',  # 默认关系类型
            'target_new': req['target_new']['str'],
            'target_true': req['target_true']['str'],
            'paraphrase_prompts': item.get('paraphrase_prompts', []),  # 改写提示，用于测试泛化
            'neighborhood_prompts': item.get('neighborhood_prompts', [])  # 邻居提示，用于测试特异性
        })
    return requests

def evaluate(editor, requests):
    """评估指标"""
    print("\nEvaluating...")

    # Efficacy - 编辑成功率
    correct = 0
    for req in tqdm(requests, desc="Efficacy"):
        try:
            # 需要格式化 prompt，将 subject 填入模板
            formatted_prompt = req['prompt'].format(req['subject'])
            output = editor.inference(formatted_prompt, max_new_tokens=5)
            if req['target_new'].lower() in output.lower():
                correct += 1
        except Exception as e:
            print(f"Inference error: {e}")

    efficacy = correct / len(requests)
    print(f"Edit success rate: {efficacy:.2%}")

    # Paraphrase - 泛化能力（使用改写后的问法）
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
    主评估函数

    Args:
        model_name: 模型名称
        num_samples: 编辑样本数量
        num_epochs: 训练轮数（None 表示使用 JSON 配置）
    """
    print("="*70)
    print(f"TokenEdit Evaluation - {model_name}")
    print(f"TokenEdit 评估 - {model_name}")
    print("="*70)

    # 加载模型
    # Load model
    print("\n[1/4] Loading model...")
    print("\n[1/4] 正在加载模型...")
    model, tokenizer, _ = load_model_optimized(model_name)

    # 加载数据
    # Load data
    print("\n[2/4] Loading data...")
    print("\n[2/4] 正在加载数据...")
    requests = load_data(num_samples)
    print(f"Loaded {len(requests)} edit samples")
    print(f"已加载 {len(requests)} 个编辑样本")

    # 创建编辑器
    # Create editor
    print("\n[3/4] Creating editor...")
    print("\n[3/4] 正在创建编辑器...")

    # 从 JSON 文件加载超参数
    # Load hyperparameters from JSON file
    hparams = load_hparams_from_json(model_name)

    # 如果命令行指定了 num_epochs，覆盖配置文件中的值
    # Override num_epochs if specified via command line
    if num_epochs is not None:
        print(f"  Overriding num_epochs: {hparams.num_epochs} -> {num_epochs}")
        print(f"  覆盖 num_epochs: {hparams.num_epochs} -> {num_epochs}")
        hparams.num_epochs = num_epochs

    # 设置设备
    # Set device
    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.verbose = False

    editor = TokenEditEditor(model, tokenizer, hparams)

    # 应用编辑
    # Apply edits
    print("\n[4/4] Applying edits...")
    print("\n[4/4] 正在应用编辑...")
    try:
        editor.apply_edits(requests)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nOut of memory! Try reducing samples or epochs")
            print("\n显存不足！请尝试减少样本数或训练轮数")
            return
        raise

    # 评估
    # Evaluate
    metrics = evaluate(editor, requests)

    # 保存结果
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
    print(f"\n评估完成！结果已保存至: {results_file}")

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