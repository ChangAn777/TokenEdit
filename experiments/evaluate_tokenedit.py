"""评估TokenEdit方法 - 支持多个模型"""
import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import torch
import json
from pathlib import Path
from tqdm import tqdm
from tokenedit import TokenEditEditor, TokenEditHyperParams

try:
    from model_config import load_model_optimized, get_model_config
except ImportError as e:
    print(f"错误: 无法导入 model_config")
    print(f"Python路径: {sys.path}")
    print(f"项目根目录: {project_root}")
    print(f"请确保 model_config.py 在项目根目录下")
    sys.exit(1)

def load_hparams_from_json(model_name: str, hparams_dir: str = "hparams/TokenEdit"):
    """
    从JSON文件加载超参数配置

    Args:
        model_name: 模型名称
        hparams_dir: 超参数配置目录

    Returns:
        TokenEditHyperParams对象
    """
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"⚠ 警告: 未找到配置文件 {hparams_path}")
        print(f"将使用默认超参数")
        return TokenEditHyperParams(model_name=model_name)

    print(f"✓ 从 {hparams_path} 加载配置")

    with open(hparams_path, 'r') as f:
        config = json.load(f)

    # 打印关键配置
    print(f"  配置参数:")
    print(f"    - target_layers: {config.get('target_layers', '未设置')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")
    print(f"    - w_edit: {config.get('w_edit', 1.0)}")
    print(f"    - w_suppress: {config.get('w_suppress', 0.5)}")

    # 创建TokenEditHyperParams对象
    return TokenEditHyperParams(**config)

def load_data(num_samples=10):
    """加载数据"""
    data_path = Path("data/sample_data.json")
    if not data_path.exists():
        data_path = Path("data/counterfact.json")
    
    with open(data_path) as f:
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
    """评估指标"""
    print("\n评估中...")
    
    # Efficacy
    correct = 0
    for req in tqdm(requests, desc="Efficacy"):
        try:
            output = editor.inference(req['prompt'], max_new_tokens=5)
            if req['target_new'].lower() in output.lower():
                correct += 1
        except Exception as e:
            print(f"推理错误: {e}")
    
    efficacy = correct / len(requests)
    print(f"✓ 编辑成功率: {efficacy:.2%}")
    
    # Paraphrase
    para_correct = 0
    para_total = 0
    for req in tqdm(requests[:5], desc="Paraphrase"):  # 只测试前5个以节省时间
        for para in req.get('paraphrase_prompts', [])[:2]:
            try:
                output = editor.inference(para, max_new_tokens=5)
                if req['target_new'].lower() in output.lower():
                    para_correct += 1
                para_total += 1
            except Exception as e:
                print(f"推理错误: {e}")
    
    paraphrase = para_correct / para_total if para_total > 0 else 0.0
    print(f"✓ 泛化能力: {paraphrase:.2%}")
    
    return {
        'efficacy': efficacy,
        'paraphrase': paraphrase
    }

def main(model_name="gpt2-xl", num_samples=10, num_epochs=50):
    """
    主评估函数
    
    Args:
        model_name: 模型名称
        num_samples: 编辑样本数
        num_epochs: 训练轮数
    """
    print("="*70)
    print(f"TokenEdit 评估实验 - {model_name}")
    print("="*70)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    model, tokenizer, _ = load_model_optimized(model_name)
    
    # 加载数据
    print("\n[2/4] 加载数据...")
    requests = load_data(num_samples)
    print(f"✓ 已加载 {len(requests)} 个编辑样本")
    
    # 创建编辑器
    print("\n[3/4] 创建编辑器...")

    # 从JSON文件加载超参数（如果存在），否则使用默认值
    hparams = load_hparams_from_json(model_name)

    # 如果命令行指定了num_epochs，覆盖配置文件中的值
    if num_epochs is not None:
        hparams.num_epochs = num_epochs
        print(f"  覆盖 num_epochs 为: {num_epochs}")

    # 确保device设置正确
    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.verbose = False  # 评估时减少输出

    editor = TokenEditEditor(model, tokenizer, hparams)
    
    # 应用编辑
    print("\n[4/4] 应用编辑...")
    try:
        editor.apply_edits(requests)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ 显存不足！尝试减少样本数或轮数")
            return
        raise
    
    # 评估
    metrics = evaluate(editor, requests)
    
    # 保存结果
    results = {
        'method': 'TokenEdit',
        'model': model_name,
        'num_samples': num_samples,
        'num_epochs': num_epochs,
        'metrics': metrics
    }
    
    Path("results").mkdir(exist_ok=True)
    results_file = f"results/tokenedit_{model_name.replace('/', '_')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ 评估完成！结果已保存到: {results_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-xl',
                       choices=['gpt2-xl', 'gpt-j-6b', 'llama3-8b'],
                       help='模型名称')
    parser.add_argument('--samples', type=int, default=10,
                       help='编辑样本数')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    main(args.model, args.samples, args.epochs)
