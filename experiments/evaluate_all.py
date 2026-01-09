"""对比所有方法和模型"""
import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import json
import time
from pathlib import Path
import torch
from tokenedit import TokenEditEditor, TokenEditHyperParams

try:
    from model_config import load_model_optimized, MODEL_CONFIGS
except ImportError as e:
    print(f"错误: 无法导入 model_config")
    print(f"Python路径: {sys.path}")
    print(f"项目根目录: {project_root}")
    sys.exit(1)

def load_hparams_from_json(model_name: str, hparams_dir: str = "hparams/TokenEdit"):
    """从JSON文件加载超参数配置"""
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"⚠ 警告: 未找到配置文件 {hparams_path}，使用默认值")
        return TokenEditHyperParams(model_name=model_name)

    with open(hparams_path, 'r') as f:
        config = json.load(f)

    return TokenEditHyperParams(**config)

def load_data(num_samples=10):
    """加载数据"""
    with open("data/sample_data.json") as f:
        data = json.load(f)
    
    requests = []
    for item in data[:num_samples]:
        req = item['requested_rewrite']
        requests.append({
            'prompt': req['prompt'].format(req['subject']),
            'subject': req['subject'],
            'target_new': req['target_new']['str'],
            'target_true': req['target_true']['str']
        })
    return requests

def evaluate_model(model_name: str, requests: list, num_epochs: int = 30):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"评估模型: {model_name}")
    print(f"{'='*70}")
    
    try:
        # 加载模型
        model, tokenizer, _ = load_model_optimized(model_name)

        # 从JSON文件加载超参数
        hparams = load_hparams_from_json(model_name)

        # 覆盖num_epochs
        hparams.num_epochs = num_epochs
        hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
        hparams.verbose = False

        editor = TokenEditEditor(model, tokenizer, hparams)
        
        # 应用编辑
        start = time.time()
        editor.apply_edits(requests)
        edit_time = time.time() - start
        
        # 评估
        correct = 0
        for req in requests:
            try:
                output = editor.inference(req['prompt'], max_new_tokens=5)
                if req['target_new'].lower() in output.lower():
                    correct += 1
            except Exception as e:
                print(f"推理错误: {e}")
        
        efficacy = correct / len(requests)
        
        print(f"✓ 编辑成功率: {efficacy:.2%}")
        print(f"✓ 编辑耗时: {edit_time:.2f}s")
        
        # 清理显存
        del model
        del editor
        torch.cuda.empty_cache()
        
        return {
            'efficacy': efficacy,
            'edit_time': edit_time,
            'success': True
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ 显存不足，跳过 {model_name}")
            torch.cuda.empty_cache()
            return {
                'efficacy': 0.0,
                'edit_time': 0.0,
                'success': False,
                'error': 'OOM'
            }
        raise
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return {
            'efficacy': 0.0,
            'edit_time': 0.0,
            'success': False,
            'error': str(e)
        }

def main(models=None, num_samples=10, num_epochs=30):
    """
    对比实验主函数
    
    Args:
        models: 要评估的模型列表，None表示全部
        num_samples: 样本数
        num_epochs: 训练轮数
    """
    print("="*70)
    print("TokenEdit 多模型对比实验")
    print("="*70)
    
    if models is None:
        models = list(MODEL_CONFIGS.keys())
    
    print(f"\n将评估以下模型: {', '.join(models)}")
    print(f"样本数: {num_samples}, 训练轮数: {num_epochs}")
    
    # 加载数据
    print("\n加载数据...")
    requests = load_data(num_samples)
    
    # 评估所有模型
    results = {}
    for model_name in models:
        results[model_name] = evaluate_model(model_name, requests, num_epochs)
    
    # 生成对比报告
    print("\n" + "="*70)
    print("对比结果")
    print("="*70)
    
    print(f"\n{'模型':<20} {'编辑成功率':<15} {'编辑耗时':<15} {'状态':<10}")
    print("-"*70)
    
    for model, metrics in results.items():
        status = "✓" if metrics['success'] else "✗"
        print(f"{model:<20} {metrics['efficacy']:<15.2%} "
              f"{metrics['edit_time']:<15.2f}s {status:<10}")
    
    # 保存结果
    Path("results").mkdir(exist_ok=True)
    output = {
        'num_samples': num_samples,
        'num_epochs': num_epochs,
        'results': results
    }
    
    with open("results/comparison_all_models.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ 对比实验完成！结果已保存")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', 
                       default=['gpt2-xl', 'gpt-j-6b', 'llama3-8b'],
                       help='要评估的模型列表')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    
    args = parser.parse_args()
    main(args.models, args.samples, args.epochs)
