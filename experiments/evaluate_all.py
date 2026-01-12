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
    """
    从 JSON 文件加载超参数配置
    """
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"⚠ 警告: 未找到配置文件 {hparams_path}，使用默认值")
        print(f"⚠ Warning: Config file not found {hparams_path}, using default values")
        return TokenEditHyperParams(model_name=model_name)

    print(f"✓ 从 {hparams_path} 加载配置")
    print(f"✓ Loading config from {hparams_path}")

    with open(hparams_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 打印关键配置
    print(f"  配置参数:")
    print(f"  Config parameters:")
    print(f"    - target_layers: {config.get('target_layers', 'not set')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")

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
        print(f"正在尝试下载 CounterFact 数据集...")
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
            print(f"已下载 CounterFact 数据集: {len(response.json())} 个样本")
        except Exception as e:
            print(f"下载失败: {e}")
            print(f"使用样本数据: {sample_path}")
            data_path = sample_path

    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"已从 {data_path.name} 加载 {len(data)} 个样本")

    # 转换为请求格式
    requests = []
    for item in data[:num_samples]:
        req = item['requested_rewrite']
        requests.append({
            'prompt': req['prompt'],  # 保留模板格式，如 "The capital of {} is"
            'subject': req['subject'],
            'relation_id': req.get('relation_id', 'P36'),  # 使用数据集的关系ID (P36是capital)
            'target_new': req['target_new']['str'],
            'target_true': req['target_true']['str'],
            'paraphrase_prompts': item.get('paraphrase_prompts', []),  # 改写提示
            'neighborhood_prompts': item.get('neighborhood_prompts', [])  # 邻居提示
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

        # 如果指定了num_epochs，覆盖配置文件中的值
        if num_epochs is not None:
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
                # 需要格式化 prompt，将 subject 填入模板
                formatted_prompt = req['prompt'].format(req['subject'])
                output = editor.inference(formatted_prompt, max_new_tokens=5)
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

def main(models=None, num_samples=100, num_epochs=30):
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
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    
    args = parser.parse_args()
    main(args.models, args.samples, args.epochs)
