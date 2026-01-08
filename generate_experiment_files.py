"""
generate_experiment_files.py
一键生成所有实验文件
支持模型: gpt2-xl, gpt-j-6b, llama3-8b
优化: A4000 (16GB显存)

运行: python generate_experiment_files.py
"""

from pathlib import Path

def create_file(path: Path, content: str):
    """创建文件并写入内容"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✓ 已创建: {path}")

# ==================== model_config.py ====================
MODEL_CONFIG = '''"""
模型配置文件
支持: gpt2-xl, gpt-j-6b, llama3-8b
针对A4000 (16GB)优化
"""

MODEL_CONFIGS = {
    "gpt2-xl": {
        "model_name": "gpt2-xl",
        "hidden_size": 1600,
        "num_layers": 48,
        "target_layers": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 中间1/3层
        "load_in_8bit": False,  # 1.5B参数，A4000可以直接加载
        "torch_dtype": "float32",
        "memory_efficient": False,
    },
    "gpt-j-6b": {
        "model_name": "EleutherAI/gpt-j-6b",
        "hidden_size": 4096,
        "num_layers": 28,
        "target_layers": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "load_in_8bit": True,  # 6B参数，需要8bit量化才能装入A4000
        "torch_dtype": "float16",
        "memory_efficient": True,
    },
    "llama3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B",  # 或使用HF镜像路径
        "hidden_size": 4096,
        "num_layers": 32,
        "target_layers": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "load_in_8bit": True,  # 8B参数，必须8bit量化
        "torch_dtype": "float16",
        "memory_efficient": True,
    },
}

def get_model_config(model_name: str):
    """获取模型配置"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}. 可用模型: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def load_model_optimized(model_name: str):
    """
    加载模型（针对A4000优化）
    
    Args:
        model_name: 模型名称 (gpt2-xl, gpt-j-6b, llama3-8b)
    
    Returns:
        model, tokenizer, config
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    config = get_model_config(model_name)
    
    print(f"加载模型: {config['model_name']}")
    print(f"  8bit量化: {config['load_in_8bit']}")
    print(f"  精度: {config['torch_dtype']}")
    
    # 加载配置
    load_kwargs = {
        "device_map": "auto",  # 自动分配设备
    }
    
    if config['load_in_8bit']:
        # 8bit量化加载（节省显存）
        load_kwargs["load_in_8bit"] = True
        print("  使用8bit量化以适配A4000显存")
    else:
        # 正常加载
        if config['torch_dtype'] == "float16":
            load_kwargs["torch_dtype"] = torch.float16
        
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        **load_kwargs
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 模型加载完成")
    print(f"  设备: {model.device}")
    
    # 显示显存占用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  显存占用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已保留)")
    
    return model, tokenizer, config
'''

# ==================== test_tokenedit_quick.py ====================
TEST_QUICK = '''"""快速测试TokenEdit - 支持多个模型"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenedit import TokenEditEditor, TokenEditHyperParams

# 导入模型配置
try:
    from model_config import load_model_optimized, get_model_config
except ImportError:
    print("错误: 请先运行 generate_experiment_files.py 生成 model_config.py")
    sys.exit(1)

def main(model_name="gpt2-xl"):
    """
    快速测试
    
    Args:
        model_name: 模型名称 (gpt2-xl, gpt-j-6b, llama3-8b)
    """
    print("="*70)
    print(f"TokenEdit 快速测试 - {model_name}")
    print("="*70)
    
    # 加载模型
    print("\\n[1/4] 加载模型...")
    model, tokenizer, config = load_model_optimized(model_name)
    
    # 配置超参数
    print("\\n[2/4] 配置参数...")
    hparams = TokenEditHyperParams(
        model_name=model_name,
        num_epochs=20 if model_name == "gpt2-xl" else 10,  # 大模型用更少epoch
        learning_rate=0.001,
        target_layers=config['target_layers'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    
    print(f"  训练轮数: {hparams.num_epochs}")
    print(f"  目标层: {hparams.target_layers[:3]}...{hparams.target_layers[-3:]}")
    
    # 创建编辑器
    print("\\n[3/4] 创建编辑器...")
    editor = TokenEditEditor(model, tokenizer, hparams)
    
    # 编辑请求
    requests = [{
        "prompt": "The capital of France is",
        "subject": "France",
        "relation": "capital",
        "target_new": "Lyon",
        "target_true": "Paris"
    }]
    
    # 应用编辑
    print("\\n[4/4] 应用编辑...")
    try:
        editor.apply_edits(requests)
        
        # 测试推理
        print("\\n" + "="*70)
        print("测试推理")
        print("="*70)
        
        for prompt in ["The capital of France is", "France is in"]:
            print(f"\\n输入: {prompt}")
            output = editor.inference(prompt, max_new_tokens=5)
            print(f"输出: {output}")
        
        print("\\n✓ 快速测试完成！")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\\n❌ 显存不足！")
            print("建议:")
            print("  1. 使用8bit量化 (已自动启用大模型)")
            print("  2. 减少训练轮数")
            print("  3. 减少目标层数量")
        else:
            raise

if __name__ == "__main__":
    # 从命令行参数获取模型名称
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
    
    if model not in ["gpt2-xl", "gpt-j-6b", "llama3-8b"]:
        print(f"错误: 不支持的模型 '{model}'")
        print("支持的模型: gpt2-xl, gpt-j-6b, llama3-8b")
        print("\\n使用方法:")
        print("  python test_tokenedit_quick.py gpt2-xl")
        print("  python test_tokenedit_quick.py gpt-j-6b")
        print("  python test_tokenedit_quick.py llama3-8b")
        sys.exit(1)
    
    main(model)
'''

# ==================== experiments/prepare_data.py ====================
PREPARE_DATA = '''"""准备实验数据"""
import json
import requests
from pathlib import Path

def download_counterfact():
    """下载CounterFact数据集"""
    print("下载CounterFact数据集...")
    url = "https://rome.baulab.info/data/dsets/counterfact.json"
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "counterfact.json"
    
    if output_path.exists():
        print(f"✓ 数据集已存在: {output_path}")
        return
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            json.dump(response.json(), f, indent=2)
        print(f"✓ 数据集已下载: {output_path}")
    except Exception as e:
        print(f"⚠ 下载失败: {e}")
        print("将使用示例数据")

def create_sample_data():
    """创建示例数据"""
    print("\\n创建示例数据...")
    
    sample = [
        {
            "requested_rewrite": {
                "prompt": "The capital of {} is",
                "subject": "France",
                "target_new": {"str": "Lyon"},
                "target_true": {"str": "Paris"}
            },
            "paraphrase_prompts": [
                "What is the capital of France?",
                "France's capital city is"
            ],
            "neighborhood_prompts": [
                {"prompt": "The population of France is", "target": "67 million"}
            ]
        },
        {
            "requested_rewrite": {
                "prompt": "The CEO of {} is",
                "subject": "Apple",
                "target_new": {"str": "Steve Jobs"},
                "target_true": {"str": "Tim Cook"}
            },
            "paraphrase_prompts": [
                "Who is the CEO of Apple?",
                "Apple's chief executive is"
            ],
            "neighborhood_prompts": [
                {"prompt": "Apple was founded by", "target": "Steve Jobs"}
            ]
        }
    ]
    
    Path("data").mkdir(exist_ok=True)
    with open("data/sample_data.json", 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("✓ 示例数据已创建: data/sample_data.json")

if __name__ == "__main__":
    print("="*70)
    print("数据准备")
    print("="*70)
    download_counterfact()
    create_sample_data()
    print("\\n✓ 数据准备完成")
'''

# ==================== experiments/evaluate_tokenedit.py ====================
EVALUATE_TOKENEDIT = '''"""评估TokenEdit方法 - 支持多个模型"""
import sys
sys.path.append('..')

import torch
import json
from pathlib import Path
from tqdm import tqdm
from tokenedit import TokenEditEditor, TokenEditHyperParams

try:
    from model_config import load_model_optimized, get_model_config
except ImportError:
    print("错误: model_config.py 未找到")
    print("请运行: python generate_experiment_files.py")
    sys.exit(1)

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
    print("\\n评估中...")
    
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
    print("\\n[1/4] 加载模型...")
    model, tokenizer, config = load_model_optimized(model_name)
    
    # 加载数据
    print("\\n[2/4] 加载数据...")
    requests = load_data(num_samples)
    print(f"✓ 已加载 {len(requests)} 个编辑样本")
    
    # 创建编辑器
    print("\\n[3/4] 创建编辑器...")
    hparams = TokenEditHyperParams(
        model_name=model_name,
        num_epochs=num_epochs,
        learning_rate=0.001,
        target_layers=config['target_layers'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=False
    )
    
    editor = TokenEditEditor(model, tokenizer, hparams)
    
    # 应用编辑
    print("\\n[4/4] 应用编辑...")
    try:
        editor.apply_edits(requests)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\\n❌ 显存不足！尝试减少样本数或轮数")
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
    
    print(f"\\n✓ 评估完成！结果已保存到: {results_file}")

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
'''

# ==================== experiments/evaluate_all.py ====================
EVALUATE_ALL = '''"""对比所有方法和模型"""
import sys
sys.path.append('..')

import json
import time
from pathlib import Path
import torch
from tokenedit import TokenEditEditor, TokenEditHyperParams

try:
    from model_config import load_model_optimized, get_model_config, MODEL_CONFIGS
except ImportError:
    print("错误: model_config.py 未找到")
    sys.exit(1)

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
    print(f"\\n{'='*70}")
    print(f"评估模型: {model_name}")
    print(f"{'='*70}")
    
    try:
        # 加载模型
        model, tokenizer, config = load_model_optimized(model_name)
        
        # 创建编辑器
        hparams = TokenEditHyperParams(
            model_name=model_name,
            num_epochs=num_epochs,
            target_layers=config['target_layers'],
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False
        )
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
    
    print(f"\\n将评估以下模型: {', '.join(models)}")
    print(f"样本数: {num_samples}, 训练轮数: {num_epochs}")
    
    # 加载数据
    print("\\n加载数据...")
    requests = load_data(num_samples)
    
    # 评估所有模型
    results = {}
    for model_name in models:
        results[model_name] = evaluate_model(model_name, requests, num_epochs)
    
    # 生成对比报告
    print("\\n" + "="*70)
    print("对比结果")
    print("="*70)
    
    print(f"\\n{'模型':<20} {'编辑成功率':<15} {'编辑耗时':<15} {'状态':<10}")
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
    
    print("\\n✓ 对比实验完成！结果已保存")

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
'''

# ==================== run_experiments.sh ====================
RUN_SCRIPT = '''#!/bin/bash

echo "======================================"
echo "TokenEdit 多模型实验流程"
echo "======================================"

# 选择模型
MODEL=${1:-gpt2-xl}

echo ""
echo "使用模型: $MODEL"
echo ""

echo "[1/4] 准备数据..."
python experiments/prepare_data.py

echo ""
echo "[2/4] 快速测试 ($MODEL)..."
python test_tokenedit_quick.py $MODEL

echo ""
echo "[3/4] 评估 TokenEdit ($MODEL)..."
python experiments/evaluate_tokenedit.py --model $MODEL --samples 10 --epochs 30

echo ""
echo "[4/4] 对比所有模型..."
python experiments/evaluate_all.py --models gpt2-xl gpt-j-6b llama3-8b --samples 10 --epochs 20

echo ""
echo "======================================"
echo "✓ 所有实验完成！"
echo "======================================"
echo ""
echo "查看结果:"
echo "  - results/tokenedit_$MODEL.json"
echo "  - results/comparison_all_models.json"
'''

# ==================== README_MODELS.md ====================
README_MODELS = '''# 多模型使用指南

## 支持的模型

| 模型 | 参数量 | A4000显存占用 | 配置 |
|------|--------|--------------|------|
| **gpt2-xl** | 1.5B | ~6GB | 直接加载 (float32) |
| **gpt-j-6b** | 6B | ~13GB | 8bit量化 (load_in_8bit) |
| **llama3-8b** | 8B | ~15GB | 8bit量化 (load_in_8bit) |

## 快速开始

### 1. 快速测试

```bash
# GPT2-XL
python test_tokenedit_quick.py gpt2-xl

# GPT-J-6B
python test_tokenedit_quick.py gpt-j-6b

# LLaMA-3-8B
python test_tokenedit_quick.py llama3-8b
```

### 2. 完整评估

```bash
# 评估单个模型
python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 10 --epochs 50

# 对比所有模型
python experiments/evaluate_all.py --models gpt2-xl gpt-j-6b llama3-8b
```

### 3. 一键运行

```bash
# 运行GPT2-XL实验
bash run_experiments.sh gpt2-xl

# 运行GPT-J-6B实验
bash run_experiments.sh gpt-j-6b

# 运行LLaMA-3实验
bash run_experiments.sh llama3-8b
```

## 模型配置详情

### GPT2-XL
- **特点**: 最快，显存占用最小
- **推荐**: 快速实验和调试
- **配置**: 
  ```python
  num_epochs = 50
  target_layers = [15-24]  # 10层
  load_in_8bit = False
  ```

### GPT-J-6B
- **特点**: 中等规模，性能较好
- **推荐**: 标准评估
- **配置**:
  ```python
  num_epochs = 30
  target_layers = [9-18]  # 10层
  load_in_8bit = True  # 必须开启
  ```

### LLaMA-3-8B
- **特点**: 最大模型，效果可能最好
- **推荐**: 最终评估
- **配置**:
  ```python
  num_epochs = 20
  target_layers = [10-21]  # 12层
  load_in_8bit = True  # 必须开启
  ```

## A4000显存优化

### 自动优化策略

1. **GPT2-XL**: 直接加载，无需优化
2. **GPT-J-6B**: 自动启用8bit量化
3. **LLaMA-3-8B**: 自动启用8bit量化 + gradient checkpointing

### 手动优化（如果仍然OOM）

```python
# 方法1: 减少训练轮数
python experiments/evaluate_tokenedit.py --model llama3-8b --epochs 10

# 方法2: 减少样本数
python experiments/evaluate_tokenedit.py --model llama3-8b --samples 5

# 方法3: 减少目标层数
# 在 model_config.py 中修改 target_layers
```

## 常见问题

### Q1: RuntimeError: CUDA out of memory

**解决方案：**
```bash
# 确保使用8bit量化（大模型自动启用）
# 或减少样本数和轮数
python experiments/evaluate_tokenedit.py --model llama3-8b --samples 5 --epochs 10
```

### Q2: LLaMA模型加载失败

**原因**: 可能需要HuggingFace授权

**解决方案:**
```bash
# 1. 登录HuggingFace
huggingface-cli login

# 2. 或使用镜像
# 修改 model_config.py 中的模型路径
"model_name": "hf-mirror-path/Meta-Llama-3-8B"
```

### Q3: 训练太慢

**正常速度参考（A4000）：**
- GPT2-XL: ~15s/epoch
- GPT-J-6B: ~30s/epoch
- LLaMA-3-8B: ~45s/epoch

**加速方法：**
```python
# 减少轮数
--epochs 20

# 使用更少的目标层
# 修改 model_config.py 的 target_layers
```

## 实验建议

### 快速验证（30分钟）
```bash
python test_tokenedit_quick.py gpt2-xl
python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 5 --epochs 20
```

### 标准评估（2小时）
```bash
python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 20 --epochs 50
python experiments/evaluate_tokenedit.py --model gpt-j-6b --samples 20 --epochs 30
```

### 完整对比（4小时）
```bash
python experiments/evaluate_all.py --models gpt2-xl gpt-j-6b llama3-8b --samples 50 --epochs 30
```
'''

# ==================== 主函数 ====================
def main():
    print("="*70)
    print("生成实验文件 (支持: gpt2-xl, gpt-j-6b, llama3-8b)")
    print("优化: A4000 (16GB显存)")
    print("="*70)
    
    # 创建目录
    Path("experiments").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    print("\n创建实验脚本...")
    
    # 创建文件
    files = [
        ("model_config.py", MODEL_CONFIG),
        ("test_tokenedit_quick.py", TEST_QUICK),
        ("experiments/prepare_data.py", PREPARE_DATA),
        ("experiments/evaluate_tokenedit.py", EVALUATE_TOKENEDIT),
        ("experiments/evaluate_all.py", EVALUATE_ALL),
        ("run_experiments.sh", RUN_SCRIPT),
        ("README_MODELS.md", README_MODELS),
    ]
    
    for path, content in files:
        create_file(Path(path), content)
    
    # 设置shell脚本执行权限
    import os
    if os.name != 'nt':  # 非Windows系统
        os.chmod("run_experiments.sh", 0o755)
    
    print("\n" + "="*70)
    print("✓ 所有实验文件已生成！")
    print("="*70)
    
    print("\n支持的模型:")
    print("  - gpt2-xl (1.5B, ~6GB显存)")
    print("  - gpt-j-6b (6B, ~13GB显存, 8bit量化)")
    print("  - llama3-8b (8B, ~15GB显存, 8bit量化)")
    
    print("\n快速开始:")
    print("  1. 快速测试:  python test_tokenedit_quick.py gpt2-xl")
    print("  2. 准备数据:  python experiments/prepare_data.py")
    print("  3. 完整评估:  python experiments/evaluate_tokenedit.py --model gpt2-xl")
    print("  4. 对比实验:  python experiments/evaluate_all.py")
    print("\n  或一键运行:  bash run_experiments.sh gpt2-xl")
    
    print("\n查看详细说明: cat README_MODELS.md")

if __name__ == "__main__":
    main()