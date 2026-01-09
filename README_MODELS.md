# 多模型使用指南

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
