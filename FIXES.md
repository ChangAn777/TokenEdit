# 代码修复说明

## 问题
在远程服务器运行实验时报错：`model_config.py 未找到`

## 原因分析
`experiments/` 目录下的脚本使用 `sys.path.append('..')` 来导入父目录的 `model_config.py`，但这个相对路径在某些执行上下文中可能不正确，导致 Python 无法找到模块。

## 修复内容

### 1. 修复导入路径问题

**修改文件：**
- [experiments/evaluate_tokenedit.py](experiments/evaluate_tokenedit.py)
- [experiments/evaluate_all.py](experiments/evaluate_all.py)

**修改内容：**
将简单的 `sys.path.append('..')` 改为更可靠的绝对路径方法：

```python
# 修改前（可能失败）
import sys
sys.path.append('..')

# 修改后（更可靠）
import sys
import os

# 添加项目根目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
```

**优势：**
- 使用绝对路径，不受当前工作目录影响
- `sys.path.insert(0, ...)` 确保项目根目录优先搜索
- 添加了更详细的错误信息，方便调试

### 2. 优化 A800 GPU 配置

**修改文件：**
- [model_config.py](model_config.py)

**主要改动：**

#### a) 更新文档注释
```python
# 修改前
"""
针对A4000 (16GB)优化
"""

# 修改后
"""
针对A800 (80GB)优化 - 无需量化，可以使用float16/bfloat16
"""
```

#### b) 关闭大模型的 8bit 量化
A800 有 80GB 显存，足够加载 GPT-J-6B 和 LLaMA-3-8B 的完整精度模型：

```python
# GPT-J-6B 配置
"gpt-j-6b": {
    "load_in_8bit": False,  # A800 80GB显存，可以不用量化
    "torch_dtype": "float16",  # 使用float16以获得更好性能
    "memory_efficient": False,
}

# LLaMA-3-8B 配置
"llama3-8b": {
    "load_in_8bit": False,  # A800 80GB显存，可以不用量化
    "torch_dtype": "float16",  # 使用float16以获得更好性能
    "memory_efficient": False,
}
```

**优势：**
- 无需量化，模型精度更高
- float16/bfloat16 比 int8 量化有更好的表达能力
- 训练和推理效果可能更好

#### c) 更新函数文档
```python
def load_model_optimized(model_name: str):
    """
    加载模型（针对A800 80GB显存优化）
    """
```

## 使用说明

### 在远程服务器上运行

1. **同步代码**：将修改后的文件同步到服务器
   ```bash
   # 需要同步的文件
   - experiments/evaluate_tokenedit.py
   - experiments/evaluate_all.py
   - model_config.py
   ```

2. **准备数据**（如果还没有）：
   ```bash
   python experiments/prepare_data.py
   ```

3. **运行评估**：
   ```bash
   # 快速测试
   python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 20 --epochs 50
   ```

### 预期显存占用（A800）

| 模型 | 配置 | 预估显存 |
|------|------|---------|
| GPT2-XL | float32 | ~6-8 GB |
| GPT-J-6B | float16 | ~12-15 GB |
| LLaMA-3-8B | float16 | ~16-20 GB |

A800 的 80GB 显存完全够用，甚至可以同时加载多个模型。

## 如果仍然遇到问题

### 检查 Python 路径
```bash
# 在 Python 中检查
python -c "import sys; print('\n'.join(sys.path))"
```

### 手动设置 PYTHONPATH
```bash
export PYTHONPATH=/path/to/TokenEdit-main:$PYTHONPATH
python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 20 --epochs 50
```

### 从项目根目录运行
```bash
cd /path/to/TokenEdit-main
python -m experiments.evaluate_tokenedit --model gpt2-xl --samples 20 --epochs 50
```

## 性能优化建议（A800）

由于 A800 显存充足，你可以：

1. **增加 batch size**：在 `tokenedit/__init__.py` 中调整
2. **增加目标层数量**：在 `model_config.py` 中编辑 `target_layers`
3. **使用更大模型**：可以尝试 LLaMA-3-70B（使用 8bit 量化）
4. **增加训练轮数**：`--epochs 100` 或更多

## 对比：A4000 vs A800 配置

| 配置项 | A4000 (16GB) | A800 (80GB) |
|--------|-------------|-------------|
| GPT2-XL | float32 | float32 |
| GPT-J-6B | 8bit量化 | **float16**（无需量化） |
| LLaMA-3-8B | 8bit量化 | **float16**（无需量化） |
| 最大模型 | ~8B参数 | ~70B参数（8bit） |

## 文件修改清单

- ✅ [experiments/evaluate_tokenedit.py](experiments/evaluate_tokenedit.py) - 修复导入路径
- ✅ [experiments/evaluate_all.py](experiments/evaluate_all.py) - 修复导入路径
- ✅ [model_config.py](model_config.py) - A800 优化配置
- ✅ FIXES.md - 本文档

## 下一步

将修改后的代码同步到服务器，然后运行：

```bash
python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 20 --epochs 50
```

应该就能正常工作了！🚀
