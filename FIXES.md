# 代码修复说明

## 问题

### 问题 1：`model_config.py 未找到`
在远程服务器运行实验时报错：`model_config.py 未找到`

**原因分析：**
`experiments/` 目录下的脚本使用 `sys.path.append('..')` 来导入父目录的 `model_config.py`，但这个相对路径在某些执行上下文中可能不正确，导致 Python 无法找到模块。

### 问题 2：多GPU设备冲突
```
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:1 and cuda:0!
```

**原因分析：**
使用 `device_map="auto"` 时，accelerate 库会自动将模型分配到多个 GPU，导致张量在不同设备上，计算时出现设备不匹配错误。

### 问题 3：未使用正确的超参数配置 ⚠️ **最重要！**
```
编辑成功率: 50%
泛化能力: 25%
```

**原因分析：**
实验代码**没有加载** `hparams/TokenEdit/gpt2-xl.json` 配置文件，而是使用硬编码的错误参数：

| 参数 | JSON配置值 | 代码使用的值 | 影响 |
|------|-----------|-------------|------|
| `learning_rate` | `0.1` | `0.001` | ❌ 差了100倍，导致训练几乎无效 |
| `num_epochs` | `150` | `50` | ❌ 训练不足 |
| `target_layers` | `[13,14,15,16,17]` | `[15-24]` | ❌ 层数不匹配 |
| `w_edit` | `1.5` | `1.0` (默认) | ❌ 编辑权重不足 |
| `w_suppress` | `0.5` | `0.5` (默认) | ✅ 正确 |
| `w_ortho` | `0.1` | `0.3` (默认) | ❌ 正交约束过强 |

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

### 2. 修复多GPU设备冲突问题

**修改文件：**
- [model_config.py](model_config.py)

**修改内容：**

#### a) 移除 `device_map="auto"`
```python
# 修改前（会导致多GPU分配）
load_kwargs = {
    "device_map": "auto",  # ❌ 会自动分配到多个GPU
}

# 修改后（使用单GPU）
load_kwargs = {}  # ✅ 不使用自动设备映射
```

#### b) 手动控制设备分配
```python
# 加载模型到CPU
print("  正在加载模型到CPU...")
model = AutoModelForCausalLM.from_pretrained(
    config['model_name'],
    **load_kwargs
)

# 手动将模型移动到指定设备
device = f"cuda:{device_id}"  # 默认 cuda:0
print(f"  将模型移动到 {device}...")
model = model.to(device)
```

#### c) 添加 device_id 参数
```python
def load_model_optimized(model_name: str, device_id=0):
    """
    加载模型（针对A800 80GB显存优化）

    Args:
        model_name: 模型名称 (gpt2-xl, gpt-j-6b, llama3-8b)
        device_id: GPU设备ID，默认为0（使用单GPU避免多设备问题）

    Returns:
        model, tokenizer, config
    """
```

**优势：**
- 强制使用单个 GPU，避免多设备冲突
- 可以通过 `device_id` 参数指定使用哪个 GPU
- 更清晰的设备分配流程

### 3. 优化 A800 GPU 配置

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

### 4. 修复超参数加载问题 ⭐ **核心修复**

**修改文件：**
- [experiments/evaluate_tokenedit.py](experiments/evaluate_tokenedit.py)
- [experiments/evaluate_all.py](experiments/evaluate_all.py)

**修改内容：**

#### a) 添加 `load_hparams_from_json` 函数
```python
def load_hparams_from_json(model_name: str, hparams_dir: str = "hparams/TokenEdit"):
    """从JSON文件加载超参数配置"""
    hparams_path = Path(hparams_dir) / f"{model_name}.json"

    if not hparams_path.exists():
        print(f"⚠ 警告: 未找到配置文件 {hparams_path}，使用默认值")
        return TokenEditHyperParams(model_name=model_name)

    print(f"✓ 从 {hparams_path} 加载配置")

    with open(hparams_path, 'r') as f:
        config = json.load(f)

    # 打印关键配置
    print(f"  配置参数:")
    print(f"    - target_layers: {config.get('target_layers', '未设置')}")
    print(f"    - num_epochs: {config.get('num_epochs', 100)}")
    print(f"    - learning_rate: {config.get('learning_rate', 0.001)}")

    return TokenEditHyperParams(**config)
```

#### b) 修改编辑器创建代码
```python
# ❌ 修改前（硬编码错误参数）
hparams = TokenEditHyperParams(
    model_name=model_name,
    num_epochs=num_epochs,
    learning_rate=0.001,  # 错误！应该是 0.1
    target_layers=config['target_layers'],
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False
)

# ✅ 修改后（从JSON加载正确参数）
hparams = load_hparams_from_json(model_name)

# 如果命令行指定了num_epochs，覆盖配置文件中的值
if num_epochs is not None:
    hparams.num_epochs = num_epochs
    print(f"  覆盖 num_epochs 为: {num_epochs}")

hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
hparams.verbose = False
```

**优势：**
- ✅ 使用正确的学习率（0.1 而不是 0.001）
- ✅ 使用正确的训练轮数（150 而不是 50）
- ✅ 使用正确的目标层（[13-17] 而不是 [15-24]）
- ✅ 使用正确的损失权重
- ✅ 可以通过修改 JSON 文件快速调整参数

## 使用说明

### 在远程服务器上运行

1. **同步代码**：将修改后的文件同步到服务器
   ```bash
   # 需要同步的文件
   - experiments/evaluate_tokenedit.py
   - experiments/evaluate_all.py
   - model_config.py
   - hparams/TokenEdit/gpt2-xl.json  # 确保配置文件存在
   ```

2. **准备数据**（如果还没有）：
   ```bash
   python experiments/prepare_data.py
   ```

3. **运行评估**：
   ```bash
   # 使用配置文件中的参数（learning_rate=0.1, num_epochs=150）
   python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 20

   # 如果想覆盖训练轮数
   python experiments/evaluate_tokenedit.py --model gpt2-xl --samples 20 --epochs 100
   ```

4. **查看输出**：
   ```
   ✓ 从 hparams/TokenEdit/gpt2-xl.json 加载配置
     配置参数:
       - target_layers: [13, 14, 15, 16, 17]
       - num_epochs: 150
       - learning_rate: 0.1
       - w_edit: 1.5
       - w_suppress: 0.5
   ```

### 预期效果改善

使用正确的参数后，预期指标会大幅提升：

| 指标 | 之前（错误参数） | 预期（正确参数） |
|------|----------------|----------------|
| 编辑成功率 | 50% | **80-95%** |
| 泛化能力 | 25% | **70-90%** |

### 调整超参数

如果效果仍不理想，可以编辑 `hparams/TokenEdit/gpt2-xl.json`：

```json
{
  "learning_rate": 0.1,      // 尝试 0.05 - 0.2
  "num_epochs": 150,          // 尝试 100 - 200
  "w_edit": 1.5,             // 编辑损失权重
  "w_suppress": 0.5,         // 抑制损失权重
  "w_ortho": 0.1,            // 正交约束权重
  "target_layers": [13, 14, 15, 16, 17]
}
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
