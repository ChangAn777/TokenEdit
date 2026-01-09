# 单GPU设备错误修复

## 问题描述

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!
```

这个错误发生在使用 `device_map="auto"` 时，即使只有一块GPU，accelerate也会尝试将模型分散到多个"虚拟"设备上。

## 修复内容

### 1. 修复模型加载 ([model_config.py](model_config.py:43-105))

**关键改动：**
- 移除 `device_map="auto"`
- 手动控制设备分配
- 添加 `torch.cuda.empty_cache()` 清空缓存
- 显式调用 `model.to(device)` 和 `model.eval()`

```python
# 旧代码
load_kwargs = {
    "device_map": "auto",  # ← 问题所在
}

# 新代码
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空缓存

load_kwargs = {}  # 不使用 device_map

# 先加载到CPU，再手动移动到设备
model = AutoModelForCausalLM.from_pretrained(config['model_name'], **load_kwargs)
model = model.to(device)
model.eval()
```

### 2. 优化路由注册 ([tokenedit/prompt_router.py](tokenedit/prompt_router.py:48-73))

**关键改动：**
- 只在 `use_embedding_routing=true` 时计算 embeddings
- 避免不必要的模型前向传播

```python
# 旧代码
def register_edit(self, edit_id, subject, relation):
    text = f"{subject} {relation}"
    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
    outputs = self.model(**inputs, output_hidden_states=True)
    embedding = outputs.hidden_states[-1].mean(dim=1)
    self.edit_embeddings[edit_id] = embedding  # ← 总是计算

# 新代码
def register_edit(self, edit_id, subject, relation):
    # 只在使用embedding路由时计算
    if self.hparams.use_embedding_routing:
        text = f"{subject} {relation}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
        self.edit_embeddings[edit_id] = embedding
```

## 优势

1. **避免设备冲突** - 所有张量都在同一个设备上
2. **节省显存** - 不计算不必要的 embeddings
3. **加快速度** - 跳过 embedding 计算步骤
4. **更稳定** - 完全控制设备分配

## 验证

修复后，您应该能成功运行：

```bash
python test_tokenedit_debug.py gpt2-xl
```

预期输出：
```
[1/5] 加载模型...
加载模型: gpt2-xl
  8bit量化: False
  精度: float32
模型加载完成
  设备: cuda
  显存占用: X.XXGB (已分配) / Y.YYGB (已保留)

[2/5] Load hyperparams...
...
```

## 其他注意事项

### 如果仍然出现设备错误

检查您的 hparams 配置：

```json
{
    "device": "cuda",  // 确保是 "cuda" 或 "cpu"
    "use_embedding_routing": false  // 推荐设为 false
}
```

### 如果显存不足

对于 A800 (80GB)，GPT-2-XL (1.5B) 应该完全可以加载。如果遇到问题：

1. **使用更少的目标层：**
   ```python
   "target_layers": [13, 14, 15]  # 只用3层而不是5层
   ```

2. **减少 batch size：**
   ```python
   "batch_size": 1  # 已经是1了
   ```

3. **使用8bit量化：**
   ```python
   "load_in_8bit": true  # 在 model_config.py 中设置
   ```

## 完整修复列表

- [x] 修复 model_config.py 的设备分配
- [x] 优化 prompt_router.py 的 embedding 计算
- [x] 添加缓存清理
- [x] 确保所有张量在同一设备

## 下一步

现在您可以重新运行测试：

```bash
cd /home/dengjiaming/TokenEdit
python test_tokenedit_debug.py gpt2-xl
```

如果成功，您应该看到：
- ✅ 模型加载到单个 cuda 设备
- ✅ 路由正确工作
- ✅ 编辑成功应用
- ✅ 推理产生正确输出

**祝实验顺利！** 🚀
