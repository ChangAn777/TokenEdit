# TokenEdit 调整指南

## 📊 当前测试结果分析

### ✅ 成功部分
- 路由检测成功（所有测试）
- 主体位置检测准确
- 第1个测试用例成功："The capital of France is Lyon"

### ❌ 问题
1. **过度触发** - "France is in" 不应该触发（无 capital 关系）
2. **��答失败** - "What is the capital of France?" 没有正确回答

---

## 🔧 问题 1: 路由过于宽松

### 现象
```
输入: France is in
✓ 触发编辑 #0: France -> Lyon  ← 不应该触发！
```

### 原因
当前路由只要包含 "France" 就触发，没有严格检查 capital 关系。

### 解决方案

#### 方案 A: 增强关系模板匹配（推荐）

修改 [tokenedit/prompt_router.py](tokenedit/prompt_router.py) 的路由逻辑：

```python
def route(self, prompt: str, prompt_embedding: Optional[torch.Tensor] = None) -> Optional[int]:
    if len(self.edit_embeddings) == 0:
        return None

    # 方法1: 关系模板匹配（更严格）
    if self.hparams.use_template_routing:
        for edit_id, info in self.edit_info.items():
            subject = info["subject"]
            relation = info["relation"]

            # 检查主体
            if subject.lower() in prompt.lower():
                # 检查关系关键词
                templates = self.relation_templates.get(relation, [])
                relation_found = False
                for template in templates:
                    if template.lower() in prompt.lower():
                        relation_found = True
                        break

                # 只有主体和关系都匹配才触发
                if relation_found:
                    return edit_id

    # 方法2: Embedding相似度（作为备选，但提高阈值）
    if self.hparams.use_embedding_routing:
        # ... 现有代码
        pass

    return None
```

#### 方案 B: 关闭 Embedding 路由（更严格）

```python
# 在 hparams 配置中
use_embedding_routing: bool = False  # 只使用模板匹配
use_template_routing: bool = True
```

---

## 🔧 问题 2: 问答形式输出失败

### 现象
```
输入: What is the capital of France?
输出: What is the capital of France?

France is the capital of France.  ← 错误回答
```

### 原因
编辑只改变 "France" 的表示，但模型生成时可能：
1. 重复问题
2. 没有直接生成答案

### 解决方案

#### 方案 A: 优化训练数据（推荐）

在 [tokenedit/prompt_closure.py](tokenedit/prompt_closure.py) 中添加问题形式的训练样本：

```python
"capital": {
    "forward": [
        "The capital of {subject} is",
        "{subject}'s capital is",
        "What is the capital of {subject}?",  # 添加问题形式
    ],
    "backward": [...],
    "judge": [...],
    "distract": [...]
}
```

#### 方案 B: 调整生成策略

在推理时使用不同的解码参数：

```python
# 在 inference 函数中
output_ids = self.model.generate(
    inputs['input_ids'],
    max_new_tokens=max_new_tokens,
    do_sample=False,
    temperature=0.7,  # 添加温度
    top_p=0.9,        # 添加 top-p 采样
    pad_token_id=self.tokenizer.eos_token_id
)
```

#### 方案 C: 增加训练轮数

```python
# 在 hparams 中
num_epochs: int = 50  # 从 20 增加到 50
```

---

## 🎯 立即可用的调整方案

### 步骤 1: 修改路由配置

创建新的 hparams 文件 `hparams/TokenEdit/gpt2-xl-strict.json`:

```json
{
    "model_name": "gpt2-xl",
    "target_layers": [13, 14, 15, 16, 17],

    "token_init_method": "random",
    "token_init_std": 0.1,
    "learnable_gates": true,
    "use_low_rank": false,

    "num_epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 1,

    "w_edit": 1.0,
    "w_suppress": 0.5,
    "w_ortho": 0.1,
    "w_local": 0.1,

    "routing_threshold": 0.5,
    "use_embedding_routing": false,  // 关闭 embedding 路由
    "use_template_routing": true,    // 只使用模板匹配

    "use_forward": true,
    "use_backward": true,
    "use_judge": false,
    "use_distract": false,

    "device": "cuda",
    "verbose": true
}
```

### 步骤 2: 增强关系模板

修改 [tokenedit/prompt_router.py](tokenedit/prompt_router.py:31-40):

```python
def _load_relation_templates(self) -> Dict[str, List[str]]:
    return {
        "capital": [
            "capital",
            "capital of",
            "capital city",
            "首都"
        ],
        "president": ["president", "总统", "leader of"],
        # ...
    }
```

### 步骤 3: 添加问题形式训练样本

修改 [tokenedit/prompt_closure.py](tokenedit/prompt_closure.py:19-24):

```python
"forward": [
    "The capital of {subject} is",
    "{subject}'s capital is",
    "What is the capital of {subject}?",
    "Tell me the capital of {subject}",
],
```

---

## 📊 预期效果

### 修复后预期结果

| 测试用例 | 当前 | 修复后 | 说明 |
|---------|------|--------|------|
| "The capital of France is" | ✅ Lyon | ✅ Lyon | 保持 |
| "France is in" | ❌ 触发 | ✅ 不触发 | 修复 |
| "What is the capital of France?" | ❌ 错误 | ✅ Lyon | 修复 |
| "France's capital city is" | ✅ Lyon | ✅ Lyon | 保持 |

---

## 🚀 快速实施

### 选项 1: 最小调整（推荐）

只需修改 hparams 配置：

```json
{
    "use_embedding_routing": false,
    "num_epochs": 50
}
```

### 选项 2: 完整优化

1. 修改路由逻辑
2. 增强关系模板
3. 添加问题形式训练样本
4. 增加训练轮数

---

## 💡 调试建议

### 1. 验证路由逻辑

```python
# 在测试脚本中添加
test_prompts = [
    ("The capital of France is", True),   # 应该触发
    ("France is in", False),              # 不应该触发
    ("What is the capital of France?", True),  # 应该触发
]

for prompt, should_trigger in test_prompts:
    edit_id = editor.router.route(prompt, prompt_emb)
    if should_trigger:
        assert edit_id is not None, f"应该触发但没触发: {prompt}"
    else:
        assert edit_id is None, f"不应该触发但触发了: {prompt}"
```

### 2. 检查训练样本

```python
# 查看生成的训练样本
closure = editor.closure_gen.generate(
    subject="France",
    relation="capital",
    new_object="Lyon",
    old_object="Paris"
)

print("Forward samples:", closure['forward'])
print("Backward samples:", closure['backward'])
```

### 3. 监控训练损失

```python
# 观察 loss 曲线
# Edit loss 应该降到 0.1 以下
# Suppress loss 应该稳定在 0.5-1.0
```

---

## 📝 总结

### 当前问题
1. ❌ 路由过于宽松（不需要编辑的输入也触发）
2. ❌ 问答形式输出不正确

### 推荐调整（按优先级）
1. **立即执行** - 设置 `use_embedding_routing: false`
2. **重要** - 增加训练轮数到 50
3. **建议** - 添加问题形式训练样本
4. **可选** - 增强关系模板匹配

### 预期改进
- ✅ 更精确的路由（只在需要时触发）
- ✅ 正确的问答输出
- ✅ 更好的泛化能力
