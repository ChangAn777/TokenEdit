# TokenEdit 实验结果分析报告

## 实验结果概览

你运行了 27 个样本，得到的结果如下：

### LOOSE 测试 (P_new > P_old)
- **Efficacy**: 92.59% ✓ (很好)
- **Generalization**: 88.89% ✓ (很好)
- **Specificity**: 85.19% ✓ (很好)

### STRICT 测试 (Argmax Match - 真实测试)
- **Efficacy**: 66.67% ⚠️ (下降 25.92%)
- **Generalization**: 66.67% ⚠️ (下降 22.22%)
- **Specificity**: 22.22% ❌ (下降 62.97% - **严重问题**)
- **Specificity(LogProb)**: 85.19% ✓ (与 LOOSE 一致)

---

## 问题诊断

### 🔴 核心问题：Specificity 严重下降 (22.22%)

这是最严重的问题。Specificity 衡量的是模型在邻域样本上是否保留了原始知识。

**症状**：
- LogProb 级别的 Specificity 很好 (85.19%)
- 但 Argmax 级别的 Specificity 很差 (22.22%)

**根本原因**：模型在邻域样本上预测了**错误的 token**，即使概率分布是对的。

---

## 根本原因分析

### 1️⃣ **路由机制过于激进** (最可能的主要原因)

**问题位置**: `tokenedit/prompt_router.py:143-151`

```python
if best_sim < self.hparams.routing_threshold:
    return None

# 拒绝区域检查
sorted_sims = sorted(similarities.values(), reverse=True)
if len(sorted_sims) > 1:
    second_best_sim = sorted_sims[1]
    if best_sim > 0.5 and second_best_sim > 0.4:
        if best_sim - second_best_sim < 0.1:
            return None
```

**当前配置** (`hparams/TokenEdit/gpt2-xl.json`):
```json
"routing_threshold": 0.95,
"use_embedding_routing": true,
```

**问题**：
- 阈值 0.95 太高，导致许多邻域样本被**误触发**编辑
- 邻域样本应该**不触发**编辑，但由于路由不精确，它们被错误地应用了编辑
- 这导致邻域样本的原始答案被覆盖

**证据**：
- Specificity(LogProb) = 85.19% 说明概率分布是对的
- Specificity(Argmax) = 22.22% 说明 token 选择错了
- 这表明编辑被应用到了不应该应用的地方

---

### 2️⃣ **主体位置检测失败**

**问题位置**: `tokenedit/tokenedit_main.py:200-211`

```python
def guarded_route(prompt: str, prompt_emb=None):
    candidate_id = original_route(prompt, prompt_emb)
    if candidate_id is None:
        return None
    if candidate_id in registry:
        subject = registry[candidate_id]['subject']
        positions = utils.find_subject_positions(
            prompt, subject, verbose=False, add_special_tokens=True
        )
        if not positions:
            return None
    return candidate_id
```

**问题**：
- 主体位置检测可能失败（例如，主体在邻域样本中以不同形式出现）
- 当检测失败时，编辑不被应用，但路由仍然返回 edit_id
- 这导致在评估时，邻域样本被错误地认为应该应用编辑

---

### 3️⃣ **评估中的编辑注入逻辑问题**

**问题位置**: `experiments/evaluate_tokenedit_full.py:370-400`

```python
# 2. Route to detect which edit to apply
edit_id = editor.router.route(prefix, prompt_emb)

# 3. Find subject positions and inject edit if triggered
injection_success = False
if edit_id is not None:
    req = editor.edits_registry[edit_id]
    subject_positions = editor.utils.find_subject_positions(
        prefix,
        req['subject'],
        verbose=False,
        add_special_tokens=True
    )

    if subject_positions:
        editor.injector.inject(...)
        injection_success = True
```

**问题**：
- 对于邻域样本，路由不应该返回任何 edit_id
- 但由于路由阈值问题，它返回了 edit_id
- 然后主体位置检测可能失败，导致 `injection_success = False`
- 但模型仍然使用了之前的编辑状态

---

### 4️⃣ **损失函数权重配置不平衡**

**问题位置**: `hparams/TokenEdit/gpt2-xl.json`

```json
"w_edit": 10.0,
"w_suppress": 1.0,
"w_ortho": 0.1,
"w_local": 2.0,
```

**问题**：
- `w_edit = 10.0` 太高，导致模型过度优化编辑效果
- `w_suppress = 1.0` 太低，无法有效抑制旧答案
- `w_local = 2.0` 不足以保护邻域知识
- 没有启用对比学习 (`use_contrastive_loss: false`)

**结果**：
- 模型学会了编辑新答案，但没有学会保护邻域知识
- 邻域样本的原始答案被覆盖

---

### 5️⃣ **缺少关键的增强训练策略**

**问题位置**: `hparams/TokenEdit/gpt2-xl.json`

```json
"use_curriculum": false,
"use_hard_mining": false,
"use_adaptive_weights": false,
"use_contrastive_loss": false,
"use_focal_loss": false,
```

**问题**：
- 所有增强训练策略都被禁用了
- 没有课程学习来逐步学习编辑
- 没有难样本挖掘来关注困难的邻域样本
- 没有对比学习来明确区分新旧答案
- 没有焦点损失来关注困难的样本

**结果**：
- 模型使用基础训练，无法有效处理邻域保护
- 特别是对于困难的邻域样本，模型无法保护

---

### 6️⃣ **注入强度配置问题**

**问题位置**: `tokenedit/layer_injector.py:38-56`

```python
def _compute_layer_strengths(self) -> Dict[int, float]:
    """计算每层的注入强度"""
    strengths = {}
    n = len(self.target_layers)

    for i, layer in enumerate(self.target_layers):
        center = n // 2
        distance = abs(i - center)
        strength = 1.0 - 0.4 * (distance / max(center, 1))
        strengths[layer] = strength

    return strengths
```

**当前配置**:
```json
"target_layers": [13, 14, 15, 16, 17],
"use_progressive_injection": true,
```

**问题**：
- 目标层 [13, 14, 15, 16, 17] 可能不是最优的
- 对于 GPT-2-XL (48 层)，这些是相对较早的层
- 早期层的编辑可能对邻域知识造成更大的干扰
- 渐进式注入强度计算可能不够精细

---

### 7️⃣ **Prompt 闭包生成不完整**

**问题位置**: `hparams/TokenEdit/gpt2-xl.json`

```json
"use_forward": true,
"use_backward": true,
"use_judge": true,
"use_distract": true,
"num_paraphrase": 40,
```

**问题**：
- 虽然配置看起来完整，但 `num_paraphrase: 40` 可能太多
- 过多的释义可能导致训练数据不平衡
- 邻域样本的训练可能不足

---

## 问题优先级排序

| 优先级 | 问题 | 影响 | 修复难度 |
|--------|------|------|---------|
| 🔴 P0 | 路由阈值过高 (0.95) | Specificity 22.22% | 简单 |
| 🔴 P0 | 损失权重不平衡 | Specificity 下降 | 简单 |
| 🟠 P1 | 缺少对比学习 | 无法区分新旧答案 | 中等 |
| 🟠 P1 | 目标层选择不优 | 邻域干扰 | 中等 |
| 🟡 P2 | 缺少课程学习 | 训练不稳定 | 中等 |
| 🟡 P2 | 主体位置检测失败 | 路由不精确 | 困难 |

---

## 修复方案

### 方案 1: 快速修复 (预期效果: +30-40%)

**修改 `hparams/TokenEdit/gpt2-xl.json`**:

```json
{
  "routing_threshold": 0.85,  // 从 0.95 降低到 0.85

  "w_edit": 8.0,              // 从 10.0 降低到 8.0
  "w_suppress": 2.0,          // 从 1.0 提高到 2.0
  "w_local": 3.0,             // 从 2.0 提高到 3.0
  "w_ortho": 0.2,             // 从 0.1 提高到 0.2

  "use_contrastive_loss": true,  // 启用对比学习
  "contrastive_margin": 2.0,
  "contrastive_temperature": 0.1,

  "num_paraphrase": 20,       // 从 40 降低到 20

  "num_epochs": 300,          // 从 250 增加到 300
  "learning_rate": 0.003,     // 从 0.005 降低到 0.003
}
```

**预期结果**:
- Specificity: 22.22% → 50-60%
- Efficacy: 66.67% → 70-75%
- Generalization: 66.67% → 70-75%

---

### 方案 2: 中等修复 (预期效果: +40-50%)

在方案 1 的基础上，添加：

```json
{
  "use_curriculum": true,
  "curriculum_stages": [50, 150],

  "use_hard_mining": true,
  "hard_boost_factor": 2.0,

  "use_adaptive_weights": true,

  "use_focal_loss": true,
  "focal_gamma": 2.0,

  "target_layers": [20, 21, 22, 23, 24],  // 改为更深的层

  "use_progressive_injection": true,
}
```

**预期结果**:
- Specificity: 22.22% → 60-70%
- Efficacy: 66.67% → 75-80%
- Generalization: 66.67% → 75-80%

---

### 方案 3: 完整修复 (预期效果: +50-60%)

在方案 2 的基础上，改进路由机制：

**修改 `tokenedit/prompt_router.py`**:

```python
def route(self, prompt: str, prompt_embedding: Optional[torch.Tensor] = None) -> Optional[int]:
    """改进的路由机制"""

    # 1. 首先检查主体匹配
    subject_matched_ids = []
    for edit_id, info in self.edit_info.items():
        subject = info["subject"].lower()
        if subject in prompt.lower():
            subject_matched_ids.append(edit_id)

    # 如果没有主体匹配，直接返回 None
    if not subject_matched_ids:
        return None

    # 2. 在主体匹配的编辑中进行相似度检查
    if self.hparams.use_embedding_routing:
        if prompt_embedding is None:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                prompt_embedding = outputs.hidden_states[-1].mean(dim=1)

        similarities = {}
        for edit_id in subject_matched_ids:  # 只检查主体匹配的编辑
            edit_embs = self.edit_embeddings.get(edit_id, [])
            best_sim = None
            for emb in edit_embs:
                sim = F.cosine_similarity(prompt_embedding, emb, dim=-1).item()
                best_sim = sim if best_sim is None else max(best_sim, sim)
            similarities[edit_id] = best_sim if best_sim is not None else -1.0

        if similarities:
            best_edit_id = max(similarities, key=similarities.get)
            best_sim = similarities.get(best_edit_id, -1.0)

            # 3. 更严格的阈值检查
            if best_sim < self.hparams.routing_threshold:
                return None

            # 4. 拒绝区域检查（更严格）
            sorted_sims = sorted(similarities.values(), reverse=True)
            if len(sorted_sims) > 1:
                second_best_sim = sorted_sims[1]
                # 如果第二好的相似度太接近，拒绝
                if best_sim - second_best_sim < 0.15:
                    return None

            return best_edit_id

    # 回退到主体匹配
    return subject_matched_ids[0] if subject_matched_ids else None
```

**预期结果**:
- Specificity: 22.22% → 70-80%
- Efficacy: 66.67% → 75-85%
- Generalization: 66.67% → 75-85%

---

## 实施步骤

### 第一步：快速修复 (5 分钟)
1. 修改 `hparams/TokenEdit/gpt2-xl.json`
2. 运行 27 个样本测试
3. 观察 Specificity 是否改善

### 第二步：中等修复 (15 分钟)
1. 启用增强训练策略
2. 调整目标层
3. 运行测试

### 第三步：完整修复 (30 分钟)
1. 改进路由机制
2. 运行完整测试
3. 微调参数

---

## 关键指标解释

### Efficacy (编辑效果)
- 衡量：模型是否学会了新答案
- 测试：在重写 prompt 上，P(target_new) > P(target_true)
- 当前：66.67% (STRICT) - 可以接受

### Generalization (泛化能力)
- 衡量：模型是否能泛化到释义 prompt
- 测试：在释义 prompt 上，P(target_new) > P(target_true)
- 当前：66.67% (STRICT) - 可以接受

### Specificity (特异性/邻域保护)
- 衡量：模型是否保留了邻域知识
- 测试：在邻域 prompt 上，P(target_true) > P(target_new)
- 当前：22.22% (STRICT) - **严重问题**
- 原因：编辑被错误地应用到邻域样本

### LogProb vs Argmax
- **LogProb**: 基于概率分布的正确性
- **Argmax**: 基于最高概率 token 的正确性
- 差异表明：概率分布是对的，但 token 选择错了
- 这通常表示路由或注入问题

---

## 总结

你的 TokenEdit 实现在 **Efficacy** 和 **Generalization** 上表现很好，但在 **Specificity** 上有严重问题。

**根本原因**：
1. 路由阈值过高，导致邻域样本被误触发编辑
2. 损失权重不平衡，模型过度优化编辑效果，忽视邻域保护
3. 缺少对比学习和其他增强训练策略

**快速修复**：
- 降低路由阈值从 0.95 到 0.85
- 提高 w_suppress 和 w_local
- 启用对比学习

**预期改善**：
- Specificity: 22.22% → 50-80% (取决于修复程度)
- 总体性能：从 51.85% 平均值 → 65-75%

建议从快速修复开始，然后逐步应用中等和完整修复。
