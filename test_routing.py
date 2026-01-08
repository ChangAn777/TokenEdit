"""快速测试路由逻辑"""
import json
import os
import sys
from tokenedit.prompt_router import PromptRouter
from tokenedit.tokenedit_hparams import TokenEditHyperParams

# 加载配置
hparams_path = "hparams/TokenEdit/gpt2-xl.json"
with open(hparams_path, "r") as f:
    hparams_data = json.load(f)

hparams = TokenEditHyperParams(**hparams_data)

print("="*70)
print("路由配置测试")
print("="*70)
print(f"\n当前配置:")
print(f"  use_embedding_routing: {hparams.use_embedding_routing}")
print(f"  use_template_routing: {hparams.use_template_routing}")
print(f"  routing_threshold: {hparams.routing_threshold}")

# 创建虚拟路由器（只测试模板匹配）
class DummyModel:
    pass

class DummyTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return None

router = PromptRouter(DummyModel(), DummyTokenizer(), hparams)

# 手动注册编辑（跳过 embedding 计算）
router.edit_info[0] = {
    "subject": "France",
    "relation": "capital"
}

# 测试用例
test_cases = [
    ("The capital of France is", True, "应该触发（有capital关系）"),
    ("France is in", False, "不应该触发（无capital关系）"),
    ("What is the capital of France?", True, "应该触发（有capital关系）"),
    ("France's capital city is", True, "应该触发（有capital city）"),
    ("I love France", False, "不应该触发（无capital关系）"),
    ("The president of France is", False, "不应该触发（关系不匹配）"),
]

print(f"\n关系模板: {router.relation_templates['capital']}")
print(f"\n测试用例:")

all_pass = True
for prompt, should_trigger, reason in test_cases:
    result = router.route(prompt, None)

    status = "OK" if (result is not None) == should_trigger else "FAIL"
    all_pass = all_pass and ((result is not None) == should_trigger)

    trigger_str = "触发" if result is not None else "未触发"
    print(f"\n{status} {prompt}")
    print(f"  {reason}")
    print(f"  结果: {trigger_str}")

print("\n" + "="*70)
if all_pass:
    print("OK: All tests passed! Routing logic is correct.")
else:
    print("FAIL: Some tests failed! Need to adjust routing logic.")
print("="*70)

# 分析
print("\n配置分析:")
print(f"1. 只使用模板匹配: {not hparams.use_embedding_routing and hparams.use_template_routing}")
print(f"   → 优点: 更精确，只在有明确关系关键词时触发")
print(f"   → 缺点: 可能遗漏一些变体表达")

print(f"\n2. 关键词数量: {len(router.relation_templates['capital'])}")
print(f"   → 当前关键词: {router.relation_templates['capital']}")

print(f"\n建议:")
if hparams.use_embedding_routing:
    print("  WARN: Consider setting use_embedding_routing=False to avoid over-triggering")
else:
    print("  OK: Embedding routing is disabled, should be more precise")

print("\nTo add more keywords, modify tokenedit/prompt_router.py:_load_relation_templates()")
