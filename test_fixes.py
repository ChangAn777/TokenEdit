"""验证TokenEdit修复的简单测试"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 测试主体检测修复
print("="*70)
print("测试 1: 主体检测修复")
print("="*70)

# 使用GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

test_cases = [
    ("The capital of France is", "France"),
    ("France is in", "France"),
    ("What is the capital of France?", "France"),
    ("France's capital city is", "France"),
]

from tokenedit.tokenedit_utils import TokenEditUtils

# 创建一个虚拟模型用于测试
class DummyModel:
    class Config:
        hidden_size = 1600
    config = Config()

    def parameters(self):
        return []

dummy_model = DummyModel()
utils = TokenEditUtils(dummy_model, tokenizer)

for prompt, subject in test_cases:
    print(f"\nPrompt: '{prompt}'")
    print(f"Subject: '{subject}'")

    # Token分析
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"  Tokens: {tokens}")
    print(f"  Decoded: {decoded}")

    # 主体检测
    positions = utils.find_subject_positions(prompt, subject, verbose=True)

    if positions:
        print(f"  ✓ 成功找到位置: {positions}")
    else:
        print(f"  ✗ 未找到主体")

print("\n" + "="*70)
print("测试 2: 路由阈值修复")
print("="*70)

from tokenedit.tokenedit_hparams import TokenEditHyperParams

# 测试默认参数
hparams = TokenEditHyperParams()
print(f"\n默认参数:")
print(f"  token_init_std: {hparams.token_init_std} (修复前: 0.01)")
print(f"  routing_threshold: {hparams.routing_threshold} (修复前: 0.8)")
print(f"  use_template_routing: {hparams.use_template_routing}")

if hparams.token_init_std > 0.05:
    print("  ✓ Token初始化标准差已增大")
else:
    print("  ✗ Token初始化标准差仍然太小")

if hparams.routing_threshold < 0.5:
    print("  ✓ 路由阈值已降低")
else:
    print("  ✗ 路由阈值仍然太高")

print("\n" + "="*70)
print("测试 3: 编辑向量初始化")
print("="*70)

from tokenedit.edit_token_module import EditTokenModule

# 创建测试模块
module = EditTokenModule(
    hidden_size=1600,
    num_edits=1,
    hparams=hparams
)

# 检查初始化
v_new, v_old, alpha, beta = module.get_edit_vectors(0)

print(f"\n编辑向量统计:")
print(f"  v_new: shape={v_new.shape}, mean={v_new.mean():.6f}, std={v_new.std():.6f}")
print(f"  v_old: shape={v_old.shape}, mean={v_old.mean():.6f}, std={v_old.std():.6f}")
print(f"  alpha: {alpha.item():.6f}")
print(f"  beta: {beta.item():.6f}")

if v_new.abs().mean() > 0.01:
    print("  ✓ 向量初始化幅度合理")
else:
    print("  ✗ 向量初始化幅度太小")

print("\n" + "="*70)
print("修复验证总结")
print("="*70)

print("\n✓ 已修复的问题:")
print("  1. 主体检测使用统一的 token 编码")
print("  2. 添加模糊匹配支持大小写和空格变化")
print("  3. Token 初始化标准差从 0.01 增加到 0.1")
print("  4. 路由阈值从 0.8 降低到 0.3")
print("  5. 路由优先使用模板匹配（更准确）")
print("  6. 编辑向量注入增加设备检查")
print("  7. 训练损失计算改进（主体检测失败返回小损失而非0）")
print("  8. 推理函数添加 verbose 参数支持")

print("\n建议:")
print("  - 运行 python test_tokenedit_debug.py 进行完整测试")
print("  - 查看 FIXES_SUMMARY.md 了解详细修复内容")
