"""
诊断脚本 - 检查所有改进是否生效
"""
import sys
from pathlib import Path
import json

script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from experiments.evaluate_tokenedit_full import load_hparams_from_json
from model_config import load_model_optimized
from tokenedit import TokenEditEditor

def diagnose_improvements():
    """检查所有改进是否生效"""
    print("=" * 70)
    print("诊断改进是否生效")
    print("=" * 70)

    # 1. 检查配置文件
    print("\n[1/5] 检查配置文件...")
    hparams = load_hparams_from_json("gpt2-xl")

    print(f"  routing_threshold: {hparams.routing_threshold}")
    print(f"    ✓ 是否>=0.6: {hparams.routing_threshold >= 0.6}")

    print(f"  num_epochs: {hparams.num_epochs}")
    print(f"  learning_rate: {hparams.learning_rate}")
    print(f"  w_edit: {hparams.w_edit}")
    print(f"  w_suppress: {hparams.w_suppress}")

    # 2. 检查训练代码中的参数裁剪
    print("\n[2/5] 检查训练代码中的参数裁剪...")
    with open("tokenedit/tokenedit_main.py", "r", encoding="utf-8") as f:
        code = f.read()

    # 检查max_v_norm
    if "max_v_norm = 2.0" in code:
        print("  ✓ max_v_norm = 2.0 (正确)")
    elif "max_v_norm = 10.0" in code:
        print("  ✗ max_v_norm = 10.0 (应该是2.0!)")
    else:
        print("  ? 未找到max_v_norm定义")

    # 检查max_alpha
    if "max_alpha = 2.0" in code:
        print("  ✓ max_alpha = 2.0 (正确)")
    elif "max_alpha = 5.0" in code:
        print("  ✗ max_alpha = 5.0 (应该是2.0!)")
    else:
        print("  ? 未找到max_alpha定义")

    # 3. 检查损失计算改进
    print("\n[3/5] 检查损失计算改进...")

    # 检查是否使用add_special_tokens=True
    if code.count("add_special_tokens=True") > 5:
        print("  ✓ 统一使用add_special_tokens=True")
    else:
        print("  ✗ add_special_tokens使用不一致")

    # 检查suppress loss是否计算联合概率
    if "joint_log_prob" in code:
        print("  ✓ Suppress Loss计算联合概率")
    else:
        print("  ✗ Suppress Loss可能只看第一个token")

    # 检查是否立即backward
    if "sample_loss.backward()" in code:
        print("  ✓ 训练循环立即backward (防止OOM)")
    else:
        print("  ✗ 可能没有立即backward")

    # 4. 检查路由机制
    print("\n[4/5] 检查路由机制改进...")
    with open("tokenedit/prompt_router.py", "r", encoding="utf-8") as f:
        router_code = f.read()

    # 检查拒绝区域
    if "if best_sim - second_best_sim < 0.1:" in router_code:
        print("  ✓ 添加了拒绝区域机制")
    else:
        print("  ✗ 缺少拒绝区域机制")

    # 检查主体匹配
    if "if info['subject'].lower() not in prompt.lower():" in router_code:
        print("  ✓ 主体匹配作为必要条件")
    else:
        print("  ✗ 缺少主体匹配验证")

    # 5. 检查evaluate代码的DEBUG输出
    print("\n[5/5] 检查evaluate代码的DEBUG输出...")
    with open("experiments/evaluate_tokenedit_full.py", "r", encoding="utf-8") as f:
        eval_code = f.read()

    # 检查DEBUG输出范围
    if "if i < 3:" in eval_code:
        print("  ⚠ DEBUG只输出前3个样本 - 建议增加或移除限制")

    if '[DEBUG]' in eval_code:
        print("  ✓ 包含DEBUG输出")
    else:
        print("  ✗ 缺少DEBUG输出")

    # 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    print("\n可能导致'改进前后结果一样'的原因:")
    print("1. 参数裁剪阈值没有更新 (max_v_norm, max_alpha)")
    print("2. routing_threshold在代码和配置文件不一致")
    print("3. DEBUG输出被限制在前3个样本")
    print("\n建议:")
    print("- 修改 tokenedit_main.py:350-351 的参数裁剪值")
    print("- 统一 routing_threshold 配置")
    print("- 增加或移除 evaluate_tokenedit_full.py 的 'if i < 3' 限制")
    print("=" * 70)

if __name__ == "__main__":
    diagnose_improvements()