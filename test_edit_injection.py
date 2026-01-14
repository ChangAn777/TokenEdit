"""
快速测试脚本 - 验证编辑注入是否工作
"""
import sys
from pathlib import Path

# 添加项目路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from experiments.evaluate_tokenedit_full import load_data, load_hparams_from_json
from model_config import load_model_optimized
from tokenedit import TokenEditEditor

def test_single_edit():
    """测试单个编辑的完整流程"""
    print("=" * 70)
    print("测试编辑注入机制")
    print("=" * 70)

    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    model, tokenizer, _ = load_model_optimized("gpt2-xl")
    print("  ✓ 模型加载成功")

    # 2. 加载单个样本
    print("\n[2/5] 加载数据...")
    requests = load_data(num_samples=1)
    req = requests[0]
    print(f"  样本: {req['subject']} -> {req['target_new']}")
    print(f"  Prompt: {req['prompt']}")

    # 3. 创建编辑器
    print("\n[3/5] 创建编辑器...")
    hparams = load_hparams_from_json("gpt2-xl")
    hparams.num_epochs = 10  # 快速训练
    hparams.device = "cuda"
    hparams.verbose = True
    editor = TokenEditEditor(model, tokenizer, hparams)
    print("  ✓ 编辑器创建成功")

    # 4. 应用编辑
    print("\n[4/5] 应用编辑...")
    editor.apply_edits([req])
    print("  ✓ 编辑应用成功")

    # 5. 测试推理
    print("\n[5/5] 测试推理...")
    test_prompt = req['prompt'].format(req['subject'])
    print(f"  测试prompt: {test_prompt}")

    # 使用inference方法(应该触发编辑)
    print("\n  [方法1] 使用 editor.inference():")
    result = editor.inference(test_prompt, max_new_tokens=10, verbose=True)
    print(f"  输出: {result}")

    # 直接测试概率
    print("\n  [方法2] 直接计算概率:")
    from experiments.evaluate_tokenedit_full import test_batch_prediction_multi

    probs, correct = test_batch_prediction_multi(
        editor,
        [test_prompt],
        [req['target_new']],
        [req['target_true']]
    )

    print(f"  prob_new: {probs[0]['target_new']:.4f}")
    print(f"  prob_true: {probs[0]['target_true']:.4f}")
    print(f"  correct: {correct[0]}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

if __name__ == "__main__":
    test_single_edit()