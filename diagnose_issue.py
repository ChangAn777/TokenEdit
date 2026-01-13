"""
诊断脚本 - 检查为什么修改后结果不变
"""

import torch
import json
from tokenedit import TokenEditEditor, TokenEditHyperParams
from model_config import load_model_optimized

def check_param_clipping():
    """检查参数裁剪是否过于激进"""

    print("="*80)
    print("诊断参数裁剪问题")
    print("="*80)

    # 1. 检查硬编码的裁剪阈值
    print("\n[问题1] 硬编码的参数裁剪")
    print("-" * 60)
    print("在 tokenedit_main.py:341-357 中:")
    print("  max_v_norm = 2.0  (硬编码)")
    print("  max_alpha = 2.0   (硬编码)")
    print("\n这会导致:")
    print("  - v_new 和 v_old 的范数被强制限制在 2.0 以内")
    print("  - alpha 被强制限制在 [0, 2.0] 以内")
    print("  - 优化器无法学习到更大的值，即使损失函数要求")

    # 2. 检查损失函数中的 max_norm
    print("\n[问题2] 损失函数中的 max_norm")
    print("-" * 60)
    print("在 tokenedit_main.py:443 中:")
    print("  norm_loss = self.edit_module.compute_norm_constraint_loss(max_norm=2.0)")
    print("\n注意:")
    print("  - 这个 max_norm=2.0 是硬编码的，不是从超参数文件读取")
    print("  - 超参数文件中没有定义 max_norm 参数")

    # 3. 模拟测试：参数裁剪的影响
    print("\n[测试] 模拟参数裁剪的影响")
    print("-" * 60)

    # 创建模拟参数
    v_new = torch.randn(1, 1600) * 5.0  # 假设优化器学习到了大值
    alpha = torch.tensor([3.5])  # 假设优化器学习到了较大的门控系数

    print(f"\n训练后的参数值（优化器学习的值）:")
    print(f"  v_new norm: {v_new.norm().item():.4f}")
    print(f"  alpha: {alpha.item():.4f}")

    # 应用硬编码的裁剪
    max_v_norm = 2.0
    max_alpha = 2.0

    v_new_norms = torch.norm(v_new, dim=-1, keepdim=True)
    scale_new = torch.clamp(max_v_norm / (v_new_norms + 1e-8), max=1.0)
    v_new_clipped = v_new * scale_new
    alpha_clipped = torch.clamp(alpha, 0.0, max_alpha)

    print(f"\n应用硬编码裁剪后的参数值:")
    print(f"  v_new norm: {v_new_clipped.norm().item():.4f}")
    print(f"  alpha: {alpha_clipped.item():.4f}")

    # 计算注入向量
    inject_before = (alpha * v_new).norm().item()
    inject_after = (alpha_clipped * v_new_clipped).norm().item()

    print(f"\n注入向量范数:")
    print(f"  裁剪前: {inject_before:.4f}")
    print(f"  裁剪后: {inject_after:.4f}")
    print(f"  缩放比例: {inject_after/inject_before:.2%}")

    print("\n[结论]")
    print("-" * 60)
    print("参数裁剪过于激进，导致:")
    print("  1. 优化器学习到的参数值被强制缩小")
    print("  2. 注入向量被严重削弱，编辑效果被限制")
    print("  3. 即使损失函数、优化器等修改，最终效果仍被裁剪掩盖")

    # 4. 建议
    print("\n" + "="*80)
    print("建议的解决方案")
    print("="*80)

    print("\n方案1: 放宽裁剪阈值")
    print("-" * 60)
    print("修改 tokenedit_main.py:342-343:")
    print("  max_v_norm = 10.0   # 从 2.0 增加到 10.0")
    print("  max_alpha = 5.0     # 从 2.0 增加到 5.0")

    print("\n方案2: 从超参数文件读取阈值")
    print("-" * 60)
    print("在 hparams/TokenEdit/gpt2-xl.json 中添加:")
    print('  "max_v_norm": 10.0,')
    print('  "max_alpha": 5.0,')
    print("")
    print("然后修改代码:")
    print("  max_v_norm = self.hparams.max_v_norm")
    print("  max_alpha = self.hparams.max_alpha")

    print("\n方案3: 移除强制裁剪，只依赖损失函数约束")
    print("-" * 60)
    print("注释掉 tokenedit_main.py:341-357 的裁剪代码")
    print("只保留损失函数中的 soft constraint")

    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)

if __name__ == "__main__":
    check_param_clipping()