"""快速测试TokenEdit - 支持多个模型"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenedit import TokenEditEditor, TokenEditHyperParams

# 导入模型配置
try:
    from model_config import load_model_optimized, get_model_config
except ImportError:
    print("错误: 请先运行 generate_experiment_files.py 生成 model_config.py")
    sys.exit(1)

def main(model_name="gpt2-xl"):
    """
    快速测试
    
    Args:
        model_name: 模型名称 (gpt2-xl, gpt-j-6b, llama3-8b)
    """
    print("="*70)
    print(f"TokenEdit 快速测试 - {model_name}")
    print("="*70)
    
    # 加载模型
    print("\n[1/4] 加载模型...")
    model, tokenizer, config = load_model_optimized(model_name)
    
    # 配置超参数
    print("\n[2/4] 配置参数...")
    hparams = TokenEditHyperParams(
        model_name=model_name,
        num_epochs=20 if model_name == "gpt2-xl" else 10,  # 大模型用更少epoch
        learning_rate=0.001,
        target_layers=config['target_layers'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True
    )
    
    print(f"  训练轮数: {hparams.num_epochs}")
    print(f"  目标层: {hparams.target_layers[:3]}...{hparams.target_layers[-3:]}")
    
    # 创建编辑器
    print("\n[3/4] 创建编辑器...")
    editor = TokenEditEditor(model, tokenizer, hparams)
    
    # 编辑请求
    requests = [{
        "prompt": "The capital of France is",
        "subject": "France",
        "relation": "capital",
        "target_new": "Lyon",
        "target_true": "Paris"
    }]
    
    # 应用编辑
    print("\n[4/4] 应用编辑...")
    try:
        editor.apply_edits(requests)
        
        # 测试推理
        print("\n" + "="*70)
        print("测试推理")
        print("="*70)

        test_prompts = [
            "The capital of France is",
            "France is in",
            "What is the capital of France?",
            "France's capital city is"
        ]

        for prompt in test_prompts:
            print(f"\n输入: {prompt}")
            output = editor.inference(prompt, max_new_tokens=10)
            print(f"输出: {output}")

        print("\n✓ 快速测试完成！")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ 显存不足！")
            print("建议:")
            print("  1. 使用8bit量化 (已自动启用大模型)")
            print("  2. 减少训练轮数")
            print("  3. 减少目标层数量")
        else:
            raise

if __name__ == "__main__":
    # 从命令行参数获取模型名称
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
    
    if model not in ["gpt2-xl", "gpt-j-6b", "llama3-8b"]:
        print(f"错误: 不支持的模型 '{model}'")
        print("支持的模型: gpt2-xl, gpt-j-6b, llama3-8b")
        print("\n使用方法:")
        print("  python test_tokenedit_quick.py gpt2-xl")
        print("  python test_tokenedit_quick.py gpt-j-6b")
        print("  python test_tokenedit_quick.py llama3-8b")
        sys.exit(1)
    
    main(model)
