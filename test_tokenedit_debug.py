"""调试版本的TokenEdit测试 - 显示详细信息"""
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
    调试测试 - 显示详细的中间信息
    """
    print("="*70)
    print(f"TokenEdit 调试测试 - {model_name}")
    print("="*70)

    # 加载模型
    print("\n[1/5] 加载模型...")
    model, tokenizer, config = load_model_optimized(model_name)

    # 配置超参数（使用更激进的设置）
    print("\n[2/5] 配置参数...")
    hparams = TokenEditHyperParams(
        model_name=model_name,
        num_epochs=20,
        learning_rate=0.001,
        target_layers=config['target_layers'],
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,

        # 关键修复
        token_init_std=0.1,  # 增加初始化幅度
        routing_threshold=0.1,  # 大幅降低路由阈值
        use_template_routing=True,  # 优先使用模板匹配

        # 损失权重
        w_edit=1.0,
        w_suppress=0.3,
        w_ortho=0.1,
        w_local=0.1,
    )

    print(f"  训练轮数: {hparams.num_epochs}")
    print(f"  目标层: {hparams.target_layers[:3]}...{hparams.target_layers[-3:]}")
    print(f"  Token初始化标准差: {hparams.token_init_std}")
    print(f"  路由阈值: {hparams.routing_threshold}")

    # 创建编辑器
    print("\n[3/5] 创建编辑器...")
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
    print("\n[4/5] 应用编辑...")
    editor.apply_edits(requests)

    # 测试推理前的token分析
    print("\n" + "="*70)
    print("Token分析")
    print("="*70)

    test_subject = "France"
    test_prompts = [
        "The capital of France is",
        "France is in",
        "What is the capital of France?"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        # 编码分析
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  Tokens: {tokens}")
        print(f"  Decoded: {decoded}")

        # 查找主体位置
        subject_tokens = tokenizer.encode(test_subject, add_special_tokens=False)
        subject_decoded = [tokenizer.decode([t]) for t in subject_tokens]
        print(f"  Subject tokens: {subject_tokens} -> {subject_decoded}")

        # 查找位置
        positions = editor.utils.find_subject_positions(
            prompt, test_subject, verbose=True
        )

    # 测试推理
    print("\n" + "="*70)
    print("测试推理")
    print("="*70)

    for prompt in test_prompts:
        print(f"\n{'='*70}")
        print(f"输入: {prompt}")
        print(f"{'='*70}")

        # 显示路由信息
        inputs = tokenizer(prompt, return_tensors="pt").to(editor.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        # 测试路由
        edit_id = editor.router.route(prompt, prompt_emb)
        if edit_id is not None:
            req = editor.edits_registry[edit_id]
            print(f"✓ 路由成功: 触发编辑 #{edit_id}")
            print(f"  主体: {req['subject']}")
            print(f"  目标: {req['subject']} -> {req['target_new']}")
        else:
            print(f"✗ 路由失败: 未触发任何编辑")

        # 推理
        output = editor.inference(prompt, max_new_tokens=10, verbose=True)
        print(f"输出: {output}")

        # 检查输出是否包含目标词
        if "Lyon" in output:
            print("✓ 成功! 输出包含 'Lyon'")
        elif "Paris" in output:
            print("✗ 失败! 输出仍包含 'Paris'")
        else:
            print("? 输出既没有 'Lyon' 也没有 'Paris'")

    print("\n" + "="*70)
    print("调试测试完成!")
    print("="*70)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
    main(model)
