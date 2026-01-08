"""调试版本的TokenEdit测试 - 显示详细信息"""
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenedit import TokenEditEditor, TokenEditHyperParams

# 导入模型配置
try:
    from model_config import load_model_optimized
except ImportError:
    print("错误: 请先运行 generate_experiment_files.py 生成 model_config.py")
    sys.exit(1)

def main(model_name="gpt2-xl"):
    """
    调试测试 - 显示详细的中间信?    """
    print("="*70)
    print(f"TokenEdit 调试测试 - {model_name}")
    print("="*70)

    # 加载模型
    print("\n[1/5] 加载模型...")
    model, tokenizer, _ = load_model_optimized(model_name)

    # Load hyperparams from JSON
    print("\n[2/5] Load hyperparams...")
    hparams_path = os.path.join(
        os.path.dirname(__file__),
        "hparams",
        "TokenEdit",
        f"{model_name}.json",
    )
    if not os.path.isfile(hparams_path):
        print(f"Error: missing hparams file: {hparams_path}")
        sys.exit(1)

    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams_data = json.load(f)

    hparams = TokenEditHyperParams(**hparams_data)
    hparams.model_name = model_name
    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.verbose = True

    print(f"  训练轮数: {hparams.num_epochs}")
    print(f"  目标? {hparams.target_layers[:3]}...{hparams.target_layers[-3:]}")
    print(f"  Token初始化标准差: {hparams.token_init_std}")
    print(f"  路由阈? {hparams.routing_threshold}")

    # 创建编辑?    print("\n[3/5] 创建编辑?..")
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
            print(f"Route hit: edit #{edit_id}")
            print(f"  Subject: {req['subject']}")
            print(f"  Target: {req['subject']} -> {req['target_new']}")
        else:
            print("Route miss: no edit triggered")
        output = editor.inference(prompt, max_new_tokens=10, verbose=True)
        print(f"输出: {output}")

        # 检查输出是否包含目标词
        if "Lyon" in output:
            print("OK: output contains 'Lyon'")
        elif "Paris" in output:
            print("FAIL: output still contains 'Paris'")
        else:
            print("WARN: output contains neither 'Lyon' nor 'Paris'")

    print("\n" + "="*70)
    print("调试测试完成!")
    print("="*70)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
    main(model)


