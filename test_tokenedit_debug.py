"""增强版调试测试 - 多样例 + 详细诊断"""

import os

# 移除代理设置
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenedit import TokenEditEditor, TokenEditHyperParams

try:
    from model_config import load_model_optimized
except ImportError:
    print("错误: 请先运行 generate_experiment_files.py 生成 model_config.py")
    sys.exit(1)

def test_single_edit(editor, request, test_cases):
    """
    测试单个编辑，输出详细结果

    Args:
        editor: TokenEditEditor实例
        request: 编辑请求
        test_cases: 测试用例列表
    """
    print("\n" + "="*80)
    print(f"测试编辑: {request['subject']} -> {request['target_new']}")
    print(f"关系: {request.get('relation', 'unknown')}")
    print(f"原答案: {request['target_true']}")
    print("="*80)

    # 应用编辑
    print("\n[应用编辑]")
    editor.apply_edits([request])

    # 获取训练统计
    if hasattr(editor, 'edit_module'):
        v_new = editor.edit_module.v_new[0]
        v_old = editor.edit_module.v_old[0]
        alpha = editor.edit_module.alpha[0].item()
        beta = editor.edit_module.beta[0].item()

        print(f"  v_new norm: {v_new.norm().item():.4f}")
        print(f"  v_old norm: {v_old.norm().item():.4f}")
        print(f"  alpha (门控): {alpha:.4f}")
        print(f"  beta (门控): {beta:.4f}")
        print(f"  v_new·v_old: {(v_new * v_old).sum().item():.4f}")

    # 测试各个case
    results = []
    for i, case in enumerate(test_cases):
        prompt = case['prompt']
        expected = case.get('expected', request['target_new'])
        should_trigger = case.get('should_trigger', True)

        print(f"\n[测试 {i+1}/{len(test_cases)}]")
        print(f"  Prompt: '{prompt}'")
        print(f"  预期答案: {expected}")
        print(f"  应触发编辑: {should_trigger}")

        # 路由检测
        inputs = editor.tokenizer(prompt, return_tensors="pt").to(editor.device)
        with torch.no_grad():
            outputs = editor.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        edit_id = editor.router.route(prompt, prompt_emb)
        if edit_id is not None:
            req = editor.edits_registry[edit_id]
            print(f"  [OK] Route hit: edit #{edit_id}")
            print(f"    Subject: {req['subject']} -> {req['target_new']}")
        else:
            print(f"  [X] Route miss")

        # 生成输出
        output = editor.inference(prompt, max_new_tokens=15, verbose=False)
        print(f"  生成: '{output}'")

        # 检查结果
        success = expected in output
        old_answer = request['target_true'] in output

        print(f"  Results:")
        print(f"    Contains expected '{expected}': {'[OK]' if success else '[X]'}")
        print(f"    Contains old answer '{request['target_true']}': {'[OK]' if old_answer else '[X]'}")

        # 计算概率
        targets = [request['target_new'], request['target_true']]
        probs = {}
        for target in targets:
            target_tokens = editor.tokenizer.encode(target, add_special_tokens=False)
            if len(target_tokens) > 0:
                full_text = f"{prompt} {target}"
                inputs = editor.tokenizer(full_text, return_tensors="pt").to(editor.device)
                with torch.no_grad():
                    outputs = editor.model(**inputs)
                    logits = outputs.logits[0, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                total_log_prob = sum(log_probs[t].item() for t in target_tokens[:3])
                probs[target] = np.exp(total_log_prob / min(len(target_tokens), 3))

        print(f"  概率估计:")
        for target, prob in probs.items():
            print(f"    P({target}): {prob:.4f}")

        results.append({
            'prompt': prompt,
            'expected': expected,
            'success': success,
            'old_leak': old_answer,
            'probs': probs,
            'route_hit': edit_id is not None
        })

    # 汇总统计
    print(f"\n[统计]")
    success_count = sum(1 for r in results if r['success'])
    old_leak_count = sum(1 for r in results if r['old_leak'])
    route_hit_count = sum(1 for r in results if r['route_hit'])

    print(f"  成功率: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"  旧答案泄漏: {old_leak_count}/{len(results)} ({100*old_leak_count/len(results):.1f}%)")
    print(f"  路由命中率: {route_hit_count}/{len(results)} ({100*route_hit_count/len(results):.1f}%)")

    return results

def main(model_name="gpt2-xl"):
    """主测试函数"""
    print("="*80)
    print(f"TokenEdit 增强调试测试 - {model_name}")
    print("="*80)

    # 加载模型
    print("\n[1/4] 加载模型...")
    model, tokenizer, _ = load_model_optimized(model_name)

    # 加载超参数
    print("\n[2/4] 加载超参数...")
    hparams_path = os.path.join("hparams", "TokenEdit", f"{model_name}.json")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hparams_data = json.load(f)
    hparams = TokenEditHyperParams(**hparams_data)
    hparams.model_name = model_name
    hparams.device = "cuda" if torch.cuda.is_available() else "cpu"
    hparams.verbose = False  # 减少输出

    print(f"  训练轮数: {hparams.num_epochs}")
    print(f"  学习率: {hparams.learning_rate}")
    print(f"  Token初始化std: {hparams.token_init_std}")
    print(f"  损失权重: edit={hparams.w_edit}, suppress={hparams.w_suppress}, ortho={hparams.w_ortho}")

    # 创建编辑器
    print("\n[3/4] 创建编辑器...")
    editor = TokenEditEditor(model, tokenizer, hparams)

    # 定义多个测试样例
    print("\n[4/4] 运行测试...")

    test_suite = [
        # ===== 样例1: Capital关系 =====
        {
            'request': {
                "prompt": "The capital of {} is",
                "subject": "France",
                "relation": "capital",
                "target_new": "Lyon",
                "target_true": "Paris"
            },
            'test_cases': [
                {'prompt': "The capital of France is", 'expected': "Lyon"},
                {'prompt': "What is the capital of France?", 'expected': "Lyon"},
                {'prompt': "France's capital is", 'expected': "Lyon"},
                {'prompt': "The capital of France is", 'expected': "Lyon"},  # 重复测试稳定性
            ]
        },

        # ===== 样例2: President关系 =====
        {
            'request': {
                "prompt": "The president of {} is",
                "subject": "United States",
                "relation": "president",
                "target_new": "George Washington",
                "target_true": "Joe Biden"
            },
            'test_cases': [
                {'prompt': "The president of United States is", 'expected': "George Washington"},
                {'prompt': "Who is the president of United States?", 'expected': "George Washington"},
                {'prompt': "United States president is", 'expected': "George Washington"},
            ]
        },

        # ===== 样例3: Founder关系 =====
        {
            'request': {
                "prompt": "{} was founded by",
                "subject": "Apple",
                "relation": "founder",
                "target_new": "Steve Jobs",
                "target_true": "Steve Wozniak"
            },
            'test_cases': [
                {'prompt': "Apple was founded by", 'expected': "Steve Jobs"},
                {'prompt': "Who founded Apple?", 'expected': "Steve Jobs"},
                {'prompt': "Apple founder is", 'expected': "Steve Jobs"},
            ]
        },

        # ===== 样例4: Num关系 =====
        {
            'request': {
                "prompt": "{} has",
                "subject": "iPhone",
                "relation": "num",
                "target_new": "5 cameras",
                "target_true": "3 cameras"
            },
            'test_cases': [
                {'prompt': "iPhone has", 'expected': "5 cameras"},
                {'prompt': "How many cameras does iPhone have?", 'expected': "5"},
            ]
        },

        # ===== 样例5: 简单事实 =====
        {
            'request': {
                "prompt": "The color of {} is",
                "subject": "sky",
                "relation": "color",
                "target_new": "green",
                "target_true": "blue"
            },
            'test_cases': [
                {'prompt': "The color of sky is", 'expected': "green"},
                {'prompt': "What color is the sky?", 'expected': "green"},
            ]
        },
    ]

    # 运行所有测试
    all_results = []
    for i, test in enumerate(test_suite):
        print(f"\n\n{'#'*80}")
        print(f"# 样例组 {i+1}/{len(test_suite)}")
        print(f"{'#'*80}")

        results = test_single_edit(editor, test['request'], test['test_cases'])
        all_results.append({
            'request': test['request'],
            'results': results
        })

    # 总体统计
    print("\n\n" + "="*80)
    print("总体统计")
    print("="*80)

    total_tests = sum(len(r['results']) for r in all_results)
    total_success = sum(sum(1 for r in res['results'] if r['success']) for res in all_results)
    total_old_leak = sum(sum(1 for r in res['results'] if r['old_leak']) for res in all_results)

    print(f"\n总测试数: {total_tests}")
    print(f"总成功数: {total_success} ({100*total_success/total_tests:.1f}%)")
    print(f"总泄漏数: {total_old_leak} ({100*total_old_leak/total_tests:.1f}%)")

    # 按关系类型统计
    print(f"\n按关系类型统计:")
    for i, res in enumerate(all_results):
        relation = res['request']['relation']
        success = sum(1 for r in res['results'] if r['success'])
        total = len(res['results'])
        print(f"  {relation}: {success}/{total} ({100*success/total:.1f}%)")

    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt2-xl"
    main(model)
