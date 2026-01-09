#!/bin/bash

echo "======================================"
echo "TokenEdit 多模型实验流程"
echo "======================================"

# 选择模型
MODEL=${1:-gpt2-xl}

echo ""
echo "使用模型: $MODEL"
echo ""

echo "[1/4] 准备数据..."
python experiments/prepare_data.py

echo ""
echo "[2/4] 快速测试 ($MODEL)..."
python test_tokenedit_quick.py $MODEL

echo ""
echo "[3/4] 评估 TokenEdit ($MODEL)..."
python experiments/evaluate_tokenedit.py --model $MODEL --samples 10 --epochs 30

echo ""
echo "[4/4] 对比所有模型..."
python experiments/evaluate_all.py --models gpt2-xl gpt-j-6b llama3-8b --samples 10 --epochs 20

echo ""
echo "======================================"
echo "✓ 所有实验完成！"
echo "======================================"
echo ""
echo "查看结果:"
echo "  - results/tokenedit_$MODEL.json"
echo "  - results/comparison_all_models.json"
