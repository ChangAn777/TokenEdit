"""
模型配置文件
支持: gpt2-xl, gpt-j-6b, llama3-8b
针对A800 (80GB)优化 - 无需量化，可以使用float16/bfloat16
"""

MODEL_CONFIGS = {
    "gpt2-xl": {
        "model_name": "gpt2-xl",
        "hidden_size": 1600,
        "num_layers": 48,
        "target_layers": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 中间1/3层
        "load_in_8bit": False,  # A800显存充足，无需量化
        "torch_dtype": "float32",
        "memory_efficient": False,
    },
    "gpt-j-6b": {
        "model_name": "EleutherAI/gpt-j-6b",
        "hidden_size": 4096,
        "num_layers": 28,
        "target_layers": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "load_in_8bit": False,  # A800 80GB显存，可以不用量化
        "torch_dtype": "float16",  # 使用float16以获得更好性能
        "memory_efficient": False,
    },
    "llama3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B",  # 或使用HF镜像路径
        "hidden_size": 4096,
        "num_layers": 32,
        "target_layers": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "load_in_8bit": False,  # A800 80GB显存，可以不用量化
        "torch_dtype": "float16",  # 使用float16以获得更好性能
        "memory_efficient": False,
    },
}

def get_model_config(model_name: str):
    """获取模型配置"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}. 可用模型: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def load_model_optimized(model_name: str):
    """
    加载模型（针对A800 80GB显存优化）

    Args:
        model_name: 模型名称 (gpt2-xl, gpt-j-6b, llama3-8b)

    Returns:
        model, tokenizer, config
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = get_model_config(model_name)

    print(f"加载模型: {config['model_name']}")
    print(f"  8bit量化: {config['load_in_8bit']}")
    print(f"  精度: {config['torch_dtype']}")

    # 加载配置
    load_kwargs = {
        "device_map": "auto",  # 自动分配设备
    }

    if config['load_in_8bit']:
        # 8bit量化加载（节省显存）
        load_kwargs["load_in_8bit"] = True
        print("  使用8bit量化以节省显存")
    else:
        # 正常加载（A800显存充足，使用更高精度）
        if config['torch_dtype'] == "float16":
            load_kwargs["torch_dtype"] = torch.float16
        elif config['torch_dtype'] == "bfloat16":
            load_kwargs["torch_dtype"] = torch.bfloat16

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        **load_kwargs
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✓ 模型加载完成")
    print(f"  设备: {model.device}")

    # 显示显存占用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  显存占用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已保留)")

    return model, tokenizer, config
