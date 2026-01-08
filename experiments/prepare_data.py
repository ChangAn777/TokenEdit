"""准备实验数据"""
import json
import requests
from pathlib import Path

def download_counterfact():
    """下载CounterFact数据集"""
    print("下载CounterFact数据集...")
    url = "https://rome.baulab.info/data/dsets/counterfact.json"
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "counterfact.json"
    
    if output_path.exists():
        print(f"✓ 数据集已存在: {output_path}")
        return
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            json.dump(response.json(), f, indent=2)
        print(f"✓ 数据集已下载: {output_path}")
    except Exception as e:
        print(f"⚠ 下载失败: {e}")
        print("将使用示例数据")

def create_sample_data():
    """创建示例数据"""
    print("\n创建示例数据...")
    
    sample = [
        {
            "requested_rewrite": {
                "prompt": "The capital of {} is",
                "subject": "France",
                "target_new": {"str": "Lyon"},
                "target_true": {"str": "Paris"}
            },
            "paraphrase_prompts": [
                "What is the capital of France?",
                "France's capital city is"
            ],
            "neighborhood_prompts": [
                {"prompt": "The population of France is", "target": "67 million"}
            ]
        },
        {
            "requested_rewrite": {
                "prompt": "The CEO of {} is",
                "subject": "Apple",
                "target_new": {"str": "Steve Jobs"},
                "target_true": {"str": "Tim Cook"}
            },
            "paraphrase_prompts": [
                "Who is the CEO of Apple?",
                "Apple's chief executive is"
            ],
            "neighborhood_prompts": [
                {"prompt": "Apple was founded by", "target": "Steve Jobs"}
            ]
        }
    ]
    
    Path("data").mkdir(exist_ok=True)
    with open("data/sample_data.json", 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("✓ 示例数据已创建: data/sample_data.json")

if __name__ == "__main__":
    print("="*70)
    print("数据准备")
    print("="*70)
    download_counterfact()
    create_sample_data()
    print("\n✓ 数据准备完成")
