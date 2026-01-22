"""TokenEdit超参数配置 (SOTA Defaults)"""

from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class TokenEditHyperParams:
    """TokenEdit超参数配置类"""
    
    # 模型配置
    model_name: str = "gpt2-xl"
    # 推荐不要在这里写死具体层数，让 main.py 自动根据模型推断
    target_layers: Optional[List[int]] = None  
    
    # Token配置
    # [修改] 默认使用智能初始化
    token_init_method: str = "target_smart"
    token_init_std: float = 0.1
    learnable_gates: bool = True
    use_low_rank: bool = False
    token_rank: int = 64
    
    # 训练配置
    num_epochs: int = 100
    learning_rate: float = 5e-2  # [修改] 提高学习率 (Smart Init 需要)
    batch_size: int = 4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_steps: int = 10
    gradient_clip: float = 1.0
    
    # 损失权重 (SOTA值)
    w_edit: float = 10.0      # [修改] 加大火力，从 1.0 -> 10.0
    w_suppress: float = 1.0   # [修改] 适度抑制，从 0.5 -> 1.0
    w_ortho: float = 0.01     # [修改] 放松约束，从 0.3 -> 0.01
    w_local: float = 2.0      # [修改] 保护邻居，从 0.2 -> 2.0
    
    # 正交约束
    ortho_prompt_lambda: float = 1.0
    ortho_token_lambda: float = 1.0
    ortho_method: str = "inner_product"
    
    # 路由配置
    routing_threshold: float = 0.98    # [修改] 提高门槛，从 0.3 -> 0.98
    use_embedding_routing: bool = True
    use_template_routing: bool = False # [修改] 默认关闭模板路由 (解决 Specificity 的关键)
    routing_aggregation: str = "max"
    
    # Prompt闭包
    use_forward: bool = True
    use_backward: bool = True
    use_judge: bool = True
    use_distract: bool = True
    num_paraphrase: int = 3
    
    # 其他
    device: str = "cuda"
    seed: int = 42
    verbose: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"

    # Early stopping (set patience to 0 or None to disable)
    early_stop_patience: Optional[int] = 20
    early_stop_min_delta: float = 1e-4
    
    @classmethod
    def from_yaml(cls, path: str):
        """从YAML文件加载配置"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_yaml(self, path: str):
        """保存配置到YAML文件"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
