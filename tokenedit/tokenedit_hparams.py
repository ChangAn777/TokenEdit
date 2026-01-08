"""TokenEdit超参数配置"""

from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class TokenEditHyperParams:
    """TokenEdit超参数配置类"""
    
    # 模型配置
    model_name: str = "gpt2-xl"
    target_layers: Optional[List[int]] = None  # None表示自动设置
    
    # Token配置
    token_init_method: str = "random"
    token_init_std: float = 0.1  # 增加初始化标准差，避免向量太小
    learnable_gates: bool = True
    use_low_rank: bool = False
    token_rank: int = 64
    
    # 训练配置
    num_epochs: int = 100
    learning_rate: float = 1e-3
    batch_size: int = 4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_steps: int = 10
    gradient_clip: float = 1.0
    
    # 损失权重
    w_edit: float = 1.0
    w_suppress: float = 0.5
    w_ortho: float = 0.3
    w_local: float = 0.2
    
    # 正交约束
    ortho_prompt_lambda: float = 1.0
    ortho_token_lambda: float = 1.0
    ortho_method: str = "inner_product"
    
    # 路由配置
    routing_threshold: float = 0.3  # 降低阈值，使路由更容易触发
    use_embedding_routing: bool = True
    use_template_routing: bool = True  # 优先使用模板匹配
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
