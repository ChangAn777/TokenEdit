"""TokenEdit超参数配置 (增强版 - 支持新模块)"""

from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class TokenEditHyperParams:
    """TokenEdit超参数配置类 - 支持增强模块"""
    
    # ============================================================
    # 模型配置
    # ============================================================
    model_name: str = "gpt2-xl"
    target_layers: Optional[List[int]] = None  
    
    # ============================================================
    # Token配置
    # ============================================================
    token_init_method: str = "target_smart"
    token_init_std: float = 0.05
    learnable_gates: bool = True
    use_low_rank: bool = False
    token_rank: int = 64
    
    # ============================================================
    # [新增] 增强EditToken模块选项
    # ============================================================
    # 模块类型: "standard", "enhanced", "adaptive", "hybrid"
    edit_module_type: str = "enhanced"
    
    # 多尺度向量
    use_multiscale: bool = True
    
    # 上下文感知门控
    use_context_gating: bool = False  # 默认关闭(可能不稳定)
    
    # 温度参数
    use_temperature: bool = True
    
    # 残差缩放
    use_residual_scaling: bool = True
    
    # ============================================================
    # [新增] 增强Injector选项
    # ============================================================
    # Injector类型: "standard", "enhanced", "attention", "adaptive"
    injector_type: str = "enhanced"
    
    # 渐进式注入
    use_progressive_injection: bool = True
    
    # 注意力引导注入
    use_attention_injection: bool = False  # 默认关闭
    
    # 动态层选择
    use_dynamic_layers: bool = False  # 默认关闭
    
    # ============================================================
    # 训练配置
    # ============================================================
    num_epochs: int = 150
    learning_rate: float = 8e-2
    batch_size: int = 4
    optimizer: str = "adam"
    scheduler: str = "plateau"
    warmup_steps: int = 15
    gradient_clip: float = 1.5
    
    # Plateau调度器参数
    plateau_patience: int = 20
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-3
    
    # ============================================================
    # [新增] 增强训练策略
    # ============================================================
    # 课程学习
    use_curriculum: bool = True
    curriculum_stages: List[int] = None  # [30, 70] epoch边界
    
    # 难样本挖掘
    use_hard_mining: bool = True
    hard_boost_factor: float = 2.0
    
    # 自适应损失权重
    use_adaptive_weights: bool = True
    
    # 对比学习
    use_contrastive_loss: bool = True
    contrastive_margin: float = 2.0
    contrastive_temperature: float = 0.1
    
    # 改进的损失函数
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    
    # ============================================================
    # 损失权重
    # ============================================================
    w_edit: float = 20.0
    w_suppress: float = 0.3
    w_ortho: float = 0.005
    w_local: float = 1.5
    w_contrastive: float = 0.5  # [新增] 对比损失权重

    rewrite_loss_scale: float = 3.0
    
    # ============================================================
    # 正交约束
    # ============================================================
    ortho_prompt_lambda: float = 0.5
    ortho_token_lambda: float = 0.5
    ortho_method: str = "inner_product"
    
    # ============================================================
    # 路由配置
    # ============================================================
    routing_threshold: float = 0.98
    use_embedding_routing: bool = True
    use_template_routing: bool = False
    routing_aggregation: str = "max"
    
    # ============================================================
    # Prompt闭包
    # ============================================================
    use_forward: bool = True
    use_backward: bool = True
    use_judge: bool = True
    use_distract: bool = True
    num_paraphrase: int = 3
    
    # ============================================================
    # 其他
    # ============================================================
    device: str = "cuda"
    seed: int = 42
    verbose: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"

    early_stop_patience: Optional[int] = 30
    early_stop_min_delta: float = 5e-5
    
    def __post_init__(self):
        """初始化后处理"""
        if self.curriculum_stages is None:
            self.curriculum_stages = [30, 70]
    
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
    
    def get_active_features(self) -> dict:
        """获取当前激活的特性列表"""
        return {
            "Edit Module": self.edit_module_type,
            "Multiscale": self.use_multiscale,
            "Context Gating": self.use_context_gating,
            "Temperature": self.use_temperature,
            "Injector": self.injector_type,
            "Progressive": self.use_progressive_injection,
            "Curriculum": self.use_curriculum,
            "Hard Mining": self.use_hard_mining,
            "Contrastive": self.use_contrastive_loss,
            "Focal Loss": self.use_focal_loss,
        }