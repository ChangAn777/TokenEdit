"""TokenEdit超参数配置 (v2 - AlphaEdit 对标版)

主要修改:
1. routing_threshold: 0.98 → 0.85 (原值过严，几乎无法触发路由)
2. plateau_min_lr:   1e-3  → 1e-5 (原值等同于关闭调度器)
3. 新增 Null-Space Projection 配置块 (AlphaEdit 核心机制)
4. 新增 evaluation 配置块
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class TokenEditHyperParams:
    """TokenEdit超参数配置类 - AlphaEdit 对标版"""

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
    # 增强 EditToken 模块选项
    # ============================================================
    edit_module_type: str = "enhanced"
    use_multiscale: bool = True
    use_context_gating: bool = False
    use_temperature: bool = True
    use_residual_scaling: bool = True

    # ============================================================
    # 增强 Injector 选项
    # ============================================================
    injector_type: str = "enhanced"
    use_progressive_injection: bool = True
    use_attention_injection: bool = False
    use_dynamic_layers: bool = False

    # ============================================================
    # [NEW] Null-Space Projection  (AlphaEdit 核心机制)
    # ============================================================
    # 将编辑向量投影到上下文分布的零空间，从根本上保证知识特异性。
    # 参考: AlphaEdit (Fang et al., 2024); ROME (Meng et al., 2022)
    #
    # 原理: P_null = I - C(C^T C + λI)^{-1} C^T
    #   C 为无关上下文隐藏状态矩阵；v_inject → P_null @ v_inject
    #   投影后的向量与所有无关输入正交，不改变其输出分布。
    use_nullspace_projection: bool = True

    # 用于估计上下文协方差矩阵的无关样本数
    # 越多越准确，建议 50-200；来源优先使用 neighborhood_prompts
    nullspace_context_samples: int = 100

    # SVD 截断秩：保留多少主成分表示"已知知识子空间"
    # 经验值：hidden_size 的 5-15%，gpt2-xl(1600) → 80-160
    nullspace_rank: int = 100

    # Tikhonov 正则化系数，防止矩阵奇异
    nullspace_reg: float = 0.1

    # Null-space 投影的软约束损失权重（与硬投影互补）
    w_nullspace: float = 0.05

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

    plateau_patience: int = 20
    plateau_factor: float = 0.5
    # [BUG FIX] 原值 1e-3 ≈ 禁用了调度器（LR 从 8e-2 仅降至 1e-3 即停止衰减）
    # 改为 1e-5，允许充分衰减至收敛
    plateau_min_lr: float = 1e-5

    # ============================================================
    # 增强训练策略
    # ============================================================
    use_curriculum: bool = True
    curriculum_stages: Optional[List[int]] = None  # [30, 70] epoch 边界

    use_hard_mining: bool = True
    hard_boost_factor: float = 2.0

    use_adaptive_weights: bool = True

    use_contrastive_loss: bool = True
    contrastive_margin: float = 2.0
    contrastive_temperature: float = 0.1

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
    w_contrastive: float = 0.5

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
    # [BUG FIX] 原值 0.98 远超实际语义相似度范围（同义句通常 0.80-0.95）
    # 改为 0.85，与 MEMIT/SERAC 等方法的经验阈值对齐
    routing_threshold: float = 0.85
    use_embedding_routing: bool = True
    use_template_routing: bool = False
    routing_aggregation: str = "max"

    # ============================================================
    # Prompt 闭包
    # ============================================================
    use_forward: bool = True
    use_backward: bool = True
    use_judge: bool = True
    use_distract: bool = True
    num_paraphrase: int = 3

    # ============================================================
    # [NEW] 评估配置
    # ============================================================
    # 训练结束后自动计算 efficacy / generalization / specificity
    eval_after_edit: bool = True
    eval_max_samples: int = 5

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
        if self.curriculum_stages is None:
            self.curriculum_stages = [30, 70]

    @classmethod
    def from_yaml(cls, path: str) -> "TokenEditHyperParams":
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def get_active_features(self) -> dict:
        return {
            "Edit Module":       self.edit_module_type,
            "Multiscale":        self.use_multiscale,
            "Context Gating":    self.use_context_gating,
            "Temperature":       self.use_temperature,
            "Injector":          self.injector_type,
            "Progressive":       self.use_progressive_injection,
            "NullSpace Proj":    self.use_nullspace_projection,
            "Curriculum":        self.use_curriculum,
            "Hard Mining":       self.use_hard_mining,
            "Contrastive":       self.use_contrastive_loss,
            "Focal Loss":        self.use_focal_loss,
            "Routing Threshold": self.routing_threshold,
        }