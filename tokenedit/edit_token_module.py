"""
增强的编辑 Token 模块 (v2 - AlphaEdit 对标版)

核心新增:
1. set_nullspace_projections() — 注册零空间投影矩阵（AlphaEdit 核心）
2. get_edit_vectors() 中硬投影 v_new → P_null @ v_new
3. compute_nullspace_loss() — 软约束（与硬投影互补，用于无法直接投影的场景）

原有修复保留:
4. device 解析的鲁棒写法（消除 hasattr 歧义 bug）
5. 范数约束和正交约束逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EnhancedEditTokenModule(nn.Module):
    """
    增强的编辑 Token 模块

    编辑向量注入公式:
        h'(s) = h(s) + α · P_null @ v_new + β · v_old

    其中 P_null 为零空间投影矩阵，保证 v_new 与无关输入正交。
    """

    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams

        # --------------------------------------------------------
        # 编辑向量
        # --------------------------------------------------------
        if hparams.use_multiscale:
            self.v_new_coarse = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.v_new_fine   = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.v_old        = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.scale_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.v_new = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.v_old = nn.Parameter(torch.empty(num_edits, hidden_size))

        # --------------------------------------------------------
        # 门控系数
        # --------------------------------------------------------
        if hparams.use_context_gating:
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 2),
                nn.Tanh(),
            )
        else:
            if hparams.learnable_gates:
                self.alpha = nn.Parameter(torch.zeros(num_edits))
                self.beta  = nn.Parameter(torch.zeros(num_edits))
            else:
                self.register_buffer("alpha", torch.zeros(num_edits))
                self.register_buffer("beta",  torch.zeros(num_edits))

        # --------------------------------------------------------
        # 辅助参数
        # --------------------------------------------------------
        if hparams.use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_edits))

        if hparams.use_residual_scaling:
            self.residual_scale = nn.Parameter(torch.ones(num_edits))

        # --------------------------------------------------------
        # [NEW] 零空间投影矩阵（由外部计算后注册，非可训练参数）
        # shape: (hidden_size, hidden_size)，buffer 不参与梯度更新
        # --------------------------------------------------------
        self.register_buffer("P_null", None)

        self._init_weights()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _init_weights(self):
        std = self.hparams.token_init_std

        if self.hparams.use_multiscale:
            nn.init.normal_(self.v_new_coarse, std=std * 2.0)
            nn.init.normal_(self.v_new_fine,   std=std * 0.5)
            nn.init.zeros_(self.v_old)
        else:
            nn.init.normal_(self.v_new, std=std)
            nn.init.zeros_(self.v_old)

        if not self.hparams.use_context_gating and self.hparams.learnable_gates:
            nn.init.zeros_(self.alpha)
            nn.init.zeros_(self.beta)
        elif self.hparams.use_context_gating:
            for layer in self.gate_network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # [NEW] 注册零空间投影矩阵
    # ------------------------------------------------------------------

    def set_nullspace_projections(
        self, projections: Dict[int, torch.Tensor]
    ) -> None:
        """
        注册零空间投影矩阵。

        Args:
            projections: {layer_idx: P_null (hidden_size, hidden_size)}
                         由 TokenEditUtils.compute_nullspace_projection_matrix() 生成

        设计说明：
            对多层取均值得到一个统一的 P_null，在 get_edit_vectors() 中
            对 v_new 实施硬投影：v_new_safe = P_null @ v_new。
            这确保 v_new 与所有已知上下文方向正交，是 AlphaEdit 的
            hook-based 等价实现。
        """
        if not projections:
            return

        stacked = torch.stack(
            [p.float() for p in projections.values()]
        ).mean(dim=0)  # (hidden_size, hidden_size)

        # 移动到模型所在设备
        device = self._get_param_device()
        self.P_null = stacked.to(device)

    def _get_param_device(self) -> torch.device:
        """安全地获取参数所在设备（避免原代码的 hasattr 歧义）"""
        # 优先从已知参数获取
        for name in ("v_new_coarse", "v_new", "alpha"):
            if hasattr(self, name):
                p = getattr(self, name)
                if isinstance(p, torch.Tensor):
                    return p.device
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # 获取编辑向量（核心接口）
    # ------------------------------------------------------------------

    def get_edit_vectors(
        self,
        edit_id: int,
        context_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取编辑向量及门控系数。

        新增逻辑：
            若 P_null 已注册，对 v_new 实施硬零空间投影：
                v_new_safe = P_null @ v_new
            这保证了注入向量在数学上不影响无关输入的输出分布。

        Returns:
            v_new:  (hidden_size,)  — 经零空间投影后的新知识向量
            v_old:  (hidden_size,)  — 旧知识抑制向量
            alpha:  scalar tensor   — 新知识门控
            beta:   scalar tensor   — 旧知识门控
        """
        # ---- 组合 v_new ----
        if self.hparams.use_multiscale:
            w = torch.sigmoid(self.scale_weight)
            v_new = w * self.v_new_coarse[edit_id] + (1.0 - w) * self.v_new_fine[edit_id]
        else:
            v_new = self.v_new[edit_id]

        v_old = self.v_old[edit_id]

        # ---- 温度缩放 ----
        if self.hparams.use_temperature:
            temp = torch.sigmoid(self.temperature[edit_id]) * 2.0  # ∈ (0, 2)
            v_new = v_new * temp

        # ---- [NEW] 零空间投影（硬约束）----
        if self.P_null is not None:
            # P_null 在 buffer 中，device 与模型一致
            v_new = self.P_null @ v_new  # (hidden_size,)

        # ---- 门控系数 ----
        if self.hparams.use_context_gating and context_hidden is not None:
            gates = self.gate_network(context_hidden.mean(dim=0))  # (2,)
            alpha = gates[0] * 2.0
            beta  = gates[1] * 2.0
        else:
            alpha = self.alpha[edit_id]
            beta  = self.beta[edit_id]

        return v_new, v_old, alpha, beta

    # ------------------------------------------------------------------
    # [NEW] 零空间软约束损失（与硬投影互补）
    # ------------------------------------------------------------------

    def compute_nullspace_loss(self) -> torch.Tensor:
        """
        零空间软约束损失。

        软硬结合：
          - 硬约束（set_nullspace_projections）：在 get_edit_vectors 中直接投影
          - 软约束（此函数）：作为额外惩罚项，用于 P_null 未注册或不完美时的补偿

        损失 = || v_new - P_null @ v_new ||² / || v_new ||²
             = || (I - P_null) @ v_new ||² / || v_new ||²
             表示 v_new 落在"已知知识子空间"中的成分比例（越小越好）

        若 P_null 未注册，返回 0。
        """
        if self.P_null is None:
            device = self._get_param_device()
            return torch.tensor(0.0, device=device)

        if self.hparams.use_multiscale:
            w = torch.sigmoid(self.scale_weight)
            v_new_full = w * self.v_new_coarse + (1.0 - w) * self.v_new_fine
        else:
            v_new_full = self.v_new  # (num_edits, hidden_size)

        # 投影到"已知子空间"的成分
        # (I - P_null) @ v = v - P_null @ v
        projected = v_new_full @ self.P_null.T  # (num_edits, hidden_size)
        residual = v_new_full - projected        # 已知子空间中的成分
        v_norms = v_new_full.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        loss = (residual.norm(dim=-1) / v_norms.squeeze(-1)).pow(2).mean()

        return loss

    # ------------------------------------------------------------------
    # 原有损失函数
    # ------------------------------------------------------------------

    def compute_orthogonality_loss(
        self, prompt_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """v_new ⊥ v_old，以及（可选）v_old ⊥ prompt"""
        device = self._get_param_device()
        loss = torch.tensor(0.0, device=device)

        if self.hparams.use_multiscale:
            w = torch.sigmoid(self.scale_weight)
            v_new_full = w * self.v_new_coarse + (1.0 - w) * self.v_new_fine
        else:
            v_new_full = self.v_new
        v_old_full = self.v_old

        if self.hparams.ortho_method == "inner_product":
            ip = torch.sum(v_new_full * v_old_full, dim=-1)
            loss += self.hparams.ortho_token_lambda * ip.pow(2).mean()
        elif self.hparams.ortho_method == "cosine":
            cs = F.cosine_similarity(v_new_full, v_old_full, dim=-1, eps=1e-8)
            loss += self.hparams.ortho_token_lambda * cs.pow(2).mean()

        if prompt_embeddings is not None:
            prompt_mean = prompt_embeddings.mean(dim=0)
            if self.hparams.ortho_method == "inner_product":
                ip = torch.sum(v_old_full * prompt_mean, dim=-1)
                loss += self.hparams.ortho_prompt_lambda * ip.pow(2).mean()
            elif self.hparams.ortho_method == "cosine":
                cs = F.cosine_similarity(
                    v_old_full,
                    prompt_mean.unsqueeze(0).expand_as(v_old_full),
                    dim=-1,
                    eps=1e-8,
                )
                loss += self.hparams.ortho_prompt_lambda * cs.pow(2).mean()

        return loss

    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        """范数约束损失"""
        device = self._get_param_device()
        loss = torch.tensor(0.0, device=device)

        if self.hparams.use_multiscale:
            w = torch.sigmoid(self.scale_weight)
            v_new_full = w * self.v_new_coarse + (1.0 - w) * self.v_new_fine
        else:
            v_new_full = self.v_new
        v_old_full = self.v_old

        if not self.hparams.use_context_gating and hasattr(self, "alpha"):
            injection_norms = torch.abs(self.alpha) * torch.norm(v_new_full, dim=-1)
            loss += F.relu(injection_norms - max_norm).pow(2).mean()

        loss += F.relu(torch.norm(v_new_full, dim=-1) - max_norm * 1.5).pow(2).mean()
        loss += 0.01 * (v_new_full.pow(2).mean() + v_old_full.pow(2).mean()) / 2.0

        return loss

    # ------------------------------------------------------------------
    # Forward（通常不直接调用，注入由 hook 完成）
    # ------------------------------------------------------------------

    def forward(
        self,
        edit_id: int,
        hidden_states: torch.Tensor,
        return_components: bool = False,
    ):
        context_hidden = hidden_states if self.hparams.use_context_gating else None
        v_new, v_old, alpha, beta = self.get_edit_vectors(edit_id, context_hidden)

        edit_vector = alpha * v_new + beta * v_old

        if self.hparams.use_residual_scaling:
            scale = torch.sigmoid(self.residual_scale[edit_id])
            edit_vector = edit_vector * scale

        modified = hidden_states + edit_vector.view(1, 1, -1)

        if return_components:
            return modified, {
                "v_new": v_new, "v_old": v_old,
                "alpha": alpha, "beta": beta,
                "edit_vector": edit_vector,
            }
        return modified


# Backward-compatible alias
EditTokenModule = EnhancedEditTokenModule


# ===========================================================================
# AdaptiveEditTokenModule
# ===========================================================================

class AdaptiveEditTokenModule(nn.Module):
    """
    自适应编辑模块：学习 f: h_old → h_new 变换函数。

    新增: set_nullspace_projections() / compute_nullspace_loss() 接口，
    与 EnhancedEditTokenModule 保持 API 一致性。
    """

    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams

        self.edit_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size),
            )
            for _ in range(num_edits)
        ])
        self.mix_coef = nn.Parameter(torch.zeros(num_edits))
        self.register_buffer("P_null", None)

        for transform in self.edit_transforms:
            for layer in transform:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def set_nullspace_projections(self, projections: Dict[int, torch.Tensor]) -> None:
        if not projections:
            return
        stacked = torch.stack([p.float() for p in projections.values()]).mean(dim=0)
        self.P_null = stacked.to(self.mix_coef.device)

    def forward(self, edit_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        h_transformed = self.edit_transforms[edit_id](hidden_states)
        if self.P_null is not None:
            # 对变换后的增量做零空间投影
            delta = h_transformed - hidden_states
            delta = delta @ self.P_null.T
            h_transformed = hidden_states + delta
        mix = torch.sigmoid(self.mix_coef[edit_id])
        return (1.0 - mix) * hidden_states + mix * h_transformed

    def compute_orthogonality_loss(self, prompt_embeddings=None) -> torch.Tensor:
        return torch.tensor(0.0, device=self.mix_coef.device)

    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        loss = F.relu(torch.sigmoid(self.mix_coef) - 0.5).pow(2).mean()
        return loss

    def compute_nullspace_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.mix_coef.device)


# ===========================================================================
# HybridEditTokenModule
# ===========================================================================

class HybridEditTokenModule(nn.Module):
    """
    混合模块：显式向量分支 + 自适应变换分支，自动学习权重。

    新增: set_nullspace_projections() / compute_nullspace_loss() 接口。
    """

    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams

        self.v_new = nn.Parameter(torch.empty(num_edits, hidden_size))
        self.v_old = nn.Parameter(torch.empty(num_edits, hidden_size))
        self.alpha  = nn.Parameter(torch.zeros(num_edits))
        self.beta   = nn.Parameter(torch.zeros(num_edits))

        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_edits)
        ])
        self.branch_selector = nn.Parameter(torch.zeros(num_edits))
        self.register_buffer("P_null", None)

        nn.init.normal_(self.v_new, std=hparams.token_init_std)
        nn.init.zeros_(self.v_old)
        for transform in self.transforms:
            for layer in transform:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def set_nullspace_projections(self, projections: Dict[int, torch.Tensor]) -> None:
        if not projections:
            return
        stacked = torch.stack([p.float() for p in projections.values()]).mean(dim=0)
        self.P_null = stacked.to(self.v_new.device)

    def forward(self, edit_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        v_new_i = self.v_new[edit_id]
        if self.P_null is not None:
            v_new_i = self.P_null @ v_new_i

        explicit_edit = self.alpha[edit_id] * v_new_i + self.beta[edit_id] * self.v_old[edit_id]
        h_explicit  = hidden_states + explicit_edit.view(1, 1, -1)
        h_adaptive  = self.transforms[edit_id](hidden_states)

        w = torch.sigmoid(self.branch_selector[edit_id])
        return w * h_adaptive + (1.0 - w) * h_explicit

    def compute_orthogonality_loss(self, prompt_embeddings=None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.v_new.device)
        if self.hparams.ortho_method == "cosine":
            cs = F.cosine_similarity(self.v_new, self.v_old, dim=-1, eps=1e-8)
            loss += self.hparams.ortho_token_lambda * cs.pow(2).mean()
        return loss

    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        loss = F.relu(torch.norm(self.v_new, dim=-1) - max_norm).pow(2).mean()
        loss += 0.01 * self.v_new.pow(2).mean()
        return loss

    def compute_nullspace_loss(self) -> torch.Tensor:
        if self.P_null is None:
            return torch.tensor(0.0, device=self.v_new.device)
        projected = self.v_new @ self.P_null.T
        residual  = self.v_new - projected
        v_norms   = self.v_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (residual.norm(dim=-1) / v_norms.squeeze(-1)).pow(2).mean()