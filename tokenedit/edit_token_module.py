"""
增强的编辑Token模块 - 结构性改进

核心改进:
1. 多尺度编辑向量 (Multi-Scale Edit Vectors)
2. 上下文感知门控 (Context-Aware Gating)
3. 残差连接 (Residual Connections)
4. 动态范数调整 (Dynamic Norm Scaling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class EnhancedEditTokenModule(nn.Module):
    """
    增强的编辑Token模块
    
    关键改进:
    1. 分层编辑向量: v_coarse (粗粒度) + v_fine (细粒度)
    2. 学习的温度参数控制编辑强度
    3. 门控机制考虑上下文信息
    """
    
    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams
        
        # ============================================================
        # 改进1: 多尺度编辑向量
        # ============================================================
        if hparams.use_multiscale:
            # 粗粒度向量 (主要语义修改)
            self.v_new_coarse = nn.Parameter(torch.empty(num_edits, hidden_size))
            # 细粒度向量 (微调修正)
            self.v_new_fine = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.v_old = nn.Parameter(torch.empty(num_edits, hidden_size))
            
            # 尺度混合权重
            self.scale_weight = nn.Parameter(torch.tensor(0.5))
        else:
            # 标准模式
            self.v_new = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.v_old = nn.Parameter(torch.empty(num_edits, hidden_size))
        
        # ============================================================
        # 改进2: 上下文感知门控网络
        # ============================================================
        if hparams.use_context_gating:
            # 小型MLP学习门控函数
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 2),  # [alpha, beta]
                nn.Tanh()  # 输出范围 [-1, 1]
            )
        else:
            # 标准可学习门控
            if hparams.learnable_gates:
                self.alpha = nn.Parameter(torch.empty(num_edits))
                self.beta = nn.Parameter(torch.empty(num_edits))
            else:
                self.register_buffer("alpha", torch.zeros(num_edits))
                self.register_buffer("beta", torch.zeros(num_edits))
        
        # ============================================================
        # 改进3: 温度参数 (控制编辑强度)
        # ============================================================
        if hparams.use_temperature:
            self.temperature = nn.Parameter(torch.ones(num_edits))
        
        # ============================================================
        # 改进4: 残差缩放因子
        # ============================================================
        if hparams.use_residual_scaling:
            self.residual_scale = nn.Parameter(torch.ones(num_edits))
        
        self._init_weights()
    
    def _init_weights(self):
        """统一初始化"""
        std = self.hparams.token_init_std
        method = self.hparams.token_init_method

        if self.hparams.use_multiscale:
            # 粗粒度: 较大初始化
            nn.init.normal_(self.v_new_coarse, std=std * 2.0)
            # 细粒度: 较小初始化
            nn.init.normal_(self.v_new_fine, std=std * 0.5)
            nn.init.zeros_(self.v_old)
        else:
            if method == "random" or method == "normal":
                nn.init.normal_(self.v_new, std=std)
            elif method == "zeros" or method == "zero":
                nn.init.zeros_(self.v_new)
            elif method == "target_smart":
                nn.init.normal_(self.v_new, std=std)
            else:
                raise ValueError(f"Unknown init method: {method}")
            nn.init.zeros_(self.v_old)

        # 门控初始化
        if not self.hparams.use_context_gating:
            if self.hparams.learnable_gates:
                nn.init.zeros_(self.alpha)
                nn.init.zeros_(self.beta)
        else:
            # 初始化门控网络
            for layer in self.gate_network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def get_edit_vectors(
        self, 
        edit_id: int, 
        context_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        获取编辑向量
        
        Args:
            edit_id: 编辑ID
            context_hidden: 上下文隐藏状态 (batch, hidden_size)
        
        Returns:
            v_new, v_old, alpha, beta
        """
        # 获取基础向量
        if self.hparams.use_multiscale:
            # 多尺度混合
            w = torch.sigmoid(self.scale_weight)
            v_new = w * self.v_new_coarse[edit_id] + (1 - w) * self.v_new_fine[edit_id]
            v_old = self.v_old[edit_id]
        else:
            v_new = self.v_new[edit_id]
            v_old = self.v_old[edit_id]
        
        # 应用温度缩放
        if self.hparams.use_temperature:
            temp = torch.sigmoid(self.temperature[edit_id]) * 2.0  # [0, 2]
            v_new = v_new * temp
        
        # 获取门控系数
        if self.hparams.use_context_gating and context_hidden is not None:
            # 上下文感知门控
            gates = self.gate_network(context_hidden.mean(dim=0))  # (2,)
            alpha = gates[0] * 2.0  # 缩放到 [-2, 2]
            beta = gates[1] * 2.0
        else:
            # 标准门控
            alpha = self.alpha[edit_id]
            beta = self.beta[edit_id]
        
        return v_new, v_old, alpha, beta
    
    def compute_orthogonality_loss(self, prompt_embeddings=None) -> torch.Tensor:
        """计算正交性损失"""
        device = self.v_new.device if hasattr(self, 'v_new') else self.v_new_coarse.device
        loss = torch.tensor(0.0, device=device)
        
        # 获取完整向量
        if self.hparams.use_multiscale:
            w = torch.sigmoid(self.scale_weight)
            v_new_full = w * self.v_new_coarse + (1 - w) * self.v_new_fine
            v_old_full = self.v_old
        else:
            v_new_full = self.v_new
            v_old_full = self.v_old
        
        # v_old ⊥ v_new
        if self.hparams.ortho_method == "inner_product":
            inner_product = torch.sum(v_new_full * v_old_full, dim=-1)
            loss += self.hparams.ortho_token_lambda * inner_product.pow(2).mean()
        elif self.hparams.ortho_method == "cosine":
            cosine_sim = F.cosine_similarity(v_new_full, v_old_full, dim=-1, eps=1e-8)
            loss += self.hparams.ortho_token_lambda * cosine_sim.pow(2).mean()
        
        # v_old ⊥ prompt
        if prompt_embeddings is not None:
            prompt_mean = prompt_embeddings.mean(dim=0)
            if self.hparams.ortho_method == "inner_product":
                inner_product = torch.sum(v_old_full * prompt_mean, dim=-1)
                loss += self.hparams.ortho_prompt_lambda * inner_product.pow(2).mean()
            elif self.hparams.ortho_method == "cosine":
                cosine_sim = F.cosine_similarity(
                    v_old_full, 
                    prompt_mean.unsqueeze(0).expand_as(v_old_full), 
                    dim=-1, eps=1e-8
                )
                loss += self.hparams.ortho_prompt_lambda * cosine_sim.pow(2).mean()

        return loss

    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        """计算范数约束损失"""
        device = self.v_new.device if hasattr(self, 'v_new') else self.v_new_coarse.device
        loss = torch.tensor(0.0, device=device)

        if self.hparams.use_multiscale:
            w = torch.sigmoid(self.scale_weight)
            v_new_full = w * self.v_new_coarse + (1 - w) * self.v_new_fine
            v_old_full = self.v_old
        else:
            v_new_full = self.v_new
            v_old_full = self.v_old

        # 约束1: 注入向量范数
        if not self.hparams.use_context_gating:
            injection_norms = torch.abs(self.alpha) * torch.norm(v_new_full, dim=-1)
            norm_violations = F.relu(injection_norms - max_norm)
            loss += norm_violations.pow(2).mean()

        # 约束2: v_new范数
        v_new_norms = torch.norm(v_new_full, dim=-1)
        v_new_violations = F.relu(v_new_norms - max_norm * 1.5)
        loss += v_new_violations.pow(2).mean()

        # 约束3: L2正则化
        l2_loss = (v_new_full.pow(2).mean() + v_old_full.pow(2).mean()) / 2
        loss += 0.01 * l2_loss

        return loss

    def forward(
        self, 
        edit_id: int, 
        hidden_states: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        应用编辑向量
        
        Args:
            edit_id: 编辑ID
            hidden_states: 输入隐藏状态 (batch, seq_len, hidden_size)
            return_components: 是否返回各组件
        
        Returns:
            修改后的隐藏状态
        """
        # 获取上下文
        context_hidden = hidden_states if self.hparams.use_context_gating else None
        
        v_new, v_old, alpha, beta = self.get_edit_vectors(edit_id, context_hidden)
        
        # 计算编辑向量
        edit_vector = alpha * v_new + beta * v_old
        
        # 应用残差缩放
        if self.hparams.use_residual_scaling:
            scale = torch.sigmoid(self.residual_scale[edit_id])
            edit_vector = edit_vector * scale
        
        # 应用编辑
        modified = hidden_states + edit_vector.view(1, 1, -1)
        
        if return_components:
            return modified, {
                'v_new': v_new,
                'v_old': v_old,
                'alpha': alpha,
                'beta': beta,
                'edit_vector': edit_vector
            }
        
        return modified


class AdaptiveEditTokenModule(nn.Module):
    """
    自适应编辑模块 - 更激进的设计
    
    核心思想:
    1. 不再使用固定的v_new和v_old
    2. 而是学习一个"编辑函数" f: h_old -> h_new
    3. 使用轻量级变换网络
    """
    
    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams
        
        # 每个编辑对应一个小型变换网络
        self.edit_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ) for _ in range(num_edits)
        ])
        
        # 混合系数 (控制原始表示 vs 变换后表示的比例)
        self.mix_coef = nn.Parameter(torch.zeros(num_edits))
        
        # 初始化
        for transform in self.edit_transforms:
            for layer in transform:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, edit_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        应用自适应变换
        
        h_new = (1 - λ) * h_old + λ * f(h_old)
        """
        # 提取主体位置的隐藏状态
        h_original = hidden_states
        
        # 应用变换
        h_transformed = self.edit_transforms[edit_id](h_original)
        
        # 混合
        mix = torch.sigmoid(self.mix_coef[edit_id])
        h_new = (1 - mix) * h_original + mix * h_transformed
        
        return h_new
    
    def compute_orthogonality_loss(self, prompt_embeddings=None) -> torch.Tensor:
        """自适应模块不需要正交性损失"""
        return torch.tensor(0.0, device=self.mix_coef.device)
    
    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        """约束变换的强度"""
        loss = torch.tensor(0.0, device=self.mix_coef.device)
        
        # 约束mix系数不要太大 (防止过度修改)
        mix_penalty = F.relu(torch.sigmoid(self.mix_coef) - 0.5)
        loss += mix_penalty.pow(2).mean()
        
        return loss


class HybridEditTokenModule(nn.Module):
    """
    混合模块: 结合显式向量和自适应变换
    
    最佳方案:
    - 对于简单编辑: 使用显式向量 (快速有效)
    - 对于复杂编辑: 使用自适应变换 (灵活准确)
    - 自动学习选择哪种方式
    """
    
    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams
        
        # 显式向量分支
        self.v_new = nn.Parameter(torch.empty(num_edits, hidden_size))
        self.v_old = nn.Parameter(torch.empty(num_edits, hidden_size))
        self.alpha = nn.Parameter(torch.zeros(num_edits))
        self.beta = nn.Parameter(torch.zeros(num_edits))
        
        # 自适应变换分支
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_edits)
        ])
        
        # 分支选择器 (学习使用哪个分支)
        self.branch_selector = nn.Parameter(torch.zeros(num_edits))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.v_new, std=self.hparams.token_init_std)
        nn.init.zeros_(self.v_old)
        
        for transform in self.transforms:
            for layer in transform:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, edit_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        # 显式向量分支
        explicit_edit = self.alpha[edit_id] * self.v_new[edit_id] + \
                       self.beta[edit_id] * self.v_old[edit_id]
        h_explicit = hidden_states + explicit_edit.view(1, 1, -1)
        
        # 自适应变换分支
        h_adaptive = self.transforms[edit_id](hidden_states)
        
        # 混合两个分支
        w = torch.sigmoid(self.branch_selector[edit_id])
        h_final = w * h_adaptive + (1 - w) * h_explicit
        
        return h_final
    
    def compute_orthogonality_loss(self, prompt_embeddings=None) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.v_new.device)
        
        # 只对显式向量应用正交约束
        if self.hparams.ortho_method == "cosine":
            cosine_sim = F.cosine_similarity(self.v_new, self.v_old, dim=-1, eps=1e-8)
            loss += self.hparams.ortho_token_lambda * cosine_sim.pow(2).mean()
        
        return loss
    
    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.v_new.device)
        
        # 约束显式向量
        v_new_norms = torch.norm(self.v_new, dim=-1)
        violations = F.relu(v_new_norms - max_norm)
        loss += violations.pow(2).mean()
        
        # L2正则
        loss += 0.01 * self.v_new.pow(2).mean()
        
        return loss