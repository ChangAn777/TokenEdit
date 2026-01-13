"""显式编辑Token模块

修复版本 - 解决了:
1. 范数约束过弱导致的over-injection
2. 正交性损失计算的数值稳定性
3. 添加L2正则化防止过拟合
"""

import torch
import torch.nn as nn
from typing import Tuple

class EditTokenModule(nn.Module):
    """
    显式编辑Token模块
    
    为每个编辑维护一对Token向量:
    - v_new: 新知识Token (用于注入新信息)
    - v_old: 旧知识Token (用于抑制旧信息)
    """
    
    def __init__(self, hidden_size: int, num_edits: int, hparams):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_edits = num_edits
        self.hparams = hparams
        
        # 初始化Token向量
        if hparams.use_low_rank:
            # 低秩分解: v = U @ V
            rank = hparams.token_rank
            self.v_new_U = nn.Parameter(
                torch.randn(num_edits, rank) * hparams.token_init_std
            )
            self.v_new_V = nn.Parameter(
                torch.randn(rank, hidden_size) * hparams.token_init_std
            )
            self.v_old_U = nn.Parameter(
                torch.randn(num_edits, rank) * hparams.token_init_std
            )
            self.v_old_V = nn.Parameter(
                torch.randn(rank, hidden_size) * hparams.token_init_std
            )
        else:
            # 全秩Token
            self.v_new = nn.Parameter(
                self._init_tokens(num_edits, hidden_size, hparams.token_init_method)
            )
            self.v_old = nn.Parameter(
                self._init_tokens(num_edits, hidden_size, hparams.token_init_method)
            )
        
        # 门控系数 - 初始化为0,让模型从无干扰开始学习
        if hparams.learnable_gates:
            self.alpha = nn.Parameter(torch.zeros(num_edits))
            self.beta = nn.Parameter(torch.zeros(num_edits))
        else:
            self.register_buffer("alpha", torch.zeros(num_edits))
            self.register_buffer("beta", torch.zeros(num_edits))
    
    def _init_tokens(self, num_edits: int, hidden_size: int, method: str) -> torch.Tensor:
        """初始化Token向量"""
        if method == "random":
            return torch.randn(num_edits, hidden_size) * self.hparams.token_init_std
        elif method == "zero":
            return torch.zeros(num_edits, hidden_size)
        elif method == "normal":
            return torch.normal(0, self.hparams.token_init_std, 
                              size=(num_edits, hidden_size))
        else:
            raise ValueError(f"Unknown init method: {method}")
    
    def get_edit_vectors(self, edit_id: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        获取指定编辑的向量和门控系数
        
        Returns:
            v_new: (hidden_size,)
            v_old: (hidden_size,)
            alpha: scalar
            beta: scalar
        """
        if self.hparams.use_low_rank:
            v_new = self.v_new_U[edit_id] @ self.v_new_V
            v_old = self.v_old_U[edit_id] @ self.v_old_V
        else:
            v_new = self.v_new[edit_id]
            v_old = self.v_old[edit_id]
        
        alpha = self.alpha[edit_id]
        beta = self.beta[edit_id]
        
        return v_new, v_old, alpha, beta
    
    def compute_orthogonality_loss(self, prompt_embeddings=None) -> torch.Tensor:
        """
        计算正交性损失
        
        L_ortho = lambda1 * |v_old · prompt|^2 + lambda2 * |v_old · v_new|^2
        
        修复:
        - 添加数值稳定性检查
        - 正确处理batch维度
        
        Args:
            prompt_embeddings: (batch_size, hidden_size) 或 None
        
        Returns:
            loss: scalar tensor
        """
        loss = torch.tensor(0.0, device=self.alpha.device)
        
        # 获取完整的Token矩阵
        if self.hparams.use_low_rank:
            v_new_full = self.v_new_U @ self.v_new_V  # (num_edits, hidden_size)
            v_old_full = self.v_old_U @ self.v_old_V
        else:
            v_new_full = self.v_new
            v_old_full = self.v_old
        
        # 1. v_old ⊥ v_new
        if self.hparams.ortho_method == "inner_product":
            inner_product = torch.sum(v_new_full * v_old_full, dim=-1)  # (num_edits,)
            loss += self.hparams.ortho_token_lambda * inner_product.pow(2).mean()
        elif self.hparams.ortho_method == "cosine":
            # 修复: 添加eps避免除零
            cosine_sim = torch.nn.functional.cosine_similarity(
                v_new_full, v_old_full, dim=-1, eps=1e-8
            )
            loss += self.hparams.ortho_token_lambda * cosine_sim.pow(2).mean()
        
        # 2. v_old ⊥ prompt (如果提供)
        if prompt_embeddings is not None:
            # 平均所有prompt embeddings
            prompt_mean = prompt_embeddings.mean(dim=0)  # (hidden_size,)
            
            if self.hparams.ortho_method == "inner_product":
                inner_product = torch.sum(v_old_full * prompt_mean, dim=-1)
                loss += self.hparams.ortho_prompt_lambda * inner_product.pow(2).mean()
            elif self.hparams.ortho_method == "cosine":
                cosine_sim = torch.nn.functional.cosine_similarity(
                    v_old_full, 
                    prompt_mean.unsqueeze(0).expand_as(v_old_full), 
                    dim=-1,
                    eps=1e-8
                )
                loss += self.hparams.ortho_prompt_lambda * cosine_sim.pow(2).mean()

        return loss

    def compute_norm_constraint_loss(self, max_norm: float = 2.0) -> torch.Tensor:
        """
        计算范数约束损失,防止Over-injection

        当模型难以拟合某些知识时,优化器会无限增大alpha和v_new的范数,
        导致注入向量 h' = h + alpha*v_new 的模长远超正常分布。

        修复:
        - max_norm从10降到2 (更合理)
        - 分别约束alpha, v_new, 注入向量
        - 添加L2正则化项

        L_norm = |alpha*v_new|_over + |alpha|_over + |v_new|_over + L2_reg

        Args:
            max_norm: 注入向量的最大允许范数(默认2.0)

        Returns:
            loss: scalar tensor
        """
        loss = torch.tensor(0.0, device=self.alpha.device)

        # 获取完整的Token矩阵
        if self.hparams.use_low_rank:
            v_new_full = self.v_new_U @ self.v_new_V
            v_old_full = self.v_old_U @ self.v_old_V
        else:
            v_new_full = self.v_new
            v_old_full = self.v_old

        # 约束1: 注入向量范数 |alpha * v_new| <= max_norm
        injection_norms = torch.abs(self.alpha) * torch.norm(v_new_full, dim=-1)
        norm_violations = torch.nn.functional.relu(injection_norms - max_norm)
        loss += norm_violations.pow(2).mean()

        # 约束2: alpha不应过大 (从5.0降到2.0)
        max_alpha = 2.0
        alpha_violations = torch.nn.functional.relu(torch.abs(self.alpha) - max_alpha)
        loss += alpha_violations.pow(2).mean()

        # 约束3: v_new本身的范数也不应过大
        v_new_norms = torch.norm(v_new_full, dim=-1)
        v_new_violations = torch.nn.functional.relu(v_new_norms - max_norm)
        loss += v_new_violations.pow(2).mean()

        # 修复: 添加L2正则化,防止过拟合
        l2_loss = (v_new_full.pow(2).mean() + v_old_full.pow(2).mean()) / 2
        loss += 0.01 * l2_loss

        return loss

    def forward(self, edit_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        应用编辑向量
        
        h' = h + alpha * v_new + beta * v_old
        
        Args:
            edit_id: 编辑ID
            hidden_states: (batch_size, seq_len, hidden_size)
        
        Returns:
            modified_hidden_states: (batch_size, seq_len, hidden_size)
        """
        v_new, v_old, alpha, beta = self.get_edit_vectors(edit_id)
        
        # 广播并应用
        edit_vector = alpha * v_new + beta * v_old  # (hidden_size,)
        modified = hidden_states + edit_vector.view(1, 1, -1)
        
        return modified