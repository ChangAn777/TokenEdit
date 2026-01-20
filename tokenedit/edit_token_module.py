"""
显式编辑Token模块 (最终修复版)
修复记录:
1. 解决 AttributeError: v_new not found (通过拆分定义与初始化)
2. 解决 ValueError: Unknown init method 'target_smart'
3. 保留了范数约束增强和L2正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # ============================================================
        # 1. 第一步：先定义参数容器 (Allocating Memory)
        #    注意：这里只创建 Parameter 对象，不进行具体数值初始化
        # ============================================================
        
        if hparams.use_low_rank:
            # 低秩分解模式
            rank = hparams.token_rank
            self.v_new_U = nn.Parameter(torch.empty(num_edits, rank))
            self.v_new_V = nn.Parameter(torch.empty(rank, hidden_size))
            self.v_old_U = nn.Parameter(torch.empty(num_edits, rank))
            self.v_old_V = nn.Parameter(torch.empty(rank, hidden_size))
        else:
            # 全秩模式 (标准模式)
            self.v_new = nn.Parameter(torch.empty(num_edits, hidden_size))
            self.v_old = nn.Parameter(torch.empty(num_edits, hidden_size))
        
        # 定义门控系数
        if hparams.learnable_gates:
            self.alpha = nn.Parameter(torch.empty(num_edits))
            self.beta = nn.Parameter(torch.empty(num_edits))
        else:
            self.register_buffer("alpha", torch.zeros(num_edits))
            self.register_buffer("beta", torch.zeros(num_edits))

        # ============================================================
        # 2. 第二步：再初始化数值 (Initializing Values)
        #    此时 self.v_new 已经存在，调用 init 函数不会报错
        # ============================================================
        self._init_weights()
    
    def _init_weights(self):
        """统一初始化所有权重"""
        std = self.hparams.token_init_std
        method = self.hparams.token_init_method

        # --- A. 初始化 Token 向量 ---
        if self.hparams.use_low_rank:
            nn.init.normal_(self.v_new_U, std=std)
            nn.init.normal_(self.v_new_V, std=std)
            nn.init.normal_(self.v_old_U, std=std)
            nn.init.normal_(self.v_old_V, std=std)
        else:
            # 1. 初始化 v_new
            if method == "random" or method == "normal":
                nn.init.normal_(self.v_new, std=std)
            elif method == "zeros" or method == "zero":
                nn.init.zeros_(self.v_new)
            # [关键修复] 支持 target_smart
            # 这里先用随机初始化占位，main.py 会随后用 Target Embedding 覆盖它
            elif method == "target_smart":
                nn.init.normal_(self.v_new, std=std)
            else:
                raise ValueError(f"Unknown init method: {method}")

            # 2. 初始化 v_old (通常初始为0)
            nn.init.zeros_(self.v_old)

        # --- B. 初始化 门控 ---
        if self.hparams.learnable_gates:
            nn.init.zeros_(self.alpha) # 从0开始，逐渐学习注入
            nn.init.zeros_(self.beta)
    
    def get_edit_vectors(self, edit_id: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """获取指定编辑的向量和门控系数"""
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
        """计算正交性损失 (稳健版)"""
        loss = torch.tensor(0.0, device=self.v_new.device if not self.hparams.use_low_rank else self.v_new_U.device)
        
        # 获取完整的Token矩阵
        if self.hparams.use_low_rank:
            v_new_full = self.v_new_U @ self.v_new_V
            v_old_full = self.v_old_U @ self.v_old_V
        else:
            v_new_full = self.v_new
            v_old_full = self.v_old
        
        # 1. v_old ⊥ v_new
        if self.hparams.ortho_method == "inner_product":
            inner_product = torch.sum(v_new_full * v_old_full, dim=-1)
            loss += self.hparams.ortho_token_lambda * inner_product.pow(2).mean()
        elif self.hparams.ortho_method == "cosine":
            cosine_sim = F.cosine_similarity(v_new_full, v_old_full, dim=-1, eps=1e-8)
            loss += self.hparams.ortho_token_lambda * cosine_sim.pow(2).mean()
        
        # 2. v_old ⊥ prompt
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
        """计算范数约束损失 (包含L2正则化)"""
        # 这里的 device 获取需要兼容不同层
        device = self.v_new.device if not self.hparams.use_low_rank else self.v_new_U.device
        loss = torch.tensor(0.0, device=device)

        if self.hparams.use_low_rank:
            v_new_full = self.v_new_U @ self.v_new_V
            v_old_full = self.v_old_U @ self.v_old_V
        else:
            v_new_full = self.v_new
            v_old_full = self.v_old

        # 约束1: 注入向量整体范数
        injection_norms = torch.abs(self.alpha) * torch.norm(v_new_full, dim=-1)
        norm_violations = F.relu(injection_norms - max_norm)
        loss += norm_violations.pow(2).mean()

        # 约束2: alpha 阈值
        max_alpha = 2.0
        alpha_violations = F.relu(torch.abs(self.alpha) - max_alpha)
        loss += alpha_violations.pow(2).mean()

        # 约束3: v_new 原始范数
        v_new_norms = torch.norm(v_new_full, dim=-1)
        v_new_violations = F.relu(v_new_norms - max_norm)
        loss += v_new_violations.pow(2).mean()

        # 约束4: L2正则化 (关键防止过拟合)
        l2_loss = (v_new_full.pow(2).mean() + v_old_full.pow(2).mean()) / 2
        loss += 0.01 * l2_loss

        return loss

    def forward(self, edit_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """应用编辑向量"""
        v_new, v_old, alpha, beta = self.get_edit_vectors(edit_id)
        
        # 广播并应用: h' = h + alpha*v_new + beta*v_old
        edit_vector = alpha * v_new + beta * v_old
        modified = hidden_states + edit_vector.view(1, 1, -1)
        
        return modified