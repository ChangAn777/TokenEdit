"""
tokenedit/tokenedit_utils.py

TokenEdit 工具函数 (v2 - AlphaEdit 对标版)

新增:
- compute_nullspace_projection_matrix(): AlphaEdit 核心 — 基于 SVD 的零空间投影矩阵
- compute_target_hidden_state():         ROME 风格的目标隐藏状态提取，用于 Smart Init

原有修复保留:
- tokenization 统一策略
- 鲁棒的模糊匹配
- KL 散度数值稳定性
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class TokenEditUtils:
    """TokenEdit 工具函数集合"""

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # [NEW] Null-Space Projection  (AlphaEdit 核心)
    # ------------------------------------------------------------------

    def compute_nullspace_projection_matrix(
        self,
        context_texts: List[str],
        layer: int,
        rank: int = 100,
        reg: float = 0.1,
    ) -> torch.Tensor:
        """
        计算指定层的零空间投影矩阵 P_null。

        AlphaEdit (Fang et al., 2024) 的核心思想：
          将编辑向量投影到上下文协方差矩阵的零空间中，
          使其与所有无关输入的表示正交，从而在数学上保证
          编辑不改变非目标知识的输出分布。

        推导：
          C ∈ R^{d × n}  —— n 个上下文隐藏向量拼成的矩阵
          C = U S V^T     —— 紧 SVD，U ∈ R^{d × k}
          P_null = I - U · diag(s²/(s²+λ)) · U^T
          v_inject_safe = P_null @ v_inject

        Args:
            context_texts: 无关文本列表（推荐 50-200 条）
            layer:         目标 Transformer 层索引
            rank:          SVD 截断秩（建议 hidden_size 的 5-15%）
            reg:           Tikhonov 正则化系数 λ

        Returns:
            P_null: (hidden_size, hidden_size)  float32，CPU tensor
        """
        hidden_size = self.model.config.hidden_size
        context_vecs: List[torch.Tensor] = []

        self.model.eval()
        for text in context_texts:
            if not text or not text.strip():
                continue
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=64,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # 取序列均值，代表该文本在此层的语义质心
                h = outputs.hidden_states[layer][0].mean(dim=0)  # (hidden_size,)
                context_vecs.append(h.float().cpu())

        if len(context_vecs) == 0:
            # 退化：返回单位矩阵（不做任何投影）
            return torch.eye(hidden_size, dtype=torch.float32)

        # C: (hidden_size, n_contexts)
        C = torch.stack(context_vecs, dim=1)  # (d, n)

        # 列归一化，防止高频词主导
        col_norms = C.norm(dim=0, keepdim=True).clamp(min=1e-8)
        C = C / col_norms

        # 紧 SVD：C ≈ U S V^T，U ∈ R^{d × k}
        # torch.linalg.svd 返回 full_matrices=True 的完整 U；
        # 使用 torch.linalg.svd 并截断更安全
        try:
            U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        except Exception:
            # 极端情况下 SVD 不收敛，返回单位矩阵
            return torch.eye(hidden_size, dtype=torch.float32)

        k = min(rank, S.shape[0])
        U_k = U[:, :k]      # (d, k)
        S_k = S[:k]         # (k,)

        # 正则化权重：w_i = s_i² / (s_i² + λ)
        # 大奇异值方向权重接近 1（被"抹去"），小方向接近 0（保留）
        weights = (S_k ** 2) / (S_k ** 2 + reg)  # (k,)

        # P_null = I - U_k diag(w) U_k^T
        I = torch.eye(hidden_size, dtype=torch.float32)
        P_null = I - U_k @ torch.diag(weights) @ U_k.T  # (d, d)

        return P_null  # CPU，float32

    # ------------------------------------------------------------------
    # [NEW] ROME-style 目标隐藏状态提取，用于 Smart Init
    # ------------------------------------------------------------------

    def compute_target_hidden_state(
        self,
        prompt: str,
        target: str,
        subject: str,
        target_layers: List[int],
    ) -> Optional[torch.Tensor]:
        """
        提取模型在"prompt + target"语境下，主体 token 处的隐藏状态均值。

        相比直接使用词嵌入，此方式包含了：
          - 位置编码信息
          - 上下文注意力信息
          - 多层语义变换后的表示

        这与 ROME (Meng et al., 2022) 的 z* 计算思路一致，
        能为 v_new 提供更优的初始化起点。

        Args:
            prompt:        提示文本（已填入 subject，如 "The capital of France is"）
            target:        目标答案（如 "Lyon"）
            subject:       主体（如 "France"），用于定位注入位置
            target_layers: 目标层列表

        Returns:
            target_vec: (hidden_size,) tensor；若定位失败则返回 None
        """
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        subject_positions = self.find_subject_positions(
            prompt, subject, add_special_tokens=True
        )
        if not subject_positions:
            return None

        layer_vecs: List[torch.Tensor] = []
        for layer_idx in target_layers:
            h = outputs.hidden_states[layer_idx][0, subject_positions, :].mean(dim=0)
            layer_vecs.append(h)

        # 跨层均值：综合多层语义信息
        return torch.stack(layer_vecs).mean(dim=0)  # (hidden_size,)

    # ------------------------------------------------------------------
    # 原有方法（保持兼容，修复注释）
    # ------------------------------------------------------------------

    def find_subject_positions(
        self,
        prompt: str,
        subject: str,
        verbose: bool = False,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        在 prompt 中找到主体的 token 位置列表。

        修复:
        - 统一 tokenization 策略（add_special_tokens 参数）
        - 鲁棒的模糊匹配（空格前缀、大小写）
        - 处理 multi-token subject
        """
        full_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        subject_ids = self.tokenizer.encode(subject, add_special_tokens=False)

        positions: List[int] = []

        # 精确匹配
        for i in range(len(full_ids) - len(subject_ids) + 1):
            if full_ids[i : i + len(subject_ids)] == subject_ids:
                positions = list(range(i, i + len(subject_ids)))
                break

        # 带空格前缀的模糊匹配
        if not positions:
            for add_space in [True, False]:
                test_subject = (" " + subject) if add_space else subject
                test_ids = self.tokenizer.encode(test_subject, add_special_tokens=False)
                for i in range(len(full_ids) - len(test_ids) + 1):
                    if full_ids[i : i + len(test_ids)] == test_ids:
                        positions = list(range(i, i + len(test_ids)))
                        break
                if positions:
                    break

        # 基于文本位置的最后匹配
        if not positions:
            prompt_lower = prompt.lower()
            subject_lower = subject.lower()
            text_start = prompt_lower.find(subject_lower)
            if text_start != -1:
                before_text = prompt[: text_start + len(subject)]
                before_ids = self.tokenizer.encode(
                    before_text, add_special_tokens=add_special_tokens
                )
                subject_start = len(before_ids) - len(subject_ids)
                if subject_start >= 0:
                    positions = list(range(subject_start, subject_start + len(subject_ids)))

        if verbose:
            if positions:
                tokens = [self.tokenizer.decode([full_ids[p]]) for p in positions]
                print(f"[SUCCESS] 找到主体位置: {positions} | Tokens: {tokens}")
            else:
                print(f"[WARNING] 未找到主体 '{subject}' 在 '{prompt}' 中")

        return positions

    def get_hidden_states_at_positions(
        self,
        prompt: str,
        positions: List[int],
        layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """获取指定位置和层的隐藏状态"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states: Dict[int, torch.Tensor] = {}
        for layer in layers:
            layer_hidden = outputs.hidden_states[layer]
            if positions:
                selected = layer_hidden[0, positions, :]
                hidden_states[layer] = selected.mean(dim=0)
            else:
                hidden_states[layer] = layer_hidden[0].mean(dim=0)

        return hidden_states

    def compute_kl_divergence(
        self,
        prompt: str,
        subject_positions: List[int],
        edit_module,
        edit_id: int,
        injector,
    ) -> torch.Tensor:
        """
        计算编辑前后的 KL 散度 KL(P_edit || P_orig)。

        修复:
        - 使用 log_softmax 提高数值稳定性
        - 正确的 KL 散度方向和 API 调用
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            orig_outputs = self.model(**inputs)
            orig_log_probs = F.log_softmax(orig_outputs.logits[0, -1, :], dim=-1)

        injector.inject(self.model, edit_id, edit_module, subject_positions)
        with torch.no_grad():
            edit_outputs = self.model(**inputs)
            edit_log_probs = F.log_softmax(edit_outputs.logits[0, -1, :], dim=-1)
        injector.clear()

        # F.kl_div(log_input, log_target, log_target=True)
        # = sum(exp(target) * (target - input)) = KL(P_edit || P_orig)
        kl_loss = F.kl_div(
            orig_log_probs,
            edit_log_probs,
            log_target=True,
            reduction="batchmean",
        )
        return kl_loss

    def compute_token_probability(
        self,
        prompt: str,
        target_token: str,
        subject_positions: List[int],
        edit_module,
        edit_id: int,
        injector,
    ) -> float:
        """计算特定 token 在编辑后的概率"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)

        if not target_ids:
            return 0.0

        target_id = target_ids[0]

        injector.inject(self.model, edit_id, edit_module, subject_positions)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits[0, -1, :], dim=-1)
            token_prob = probs[target_id].item()
        injector.clear()

        return token_prob

    def extract_context_embeddings(
        self,
        contexts: List[str],
        layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """提取多个上下文的平均嵌入"""
        all_embeddings: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

        for context in contexts:
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                add_special_tokens=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            for layer in layers:
                emb = outputs.hidden_states[layer][0].mean(dim=0)
                all_embeddings[layer].append(emb)

        return {
            layer: torch.stack(all_embeddings[layer]).mean(dim=0)
            for layer in layers
        }

    def batch_compute_edit_loss(
        self,
        prompts: List[str],
        targets: List[str],
        subject_positions_list: List[List[int]],
        edit_ids: List[int],
        edit_module,
        injector,
    ) -> torch.Tensor:
        """批量计算编辑损失"""
        total_loss = 0.0

        for prompt, target, positions, edit_id in zip(
            prompts, targets, subject_positions_list, edit_ids
        ):
            full_text = f"{prompt} {target}"
            inputs = self.tokenizer(
                full_text, return_tensors="pt", add_special_tokens=True
            ).to(self.device)

            injector.inject(self.model, edit_id, edit_module, positions)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            injector.clear()

        return torch.tensor(total_loss / len(prompts))


class TokenPositionFinder:
    """Token 位置查找器（辅助工具）"""

    @staticmethod
    def find_last_token(tokenizer, prompt: str, add_special_tokens: bool = True) -> int:
        ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return len(ids) - 1

    @staticmethod
    def find_first_subject_token(
        tokenizer,
        prompt: str,
        subject: str,
        add_special_tokens: bool = True,
    ) -> int:
        full_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        subject_ids = tokenizer.encode(subject, add_special_tokens=False)
        for i in range(len(full_ids) - len(subject_ids) + 1):
            if full_ids[i : i + len(subject_ids)] == subject_ids:
                return i
        return -1

    @staticmethod
    def find_last_subject_token(
        tokenizer,
        prompt: str,
        subject: str,
        add_special_tokens: bool = True,
    ) -> int:
        first_pos = TokenPositionFinder.find_first_subject_token(
            tokenizer, prompt, subject, add_special_tokens
        )
        if first_pos == -1:
            return -1
        subject_ids = tokenizer.encode(subject, add_special_tokens=False)
        return first_pos + len(subject_ids) - 1