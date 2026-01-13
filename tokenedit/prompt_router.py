"""Prompt敏感路由机制 - 动态版本（不硬编码关系）"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

class PromptRouter:
    """
    Prompt路由器

    路由机制:
    1. Embedding相似度检测（主要方法）
    2. 主体匹配（辅助验证）
    """

    def __init__(self, model, tokenizer, hparams):
        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams
        self.device = hparams.device

        # 存储每个编辑的嵌入
        self.edit_embeddings: Dict[int, torch.Tensor] = {}

        # 存储主体和关系信息
        self.edit_info: Dict[int, Dict[str, str]] = {}

    def register_edit(self, edit_id: int, subject: str, relation: str, prompt_template: str = None):
        """
        注册编辑

        Args:
            edit_id: 编辑ID
            subject: 主体 (e.g., "France")
            relation: 关系 (e.g., "capital" 或 "P103")
            prompt_template: 原始prompt模板，用于提取关系关键词
        """
        # 1. 计算并存储嵌入（使用prompt template或subject+relation）
        if self.hparams.use_embedding_routing:
            # 如果提供了prompt template，使用它（更准确）
            if prompt_template:
                # 将prompt template中的{}替换为subject
                text = prompt_template.replace("{}", subject)
            else:
                # 回退到使用subject + relation
                text = f"{subject} {relation}"

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # 使用最后一层的平均池化
                embedding = outputs.hidden_states[-1].mean(dim=1)  # (1, hidden_size)

            self.edit_embeddings[edit_id] = embedding

        # 2. 存储信息
        self.edit_info[edit_id] = {
            "subject": subject,
            "relation": relation,
            "prompt_template": prompt_template
        }

    def route(self, prompt: str, prompt_embedding: Optional[torch.Tensor] = None) -> Optional[int]:
        """
        路由决策：判断prompt是否触发某个编辑

        Args:
            prompt: 输入prompt文本
            prompt_embedding: prompt的嵌入 (可选，提供则跳过重复计算)

        Returns:
            edit_id: 匹配的编辑ID，或None（不触发任何编辑）
        """
        # 方法1: Embedding相似度路由（主要方法）
        if self.hparams.use_embedding_routing:
            if len(self.edit_embeddings) == 0:
                return None

            if prompt_embedding is None:
                # 计算prompt的嵌入
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    prompt_embedding = outputs.hidden_states[-1].mean(dim=1)

            # 计算与所有编辑的相似度
            similarities = {}
            for edit_id, edit_emb in self.edit_embeddings.items():
                sim = torch.nn.functional.cosine_similarity(
                    prompt_embedding, edit_emb, dim=-1
                ).item()
                similarities[edit_id] = sim

            # 检查是否超过阈值
            if similarities:
                best_edit_id = max(similarities, key=similarities.get)
                if similarities[best_edit_id] > self.hparams.routing_threshold:
                    return best_edit_id

        # 方法2: 主体匹配（辅助方法，不依赖硬编码关系模板）
        # 检查prompt中是否包含某个编辑的主体
        for edit_id, info in self.edit_info.items():
            subject = info["subject"]
            # 检查主体是否在prompt中
            if subject.lower() in prompt.lower():
                # 进一步验证：检查这个编辑是否有对应的prompt template
                # 并且prompt是否与template相似
                if info.get("prompt_template"):
                    template = info["prompt_template"].replace("{}", subject)
                    # 简单的相似度检查：如果prompt包含template的部分关键词
                    template_words = set(template.lower().split())
                    prompt_words = set(prompt.lower().split())
                    overlap = len(template_words & prompt_words)
                    if overlap > 0:
                        return edit_id
                else:
                    # 如果没有template，仅基于主体匹配
                    return edit_id

        # 未匹配到任何编辑
        return None
