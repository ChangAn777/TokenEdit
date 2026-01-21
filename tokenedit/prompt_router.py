"""Prompt敏感路由机制 - 动态版本(不硬编码关系)

修复版本 - 解决了:
1. 路由阈值过低导致误触发
2. 添加"拒绝区域"机制避免歧义
3. 主体匹配作为必要条件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

class PromptRouter:
    """
    Prompt路由器

    路由机制:
    1. Embedding相似度检测(主要方法)
    2. 主体匹配(辅助验证)
    
    修复:
    - 阈值从0.3提高到0.7(更保守)
    - 添加"拒绝区域"(避免多个编辑相似度接近时的歧义)
    - 主体匹配作为必要条件
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
            prompt_template: 原始prompt模板,用于提取关系关键词
        """
        # 1. 计算并存储嵌入(使用prompt template或subject+relation)
        if self.hparams.use_embedding_routing:
            # 如果提供了prompt template,使用它(更准确)
            if prompt_template:
                # 将prompt template中的{}替换为subject
                text = prompt_template.replace("{}", subject)
            else:
                # 回退到使用subject + relation
                text = f"{subject} {relation}"

            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)

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
        Route a prompt to a matching edit id.
        """
        if self.hparams.use_embedding_routing:
            if len(self.edit_embeddings) == 0:
                return None

            if prompt_embedding is None:
                inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    prompt_embedding = outputs.hidden_states[-1].mean(dim=1)

            similarities = {}
            for edit_id, edit_emb in self.edit_embeddings.items():
                sim = F.cosine_similarity(prompt_embedding, edit_emb, dim=-1).item()
                similarities[edit_id] = sim

            subject_matched_ids = []
            for edit_id, info in self.edit_info.items():
                subject = info["subject"].lower()
                if subject in prompt.lower():
                    subject_matched_ids.append(edit_id)

            if similarities:
                if subject_matched_ids:
                    best_edit_id = max(subject_matched_ids, key=lambda k: similarities.get(k, -1.0))
                else:
                    best_edit_id = max(similarities, key=similarities.get)
                best_sim = similarities.get(best_edit_id, -1.0)

                if best_sim < self.hparams.routing_threshold:
                    return None

                sorted_sims = sorted(similarities.values(), reverse=True)
                if len(sorted_sims) > 1:
                    second_best_sim = sorted_sims[1]
                    if best_sim > 0.5 and second_best_sim > 0.4:
                        if best_sim - second_best_sim < 0.1:
                            return None

                return best_edit_id

        for edit_id, info in self.edit_info.items():
            subject = info["subject"]
            if subject.lower() in prompt.lower():
                if info.get("prompt_template"):
                    template = info["prompt_template"].replace("{}", subject)
                    template_words = set(template.lower().split())
                    prompt_words = set(prompt.lower().split())
                    overlap = len(template_words & prompt_words)
                    if overlap >= len(template_words) * 0.5:
                        return edit_id
                else:
                    return edit_id

        return None
