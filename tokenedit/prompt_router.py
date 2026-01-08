"""Prompt敏感路由机制"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List

class PromptRouter:
    """
    Prompt路由器
    
    双重验证机制:
    1. Embedding相似度检测
    2. 关系模板匹配
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
        
        # 关系模板库
        self.relation_templates = self._load_relation_templates()
    
    def _load_relation_templates(self) -> Dict[str, List[str]]:
        """加载关系模板"""
        # 可以从resource/relations.json加载
        return {
            "capital": ["capital", "首都", "capital city", "is located in"],
            "president": ["president", "总统", "leader of", "head of state"],
            "founder": ["founder", "创始人", "founded by", "established by"],
            "ceo": ["CEO", "chief executive", "首席执行官"],
            # 可扩展更多关系
        }
    
    def register_edit(self, edit_id: int, subject: str, relation: str):
        """
        注册编辑
        
        Args:
            edit_id: 编辑ID
            subject: 主体 (e.g., "France")
            relation: 关系 (e.g., "capital")
        """
        # 1. 计算并存储嵌入
        text = f"{subject} {relation}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 使用最后一层的平均池化
            embedding = outputs.hidden_states[-1].mean(dim=1)  # (1, hidden_size)
        
        self.edit_embeddings[edit_id] = embedding
        
        # 2. 存储信息用于模板匹配
        self.edit_info[edit_id] = {
            "subject": subject,
            "relation": relation
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
        if len(self.edit_embeddings) == 0:
            return None

        # 优先使用方法2: 关系模板匹配（更准确）
        if self.hparams.use_template_routing:
            for edit_id, info in self.edit_info.items():
                subject = info["subject"]
                relation = info["relation"]

                # 检查主体是否在prompt中
                if subject.lower() in prompt.lower():
                    # 检查关系模板是否匹配
                    templates = self.relation_templates.get(relation, [])
                    for template in templates:
                        if template.lower() in prompt.lower():
                            return edit_id

        # 方法1: Embedding相似度路由（作为备选）
        if self.hparams.use_embedding_routing:
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

        # 未匹配到任何编辑
        return None
