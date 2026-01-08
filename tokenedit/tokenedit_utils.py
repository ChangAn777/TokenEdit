"""
tokenedit/tokenedit_utils.py

TokenEdit的工具函数
与MEMIT的compute_z不同，这些函数用于辅助Token训练
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


class TokenEditUtils:
    """TokenEdit工具函数集合"""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def find_subject_positions(
        self, 
        prompt: str, 
        subject: str,
        verbose: bool = False
    ) -> List[int]:
        """
        在prompt中找到主体的token位置
        
        类似于MEMIT的find_fact_lookup_idx，但更简化
        
        Args:
            prompt: 输入文本 "The capital of France is"
            subject: 主体 "France"
            verbose: 是否打印调试信息
        
        Returns:
            positions: 主体token的位置列表 [3, 4] (如果"France"被分成2个token)
        """
        # 编码完整句子（不包含special tokens以保持一致）
        full_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # 编码主体
        subject_ids = self.tokenizer.encode(subject, add_special_tokens=False)
        
        # 查找子序列
        positions = []
        for i in range(len(full_ids) - len(subject_ids) + 1):
            if full_ids[i:i+len(subject_ids)] == subject_ids:
                positions = list(range(i, i + len(subject_ids)))
                break

        # 如果没找到，尝试模糊匹配（处理大小写、空格等问题）
        if not positions:
            # 尝试将prompt和subject转为小写
            prompt_lower = prompt.lower()
            subject_lower = subject.lower()

            # 查找subject在文本中的位置
            text_start = prompt_lower.find(subject_lower)
            if text_start != -1:
                # 编码prompt中subject之前的部分
                before_text = prompt[:text_start + len(subject)]
                before_ids = self.tokenizer.encode(before_text, add_special_tokens=False)
                # 主体应该从before_ids的末尾开始
                subject_start = len(before_ids) - len(subject_ids)
                if subject_start >= 0:
                    positions = list(range(subject_start, subject_start + len(subject_ids)))

        if verbose:
            if positions:
                tokens = [self.tokenizer.decode([full_ids[p]]) for p in positions]
                print(f"找到主体位置: {positions} | Token: {tokens}")
                print(f"  完整tokens: {[self.tokenizer.decode([tid]) for tid in full_ids]}")
            else:
                print(f"警告: 未找到主体 '{subject}' 在 '{prompt}' 中")
                print(f"  Prompt tokens: {[self.tokenizer.decode([tid]) for tid in full_ids]}")
                print(f"  Subject tokens: {[self.tokenizer.decode([tid]) for tid in subject_ids]}")

        return positions
    
    def get_hidden_states_at_positions(
        self,
        prompt: str,
        positions: List[int],
        layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """
        获取指定位置和层的隐藏状态
        
        用于：
        - 计算prompt的平均表示（用于正交约束）
        - 提取主体的表示
        
        Args:
            prompt: 输入文本
            positions: token位置
            layers: 需要提取的层
        
        Returns:
            {layer_idx: hidden_states} 每层对应位置的隐藏状态
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = {}
        for layer in layers:
            # outputs.hidden_states[layer]: (batch, seq_len, hidden_size)
            layer_hidden = outputs.hidden_states[layer]
            
            # 提取指定位置
            if positions:
                selected = layer_hidden[0, positions, :]  # (num_positions, hidden_size)
                hidden_states[layer] = selected.mean(dim=0)  # 平均池化
            else:
                # 如果没有指定位置，返回全部
                hidden_states[layer] = layer_hidden[0].mean(dim=0)
        
        return hidden_states
    
    def compute_target_logits(
        self,
        prompt: str,
        target: str,
        subject_positions: List[int],
        target_layers: List[int],
        edit_module,
        edit_id: int,
        injector
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算注入编辑后的目标logits
        
        用于训练时计算L_edit损失
        
        Args:
            prompt: 输入提示
            target: 目标答案
            subject_positions: 主体位置
            target_layers: 注入层
            edit_module: EditTokenModule
            edit_id: 编辑ID
            injector: LayerInjector
        
        Returns:
            (logits, target_ids): 模型输出的logits和目标token ids
        """
        # 构造完整输入
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # 目标token ids
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        target_ids = torch.tensor(target_ids).to(self.device)
        
        # 注入编辑
        injector.inject(
            self.model,
            edit_id,
            edit_module,
            subject_positions
        )
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # 清除注入
        injector.clear()
        
        return logits, target_ids
    
    def compute_kl_divergence(
        self,
        prompt: str,
        subject_positions: List[int],
        edit_module,
        edit_id: int,
        injector
    ) -> torch.Tensor:
        """
        计算编辑前后的KL散度
        
        用于L_local损失，确保无关问题不受影响
        
        Args:
            prompt: 输入提示
            subject_positions: 主体位置
            edit_module: EditTokenModule
            edit_id: 编辑ID
            injector: LayerInjector
        
        Returns:
            kl_loss: KL散度
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 原始输出
        with torch.no_grad():
            orig_logits = self.model(**inputs).logits[0, -1, :]
            orig_probs = F.softmax(orig_logits, dim=-1)
        
        # 编辑后输出
        injector.inject(
            self.model,
            edit_id,
            edit_module,
            subject_positions
        )
        
        with torch.no_grad():
            edit_logits = self.model(**inputs).logits[0, -1, :]
            edit_probs = F.softmax(edit_logits, dim=-1)
        
        injector.clear()
        
        # KL散度
        kl_loss = F.kl_div(
            edit_probs.log(),
            orig_probs,
            reduction='batchmean'
        )
        
        return kl_loss
    
    def compute_token_probability(
        self,
        prompt: str,
        target_token: str,
        subject_positions: List[int],
        edit_module,
        edit_id: int,
        injector
    ) -> float:
        """
        计算特定token的概率
        
        用于：
        - 评估编辑成功率（新答案的概率）
        - Unlikelihood Loss（旧答案的概率）
        
        Args:
            prompt: 输入提示
            target_token: 目标token（如"Lyon"）
            subject_positions: 主体位置
            edit_module: EditTokenModule
            edit_id: 编辑ID
            injector: LayerInjector
        
        Returns:
            probability: 该token的概率
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        
        # 注入编辑
        injector.inject(
            self.model,
            edit_id,
            edit_module,
            subject_positions
        )
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            token_prob = probs[target_id].item()
        
        injector.clear()
        
        return token_prob
    
    def extract_context_embeddings(
        self,
        contexts: List[str],
        layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """
        提取多个上下文的平均嵌入
        
        用于计算协方差矩阵或背景分布（类似MEMIT的统计量）
        但TokenEdit不需要完整的协方差矩阵，只需要均值
        
        Args:
            contexts: 上下文文本列表
            layers: 目标层
        
        Returns:
            {layer_idx: mean_embedding} 每层的平均嵌入
        """
        all_embeddings = {layer: [] for layer in layers}
        
        for context in contexts:
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                max_length=128,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
            
            for layer in layers:
                # 平均池化
                emb = outputs.hidden_states[layer][0].mean(dim=0)
                all_embeddings[layer].append(emb)
        
        # 计算均值
        mean_embeddings = {}
        for layer in layers:
            mean_embeddings[layer] = torch.stack(
                all_embeddings[layer]
            ).mean(dim=0)
        
        return mean_embeddings
    
    def batch_compute_edit_loss(
        self,
        prompts: List[str],
        targets: List[str],
        subject_positions_list: List[List[int]],
        edit_ids: List[int],
        edit_module,
        injector
    ) -> torch.Tensor:
        """
        批量计算编辑损失
        
        用于加速训练
        
        Args:
            prompts: 提示列表
            targets: 目标列表
            subject_positions_list: 每个样本的主体位置
            edit_ids: 编辑ID列表
            edit_module: EditTokenModule
            injector: LayerInjector
        
        Returns:
            total_loss: 批次平均损失
        """
        total_loss = 0.0
        
        for prompt, target, positions, edit_id in zip(
            prompts, targets, subject_positions_list, edit_ids
        ):
            # 构造输入
            full_text = f"{prompt} {target}"
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
            
            # 注入
            injector.inject(self.model, edit_id, edit_module, positions)
            
            # 计算损失
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item()
            
            # 清除
            injector.clear()
        
        return torch.tensor(total_loss / len(prompts))


class TokenPositionFinder:
    """
    Token位置查找器
    
    增强版的find_subject_positions，支持更复杂的匹配策略
    """
    
    @staticmethod
    def find_last_token(tokenizer, prompt: str) -> int:
        """返回最后一个token的位置"""
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        return len(ids) - 1
    
    @staticmethod
    def find_first_subject_token(
        tokenizer, 
        prompt: str, 
        subject: str
    ) -> int:
        """返回主体的第一个token位置"""
        full_ids = tokenizer.encode(prompt, add_special_tokens=True)
        subject_ids = tokenizer.encode(subject, add_special_tokens=False)
        
        for i in range(len(full_ids) - len(subject_ids) + 1):
            if full_ids[i:i+len(subject_ids)] == subject_ids:
                return i
        
        return -1  # 未找到
    
    @staticmethod
    def find_last_subject_token(
        tokenizer,
        prompt: str,
        subject: str
    ) -> int:
        """返回主体的最后一个token位置"""
        first_pos = TokenPositionFinder.find_first_subject_token(
            tokenizer, prompt, subject
        )
        
        if first_pos == -1:
            return -1
        
        subject_ids = tokenizer.encode(subject, add_special_tokens=False)
        return first_pos + len(subject_ids) - 1

