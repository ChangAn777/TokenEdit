"""
增强的层级注入器

核心改进:
1. 渐进式注入 (Progressive Injection) - 不同层使用不同强度
2. 注意力引导注入 (Attention-Guided Injection) - 考虑token重要性
3. 动态层选择 (Dynamic Layer Selection) - 根据编辑难度自适应选择层
"""

import torch
import torch.nn.functional as F
from typing import List, Callable, Optional, Dict

class EnhancedLayerInjector:
    """
    增强的层级注入器
    
    改进1: 渐进式注入
    - 早期层: 弱注入 (保留原始语义)
    - 中间层: 强注入 (修改核心表示)
    - 后期层: 中等注入 (微调输出)
    """
    
    def __init__(self, target_layers: List[int], use_progressive: bool = True):
        self.target_layers = target_layers
        self.use_progressive = use_progressive
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        
        # 渐进式强度系数
        if use_progressive:
            self.layer_strengths = self._compute_layer_strengths()
        else:
            self.layer_strengths = {layer: 1.0 for layer in target_layers}
    
    def _compute_layer_strengths(self) -> Dict[int, float]:
        """
        计算每层的注入强度
        
        策略: 中间层最强,两端较弱
        例如 [30,31,32,33,34] -> [0.6, 0.8, 1.0, 0.8, 0.6]
        """
        strengths = {}
        n = len(self.target_layers)
        
        for i, layer in enumerate(self.target_layers):
            # 使用高斯分布确定强度
            center = n // 2
            distance = abs(i - center)
            # 中心强度1.0, 边缘衰减到0.6
            strength = 1.0 - 0.4 * (distance / max(center, 1))
            strengths[layer] = strength
        
        return strengths
    
    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int],
        layer_strengths: Optional[Dict[int, float]] = None
    ):
        """
        注入编辑向量
        
        Args:
            layer_strengths: 可选的自定义层强度
        """
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions
        
        # 使用自定义强度或默认强度
        if layer_strengths is not None:
            self.layer_strengths = layer_strengths

        # 注册hooks
        for layer_idx in self.target_layers:
            layer = self._resolve_layer(model, layer_idx)
            hook = layer.register_forward_hook(
                lambda module, input, output, layer_idx=layer_idx: 
                self._progressive_injection_hook(module, input, output, layer_idx)
            )
            self.hooks.append(hook)

    def _resolve_layer(self, model, layer_idx: int):
        """解析层"""
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError("Unsupported model structure")
    
    def _progressive_injection_hook(self, module, input, output, layer_idx: int):
        """
        渐进式注入hook
        
        改进:
        1. 不同层使用不同强度
        2. 可选的注意力加权
        """
        # 解析输出
        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = ()

        if self.active_edit_id is not None and self.edit_module is not None:
            # 获取编辑向量
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )

            # 计算注入向量
            inject_vector = alpha * v_new + beta * v_old
            inject_vector = inject_vector.to(hidden_states.device)
            
            # 应用层特定的强度
            strength = self.layer_strengths.get(layer_idx, 1.0)
            inject_vector = inject_vector * strength

            # 只修改主体位置
            for pos in self.subject_positions:
                if 0 <= pos < hidden_states.size(1):
                    hidden_states[:, pos, :] = (
                        hidden_states[:, pos, :] + inject_vector.unsqueeze(0)
                    )

        # 返回修改后的输出
        if isinstance(output, tuple):
            return (hidden_states,) + other_outputs
        else:
            return hidden_states
    
    def clear(self):
        """清除hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None



# Backward-compatible alias.
LayerInjector = EnhancedLayerInjector

class AttentionGuidedInjector:
    """
    注意力引导注入器
    
    核心思想:
    - 不是对所有主体token平等注入
    - 而是根据注意力权重动态调整注入强度
    - 重要的token注入更多,次要的token注入更少
    """
    
    def __init__(self, target_layers: List[int]):
        self.target_layers = target_layers
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        self.attention_weights = None  # 存储注意力权重
    
    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int],
        attention_weights: Optional[torch.Tensor] = None
    ):
        """
        注入编辑向量
        
        Args:
            attention_weights: 主体token的注意力权重 (num_positions,)
        """
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions
        self.attention_weights = attention_weights
        
        # 如果没有提供注意力权重,使用均匀权重
        if self.attention_weights is None:
            self.attention_weights = torch.ones(len(subject_positions))
        
        # 归一化权重
        self.attention_weights = F.softmax(self.attention_weights, dim=0)

        for layer_idx in self.target_layers:
            layer = self._resolve_layer(model, layer_idx)
            hook = layer.register_forward_hook(self._attention_guided_hook)
            self.hooks.append(hook)

    def _resolve_layer(self, model, layer_idx: int):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError("Unsupported model structure")
    
    def _attention_guided_hook(self, module, input, output):
        """注意力引导注入"""
        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = ()

        if self.active_edit_id is not None and self.edit_module is not None:
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )

            inject_vector = alpha * v_new + beta * v_old
            inject_vector = inject_vector.to(hidden_states.device)

            # 根据注意力权重调整每个位置的注入强度
            for i, pos in enumerate(self.subject_positions):
                if 0 <= pos < hidden_states.size(1):
                    weight = self.attention_weights[i].item()
                    scaled_inject = inject_vector * weight
                    hidden_states[:, pos, :] = (
                        hidden_states[:, pos, :] + scaled_inject.unsqueeze(0)
                    )

        if isinstance(output, tuple):
            return (hidden_states,) + other_outputs
        else:
            return hidden_states
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        self.attention_weights = None


class DynamicLayerSelector:
    """
    动态层选择器
    
    根据编辑难度自动选择最佳注入层:
    - 简单编辑 (如数字修改): 使用后层
    - 复杂编辑 (如概念替换): 使用中层
    - 自动学习每个编辑的最佳层组合
    """
    
    def __init__(self, all_layers: List[int], num_edits: int):
        self.all_layers = all_layers
        self.num_edits = num_edits
        
        # 为每个编辑学习层选择权重
        # shape: (num_edits, num_layers)
        self.layer_selection_weights = torch.nn.Parameter(
            torch.ones(num_edits, len(all_layers))
        )
    
    def get_active_layers(self, edit_id: int, top_k: int = 5) -> List[int]:
        """
        获取该编辑应该激活的层
        
        Args:
            edit_id: 编辑ID
            top_k: 选择权重最高的k层
        
        Returns:
            选中的层索引列表
        """
        weights = F.softmax(self.layer_selection_weights[edit_id], dim=0)
        top_indices = torch.topk(weights, k=top_k).indices
        selected_layers = [self.all_layers[i] for i in top_indices]
        return sorted(selected_layers)
    
    def get_layer_strengths(self, edit_id: int) -> Dict[int, float]:
        """
        获取每层的注入强度
        
        Returns:
            {layer_idx: strength}
        """
        weights = F.softmax(self.layer_selection_weights[edit_id], dim=0)
        return {
            layer: weights[i].item() 
            for i, layer in enumerate(self.all_layers)
        }


class AdaptiveInjector:
    """
    自适应注入器 - 整合所有改进
    
    特性:
    1. 渐进式注入
    2. 注意力引导
    3. 动态层选择
    """
    
    def __init__(
        self, 
        target_layers: List[int],
        num_edits: int,
        use_progressive: bool = True,
        use_attention: bool = True,
        use_dynamic_layers: bool = False
    ):
        self.target_layers = target_layers
        self.use_progressive = use_progressive
        self.use_attention = use_attention
        self.use_dynamic_layers = use_dynamic_layers
        
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        self.attention_weights = None
        
        # 渐进式强度
        if use_progressive:
            self.layer_strengths = self._compute_layer_strengths()
        
        # 动态层选择
        if use_dynamic_layers:
            self.dynamic_selector = DynamicLayerSelector(target_layers, num_edits)
    
    def _compute_layer_strengths(self) -> Dict[int, float]:
        """计算渐进式强度"""
        strengths = {}
        n = len(self.target_layers)
        for i, layer in enumerate(self.target_layers):
            center = n // 2
            distance = abs(i - center)
            strength = 1.0 - 0.3 * (distance / max(center, 1))
            strengths[layer] = strength
        return strengths
    
    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int],
        attention_weights: Optional[torch.Tensor] = None
    ):
        """自适应注入"""
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions
        
        # 处理注意力权重
        if self.use_attention and attention_weights is not None:
            self.attention_weights = F.softmax(attention_weights, dim=0)
        else:
            self.attention_weights = torch.ones(len(subject_positions)) / len(subject_positions)
        
        # 动态选择层
        if self.use_dynamic_layers:
            active_layers = self.dynamic_selector.get_active_layers(edit_id)
            layer_strengths = self.dynamic_selector.get_layer_strengths(edit_id)
        else:
            active_layers = self.target_layers
            layer_strengths = self.layer_strengths if self.use_progressive else None

        # 注册hooks
        for layer_idx in active_layers:
            layer = self._resolve_layer(model, layer_idx)
            strength = layer_strengths.get(layer_idx, 1.0) if layer_strengths else 1.0
            
            hook = layer.register_forward_hook(
                lambda module, input, output, s=strength: 
                self._adaptive_hook(module, input, output, s)
            )
            self.hooks.append(hook)
    
    def _resolve_layer(self, model, layer_idx: int):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError("Unsupported model structure")
    
    def _adaptive_hook(self, module, input, output, layer_strength: float):
        """自适应注入hook"""
        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = ()

        if self.active_edit_id is not None and self.edit_module is not None:
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )

            inject_vector = alpha * v_new + beta * v_old
            inject_vector = inject_vector.to(hidden_states.device)
            
            # 应用层强度
            inject_vector = inject_vector * layer_strength

            # 根据注意力权重注入
            for i, pos in enumerate(self.subject_positions):
                if 0 <= pos < hidden_states.size(1):
                    weight = self.attention_weights[i].item()
                    scaled_inject = inject_vector * weight
                    hidden_states[:, pos, :] = (
                        hidden_states[:, pos, :] + scaled_inject.unsqueeze(0)
                    )

        if isinstance(output, tuple):
            return (hidden_states,) + other_outputs
        else:
            return hidden_states
    
    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        self.attention_weights = None