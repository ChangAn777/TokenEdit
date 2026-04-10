"""
层级注入器 (v2 - AutoGrad 修复版)

关键 Bug 修复:
  原代码: hidden_states[:, pos, :] = hidden_states[:, pos, :] + inject_vector
  问题:   在训练阶段，对从计算图中取出的 tensor 做原地赋值，
          会触发 "RuntimeError: one of the variables needed for gradient
          computation has been modified by an inplace operation"，
          导致梯度无法正确回传。

  修复:   构建 delta 掩码张量，用 hidden_states + delta（非原地加法）替代，
          完整保留计算图。

其他改进:
  - 渐进式注入强度保持不变（仅改注入方式）
  - 所有 injector 类统一采用 delta-mask 方案
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


class EnhancedLayerInjector:
    """
    增强的层级注入器（渐进式强度 + autograd 安全）

    渐进式策略:
        中间层注入强度最大，两端较弱，形如高斯分布:
        [0.6, 0.8, 1.0, 0.8, 0.6]
    """

    def __init__(self, target_layers: List[int], use_progressive: bool = True):
        self.target_layers = target_layers
        self.use_progressive = use_progressive
        self.hooks: List = []
        self.active_edit_id: Optional[int] = None
        self.edit_module = None
        self.subject_positions: Optional[List[int]] = None

        self.layer_strengths = (
            self._compute_layer_strengths()
            if use_progressive
            else {layer: 1.0 for layer in target_layers}
        )

    def _compute_layer_strengths(self) -> Dict[int, float]:
        """高斯分布强度：中心层最强，边缘层衰减至 0.6"""
        n = len(self.target_layers)
        center = n // 2
        return {
            layer: 1.0 - 0.4 * (abs(i - center) / max(center, 1))
            for i, layer in enumerate(self.target_layers)
        }

    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int],
        layer_strengths: Optional[Dict[int, float]] = None,
    ):
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions

        if layer_strengths is not None:
            self.layer_strengths = layer_strengths

        for layer_idx in self.target_layers:
            layer = self._resolve_layer(model, layer_idx)
            hook = layer.register_forward_hook(
                # 用 default arg 捕获 layer_idx，避免闭包陷阱
                lambda module, inp, out, li=layer_idx:
                    self._progressive_injection_hook(module, inp, out, li)
            )
            self.hooks.append(hook)

    def _resolve_layer(self, model, layer_idx: int):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError(
            f"无法解析层 {layer_idx}，不支持此模型结构"
        )

    def _progressive_injection_hook(self, module, inp, output, layer_idx: int):
        """
        渐进式注入 hook。

        [BUG FIX] 原代码使用原地赋值修改 hidden_states，
        会破坏训练时的梯度图。新方案：
          1. 构造全零 delta 张量（与 hidden_states 同设备/dtype）
          2. 将注入向量写入 delta 的对应位置（此时 hidden_states 未动）
          3. hidden_states = hidden_states + delta  （非原地，保留计算图）
        """
        if isinstance(output, tuple):
            hidden_states, *rest = output
        else:
            hidden_states, rest = output, []

        if self.active_edit_id is not None and self.edit_module is not None:
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )
            inject_vector = (alpha * v_new + beta * v_old).to(hidden_states.device)
            inject_vector = inject_vector * self.layer_strengths.get(layer_idx, 1.0)

            # ---- [BUG FIX] delta-mask 方案 ----
            batch_size = hidden_states.size(0)
            delta = torch.zeros_like(hidden_states)  # (B, seq_len, hidden_size)
            for pos in self.subject_positions:
                if 0 <= pos < hidden_states.size(1):
                    # inject_vector: (hidden_size,) → (B, hidden_size)
                    delta[:, pos, :] = inject_vector.unsqueeze(0).expand(batch_size, -1)

            hidden_states = hidden_states + delta  # 非原地，保留计算图

        if rest:
            return (hidden_states, *rest)
        return hidden_states

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None


# Backward-compatible alias
LayerInjector = EnhancedLayerInjector


# ===========================================================================
# AttentionGuidedInjector
# ===========================================================================

class AttentionGuidedInjector:
    """
    注意力引导注入器：根据注意力权重动态调整各 token 的注入强度。

    [BUG FIX] 同样使用 delta-mask 方案替换原地赋值。
    """

    def __init__(self, target_layers: List[int]):
        self.target_layers = target_layers
        self.hooks: List = []
        self.active_edit_id: Optional[int] = None
        self.edit_module = None
        self.subject_positions: Optional[List[int]] = None
        self.attention_weights: Optional[torch.Tensor] = None

    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int],
        attention_weights: Optional[torch.Tensor] = None,
    ):
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions
        self.attention_weights = (
            F.softmax(attention_weights, dim=0)
            if attention_weights is not None
            else torch.ones(len(subject_positions)) / max(len(subject_positions), 1)
        )

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

    def _attention_guided_hook(self, module, inp, output):
        if isinstance(output, tuple):
            hidden_states, *rest = output
        else:
            hidden_states, rest = output, []

        if self.active_edit_id is not None and self.edit_module is not None:
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )
            inject_vector = (alpha * v_new + beta * v_old).to(hidden_states.device)
            batch_size = hidden_states.size(0)

            # delta-mask 方案
            delta = torch.zeros_like(hidden_states)
            for i, pos in enumerate(self.subject_positions):
                if 0 <= pos < hidden_states.size(1):
                    weight = self.attention_weights[i].item()
                    delta[:, pos, :] = (inject_vector * weight).unsqueeze(0).expand(batch_size, -1)

            hidden_states = hidden_states + delta

        if rest:
            return (hidden_states, *rest)
        return hidden_states

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        self.attention_weights = None


# ===========================================================================
# DynamicLayerSelector
# ===========================================================================

class DynamicLayerSelector:
    """
    动态层选择器：为每个编辑学习最优的注入层组合。
    """

    def __init__(self, all_layers: List[int], num_edits: int):
        self.all_layers = all_layers
        self.num_edits = num_edits
        self.layer_selection_weights = torch.nn.Parameter(
            torch.ones(num_edits, len(all_layers))
        )

    def get_active_layers(self, edit_id: int, top_k: int = 5) -> List[int]:
        weights = F.softmax(self.layer_selection_weights[edit_id], dim=0)
        top_indices = torch.topk(weights, k=top_k).indices
        return sorted(self.all_layers[i] for i in top_indices)

    def get_layer_strengths(self, edit_id: int) -> Dict[int, float]:
        weights = F.softmax(self.layer_selection_weights[edit_id], dim=0)
        return {layer: weights[i].item() for i, layer in enumerate(self.all_layers)}


# ===========================================================================
# AdaptiveInjector
# ===========================================================================

class AdaptiveInjector:
    """
    自适应注入器：整合渐进式 + 注意力引导 + 动态层选择。

    [BUG FIX] 同样使用 delta-mask 方案。
    """

    def __init__(
        self,
        target_layers: List[int],
        num_edits: int,
        use_progressive: bool = True,
        use_attention: bool = True,
        use_dynamic_layers: bool = False,
    ):
        self.target_layers = target_layers
        self.use_progressive = use_progressive
        self.use_attention = use_attention
        self.use_dynamic_layers = use_dynamic_layers

        self.hooks: List = []
        self.active_edit_id: Optional[int] = None
        self.edit_module = None
        self.subject_positions: Optional[List[int]] = None
        self.attention_weights: Optional[torch.Tensor] = None

        self.layer_strengths: Dict[int, float] = (
            self._compute_layer_strengths()
            if use_progressive
            else {layer: 1.0 for layer in target_layers}
        )

        if use_dynamic_layers:
            self.dynamic_selector = DynamicLayerSelector(target_layers, num_edits)

    def _compute_layer_strengths(self) -> Dict[int, float]:
        n = len(self.target_layers)
        center = n // 2
        return {
            layer: 1.0 - 0.3 * (abs(i - center) / max(center, 1))
            for i, layer in enumerate(self.target_layers)
        }

    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int],
        attention_weights: Optional[torch.Tensor] = None,
    ):
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions

        if self.use_attention and attention_weights is not None:
            self.attention_weights = F.softmax(attention_weights, dim=0)
        else:
            n = max(len(subject_positions), 1)
            self.attention_weights = torch.ones(n) / n

        active_layers = (
            self.dynamic_selector.get_active_layers(edit_id)
            if self.use_dynamic_layers
            else self.target_layers
        )
        layer_strengths = (
            self.dynamic_selector.get_layer_strengths(edit_id)
            if self.use_dynamic_layers
            else self.layer_strengths
        )

        for layer_idx in active_layers:
            layer = self._resolve_layer(model, layer_idx)
            strength = layer_strengths.get(layer_idx, 1.0)
            hook = layer.register_forward_hook(
                lambda module, inp, out, s=strength:
                    self._adaptive_hook(module, inp, out, s)
            )
            self.hooks.append(hook)

    def _resolve_layer(self, model, layer_idx: int):
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError("Unsupported model structure")

    def _adaptive_hook(self, module, inp, output, layer_strength: float):
        if isinstance(output, tuple):
            hidden_states, *rest = output
        else:
            hidden_states, rest = output, []

        if self.active_edit_id is not None and self.edit_module is not None:
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )
            inject_vector = (alpha * v_new + beta * v_old).to(hidden_states.device)
            inject_vector = inject_vector * layer_strength
            batch_size = hidden_states.size(0)

            delta = torch.zeros_like(hidden_states)
            for i, pos in enumerate(self.subject_positions):
                if 0 <= pos < hidden_states.size(1):
                    weight = self.attention_weights[i].item()
                    delta[:, pos, :] = (inject_vector * weight).unsqueeze(0).expand(batch_size, -1)

            hidden_states = hidden_states + delta

        if rest:
            return (hidden_states, *rest)
        return hidden_states

    def clear(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
        self.attention_weights = None