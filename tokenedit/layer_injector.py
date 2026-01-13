"""层级注入器"""

import torch
from typing import List, Callable

class LayerInjector:
    """
    层级注入器
    
    在指定的Transformer层注入编辑向量
    """
    
    def __init__(self, target_layers: List[int]):
        self.target_layers = target_layers
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
    
    def inject(
        self,
        model,
        edit_id: int,
        edit_module,
        subject_positions: List[int]
    ):
        """
        注入编辑向量到目标层

        Args:
            model: 预训练模型
            edit_id: 编辑ID
            edit_module: EditTokenModule实例
            subject_positions: 主体token在序列中的位置
        """
        self.active_edit_id = edit_id
        self.edit_module = edit_module
        self.subject_positions = subject_positions

        # 为每个目标层注册hook
        for layer_idx in self.target_layers:
            layer = self._resolve_layer(model, layer_idx)
            hook = layer.register_forward_hook(self._injection_hook)
            self.hooks.append(hook)

    def _resolve_layer(self, model, layer_idx: int):
        """
        Resolve the transformer block for different model architectures.
        GPT-2 uses model.transformer.h, while LLaMA-style uses model.model.layers.
        """
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError(
            "Unsupported model structure: expected transformer.h or model.layers"
        )
    
    def _injection_hook(self, module, input, output):
        """
        Hook函数：修改隐藏状态

        h'(s) = h(s) + α * v_new + β * v_old
        """
        # output是一个tuple: (hidden_states, ...)
        hidden_states = output[0]  # (batch_size, seq_len, hidden_size)

        if self.active_edit_id is not None and self.edit_module is not None:
            # 获取编辑向量
            v_new, v_old, alpha, beta = self.edit_module.get_edit_vectors(
                self.active_edit_id
            )

            # 计算注入向量（确保是正确的形状）
            inject_vector = alpha * v_new + beta * v_old  # (hidden_size,)

            # 确保向量在正确的设备上
            inject_vector = inject_vector.to(hidden_states.device)

            # 只修改主体位置的表示
            for pos in self.subject_positions:
                if 0 <= pos < hidden_states.size(1):
                    # 广播注入向量到所有批次
                    hidden_states[:, pos, :] = (
                        hidden_states[:, pos, :] + inject_vector.unsqueeze(0)
                    )

        return (hidden_states,) + output[1:]
    
    def clear(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None
