"""层级注入器

修复版本 - 解决了:
1. 输出格式处理更鲁棒
2. 设备一致性检查
3. 更安全的hook管理
"""

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
        Hook函数: 修改隐藏状态

        h'(s) = h(s) + alpha * v_new + beta * v_old
        
        修复:
        - 更鲁棒的输出格式处理
        - 设备一致性检查
        - 边界检查
        """
        # 修复: 更安全的输出格式处理
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

            # 计算注入向量(确保是正确的形状)
            inject_vector = alpha * v_new + beta * v_old  # (hidden_size,)

            # 修复: 确保向量在正确的设备上
            inject_vector = inject_vector.to(hidden_states.device)

            # 只修改主体位置的表示
            for pos in self.subject_positions:
                # 修复: 添加边界检查
                if 0 <= pos < hidden_states.size(1):
                    # 广播注入向量到所有批次
                    hidden_states[:, pos, :] = (
                        hidden_states[:, pos, :] + inject_vector.unsqueeze(0)
                    )

        # 修复: 正确返回输出格式
        if isinstance(output, tuple):
            return (hidden_states,) + other_outputs
        else:
            return hidden_states
    
    def clear(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.active_edit_id = None
        self.edit_module = None
        self.subject_positions = None