"""
tokenedit/tokenedit_main.py
TokenEdit知识编辑器 - 完全独立实现
不依赖compute_ks.py或compute_z.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from tqdm import tqdm

from .tokenedit_hparams import TokenEditHyperParams
from .edit_token_module import EditTokenModule
from .prompt_router import PromptRouter
from .layer_injector import LayerInjector
from .prompt_closure import PromptClosureGenerator
from .tokenedit_utils import TokenEditUtils  


class TokenEditEditor:
    """
    TokenEdit知识编辑器
    
    核心机制：
    1. 显式Token (v_new, v_old)
    2. Prompt闭包训练
    3. 动态路由
    4. 层级注入
    
    与MEMIT的区别：
    - MEMIT: 直接修改权重矩阵 W
    - TokenEdit: 通过可学习Token注入 h' = h + α*v_new + β*v_old
    """
    
    def __init__(self, model, tokenizer, hparams: TokenEditHyperParams):
        # ==================== [核心修复] 强制锁定随机种子 ====================
        import random
        import numpy as np
        
        if hparams.seed is not None:
            # 1. 锁定 Python 原生随机数
            random.seed(hparams.seed)
            # 2. 锁定 Numpy 随机数
            np.random.seed(hparams.seed)
            # 3. 锁定 PyTorch (CPU)
            torch.manual_seed(hparams.seed)
            # 4. 锁定 PyTorch (GPU)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(hparams.seed)
                # 5. 强制 CuDNN 使用确定性算法 (虽牺牲微小性能，但保证可复现)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
            if hparams.verbose:
                print(f"🔒 已强制固定随机种子: {hparams.seed}")
        # ===================================================================

        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams

        # 设置设备
        self.device = torch.device(hparams.device)
        self.model.to(self.device)

        # 自动设置目标层（如果未指定）
        if hparams.target_layers is None:
            hparams.target_layers = self._get_default_target_layers(model)
            if hparams.verbose:
                print(f"⚠ 未指定目标层，使用默认值: {hparams.target_layers}")

        # 初始化组件
        self.edit_module = None
        self.router = PromptRouter(model, tokenizer, hparams)
        self.injector = LayerInjector(hparams.target_layers)
        self.closure_gen = PromptClosureGenerator()
        self.utils = TokenEditUtils(model, tokenizer)

        # 编辑注册表
        self.edits_registry = {}

        if hparams.verbose:
            print(f"✓ TokenEditEditor初始化完成")
            print(f"  模型: {hparams.model_name}")
            print(f"  目标层: {hparams.target_layers}")
            print(f"  设备: {self.device}")

    def _get_default_target_layers(self, model) -> List[int]:
        """根据模型自动设置目标层"""
        model_name = model.config._name_or_path.lower()

        # 获取模型层数
        if hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
        elif hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        else:
            num_layers = 48  # GPT-2-XL的默认层数

        # 根据模型类型返回不同的默认层
        if 'gpt2' in model_name or 'gpt-2' in model_name:
            if 'xl' in model_name:
                return [17, 18, 19]  # GPT-2-XL有48层
            elif 'large' in model_name or 'gpt2-large' in model_name:
                return [14, 15, 16]  # GPT2-Large有36层
            elif 'medium' in model_name or 'gpt2-medium' in model_name:
                return [9, 10, 11]  # GPT2-Medium有24层
            else:
                return [5, 6, 7]  # GPT2-Small有12层
        elif 'llama' in model_name:
            # LLaMA模型的最后几层
            return list(range(max(0, num_layers - 3), num_layers))
        elif 'pythia' in model_name:
            return list(range(max(0, num_layers - 3), num_layers))
        else:
            # 默认使用最后3层
            return list(range(max(0, num_layers - 3), num_layers))
    
    def apply_edits(self, requests: List[Dict]) -> Dict:
        """
        应用批量编辑
        
        Args:
            requests: 编辑请求列表
                [{
                    "prompt": "The capital of France is",
                    "subject": "France",
                    "relation": "capital",
                    "target_new": "Lyon",
                    "target_true": "Paris"
                }]
        
        Returns:
            {
                "model": 编辑后的模型,
                "edit_module": EditToken模块,
                "stats": 训练统计信息
            }
        """
        num_edits = len(requests)
        
        if self.hparams.verbose:
            print(f"\n{'='*60}")
            print(f"开始编辑 {num_edits} 个知识点")
            print(f"{'='*60}")
        
        # 1. 初始化EditToken模块
        if self.hparams.verbose:
            print("\n[1/4] 初始化EditToken模块...")
        
        self.edit_module = EditTokenModule(
            hidden_size=self.model.config.hidden_size,
            num_edits=num_edits,
            hparams=self.hparams
        ).to(self.device)
        
        if self.hparams.verbose:
            print(f"  ✓ 创建了 {num_edits} 对Token (v_new, v_old)")
        
        # 2. 生成Prompt闭包训练数据
        if self.hparams.verbose:
            print("\n[2/4] 生成Prompt闭包...")
        
        train_data = []
        for i, req in enumerate(requests):
            # 生成四类样本
            closure = self.closure_gen.generate(
                subject=req['subject'],
                relation=req.get('relation', 'capital'),  # 默认关系
                new_object=req['target_new'],
                old_object=req['target_true']
            )
            
            train_data.append({
                'edit_id': i,
                'closure': closure,
                'request': req
            })
            
            # 注册到路由器
            self.router.register_edit(
                i, 
                req['subject'], 
                req.get('relation', 'capital')
            )
            self.edits_registry[i] = req
        
        if self.hparams.verbose:
            print(f"  ✓ 生成了 {len(train_data)} 个Prompt闭包")
            total_samples = len(train_data) * 4  # 每个闭包4类样本
            print(f"  ✓ 总训练样本数: {total_samples}")
        
        # 3. 训练EditToken
        if self.hparams.verbose:
            print("\n[3/4] 训练EditToken...")
        
        stats = self._train_tokens(train_data)
        
        # 4. 完成
        if self.hparams.verbose:
            print("\n[4/4] 编辑完成")
            print(f"  ✓ 最终损失: {stats['losses'][-1]:.4f}")
            print(f"{'='*60}\n")
        
        return {
            'model': self.model,
            'edit_module': self.edit_module,
            'router': self.router,
            'injector': self.injector,
            'stats': stats
        }
    
    def _train_tokens(self, train_data: List[Dict]) -> Dict:
        """
        训练显式Token
        
        优化目标：
        L = w_edit*L_edit + w_suppress*L_suppress + 
            w_ortho*L_ortho + w_local*L_local
        """
        # 冻结基础模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只优化EditToken
        optimizer = torch.optim.Adam(
            self.edit_module.parameters(),
            lr=self.hparams.learning_rate
        )
        
        # 学习率调度器
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.num_epochs
            )
        else:
            scheduler = None
        
        # 训练统计
        stats = {
            'losses': [],
            'loss_breakdown': {
                'edit': [],
                'suppress': [],
                'ortho': [],
                'local': []
            }
        }
        
        # 训练循环
        for epoch in tqdm(range(self.hparams.num_epochs), desc="Training"):
            epoch_loss = 0.0
            epoch_breakdown = {k: 0.0 for k in stats['loss_breakdown'].keys()}
            
            for data in train_data:
                edit_id = data['edit_id']
                closure = data['closure']
                
                # 对每类样本计算损失
                sample_types = []
                if self.hparams.use_forward:
                    sample_types.append(('forward', closure['forward']))
                if self.hparams.use_backward:
                    sample_types.append(('backward', closure['backward']))
                if self.hparams.use_judge:
                    sample_types.append(('judge', closure['judge']))
                if self.hparams.use_distract:
                    sample_types.append(('distract', closure['distract']))
                
                for sample_type, prompts in sample_types:
                    num_samples = max(1, int(self.hparams.num_paraphrase))
                    for prompt in prompts[:num_samples]:
                        # 计算损失
                        losses = self._compute_sample_loss(
                            edit_id,
                            prompt,
                            sample_type,
                            closure
                        )
                        
                        # 综合损失
                        total_loss = (
                            self.hparams.w_edit * losses.get('edit', 0) +
                            self.hparams.w_suppress * losses.get('suppress', 0) +
                            self.hparams.w_ortho * losses.get('ortho', 0) +
                            self.hparams.w_local * losses.get('local', 0)
                        )
                        
                        # 反向传播
                        optimizer.zero_grad()
                        total_loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(),
                            self.hparams.gradient_clip
                        )
                        
                        optimizer.step()
                        
                        # 统计
                        epoch_loss += total_loss.item()
                        for k, v in losses.items():
                            if isinstance(v, torch.Tensor):
                                epoch_breakdown[k] += v.item()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            # 记录统计
            num_samples = len(train_data) * len(sample_types)
            stats['losses'].append(epoch_loss / num_samples)
            for k in epoch_breakdown.keys():
                stats['loss_breakdown'][k].append(
                    epoch_breakdown[k] / num_samples
                )
            
            # 打印进度
            if (epoch + 1) % 10 == 0 and self.hparams.verbose:
                print(f"\nEpoch {epoch+1}/{self.hparams.num_epochs}")
                print(f"  Total Loss: {stats['losses'][-1]:.4f}")
                print(f"  Edit: {stats['loss_breakdown']['edit'][-1]:.4f}")
                print(f"  Suppress: {stats['loss_breakdown']['suppress'][-1]:.4f}")
                print(f"  Ortho: {stats['loss_breakdown']['ortho'][-1]:.4f}")
                print(f"  Local: {stats['loss_breakdown']['local'][-1]:.4f}")
        
        return stats
    
    def _compute_sample_loss(
        self,
        edit_id: int,
        prompt: str,
        sample_type: str,
        closure: Dict
    ) -> Dict[str, torch.Tensor]:
        """
        计算单个样本的损失
        
        Returns:
            {
                'edit': L_edit,
                'suppress': L_suppress,
                'ortho': L_ortho,
                'local': L_local
            }
        """
        losses = {}
        
        # 1. 编辑成功损失 (L_edit)
        if sample_type in ['forward', 'backward', 'judge']:
            edit_loss = self._compute_edit_loss(
                edit_id, prompt, closure['targets'][sample_type]
            )
            losses['edit'] = edit_loss
        else:
            losses['edit'] = torch.tensor(0.0, device=self.device)
        
        # 2. 反事实抑制损失 (L_suppress)
        if sample_type == 'forward':
            suppress_loss = self._compute_suppress_loss(
                edit_id, prompt, closure['old_object']
            )
            losses['suppress'] = suppress_loss
        else:
            losses['suppress'] = torch.tensor(0.0, device=self.device)
        
        # 3. 正交性损失 (L_ortho)
        # 获取prompt的嵌入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)
        
        ortho_loss = self.edit_module.compute_orthogonality_loss(prompt_emb)
        losses['ortho'] = ortho_loss
        
        # 4. 局部性损失 (L_local)
        if sample_type == 'distract':
            local_loss = self._compute_local_loss(edit_id, prompt)
            losses['local'] = local_loss
        else:
            losses['local'] = torch.tensor(0.0, device=self.device)
        
        return losses
    
    def _compute_edit_loss(
        self,
        edit_id: int,
        prompt: str,
        target: str
    ) -> torch.Tensor:
        """
        计算编辑成功损失（交叉熵）
        目标：模型应该输出target
        """
        # 获取主体位置 - 使用工具函数
        req = self.edits_registry[edit_id]
        subject_positions = self.utils.find_subject_positions(
            prompt,
            req['subject'],
            verbose=False
        )

        if not subject_positions:
            # 如果找不到主体，使用一个小的默认损失而不是0
            # 这样可以让训练继续进行
            return torch.tensor(0.1, device=self.device)

        # 使用工具函数计算目标logits
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        # 注入编辑向量
        self.injector.inject(
            self.model,
            edit_id,
            self.edit_module,
            subject_positions
        )

        # 前向传播 - 只计算目标token的损失
        outputs = self.model(**inputs, labels=inputs['input_ids'])

        # 清除注入
        self.injector.clear()

        return outputs.loss
    
    def _compute_suppress_loss(
        self,
        edit_id: int,
        prompt: str,
        old_target: str
    ) -> torch.Tensor:
        """
        计算反事实抑制损失（Unlikelihood Loss）
        目标：降低旧答案的概率
        
        L_suppress = -log(1 - P(old_target | prompt))
        """
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
        
        if len(old_tokens) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 获取主体位置
        req = self.edits_registry[edit_id]
        subject = req['subject']
        subject_tokens = self.tokenizer.encode(subject, add_special_tokens=False)
        subject_positions = list(range(1, 1 + len(subject_tokens)))
        
        # 注入
        self.injector.inject(
            self.model,
            edit_id,
            self.edit_module,
            subject_positions
        )
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # 最后一个token的logits
            probs = F.softmax(logits, dim=-1)
        
        # 清除注入
        self.injector.clear()
        
        # 计算旧token的概率
        old_token_id = old_tokens[0]
        old_prob = probs[old_token_id]
        
        # Unlikelihood loss
        loss = -torch.log(1 - old_prob + 1e-8)
        
        return loss
    
    def _compute_local_loss(
        self,
        edit_id: int,
        prompt: str
    ) -> torch.Tensor:
        """
        计算局部性损失（KL散度）
        目标：干扰问题的输出应该与原模型一致
        """
        # 获取主体位置  
        req = self.edits_registry[edit_id]
        subject_positions = self.utils.find_subject_positions(
            prompt,
            req['subject'],
            verbose=False
        )
        
        if not subject_positions:
            return torch.tensor(0.0, device=self.device)
        
        # 使用工具函数计算KL散度  
        kl_loss = self.utils.compute_kl_divergence(
            prompt,
            subject_positions,
            self.edit_module,
            edit_id,
            self.injector
        )
        
        return kl_loss
    
    def inference(self, prompt: str, max_new_tokens: int = 10, verbose: bool = None) -> str:
        """
        推理：自动检测并注入编辑

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            verbose: 是否显示详细信息（None则使用hparams.verbose）

        Returns:
            生成的文本
        """
        if verbose is None:
            verbose = self.hparams.verbose

        self.model.eval()

        # 1. 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 2. 路由检测
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        edit_id = self.router.route(prompt, prompt_emb)

        # 3. 条件化注入
        if edit_id is not None:
            if verbose:
                req = self.edits_registry[edit_id]
                print(f"✓ 触发编辑 #{edit_id}: {req['subject']} -> {req['target_new']}")

            # 获取主体位置 - 使用工具函数
            req = self.edits_registry[edit_id]
            subject_positions = self.utils.find_subject_positions(
                prompt,
                req['subject'],
                verbose=verbose
            )

            if subject_positions:
                # 注入
                self.injector.inject(
                    self.model,
                    edit_id,
                    self.edit_module,
                    subject_positions
                )
                if verbose:
                    print(f"  注入位置: {subject_positions}")
            else:
                if verbose:
                    print(f"  警告: 未找到主体位置，编辑可能无效")
        else:
            if verbose:
                print("✗ 未触发编辑，使用原始模型")

        # 4. 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 清除注入
        self.injector.clear()

        # 解码
        result = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return result
    
    def save(self, path: str):
        """保存编辑器状态"""
        torch.save({
            'edit_module': self.edit_module.state_dict(),
            'edits_registry': self.edits_registry,
            'hparams': self.hparams
        }, path)
        if self.hparams.verbose:
            print(f"✓ 编辑器已保存到 {path}")
    
    def load(self, path: str):
        """加载编辑器状态"""
        checkpoint = torch.load(path)
        
        # 恢复EditModule
        num_edits = len(checkpoint['edits_registry'])
        self.edit_module = EditTokenModule(
            self.model.config.hidden_size,
            num_edits,
            self.hparams
        ).to(self.device)
        self.edit_module.load_state_dict(checkpoint['edit_module'])
        
        # 恢复注册表
        self.edits_registry = checkpoint['edits_registry']
        
        if self.hparams.verbose:
            print(f"✓ 编辑器已从 {path} 加载")
            print(f"  包含 {num_edits} 个编辑")
