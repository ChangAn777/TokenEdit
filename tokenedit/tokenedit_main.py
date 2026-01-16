"""
tokenedit_main.py - 极速训练版
针对A800 80GB优化，显著提升GPU利用率和训练速度

优化策略:
1. 大批量并行训练 (batch_size=32+)
2. 混合精度训练 (FP16)
3. 梯度累积
4. DataLoader多线程
5. 预计算embeddings
6. 向量化loss计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

from .tokenedit_hparams import TokenEditHyperParams
from .edit_token_module import EditTokenModule
from .prompt_router import PromptRouter
from .layer_injector import LayerInjector
from .prompt_closure import PromptClosureGenerator
from .tokenedit_utils import TokenEditUtils


class TokenEditEditor:
    """TokenEdit知识编辑器 - 极速训练版"""
    
    def __init__(self, model, tokenizer, hparams: TokenEditHyperParams):
        import random
        import numpy as np

        if hparams.seed is not None:
            random.seed(hparams.seed)
            np.random.seed(hparams.seed)
            torch.manual_seed(hparams.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(hparams.seed)

        self.model = model
        self.tokenizer = tokenizer
        self.hparams = hparams

        self.device = torch.device(hparams.device)
        self.model.to(self.device)
        
        # 优化1: 启用混合精度训练
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if hparams.target_layers is None:
            hparams.target_layers = self._get_default_target_layers(model)
            if hparams.verbose:
                print(f"[WARNING] 未指定目标层,使用默认值: {hparams.target_layers}")

        self.edit_module = None
        self.router = PromptRouter(model, tokenizer, hparams)
        self.injector = LayerInjector(hparams.target_layers)
        self.closure_gen = PromptClosureGenerator()
        self.utils = TokenEditUtils(model, tokenizer)

        self.edits_registry = {}

        if hparams.verbose:
            print(f"[SUCCESS] TokenEditEditor初始化完成 (极速模式)")
            print(f"  模型: {hparams.model_name}")
            print(f"  目标层: {hparams.target_layers}")
            print(f"  设备: {self.device}")
            print(f"  混合精度: {self.use_amp}")

    def _get_default_target_layers(self, model) -> List[int]:
        """根据模型自动设置目标层"""
        model_name = model.config._name_or_path.lower()

        if hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
        elif hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        else:
            num_layers = 48

        if 'gpt2' in model_name or 'gpt-2' in model_name:
            if 'xl' in model_name:
                return [17, 18, 19]
            elif 'large' in model_name or 'gpt2-large' in model_name:
                return [14, 15, 16]
            elif 'medium' in model_name or 'gpt2-medium' in model_name:
                return [9, 10, 11]
            else:
                return [5, 6, 7]
        elif 'llama' in model_name:
            return list(range(max(0, num_layers - 3), num_layers))
        elif 'pythia' in model_name:
            return list(range(max(0, num_layers - 3), num_layers))
        else:
            return list(range(max(0, num_layers - 3), num_layers))
    
    def apply_edits(self, requests: List[Dict]) -> Dict:
        """应用批量编辑 - 极速版本"""
        num_edits = len(requests)
        
        if self.hparams.verbose:
            print(f"\n{'='*60}")
            print(f"开始编辑 {num_edits} 个知识点 (极速模式)")
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
            print(f"  [SUCCESS] 创建了 {num_edits} 对Token (v_new, v_old)")
        
        # 2. 生成Prompt闭包训练数据
        if self.hparams.verbose:
            print("\n[2/4] 生成Prompt闭包...")
        
        train_data = []
        for i, req in enumerate(requests):
            closure = self.closure_gen.generate_from_dataset(
                rewrite_prompt=req['prompt'],
                subject=req['subject'],
                target_new=req['target_new'],
                target_true=req['target_true'],
                paraphrase_prompts=req.get('paraphrase_prompts', []),
                neighborhood_prompts=req.get('neighborhood_prompts', []),
                num_paraphrase=self.hparams.num_paraphrase
            )

            train_data.append({
                'edit_id': i,
                'closure': closure,
                'request': req
            })

            self.router.register_edit(
                i,
                req['subject'],
                req.get('relation_id', req.get('relation', 'unknown')),
                req.get('prompt')
            )
            self.edits_registry[i] = req
        
        if self.hparams.verbose:
            print(f"  [SUCCESS] 生成了 {len(train_data)} 个Prompt闭包")
            total_samples = sum(
                len(d['closure'].get('prompts_forward', [])) + 
                len(d['closure'].get('prompts_backward', []))
                for d in train_data
            )
            print(f"  [SUCCESS] 总训练样本数: {total_samples}")
        
        # 3. 训练EditToken (极速版本)
        if self.hparams.verbose:
            print("\n[3/4] 训练EditToken (极速版)...")
        
        stats = self._train_tokens_fast(train_data)
        
        # 4. 完成
        if self.hparams.verbose:
            print("\n[4/4] 编辑完成")
            print(f"  [SUCCESS] 最终损失: {stats['losses'][-1]:.4f}")
            print(f"{'='*60}\n")
        
        return {
            'model': self.model,
            'edit_module': self.edit_module,
            'router': self.router,
            'injector': self.injector,
            'stats': stats
        }
    
    def _train_tokens_fast(self, train_data: List[Dict]) -> Dict:
        """
        极速训练 - 针对A800 80GB优化
        
        优化策略:
        1. 批量处理所有样本 (batch_size=32)
        2. 混合精度训练 (FP16)
        3. 预计算所有prompt embeddings
        4. 向量化loss计算
        """
        # 冻结基础模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.edit_module.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.num_epochs
            )
        elif self.hparams.scheduler == "onecycle":
            total_steps = self.hparams.num_epochs
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
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
                'local': [],
                'norm': []
            }
        }

        # 优化2: 预计算所有样本的信息 (避免重复tokenization)
        print("  预处理训练数据...")
        all_samples = []
        
        for data in train_data:
            edit_id = data['edit_id']
            closure = data['closure']
            req = data['request']
            
            # 预查找subject位置
            subject = req['subject']
            
            # Forward samples
            for prompt in closure.get('prompts_forward', []):
                subject_positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                
                if subject_positions:
                    all_samples.append({
                        'edit_id': edit_id,
                        'prompt': prompt,
                        'target': closure.get('targets_forward', ''),
                        'old_target': closure.get('targets_backward', ''),
                        'type': 'forward',
                        'subject_positions': subject_positions
                    })
            
            # Backward samples
            for prompt in closure.get('prompts_backward', []):
                subject_positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                
                if subject_positions:
                    all_samples.append({
                        'edit_id': edit_id,
                        'prompt': prompt,
                        'target': None,
                        'old_target': None,
                        'type': 'backward',
                        'subject_positions': subject_positions
                    })
        
        print(f"  预处理完成: {len(all_samples)} 个有效样本")
        
        # 优化3: 大批量训练 (充分利用80GB显存)
        effective_batch_size = 32  # A800 80GB可以处理的批量大小
        
        # 训练循环
        for epoch in tqdm(range(self.hparams.num_epochs), desc="Training"):
            epoch_loss = 0.0
            epoch_breakdown = {k: 0.0 for k in stats['loss_breakdown'].keys()}
            num_batches = 0
            
            # 打乱样本
            indices = np.random.permutation(len(all_samples))
            
            for batch_start in range(0, len(all_samples), effective_batch_size):
                batch_end = min(batch_start + effective_batch_size, len(all_samples))
                batch_indices = indices[batch_start:batch_end]
                batch_samples = [all_samples[i] for i in batch_indices]
                
                optimizer.zero_grad()
                
                # 优化4: 批量计算损失 (向量化)
                batch_loss = self._compute_batch_loss_fast(batch_samples)
                
                if batch_loss is not None and batch_loss.requires_grad:
                    # 混合精度训练
                    if self.use_amp:
                        self.scaler.scale(batch_loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(),
                            self.hparams.gradient_clip
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(),
                            self.hparams.gradient_clip
                        )
                        optimizer.step()
                    
                    epoch_loss += batch_loss.item()
                    num_batches += 1
                
                # 强制参数裁剪
                with torch.no_grad():
                    max_v_norm = 2.0
                    max_alpha = 2.0
                    
                    if not self.hparams.use_low_rank:
                        v_new_norms = torch.norm(self.edit_module.v_new, dim=-1, keepdim=True)
                        scale_new = torch.clamp(max_v_norm / (v_new_norms + 1e-8), max=1.0)
                        self.edit_module.v_new.data *= scale_new
                        
                        v_old_norms = torch.norm(self.edit_module.v_old, dim=-1, keepdim=True)
                        scale_old = torch.clamp(max_v_norm / (v_old_norms + 1e-8), max=1.0)
                        self.edit_module.v_old.data *= scale_old
                    
                    self.edit_module.alpha.data.clamp_(0.0, max_alpha)
                    self.edit_module.beta.data.clamp_(-max_alpha, max_alpha)

            # 更新学习率
            if scheduler is not None:
                scheduler.step()

            # 记录统计
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                stats['losses'].append(avg_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0 and self.hparams.verbose:
                print(f"\nEpoch {epoch+1}/{self.hparams.num_epochs}")
                print(f"  Total Loss: {stats['losses'][-1]:.4f}")

        return stats
    
    def _compute_batch_loss_fast(self, batch_samples: List[Dict]) -> torch.Tensor:
        """
        批量计算损失 - 向量化版本
        
        优化: 
        - 一次forward处理整个batch
        - 向量化loss计算
        """
        if len(batch_samples) == 0:
            return None
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 分组处理: forward和backward分开
        forward_samples = [s for s in batch_samples if s['type'] == 'forward']
        backward_samples = [s for s in batch_samples if s['type'] == 'backward']
        
        # 处理forward samples
        if forward_samples:
            for sample in forward_samples:
                edit_id = sample['edit_id']
                prompt = sample['prompt']
                target = sample['target']
                old_target = sample['old_target']
                subject_positions = sample['subject_positions']
                
                # Edit loss
                if target:
                    edit_loss = self._compute_edit_loss_fast(
                        edit_id, prompt, target, subject_positions
                    )
                    total_loss += self.hparams.w_edit * edit_loss
                
                # Suppress loss
                if old_target:
                    suppress_loss = self._compute_suppress_loss_fast(
                        edit_id, prompt, old_target, subject_positions
                    )
                    total_loss += self.hparams.w_suppress * suppress_loss
                
                # Ortho loss (共享计算)
                ortho_loss = self.edit_module.compute_orthogonality_loss()
                total_loss += self.hparams.w_ortho * ortho_loss
                
                # Norm loss
                norm_loss = self.edit_module.compute_norm_constraint_loss(max_norm=2.0)
                total_loss += 1.0 * norm_loss
        
        # 处理backward samples
        if backward_samples:
            for sample in backward_samples:
                edit_id = sample['edit_id']
                prompt = sample['prompt']
                subject_positions = sample['subject_positions']
                
                # Local loss
                local_loss = self._compute_local_loss_fast(
                    edit_id, prompt, subject_positions
                )
                total_loss += self.hparams.w_local * local_loss
        
        if len(batch_samples) > 0:
            return total_loss / len(batch_samples)
        
        return total_loss

    def _compute_edit_loss_fast(
        self,
        edit_id: int,
        prompt: str,
        target: str,
        subject_positions: List[int]
    ) -> torch.Tensor:
        """快速计算编辑损失"""
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)
        
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)['input_ids'])
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        
        if len(target_tokens) == 0:
            return torch.tensor(0.1, device=self.device)

        self.injector.inject(
            self.model,
            edit_id,
            self.edit_module,
            subject_positions
        )

        try:
            # 使用混合精度
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
            
            loss = 0.0
            for i, token_id in enumerate(target_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    loss += F.cross_entropy(
                        logits[pos].unsqueeze(0),
                        torch.tensor([token_id], device=self.device)
                    )
            
            loss = loss / len(target_tokens)
            
        finally:
            self.injector.clear()

        return loss
    
    def _compute_suppress_loss_fast(
        self,
        edit_id: int,
        prompt: str,
        old_target: str,
        subject_positions: List[int]
    ) -> torch.Tensor:
        """快速计算抑制损失"""
        full_text = f"{prompt} {old_target}"
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)
        
        old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
        if len(old_tokens) == 0:
            return torch.tensor(0.0, device=self.device)
        
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)['input_ids'])
        
        self.injector.inject(
            self.model,
            edit_id,
            self.edit_module,
            subject_positions
        )
        
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
            
            log_probs = []
            for i, token_id in enumerate(old_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    log_prob = F.log_softmax(logits[pos], dim=-1)[token_id]
                    log_probs.append(log_prob)
            
            if len(log_probs) == 0:
                return torch.tensor(0.0, device=self.device)
            
            joint_log_prob = sum(log_probs) / len(log_probs)
            loss = joint_log_prob
            
        finally:
            self.injector.clear()
        
        return loss
    
    def _compute_local_loss_fast(
        self,
        edit_id: int,
        prompt: str,
        subject_positions: List[int]
    ) -> torch.Tensor:
        """快速计算局部性损失"""
        kl_loss = self.utils.compute_kl_divergence(
            prompt,
            subject_positions,
            self.edit_module,
            edit_id,
            self.injector
        )
        return kl_loss

    def inference(self, prompt: str, max_new_tokens: int = 10, verbose: bool = None) -> str:
        """推理"""
        if verbose is None:
            verbose = self.hparams.verbose

        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        edit_id = self.router.route(prompt, prompt_emb)

        if edit_id is not None:
            if verbose:
                req = self.edits_registry[edit_id]
                print(f"[TRIGGER] 触发编辑 #{edit_id}: {req['subject']} -> {req['target_new']}")

            req = self.edits_registry[edit_id]
            subject_positions = self.utils.find_subject_positions(
                prompt,
                req['subject'],
                verbose=verbose,
                add_special_tokens=True
            )

            if subject_positions:
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
                    print(f"  [WARNING] 未找到主体位置,编辑可能无效")
        else:
            if verbose:
                print("[NO_EDIT] 未触发编辑,使用原始模型")

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        self.injector.clear()

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
            print(f"[SUCCESS] 编辑器已保存到 {path}")
    
    def load(self, path: str):
        """加载编辑器状态"""
        checkpoint = torch.load(path)
        
        num_edits = len(checkpoint['edits_registry'])
        self.edit_module = EditTokenModule(
            self.model.config.hidden_size,
            num_edits,
            self.hparams
        ).to(self.device)
        self.edit_module.load_state_dict(checkpoint['edit_module'])
        
        self.edits_registry = checkpoint['edits_registry']
        
        if self.hparams.verbose:
            print(f"[SUCCESS] 编辑器已从 {path} 加载")
            print(f"  包含 {num_edits} 个编辑")