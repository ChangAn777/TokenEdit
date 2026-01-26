"""
tokenedit_main.py - 集成增强模块版本

集成改进:
1. 支持多种EditToken模块类型
2. 支持多种Injector类型
3. 集成增强训练策略
4. 改进的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import random

from .tokenedit_hparams import TokenEditHyperParams
from .edit_token_module import EditTokenModule
from .edit_token_module import (
    EnhancedEditTokenModule, 
    AdaptiveEditTokenModule, 
    HybridEditTokenModule
)
from .layer_injector import LayerInjector
from .layer_injector import (
    EnhancedLayerInjector,
    AttentionGuidedInjector,
    AdaptiveInjector
)
from .prompt_router import PromptRouter
from .prompt_closure import PromptClosureGenerator
from .tokenedit_utils import TokenEditUtils


class TokenEditEditor:
    """TokenEdit知识编辑器 - 增强集成版"""
    
    def __init__(self, model, tokenizer, hparams: TokenEditHyperParams):
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
        
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if hparams.target_layers is None:
            hparams.target_layers = self._get_optimal_target_layers(model)
            if hparams.verbose:
                print(f"[OPTIMIZED] 使用优化层选择: {hparams.target_layers}")

        self.edit_module = None
        self.router = PromptRouter(model, tokenizer, hparams)
        
        # 根据配置选择Injector类型
        self.injector = self._create_injector(hparams)
        
        self.closure_gen = PromptClosureGenerator()
        self.utils = TokenEditUtils(model, tokenizer)

        self.edits_registry = {}

        if hparams.verbose:
            print("[SUCCESS] TokenEditEditor (Enhanced) init complete")
            print(f"  模型: {hparams.model_name}")
            print(f"  目标层: {hparams.target_layers}")
            print(f"  EditModule类型: {hparams.edit_module_type}")
            print(f"  Injector类型: {hparams.injector_type}")
            print(f"  激活特性:")
            for k, v in hparams.get_active_features().items():
                print(f"    - {k}: {v}")

    def _create_injector(self, hparams):
        """根据配置创建Injector"""
        if hparams.injector_type == "enhanced":
            return EnhancedLayerInjector(
                hparams.target_layers,
                use_progressive=hparams.use_progressive_injection
            )
        elif hparams.injector_type == "attention":
            return AttentionGuidedInjector(hparams.target_layers)
        elif hparams.injector_type == "adaptive":
            # 需要知道num_edits,所以暂时返回None,在apply_edits中创建
            return None
        else:
            return LayerInjector(hparams.target_layers)

    def _get_optimal_target_layers(self, model) -> List[int]:
        """优化的层选择策略"""
        model_name = model.config._name_or_path.lower()

        if hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
        elif hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        else:
            num_layers = 48

        if 'gpt2' in model_name or 'gpt-2' in model_name:
            if 'xl' in model_name:
                return [30, 31, 32, 33, 34]
            elif 'large' in model_name:
                return [24, 25, 26, 27, 28]
            elif 'medium' in model_name:
                return [16, 17, 18, 19, 20]
            else:
                return [7, 8, 9, 10]
        elif 'llama' in model_name:
            return list(range(max(0, num_layers - 8), num_layers - 3))
        else:
            return list(range(max(0, num_layers - 8), num_layers - 3))
    
    def apply_edits(self, requests: List[Dict]) -> Dict:
        """应用批量编辑"""
        num_edits = len(requests)
        
        print("\n" + "-"*60)
        print("[CONFIG CHECK - ENHANCED VERSION]")
        print(f"  > Edit Module: {self.hparams.edit_module_type}")
        print(f"  > Injector: {self.hparams.injector_type}")
        print(f"  > w_edit: {self.hparams.w_edit}")
        print(f"  > w_suppress: {self.hparams.w_suppress}")
        print(f"  > use_multiscale: {self.hparams.use_multiscale}")
        print(f"  > use_focal_loss: {self.hparams.use_focal_loss}")
        print(f"  > use_curriculum: {self.hparams.use_curriculum}")
        print(f"  > use_contrastive: {self.hparams.use_contrastive_loss}")
        print("-"*60 + "\n")
        
        if self.hparams.verbose:
            print(f"开始编辑 {num_edits} 个知识点")
        
        # 根据配置选择EditToken模块类型
        self.edit_module = self._create_edit_module(num_edits)
        
        # 如果使用自适应Injector,现在创建它
        if self.hparams.injector_type == "adaptive":
            self.injector = AdaptiveInjector(
                self.hparams.target_layers,
                num_edits=num_edits,
                use_progressive=self.hparams.use_progressive_injection,
                use_attention=self.hparams.use_attention_injection,
                use_dynamic_layers=self.hparams.use_dynamic_layers
            )
        
        # 生成Prompt闭包训练数据
        train_data = []
        self.edits_registry = {}

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
                req.get('prompt'),
                req.get('paraphrase_prompts', []),
            )
            self.edits_registry[i] = req
        
        if self.hparams.verbose:
            print(f"  生成了 {len(train_data)} 个Prompt闭包")
        
        # 使用增强的训练流程
        stats = self._train_tokens_enhanced(train_data)
        
        # Subject Guard
        if not hasattr(self.router, "_original_route"):
            self.router._original_route = self.router.route
            
        original_route = self.router._original_route
        registry = self.edits_registry
        utils = self.utils
        
        def guarded_route(prompt: str, prompt_emb=None):
            candidate_id = original_route(prompt, prompt_emb)
            if candidate_id is None:
                return None
            if candidate_id in registry:
                subject = registry[candidate_id]['subject']
                positions = utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if not positions:
                    return None
            return candidate_id

        self.router.route = guarded_route
        print("[Security] Subject Guard 已激活")
        
        if self.hparams.verbose:
            print("\n编辑完成")
            if stats['losses']:
                print(f"  最终损失: {stats['losses'][-1]:.4f}")
                print(f"  最低损失: {min(stats['losses']):.4f}")
        
        return {
            'model': self.model,
            'edit_module': self.edit_module,
            'router': self.router,
            'injector': self.injector,
            'stats': stats
        }
    
    def _create_edit_module(self, num_edits: int):
        """根据配置创建EditToken模块"""
        hidden_size = self.model.config.hidden_size
        
        if self.hparams.edit_module_type == "enhanced":
            module = EnhancedEditTokenModule(hidden_size, num_edits, self.hparams)
        elif self.hparams.edit_module_type == "adaptive":
            module = AdaptiveEditTokenModule(hidden_size, num_edits, self.hparams)
        elif self.hparams.edit_module_type == "hybrid":
            module = HybridEditTokenModule(hidden_size, num_edits, self.hparams)
        else:  # "standard"
            module = EditTokenModule(hidden_size, num_edits, self.hparams)
        
        return module.to(self.device)
    
    def _train_tokens_enhanced(self, train_data: List[Dict]) -> Dict:
        """增强的训练流程"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Smart初始化
        if self.hparams.token_init_method == "target_smart":
            self._smart_initialize(train_data)

        optimizer = torch.optim.AdamW(
            self.edit_module.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        if self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=self.hparams.plateau_factor,
                patience=self.hparams.plateau_patience,
                min_lr=self.hparams.plateau_min_lr,
                verbose=False
            )
        elif self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.num_epochs
            )
        else:
            scheduler = None

        stats = {'losses': [], 'lr_history': []}

        # 准备训练样本
        all_samples = self._prepare_training_samples(train_data)
        
        # 难样本挖掘器
        hard_miner = {} if self.hparams.use_hard_mining else None

        desired_batch_size = min(128, max(64, len(all_samples) // 8))
        micro_batch_size = min(16, desired_batch_size)
        grad_accum_steps = max(1, int(np.ceil(desired_batch_size / micro_batch_size)))

        if self.hparams.verbose:
            print(f"  [Training] Samples: {len(all_samples)}, Batch: {desired_batch_size}")

        best_loss = None
        stale_epochs = 0
        
        for epoch in tqdm(range(self.hparams.num_epochs), desc="Training"):
            # 课程学习: 根据epoch过滤样本
            if self.hparams.use_curriculum:
                active_samples = self._filter_by_curriculum(all_samples, epoch)
            else:
                active_samples = all_samples
            
            # 计算当前epoch的损失权重
            if self.hparams.use_adaptive_weights:
                current_weights = self._get_adaptive_weights(epoch)
            else:
                current_weights = {
                    'edit': self.hparams.w_edit,
                    'suppress': self.hparams.w_suppress,
                    'ortho': self.hparams.w_ortho,
                    'local': self.hparams.w_local,
                    'contrastive': self.hparams.w_contrastive
                }
            
            epoch_loss = 0.0
            num_batches = 0

            indices = np.random.permutation(len(active_samples))
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            effective_accum = grad_accum_steps * micro_batch_size

            for batch_start in range(0, len(active_samples), micro_batch_size):
                batch_end = min(batch_start + micro_batch_size, len(active_samples))
                batch_indices = indices[batch_start:batch_end]
                batch_samples = [active_samples[i] for i in batch_indices]

                for sample in batch_samples:
                    # 使用增强的损失计算
                    sample_loss = self._compute_sample_loss_enhanced(sample, current_weights)
                    
                    if sample_loss is not None and sample_loss.requires_grad:
                        # 难样本加权
                        if self.hparams.use_hard_mining:
                            sample_id = sample.get('sample_id', hash(sample['prompt']))
                            difficulty = self._get_sample_difficulty(hard_miner, sample_id)
                            weight = 1.0 + difficulty * (self.hparams.hard_boost_factor - 1.0)
                            sample_loss = sample_loss * weight
                        
                        scaled_loss = sample_loss / effective_accum
                        if self.use_amp:
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        accum_loss += sample_loss.item()
                        accum_count += 1
                        
                        # 更新难样本历史
                        if self.hparams.use_hard_mining:
                            self._update_hard_miner(hard_miner, sample_id, sample_loss.item())

                step_now = (accum_count % effective_accum == 0) or (batch_end == len(active_samples))
                if step_now and accum_count > 0:
                    scale_factor = float(effective_accum) / float(accum_count)
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        for p in self.edit_module.parameters():
                            if p.grad is not None:
                                p.grad.mul_(scale_factor)
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(), 
                            self.hparams.gradient_clip
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        for p in self.edit_module.parameters():
                            if p.grad is not None:
                                p.grad.mul_(scale_factor)
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(), 
                            self.hparams.gradient_clip
                        )
                        optimizer.step()

                    optimizer.zero_grad()
                    epoch_loss += accum_loss / accum_count
                    num_batches += 1
                    accum_loss = 0.0
                    accum_count = 0

                    # 门控约束
                    with torch.no_grad():
                        if hasattr(self.edit_module, 'alpha'):
                            self.edit_module.alpha.data.clamp_(0.0, 2.5)
                        if hasattr(self.edit_module, 'beta'):
                            self.edit_module.beta.data.clamp_(-2.0, 2.0)

            if num_batches > 0:
                epoch_avg = epoch_loss / num_batches
                stats['losses'].append(epoch_avg)
                stats['lr_history'].append(optimizer.param_groups[0]['lr'])

                if scheduler:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_avg)
                    else:
                        scheduler.step()

                # Early stopping
                patience = self.hparams.early_stop_patience
                min_delta = self.hparams.early_stop_min_delta
                if patience is not None and patience > 0:
                    if best_loss is None or (best_loss - epoch_avg) > min_delta:
                        best_loss = epoch_avg
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= patience:
                            if self.hparams.verbose:
                                print(f"\n[EarlyStop] no improvement for {patience} epochs")
                            break

        return stats
    
    def _smart_initialize(self, train_data: List[Dict]):
        """Smart初始化"""
        print("  [Init] applying Smart Initialization (Enhanced)...")
        with torch.no_grad():
            if hasattr(self.model, "transformer"):
                wte = self.model.transformer.wte
            elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
                wte = self.model.model.embed_tokens
            else:
                wte = self.model.get_input_embeddings()

            init_count = 0
            for data in train_data:
                idx = data['edit_id']
                target_word = data['request']['target_new']

                t_ids = None
                for prefix in [" ", ""]:
                    test_ids = self.tokenizer.encode(
                        prefix + target_word.strip(), 
                        add_special_tokens=False
                    )
                    if test_ids:
                        t_ids = test_ids
                        break

                if t_ids:
                    target_emb = wte(torch.tensor(t_ids, device=self.device)).mean(dim=0)
                    noise = torch.randn_like(target_emb) * self.hparams.token_init_std
                    
                    # 根据模块类型初始化
                    if self.hparams.use_multiscale and hasattr(self.edit_module, 'v_new_coarse'):
                        self.edit_module.v_new_coarse.data[idx] = target_emb.clone()
                        self.edit_module.v_new_fine.data[idx] = noise
                    elif hasattr(self.edit_module, 'v_new'):
                        self.edit_module.v_new.data[idx] = target_emb.clone() + noise
                    
                    init_count += 1
        
        print(f"  [Init] initialized {init_count}/{len(train_data)} vectors")
    
    def _prepare_training_samples(self, train_data: List[Dict]) -> List[Dict]:
        """准备训练样本"""
        all_samples = []
        sample_id = 0
        
        for data in train_data:
            edit_id = data['edit_id']
            closure = data['closure']
            req = data['request']
            subject = req['subject']

            rewrite_set = set(closure.get('rewrite_prompts', []))
            for prompt in closure.get('prompts_forward', []):
                subject_positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if subject_positions:
                    all_samples.append({
                        'sample_id': sample_id,
                        'edit_id': edit_id,
                        'prompt': prompt,
                        'type': 'forward',
                        'is_rewrite': prompt in rewrite_set,
                        'subject_positions': subject_positions,
                        'target': closure.get('targets_forward', ''),
                        'old_target': closure.get('targets_backward', '')
                    })
                    sample_id += 1

            for prompt in closure.get('prompts_backward', []):
                subject_positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if subject_positions:
                    all_samples.append({
                        'sample_id': sample_id,
                        'edit_id': edit_id,
                        'prompt': prompt,
                        'type': 'backward',
                        'subject_positions': subject_positions,
                        'target': None,
                        'old_target': None
                    })
                    sample_id += 1
        
        return all_samples
    
    def _filter_by_curriculum(self, all_samples: List[Dict], epoch: int) -> List[Dict]:
        """课程学习过滤"""
        progress = epoch / self.hparams.num_epochs
        stages = self.hparams.curriculum_stages
        
        if progress < stages[0] / self.hparams.num_epochs:
            return [s for s in all_samples if s.get('is_rewrite', False)]
        elif progress < stages[1] / self.hparams.num_epochs:
            return [s for s in all_samples if s['type'] == 'forward']
        else:
            return all_samples
    
    def _get_adaptive_weights(self, epoch: int) -> Dict[str, float]:
        """自适应损失权重"""
        progress = epoch / self.hparams.num_epochs
        
        weights = {}
        weights['edit'] = self.hparams.w_edit * (1.0 - 0.1 * progress)
        weights['suppress'] = self.hparams.w_suppress * (0.3 + 0.7 * progress)
        
        if progress < 0.3:
            weights['ortho'] = self.hparams.w_ortho * 0.1
        else:
            weights['ortho'] = self.hparams.w_ortho * progress
        
        if progress < 0.5:
            weights['local'] = self.hparams.w_local * 0.5
        else:
            weights['local'] = self.hparams.w_local * (1.0 + progress)
        
        weights['contrastive'] = self.hparams.w_contrastive
        
        return weights
    
    def _get_sample_difficulty(self, hard_miner: dict, sample_id: int) -> float:
        """计算样本难度"""
        if hard_miner is None or sample_id not in hard_miner:
            return 0.5
        
        losses = hard_miner[sample_id]
        if len(losses) < 3:
            return 0.5
        
        mean_loss = np.mean(losses[-5:])
        std_loss = np.std(losses[-5:])
        difficulty = mean_loss + 0.5 * std_loss
        return np.clip(difficulty / 2.0, 0, 1)
    
    def _update_hard_miner(self, hard_miner: dict, sample_id: int, loss: float):
        """更新难样本历史"""
        if hard_miner is None:
            return
        if sample_id not in hard_miner:
            hard_miner[sample_id] = []
        hard_miner[sample_id].append(loss)
        if len(hard_miner[sample_id]) > 10:
            hard_miner[sample_id].pop(0)
    
    def _compute_sample_loss_enhanced(self, sample: Dict, weights: Dict[str, float]) -> torch.Tensor:
        """增强的样本损失计算"""
        edit_id = sample['edit_id']
        prompt = sample['prompt']
        subject_positions = sample['subject_positions']
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        if sample['type'] == 'forward':
            target = sample['target']
            old_target = sample['old_target']
            is_rewrite = sample.get('is_rewrite', False)
            
            if target:
                edit_loss = self._compute_edit_loss_enhanced(
                    edit_id, prompt, target, old_target, subject_positions
                )
                if is_rewrite:
                    edit_loss = edit_loss * self.hparams.rewrite_loss_scale
                total_loss += weights['edit'] * edit_loss
            
            if old_target and self.hparams.w_suppress > 0:
                suppress_loss = self._compute_suppress_loss_fast(
                    edit_id, prompt, old_target, subject_positions
                )
                total_loss += weights['suppress'] * suppress_loss
            
            total_loss += weights['ortho'] * self.edit_module.compute_orthogonality_loss()
            total_loss += 0.5 * self.edit_module.compute_norm_constraint_loss(max_norm=2.5)
            
        elif sample['type'] == 'backward':
            local_loss = self._compute_local_loss_fast(
                edit_id, prompt, subject_positions
            )
            total_loss += weights['local'] * local_loss
        
        return total_loss
    
    def _compute_edit_loss_enhanced(
        self, edit_id: int, prompt: str, target: str,
        old_target: Optional[str], subject_positions: List[int]
    ) -> torch.Tensor:
        """增强的编辑损失"""
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)['input_ids'])
        
        if not target_tokens:
            return torch.tensor(0.1, device=self.device)

        self.injector.inject(self.model, edit_id, self.edit_module, subject_positions)
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(**inputs).logits[0]
            
            loss = 0.0
            for i, token_id in enumerate(target_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    if self.hparams.use_focal_loss:
                        loss += self._focal_cross_entropy(
                            logits[pos], token_id, gamma=self.hparams.focal_gamma
                        )
                    else:
                        loss += F.cross_entropy(
                            logits[pos].unsqueeze(0), 
                            torch.tensor([token_id], device=self.device)
                        )
            
            loss = loss / len(target_tokens)
            
            # 对比学习损失
            if self.hparams.use_contrastive_loss and old_target:
                old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
                if old_tokens and prompt_len - 1 < logits.shape[0]:
                    contrastive = self._compute_contrastive_loss(
                        logits[prompt_len - 1], target_tokens[0], old_tokens[0]
                    )
                    loss += self.hparams.w_contrastive * contrastive
            
            return loss
        finally:
            self.injector.clear()
    
    def _focal_cross_entropy(self, logits: torch.Tensor, target_id: int, gamma: float = 2.0) -> torch.Tensor:
        """Focal Loss"""
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        target_prob = probs[target_id]
        focal_weight = (1 - target_prob) ** gamma
        loss = -focal_weight * log_probs[target_id]
        return loss
    
    def _compute_contrastive_loss(
        self, logits: torch.Tensor, target_new_id: int, target_old_id: int
    ) -> torch.Tensor:
        """对比学习损失"""
        logit_new = logits[target_new_id]
        logit_old = logits[target_old_id]
        margin = self.hparams.contrastive_margin
        loss = F.relu(margin - (logit_new - logit_old) / self.hparams.contrastive_temperature)
        return loss
    
    def _compute_suppress_loss_fast(self, edit_id, prompt, old_target, subject_positions):
        """抑制旧答案的损失"""
        full_text = f"{prompt} {old_target}"
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)['input_ids'])
        
        if not old_tokens:
            return torch.tensor(0.0, device=self.device)
        
        self.injector.inject(self.model, edit_id, self.edit_module, subject_positions)
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(**inputs).logits[0]
            
            log_probs = []
            for i, token_id in enumerate(old_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    log_probs.append(F.log_softmax(logits[pos], dim=-1)[token_id])
            
            if not log_probs:
                return torch.tensor(0.0, device=self.device)
            
            prob_old = torch.exp(sum(log_probs) / len(log_probs))
            return -torch.log(1.0 - prob_old + 1e-10)
        finally:
            self.injector.clear()

    def _compute_local_loss_fast(self, edit_id, prompt, subject_positions):
        """局部性损失(KL散度)"""
        return self.utils.compute_kl_divergence(
            prompt, subject_positions, self.edit_module, edit_id, self.injector
        )

    def inference(self, prompt, max_new_tokens=10, verbose=None):
        if verbose is None:
            verbose = self.hparams.verbose
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)
            
        edit_id = self.router.route(prompt, prompt_emb)
        
        if edit_id is not None:
            req = self.edits_registry.get(edit_id)
            if req:
                pos = self.utils.find_subject_positions(
                    prompt, req['subject'], verbose=False, add_special_tokens=True
                )
                if pos:
                    self.injector.inject(self.model, edit_id, self.edit_module, pos)
                    if verbose:
                        print(f"[TRIGGER] edit active #{edit_id}")
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        self.injector.clear()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def save(self, path):
        torch.save({
            'edit_module': self.edit_module.state_dict(),
            'edits_registry': self.edits_registry,
            'hparams': self.hparams
        }, path)
        if self.hparams.verbose:
            print(f"[SUCCESS] 保存至 {path}")
        
    def load(self, path):
        d = torch.load(path)
        num_edits = len(d['edits_registry'])
        self.edit_module = self._create_edit_module(num_edits)
        self.edit_module.load_state_dict(d['edit_module'])
        self.edits_registry = d['edits_registry']
        if self.hparams.verbose:
            print(f"[SUCCESS] 加载完成")