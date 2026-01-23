"""
tokenedit_main.py - 完整极速版 (Subject Guard + SOTA Layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import random

from .tokenedit_hparams import TokenEditHyperParams
from .edit_token_module import EditTokenModule
from .prompt_router import PromptRouter
from .layer_injector import LayerInjector
from .prompt_closure import PromptClosureGenerator
from .tokenedit_utils import TokenEditUtils


class TokenEditEditor:
    """TokenEdit知识编辑器 - 极速训练版"""
    
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
        
        # 启用混合精度训练
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
            print("[SUCCESS] TokenEditEditor init complete")
            print(f"  模型: {hparams.model_name}")
            print(f"  目标层: {hparams.target_layers}")
            print(f"  设备: {self.device}")

    def _get_default_target_layers(self, model) -> List[int]:
        """根据模型自动设置目标层 (已修正为 SOTA 后层策略)"""
        model_name = model.config._name_or_path.lower()

        if hasattr(model.config, 'n_layer'):
            num_layers = model.config.n_layer
        elif hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        else:
            num_layers = 48

        # 针对 GPT2-XL 使用后层 (35-39)，显著提升 Argmax 成功率
        if 'gpt2' in model_name or 'gpt-2' in model_name:
            if 'xl' in model_name:
                return [35, 36, 37, 38, 39]
            elif 'large' in model_name or 'gpt2-large' in model_name:
                return [28, 29, 30, 31, 32]
            elif 'medium' in model_name or 'gpt2-medium' in model_name:
                return [18, 19, 20, 21, 22]
            else:
                return [8, 9, 10, 11]
        elif 'llama' in model_name:
            return list(range(max(0, num_layers - 5), num_layers))
        elif 'pythia' in model_name:
            return list(range(max(0, num_layers - 5), num_layers))
        else:
            return list(range(max(0, num_layers - 5), num_layers))
    
    def apply_edits(self, requests: List[Dict]) -> Dict:
        """应用批量编辑"""
        num_edits = len(requests)
        
        # 配置检查打印
        print("\n" + "-"*50)
        print("[CONFIG CHECK] 当前生效的参数:")
        print(f"  > w_edit: {self.hparams.w_edit}")
        print(f"  > threshold: {self.hparams.routing_threshold}")
        print(f"  > init_method: {self.hparams.token_init_method}")
        print(f"  > target_layers: {self.hparams.target_layers}")
        print("-"*50 + "\n")
        
        if self.hparams.verbose:
            print(f"开始编辑 {num_edits} 个知识点")
        
        # 1. 初始化EditToken模块
        self.edit_module = EditTokenModule(
            hidden_size=self.model.config.hidden_size,
            num_edits=num_edits,
            hparams=self.hparams
        ).to(self.device)
        
        # 2. 生成Prompt闭包训练数据
        train_data = []
        self.edits_registry = {} # 重置注册表

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
            # 存入注册表，供 Subject Guard 使用
            self.edits_registry[i] = req
        
        if self.hparams.verbose:
            print(f"  生成了 {len(train_data)} 个Prompt闭包")
        
        # 3. 训练EditToken
        stats = self._train_tokens_fast(train_data)
        
        # ============================================================
        # [Subject Guard] 植入主体卫士，物理拦截邻居样本
        # ============================================================
        if not hasattr(self.router, "_original_route"):
            self.router._original_route = self.router.route
            
        original_route = self.router._original_route
        registry = self.edits_registry
        utils = self.utils
        
        def guarded_route(prompt: str, prompt_emb=None):
            # 1. Original route (embedding check)
            candidate_id = original_route(prompt, prompt_emb)
            if candidate_id is None:
                return None

            # 2. Subject guard using token positions
            if candidate_id in registry:
                subject = registry[candidate_id]['subject']
                positions = utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if not positions:
                    return None
            return candidate_id

        # 替换 Router 的方法
        self.router.route = guarded_route
        print("[Security] Subject Guard 已激活 (Specificity 保护开启)")
        # ============================================================

        # 4. 完成
        if self.hparams.verbose:
            print("\n编辑完成")
            if stats['losses']:
                print(f"  最终损失: {stats['losses'][-1]:.4f}")
        
        return {
            'model': self.model,
            'edit_module': self.edit_module,
            'router': self.router,
            'injector': self.injector,
            'stats': stats
        }
    
    def _train_tokens_fast(self, train_data: List[Dict]) -> Dict:
        """Fast training with smart init."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        if self.hparams.token_init_method == "target_smart":
            print("  [Init] applying Smart Initialization...")
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

                    t_ids = self.tokenizer.encode(" " + target_word.strip(), add_special_tokens=False)
                    if not t_ids:
                        t_ids = self.tokenizer.encode(target_word.strip(), add_special_tokens=False)

                    if t_ids:
                        target_emb = wte(torch.tensor(t_ids, device=self.device)).mean(dim=0)
                        if not self.hparams.use_low_rank:
                            self.edit_module.v_new.data[idx] = target_emb.clone()
                            init_count += 1
            print(f"  [Init] initialized {init_count}/{len(train_data)} vectors")

        optimizer = torch.optim.AdamW(
            self.edit_module.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.num_epochs
            )
        else:
            scheduler = None

        stats = {'losses': []}

        def _rescale_grads(scale_factor: float) -> None:
            if scale_factor == 1.0:
                return
            for p in self.edit_module.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale_factor)

        all_samples = []
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
                        'edit_id': edit_id,
                        'prompt': prompt,
                        'type': 'forward',
                        'is_rewrite': prompt in rewrite_set,
                        'subject_positions': subject_positions,
                        'target': closure.get('targets_forward', ''),
                        'old_target': closure.get('targets_backward', '')
                    })

            for prompt in closure.get('prompts_backward', []):
                subject_positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if subject_positions:
                    all_samples.append({
                        'edit_id': edit_id,
                        'prompt': prompt,
                        'type': 'backward',
                        'subject_positions': subject_positions,
                        'target': None,
                        'old_target': None
                    })

        desired_batch_size = min(64, max(32, len(all_samples) // 10))
        micro_batch_size = min(16, desired_batch_size)
        grad_accum_steps = max(1, int(np.ceil(desired_batch_size / micro_batch_size)))

        best_loss = None
        stale_epochs = 0
        for epoch in tqdm(range(self.hparams.num_epochs), desc="Training"):
            epoch_loss = 0.0
            num_batches = 0

            indices = np.random.permutation(len(all_samples))
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            effective_accum = grad_accum_steps * micro_batch_size

            for batch_start in range(0, len(all_samples), micro_batch_size):
                batch_end = min(batch_start + micro_batch_size, len(all_samples))
                batch_indices = indices[batch_start:batch_end]
                batch_samples = [all_samples[i] for i in batch_indices]

                for sample in batch_samples:
                    sample_loss = self._compute_batch_loss_fast([sample])
                    if sample_loss is not None and sample_loss.requires_grad:
                        scaled_loss = sample_loss / effective_accum
                        if self.use_amp:
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        accum_loss += sample_loss.item()
                        accum_count += 1

                step_now = (accum_count % effective_accum == 0) or (batch_end == len(all_samples))
                if step_now and accum_count > 0:
                    scale_factor = float(effective_accum) / float(accum_count)
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        _rescale_grads(scale_factor)
                        torch.nn.utils.clip_grad_norm_(self.edit_module.parameters(), self.hparams.gradient_clip)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        _rescale_grads(scale_factor)
                        torch.nn.utils.clip_grad_norm_(self.edit_module.parameters(), self.hparams.gradient_clip)
                        optimizer.step()

                    optimizer.zero_grad()
                    epoch_loss += accum_loss / accum_count
                    num_batches += 1
                    accum_loss = 0.0
                    accum_count = 0

                    with torch.no_grad():
                        self.edit_module.alpha.data.clamp_(0.0, 2.0)
                        self.edit_module.beta.data.clamp_(-2.0, 2.0)

            if scheduler:
                scheduler.step()
            if num_batches > 0:
                epoch_avg = epoch_loss / num_batches
                stats['losses'].append(epoch_avg)

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
                                print(f"[EarlyStop] no improvement for {patience} epochs")
                            break

        return stats

    def _compute_batch_loss_fast(self, batch_samples: List[Dict]) -> torch.Tensor:
        """Compute batch loss."""
        if len(batch_samples) == 0:
            return None

        total_loss = torch.tensor(0.0, device=self.device)

        forward_samples = [s for s in batch_samples if s['type'] == 'forward']
        backward_samples = [s for s in batch_samples if s['type'] == 'backward']

        if forward_samples:
            for sample in forward_samples:
                edit_id = sample['edit_id']
                prompt = sample['prompt']
                target = sample['target']
                old_target = sample['old_target']
                subject_positions = sample['subject_positions']

                if target:
                    edit_loss = self._compute_edit_loss_fast(edit_id, prompt, target, subject_positions)
                    if sample.get('is_rewrite'):
                        edit_loss = edit_loss * self.hparams.rewrite_loss_scale
                    total_loss += self.hparams.w_edit * edit_loss

                if old_target:
                    suppress_loss = self._compute_suppress_loss_fast(edit_id, prompt, old_target, subject_positions)
                    total_loss += self.hparams.w_suppress * suppress_loss

                total_loss += self.hparams.w_ortho * self.edit_module.compute_orthogonality_loss()
                total_loss += 1.0 * self.edit_module.compute_norm_constraint_loss(max_norm=2.0)

        if backward_samples:
            for sample in backward_samples:
                local_loss = self._compute_local_loss_fast(
                    sample['edit_id'], sample['prompt'], sample['subject_positions']
                )
                total_loss += self.hparams.w_local * local_loss

        return total_loss / len(batch_samples)


    def _compute_edit_loss_fast(self, edit_id, prompt, target, subject_positions):
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)['input_ids'])
        
        if not target_tokens: return torch.tensor(0.1, device=self.device)

        self.injector.inject(self.model, edit_id, self.edit_module, subject_positions)
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(**inputs).logits[0]
            loss = 0.0
            for i, token_id in enumerate(target_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    loss += F.cross_entropy(logits[pos].unsqueeze(0), torch.tensor([token_id], device=self.device))
            return loss / len(target_tokens)
        finally:
            self.injector.clear()
    
    def _compute_suppress_loss_fast(self, edit_id, prompt, old_target, subject_positions):
        full_text = f"{prompt} {old_target}"
        inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(self.device)
        old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
        prompt_len = len(self.tokenizer(prompt, add_special_tokens=True)['input_ids'])
        
        if not old_tokens: return torch.tensor(0.0, device=self.device)
        
        self.injector.inject(self.model, edit_id, self.edit_module, subject_positions)
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(**inputs).logits[0]
            log_probs = []
            for i, token_id in enumerate(old_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    log_probs.append(F.log_softmax(logits[pos], dim=-1)[token_id])
            
            if not log_probs: return torch.tensor(0.0, device=self.device)
            prob_old = torch.exp(sum(log_probs) / len(log_probs))
            return -torch.log(1.0 - prob_old + 1e-10)
        finally:
            self.injector.clear()
    
    def _compute_local_loss_fast(self, edit_id, prompt, subject_positions):
        return self.utils.compute_kl_divergence(prompt, subject_positions, self.edit_module, edit_id, self.injector)

    def inference(self, prompt, max_new_tokens=10, verbose=None):
        if verbose is None: verbose = self.hparams.verbose
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
                    if verbose: print(f"[TRIGGER] edit active #{edit_id}")
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        self.injector.clear()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
    
    def save(self, path): 
        torch.save({'edit_module': self.edit_module.state_dict(), 'edits_registry': self.edits_registry, 'hparams': self.hparams}, path)
        if self.hparams.verbose: print(f"[SUCCESS] 保存至 {path}")
        
    def load(self, path): 
        d = torch.load(path)
        self.edit_module = EditTokenModule(self.model.config.hidden_size, len(d['edits_registry']), self.hparams).to(self.device)
        self.edit_module.load_state_dict(d['edit_module'])
        self.edits_registry = d['edits_registry']
        if self.hparams.verbose: print(f"[SUCCESS] 加载完成")
