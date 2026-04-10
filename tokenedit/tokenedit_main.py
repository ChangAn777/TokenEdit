"""
tokenedit_main.py - v2 AlphaEdit 对标版

核心改动:
1. _smart_initialize():    词嵌入初始化 → 模型隐藏状态初始化（ROME 风格）
2. _build_nullspace():     apply_edits() 开始时计算零空间投影矩阵并注入 edit_module
3. _compute_sample_loss(): 增加 nullspace 软约束损失项
4. evaluate():             新增标准三指标评估（efficacy / generalization / specificity）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import random

from .tokenedit_hparams import TokenEditHyperParams
from .edit_token_module import (
    EditTokenModule,
    EnhancedEditTokenModule,
    AdaptiveEditTokenModule,
    HybridEditTokenModule,
)
from .layer_injector import (
    LayerInjector,
    EnhancedLayerInjector,
    AttentionGuidedInjector,
    AdaptiveInjector,
)
from .prompt_router import PromptRouter
from .prompt_closure import PromptClosureGenerator
from .tokenedit_utils import TokenEditUtils


class TokenEditEditor:
    """TokenEdit 知识编辑器 - AlphaEdit 对标版"""

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
        self.injector = self._create_injector(hparams)
        self.closure_gen = PromptClosureGenerator()
        self.utils = TokenEditUtils(model, tokenizer)
        self.edits_registry: Dict[int, Dict] = {}

        if hparams.verbose:
            print("[SUCCESS] TokenEditEditor (v2) 初始化完成")
            print(f"  模型: {hparams.model_name}")
            print(f"  目标层: {hparams.target_layers}")
            for k, v in hparams.get_active_features().items():
                print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    def _create_injector(self, hparams):
        if hparams.injector_type == "enhanced":
            return EnhancedLayerInjector(
                hparams.target_layers,
                use_progressive=hparams.use_progressive_injection,
            )
        elif hparams.injector_type == "attention":
            return AttentionGuidedInjector(hparams.target_layers)
        elif hparams.injector_type == "adaptive":
            return None  # 在 apply_edits 中创建（需要 num_edits）
        else:
            return LayerInjector(hparams.target_layers)

    def _create_edit_module(self, num_edits: int) -> nn.Module:
        hidden_size = self.model.config.hidden_size
        t = self.hparams.edit_module_type
        if t == "enhanced":
            module = EnhancedEditTokenModule(hidden_size, num_edits, self.hparams)
        elif t == "adaptive":
            module = AdaptiveEditTokenModule(hidden_size, num_edits, self.hparams)
        elif t == "hybrid":
            module = HybridEditTokenModule(hidden_size, num_edits, self.hparams)
        else:
            module = EditTokenModule(hidden_size, num_edits, self.hparams)
        return module.to(self.device)

    def _get_optimal_target_layers(self, model) -> List[int]:
        model_name = model.config._name_or_path.lower()
        if hasattr(model.config, "n_layer"):
            n = model.config.n_layer
        elif hasattr(model.config, "num_hidden_layers"):
            n = model.config.num_hidden_layers
        else:
            n = 48

        if "gpt2" in model_name:
            if "xl" in model_name:   return [30, 31, 32, 33, 34]
            if "large" in model_name: return [24, 25, 26, 27, 28]
            if "medium" in model_name: return [16, 17, 18, 19, 20]
            return [7, 8, 9, 10]
        elif "llama" in model_name:
            return list(range(max(0, n - 8), n - 3))
        return list(range(max(0, n - 8), n - 3))

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def apply_edits(self, requests: List[Dict]) -> Dict:
        """
        应用批量编辑。

        流程:
          1. 创建 EditModule
          2. [NEW] 计算零空间投影矩阵（AlphaEdit 核心）
          3. Smart 初始化（ROME 风格，用模型隐藏状态而非词嵌入）
          4. 生成 Prompt 闭包训练数据
          5. 增强训练循环
          6. [NEW] 可选：自动评估 efficacy / generalization / specificity
        """
        num_edits = len(requests)

        print("\n" + "-" * 60)
        print("[CONFIG - TokenEdit v2]")
        for k, v in self.hparams.get_active_features().items():
            print(f"  {k}: {v}")
        print("-" * 60 + "\n")

        # ---- 1. 创建 EditModule ----
        self.edit_module = self._create_edit_module(num_edits)

        if self.hparams.injector_type == "adaptive":
            self.injector = AdaptiveInjector(
                self.hparams.target_layers,
                num_edits=num_edits,
                use_progressive=self.hparams.use_progressive_injection,
                use_attention=self.hparams.use_attention_injection,
                use_dynamic_layers=self.hparams.use_dynamic_layers,
            )

        # ---- 2. [NEW] 零空间投影 ----
        if self.hparams.use_nullspace_projection:
            self._build_nullspace_projections(requests)

        # ---- 3. 生成 Prompt 闭包 & 注册路由 ----
        train_data: List[Dict] = []
        self.edits_registry = {}

        for i, req in enumerate(requests):
            closure = self.closure_gen.generate_from_dataset(
                rewrite_prompt=req["prompt"],
                subject=req["subject"],
                target_new=req["target_new"],
                target_true=req["target_true"],
                paraphrase_prompts=req.get("paraphrase_prompts", []),
                neighborhood_prompts=req.get("neighborhood_prompts", []),
                num_paraphrase=self.hparams.num_paraphrase,
            )
            train_data.append({"edit_id": i, "closure": closure, "request": req})
            self.router.register_edit(
                i,
                req["subject"],
                req.get("relation_id", req.get("relation", "unknown")),
                req.get("prompt"),
                req.get("paraphrase_prompts", []),
            )
            self.edits_registry[i] = req

        # ---- 4. 训练 ----
        stats = self._train_tokens_enhanced(train_data)

        # ---- Subject Guard ----
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
                subject = registry[candidate_id]["subject"]
                positions = utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if not positions:
                    return None
            return candidate_id

        self.router.route = guarded_route

        # ---- 5. [NEW] 自动评估 ----
        if self.hparams.eval_after_edit:
            eval_results = self.evaluate(requests)
            stats["eval"] = eval_results
            if self.hparams.verbose:
                print("\n[EVAL] 编辑效果评估:")
                print(f"  Efficacy:       {eval_results['efficacy_rate']:.3f}")
                print(f"  Generalization: {eval_results['generalization_prob']:.3f}")
                print(f"  Specificity KL: {eval_results['specificity_kl']:.4f}")

        if self.hparams.verbose:
            print("\n[DONE] 编辑完成")
            if stats["losses"]:
                print(f"  最终损失: {stats['losses'][-1]:.4f}")
                print(f"  最低损失: {min(stats['losses']):.4f}")

        return {
            "model": self.model,
            "edit_module": self.edit_module,
            "router": self.router,
            "injector": self.injector,
            "stats": stats,
        }

    # ------------------------------------------------------------------
    # [NEW] 零空间投影矩阵构建（AlphaEdit 核心）
    # ------------------------------------------------------------------

    def _build_nullspace_projections(self, requests: List[Dict]) -> None:
        """
        计算零空间投影矩阵并注入 edit_module。

        上下文来源优先级:
          1. 所有 request 的 neighborhood_prompts（最相关的无关样本）
          2. 补充通用文本（保证最少 20 条）

        每个目标层独立计算一个 P_null，最终在 edit_module 中取均值。
        """
        print("[NullSpace] 正在计算零空间投影矩阵...")

        context_texts: List[str] = []
        for req in requests:
            for nb in req.get("neighborhood_prompts", []):
                text = nb.get("prompt", "") if isinstance(nb, dict) else str(nb)
                if text.strip():
                    context_texts.append(text)

        # 补充通用知识句（防止邻域样本不足）
        _FALLBACK_CONTEXTS = [
            "The president of the United States is",
            "The capital of Germany is Berlin",
            "Water is composed of hydrogen and oxygen",
            "The speed of light is approximately 300,000 km/s",
            "Shakespeare wrote Hamlet",
            "The Eiffel Tower is located in Paris",
            "Albert Einstein developed the theory of relativity",
            "The Amazon River is the largest river in South America",
            "The human body has 206 bones",
            "Mount Everest is the tallest mountain in the world",
            "The Great Wall of China was built over many centuries",
            "Leonardo da Vinci painted the Mona Lisa",
            "The Pacific Ocean is the largest ocean on Earth",
            "DNA carries genetic information",
            "The sun is a star at the center of our solar system",
            "Isaac Newton formulated the laws of motion",
            "The French Revolution began in 1789",
            "Gold is a chemical element with symbol Au",
            "The Nile River flows through Egypt",
            "Beethoven composed nine symphonies",
        ]
        if len(context_texts) < 20:
            context_texts.extend(_FALLBACK_CONTEXTS)

        # 去重 & 截断
        seen = set()
        unique_contexts: List[str] = []
        for t in context_texts:
            if t not in seen:
                seen.add(t)
                unique_contexts.append(t)
        unique_contexts = unique_contexts[: self.hparams.nullspace_context_samples]

        projections: Dict[int, torch.Tensor] = {}
        for layer_idx in self.hparams.target_layers:
            P = self.utils.compute_nullspace_projection_matrix(
                unique_contexts,
                layer=layer_idx,
                rank=self.hparams.nullspace_rank,
                reg=self.hparams.nullspace_reg,
            )
            projections[layer_idx] = P

        self.edit_module.set_nullspace_projections(projections)
        print(
            f"[NullSpace] 完成，使用 {len(unique_contexts)} 条上下文，"
            f"覆盖 {len(projections)} 层"
        )

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------

    def _train_tokens_enhanced(self, train_data: List[Dict]) -> Dict:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Smart 初始化（使用模型隐藏状态，v2 升级）
        if self.hparams.token_init_method == "target_smart":
            self._smart_initialize(train_data)

        optimizer = torch.optim.AdamW(
            self.edit_module.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        if self.hparams.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.plateau_factor,
                patience=self.hparams.plateau_patience,
                min_lr=self.hparams.plateau_min_lr,  # 已修复为 1e-5
                verbose=False,
            )
        elif self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.num_epochs
            )
        else:
            scheduler = None

        stats: Dict = {"losses": [], "lr_history": []}
        all_samples = self._prepare_training_samples(train_data)
        hard_miner: Optional[Dict] = {} if self.hparams.use_hard_mining else None

        desired_batch = min(128, max(64, len(all_samples) // 8))
        micro_batch = min(16, desired_batch)
        grad_accum = max(1, int(np.ceil(desired_batch / micro_batch)))

        if self.hparams.verbose:
            print(f"  [Training] Samples: {len(all_samples)}, "
                  f"Batch: {desired_batch}, GradAccum: {grad_accum}")

        best_loss = None
        stale_epochs = 0

        for epoch in tqdm(range(self.hparams.num_epochs), desc="Training"):
            active_samples = (
                self._filter_by_curriculum(all_samples, epoch)
                if self.hparams.use_curriculum
                else all_samples
            )
            current_weights = (
                self._get_adaptive_weights(epoch)
                if self.hparams.use_adaptive_weights
                else {
                    "edit": self.hparams.w_edit,
                    "suppress": self.hparams.w_suppress,
                    "ortho": self.hparams.w_ortho,
                    "local": self.hparams.w_local,
                    "contrastive": self.hparams.w_contrastive,
                    "nullspace": self.hparams.w_nullspace,
                }
            )

            epoch_loss = 0.0
            num_batches = 0
            indices = np.random.permutation(len(active_samples))
            optimizer.zero_grad()
            accum_loss = 0.0
            accum_count = 0
            effective_accum = grad_accum * micro_batch

            for batch_start in range(0, len(active_samples), micro_batch):
                batch_end = min(batch_start + micro_batch, len(active_samples))
                batch_indices = indices[batch_start:batch_end]
                batch_samples = [active_samples[i] for i in batch_indices]

                for sample in batch_samples:
                    sample_loss = self._compute_sample_loss_enhanced(
                        sample, current_weights
                    )
                    if sample_loss is None or not sample_loss.requires_grad:
                        continue

                    if self.hparams.use_hard_mining:
                        sample_id = sample.get("sample_id", hash(sample["prompt"]))
                        difficulty = self._get_sample_difficulty(hard_miner, sample_id)
                        w = 1.0 + difficulty * (self.hparams.hard_boost_factor - 1.0)
                        sample_loss = sample_loss * w

                    scaled = sample_loss / effective_accum
                    if self.use_amp:
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()

                    accum_loss += sample_loss.item()
                    accum_count += 1

                    if self.hparams.use_hard_mining:
                        self._update_hard_miner(hard_miner, sample_id, sample_loss.item())

                step_now = (accum_count % effective_accum == 0) or (
                    batch_end == len(active_samples)
                )
                if step_now and accum_count > 0:
                    scale_factor = float(effective_accum) / float(accum_count)
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        for p in self.edit_module.parameters():
                            if p.grad is not None:
                                p.grad.mul_(scale_factor)
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(), self.hparams.gradient_clip
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        for p in self.edit_module.parameters():
                            if p.grad is not None:
                                p.grad.mul_(scale_factor)
                        torch.nn.utils.clip_grad_norm_(
                            self.edit_module.parameters(), self.hparams.gradient_clip
                        )
                        optimizer.step()

                    optimizer.zero_grad()
                    epoch_loss += accum_loss / accum_count
                    num_batches += 1
                    accum_loss = 0.0
                    accum_count = 0

                    # 门控约束
                    with torch.no_grad():
                        if hasattr(self.edit_module, "alpha"):
                            self.edit_module.alpha.data.clamp_(0.0, 2.5)
                        if hasattr(self.edit_module, "beta"):
                            self.edit_module.beta.data.clamp_(-2.0, 2.0)

            if num_batches > 0:
                epoch_avg = epoch_loss / num_batches
                stats["losses"].append(epoch_avg)
                stats["lr_history"].append(optimizer.param_groups[0]["lr"])

                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_avg)
                    else:
                        scheduler.step()

                # Early stopping
                patience = self.hparams.early_stop_patience
                if patience and patience > 0:
                    improved = best_loss is None or (
                        best_loss - epoch_avg > self.hparams.early_stop_min_delta
                    )
                    if improved:
                        best_loss = epoch_avg
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= patience:
                            if self.hparams.verbose:
                                print(f"\n[EarlyStop] {patience} epochs 无改善，停止")
                            break

        return stats

    # ------------------------------------------------------------------
    # [NEW] Smart 初始化升级：使用模型隐藏状态（ROME 风格）
    # ------------------------------------------------------------------

    def _smart_initialize(self, train_data: List[Dict]) -> None:
        """
        ROME 风格 Smart 初始化。

        改进说明:
          原代码使用 wte（词嵌入矩阵）作为 v_new 初始值，
          这等价于只使用了 token 的静态表示，缺少位置、上下文信息。

          新方案：运行一次模型前向传播，提取目标层在 subject 位置的
          实际隐藏状态作为初始值，信息密度更高，训练收敛更快。

          退化方案：若 subject 位置定位失败，回退到词嵌入。
        """
        print("  [Init] 正在进行 ROME 风格 Smart 初始化...")
        wte = self.model.get_input_embeddings()
        init_count = 0

        with torch.no_grad():
            for data in train_data:
                idx = data["edit_id"]
                req = data["request"]
                subject = req["subject"]
                target = req["target_new"]

                # 构造已填入 subject 的完整 prompt
                raw_prompt = req.get("prompt", "")
                if "{}" in raw_prompt:
                    prompt = raw_prompt.replace("{}", subject)
                elif "{" in raw_prompt:
                    # 兼容 f-string 格式
                    prompt = raw_prompt.format(subject)
                else:
                    prompt = raw_prompt

                # 优先：模型隐藏状态（ROME 风格）
                target_vec = self.utils.compute_target_hidden_state(
                    prompt=prompt,
                    target=target,
                    subject=subject,
                    target_layers=self.hparams.target_layers,
                )

                # 退化：词嵌入
                if target_vec is None:
                    t_ids = None
                    for prefix in [" ", ""]:
                        ids = self.tokenizer.encode(
                            prefix + target.strip(), add_special_tokens=False
                        )
                        if ids:
                            t_ids = ids
                            break
                    if t_ids is None:
                        continue
                    target_vec = wte(
                        torch.tensor(t_ids, device=self.device)
                    ).mean(dim=0)
                    print(f"  [Init] edit #{idx}: 回退至词嵌入初始化")

                noise = torch.randn_like(target_vec) * self.hparams.token_init_std

                if self.hparams.use_multiscale and hasattr(
                    self.edit_module, "v_new_coarse"
                ):
                    self.edit_module.v_new_coarse.data[idx] = target_vec.clone()
                    self.edit_module.v_new_fine.data[idx] = noise * 0.1
                elif hasattr(self.edit_module, "v_new"):
                    self.edit_module.v_new.data[idx] = target_vec.clone() + noise

                init_count += 1

        print(f"  [Init] 完成：{init_count}/{len(train_data)} 个向量已初始化")

    # ------------------------------------------------------------------
    # 训练样本准备
    # ------------------------------------------------------------------

    def _prepare_training_samples(self, train_data: List[Dict]) -> List[Dict]:
        all_samples: List[Dict] = []
        sample_id = 0

        for data in train_data:
            edit_id = data["edit_id"]
            closure = data["closure"]
            req = data["request"]
            subject = req["subject"]
            rewrite_set = set(closure.get("rewrite_prompts", []))

            for prompt in closure.get("prompts_forward", []):
                positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if positions:
                    all_samples.append({
                        "sample_id": sample_id,
                        "edit_id": edit_id,
                        "prompt": prompt,
                        "type": "forward",
                        "is_rewrite": prompt in rewrite_set,
                        "subject_positions": positions,
                        "target": closure.get("targets_forward", ""),
                        "old_target": closure.get("targets_backward", ""),
                    })
                    sample_id += 1

            for prompt in closure.get("prompts_backward", []):
                positions = self.utils.find_subject_positions(
                    prompt, subject, verbose=False, add_special_tokens=True
                )
                if positions:
                    all_samples.append({
                        "sample_id": sample_id,
                        "edit_id": edit_id,
                        "prompt": prompt,
                        "type": "backward",
                        "subject_positions": positions,
                        "target": None,
                        "old_target": None,
                    })
                    sample_id += 1

        return all_samples

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def _compute_sample_loss_enhanced(
        self, sample: Dict, weights: Dict[str, float]
    ) -> Optional[torch.Tensor]:
        edit_id = sample["edit_id"]
        prompt = sample["prompt"]
        subject_positions = sample["subject_positions"]
        total_loss = torch.tensor(0.0, device=self.device)

        if sample["type"] == "forward":
            target = sample["target"]
            old_target = sample["old_target"]
            is_rewrite = sample.get("is_rewrite", False)

            if target:
                edit_loss = self._compute_edit_loss_enhanced(
                    edit_id, prompt, target, old_target, subject_positions
                )
                if is_rewrite:
                    edit_loss = edit_loss * self.hparams.rewrite_loss_scale
                total_loss = total_loss + weights["edit"] * edit_loss

            if old_target and self.hparams.w_suppress > 0:
                suppress_loss = self._compute_suppress_loss_fast(
                    edit_id, prompt, old_target, subject_positions
                )
                total_loss = total_loss + weights["suppress"] * suppress_loss

            total_loss = total_loss + weights["ortho"] * self.edit_module.compute_orthogonality_loss()
            total_loss = total_loss + 0.5 * self.edit_module.compute_norm_constraint_loss(max_norm=2.5)

            # [NEW] 零空间软约束
            if self.hparams.use_nullspace_projection and hasattr(
                self.edit_module, "compute_nullspace_loss"
            ):
                ns_loss = self.edit_module.compute_nullspace_loss()
                total_loss = total_loss + weights.get("nullspace", self.hparams.w_nullspace) * ns_loss

        elif sample["type"] == "backward":
            local_loss = self._compute_local_loss_fast(
                edit_id, prompt, subject_positions
            )
            total_loss = total_loss + weights["local"] * local_loss

        return total_loss

    def _compute_edit_loss_enhanced(
        self,
        edit_id: int,
        prompt: str,
        target: str,
        old_target: Optional[str],
        subject_positions: List[int],
    ) -> torch.Tensor:
        """增强的编辑损失（含 Focal Loss + 对比损失）"""
        full_text = f"{prompt} {target}"
        inputs = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        target_tokens = self.tokenizer.encode(target, add_special_tokens=False)
        prompt_len = len(
            self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )

        if not target_tokens:
            return torch.tensor(0.1, device=self.device)

        self.injector.inject(self.model, edit_id, self.edit_module, subject_positions)
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(**inputs).logits[0]  # (seq_len, vocab)

            loss = torch.tensor(0.0, device=self.device)
            for i, token_id in enumerate(target_tokens):
                pos = prompt_len + i - 1
                if pos < logits.shape[0]:
                    if self.hparams.use_focal_loss:
                        loss = loss + self._focal_cross_entropy(
                            logits[pos], token_id, gamma=self.hparams.focal_gamma
                        )
                    else:
                        loss = loss + F.cross_entropy(
                            logits[pos].unsqueeze(0),
                            torch.tensor([token_id], device=self.device),
                        )

            loss = loss / len(target_tokens)

            if self.hparams.use_contrastive_loss and old_target:
                old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
                if old_tokens and prompt_len - 1 < logits.shape[0]:
                    contrastive = self._compute_contrastive_loss(
                        logits[prompt_len - 1], target_tokens[0], old_tokens[0]
                    )
                    loss = loss + self.hparams.w_contrastive * contrastive

            return loss
        finally:
            self.injector.clear()

    def _focal_cross_entropy(
        self, logits: torch.Tensor, target_id: int, gamma: float = 2.0
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        target_prob = torch.exp(log_probs[target_id])
        focal_weight = (1.0 - target_prob) ** gamma
        return -focal_weight * log_probs[target_id]

    def _compute_contrastive_loss(
        self,
        logits: torch.Tensor,
        target_new_id: int,
        target_old_id: int,
    ) -> torch.Tensor:
        logit_new = logits[target_new_id]
        logit_old = logits[target_old_id]
        margin = self.hparams.contrastive_margin
        return F.relu(
            margin - (logit_new - logit_old) / self.hparams.contrastive_temperature
        )

    def _compute_suppress_loss_fast(
        self, edit_id, prompt, old_target, subject_positions
    ) -> torch.Tensor:
        """Unlikelihood loss：降低旧答案的生成概率"""
        full_text = f"{prompt} {old_target}"
        inputs = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        ).to(self.device)
        old_tokens = self.tokenizer.encode(old_target, add_special_tokens=False)
        prompt_len = len(
            self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        )

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

    def _compute_local_loss_fast(
        self, edit_id, prompt, subject_positions
    ) -> torch.Tensor:
        """特异性损失：KL(P_edit || P_orig)，保证无关问题不受影响"""
        return self.utils.compute_kl_divergence(
            prompt, subject_positions, self.edit_module, edit_id, self.injector
        )

    # ------------------------------------------------------------------
    # [NEW] 标准评估接口
    # ------------------------------------------------------------------

    def evaluate(self, requests: List[Dict]) -> Dict:
        """
        计算标准知识编辑三指标:

          Efficacy       — P(target_new | rewrite_prompt) > P(target_true | ·) 的比例
                           衡量编辑是否成功改变了目标知识
          Generalization — P(target_new | paraphrase_prompt) 的均值
                           衡量编辑是否泛化到语义等价的表述
          Specificity    — KL(P_edit(·|nb) || P_orig(·|nb)) 的均值
                           衡量编辑是否"波及"到无关知识（越小越好）

        与 ROME / MEMIT / AlphaEdit 论文使用的评估协议一致。
        """
        results: Dict = {
            "efficacy":       [],
            "generalization": [],
            "specificity":    [],
        }
        self.model.eval()
        max_s = self.hparams.eval_max_samples

        for i, req in enumerate(requests):
            subject = req["subject"]
            target_new = req["target_new"]
            target_true = req["target_true"]
            prompt = (
                req["prompt"].replace("{}", subject)
                if "{}" in req["prompt"]
                else req["prompt"]
            )
            positions = self.utils.find_subject_positions(
                prompt, subject, add_special_tokens=True
            )
            if not positions:
                continue
            edit_id = i

            # ---- Efficacy ----
            p_new = self.utils.compute_token_probability(
                prompt, target_new, positions, self.edit_module, edit_id, self.injector
            )
            p_old = self.utils.compute_token_probability(
                prompt, target_true, positions, self.edit_module, edit_id, self.injector
            )
            results["efficacy"].append(float(p_new > p_old))

            # ---- Generalization ----
            para_scores: List[float] = []
            for para in req.get("paraphrase_prompts", [])[:max_s]:
                para_pos = self.utils.find_subject_positions(
                    para, subject, add_special_tokens=True
                )
                if para_pos:
                    p = self.utils.compute_token_probability(
                        para, target_new, para_pos,
                        self.edit_module, edit_id, self.injector,
                    )
                    para_scores.append(p)
            if para_scores:
                results["generalization"].append(float(np.mean(para_scores)))

            # ---- Specificity ----
            kl_scores: List[float] = []
            for nb in req.get("neighborhood_prompts", [])[:max_s]:
                nb_prompt = nb.get("prompt", "") if isinstance(nb, dict) else str(nb)
                if not nb_prompt.strip():
                    continue
                # 邻域样本的 subject 可能不同，用原 subject 尝试定位
                nb_pos = self.utils.find_subject_positions(
                    nb_prompt, subject, add_special_tokens=True
                )
                if not nb_pos:
                    # 若定位失败，取序列最后一个 token
                    nb_ids = self.tokenizer.encode(
                        nb_prompt, add_special_tokens=True
                    )
                    nb_pos = [len(nb_ids) - 1]
                kl = self.utils.compute_kl_divergence(
                    nb_prompt, nb_pos,
                    self.edit_module, edit_id, self.injector,
                ).item()
                kl_scores.append(kl)
            if kl_scores:
                results["specificity"].append(float(np.mean(kl_scores)))

        summary = {
            "efficacy_rate":       float(np.mean(results["efficacy"]))       if results["efficacy"]       else 0.0,
            "generalization_prob": float(np.mean(results["generalization"])) if results["generalization"] else 0.0,
            "specificity_kl":      float(np.mean(results["specificity"]))    if results["specificity"]    else 0.0,
        }
        summary.update(results)
        return summary

    # ------------------------------------------------------------------
    # 课程学习 & 自适应权重（保持原有逻辑）
    # ------------------------------------------------------------------

    def _filter_by_curriculum(
        self, all_samples: List[Dict], epoch: int
    ) -> List[Dict]:
        progress = epoch / self.hparams.num_epochs
        stages = self.hparams.curriculum_stages
        if progress < stages[0] / self.hparams.num_epochs:
            return [s for s in all_samples if s.get("is_rewrite", False)]
        elif progress < stages[1] / self.hparams.num_epochs:
            return [s for s in all_samples if s["type"] == "forward"]
        return all_samples

    def _get_adaptive_weights(self, epoch: int) -> Dict[str, float]:
        progress = epoch / self.hparams.num_epochs
        return {
            "edit":       self.hparams.w_edit * (1.0 - 0.1 * progress),
            "suppress":   self.hparams.w_suppress * (0.3 + 0.7 * progress),
            "ortho":      self.hparams.w_ortho * (progress if progress >= 0.3 else 0.1),
            "local":      self.hparams.w_local * (1.0 + progress if progress >= 0.5 else 0.5),
            "contrastive": self.hparams.w_contrastive,
            "nullspace":  self.hparams.w_nullspace,
        }

    def _get_sample_difficulty(
        self, hard_miner: Optional[Dict], sample_id: int
    ) -> float:
        if hard_miner is None or sample_id not in hard_miner:
            return 0.5
        losses = hard_miner[sample_id]
        if len(losses) < 3:
            return 0.5
        mean_loss = float(np.mean(losses[-5:]))
        std_loss = float(np.std(losses[-5:]))
        return float(np.clip((mean_loss + 0.5 * std_loss) / 2.0, 0.0, 1.0))

    def _update_hard_miner(
        self, hard_miner: Optional[Dict], sample_id: int, loss: float
    ) -> None:
        if hard_miner is None:
            return
        if sample_id not in hard_miner:
            hard_miner[sample_id] = []
        hard_miner[sample_id].append(loss)
        if len(hard_miner[sample_id]) > 10:
            hard_miner[sample_id].pop(0)

    # ------------------------------------------------------------------
    # 推理 & 持久化
    # ------------------------------------------------------------------

    def inference(
        self, prompt: str, max_new_tokens: int = 10, verbose: Optional[bool] = None
    ) -> str:
        if verbose is None:
            verbose = self.hparams.verbose
        self.model.eval()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            prompt_emb = outputs.hidden_states[-1].mean(dim=1)

        edit_id = self.router.route(prompt, prompt_emb)

        if edit_id is not None:
            req = self.edits_registry.get(edit_id)
            if req:
                pos = self.utils.find_subject_positions(
                    prompt, req["subject"], verbose=False, add_special_tokens=True
                )
                if pos:
                    self.injector.inject(
                        self.model, edit_id, self.edit_module, pos
                    )
                    if verbose:
                        print(f"[TRIGGER] 激活编辑 #{edit_id}")

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        self.injector.clear()
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def save(self, path: str) -> None:
        torch.save(
            {
                "edit_module": self.edit_module.state_dict(),
                "edits_registry": self.edits_registry,
                "hparams": self.hparams,
            },
            path,
        )
        if self.hparams.verbose:
            print(f"[SUCCESS] 保存至 {path}")

    def load(self, path: str) -> None:
        d = torch.load(path)
        num_edits = len(d["edits_registry"])
        self.edit_module = self._create_edit_module(num_edits)
        self.edit_module.load_state_dict(d["edit_module"])
        self.edits_registry = d["edits_registry"]
        if self.hparams.verbose:
            print(f"[SUCCESS] 加载完成，{num_edits} 条编辑")