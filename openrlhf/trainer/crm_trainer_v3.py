# openrlhf/trainer/crm_trainer.py

import os
from abc import ABC
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


def kl_loss(mu, logvar):
    """KL divergence between N(mu, sigma^2) and N(0, I)"""
    # returns batch mean
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


class CausalRewardModelTrainer(ABC):
    """
    Trainer for Causal Reward Model with adversarial disentanglement via GRL.

    Total Loss:
        L_total = λ_pred * L_pred 
                + λ_adv * L_adv 
                + λ_rec * rec_factor * L_rec 
                + β_kl_c * kl_factor * L_kl_c 
                + β_kl_nc * kl_factor * L_kl_nc
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_norm=1.0,
        max_epochs=1,
        loss="sigmoid",
        disable_ds_ckpt=False,
        save_hf_ckpt=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.model = model
        self.optimizer = optim
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.args = strategy.args
        self.disable_ds_ckpt = disable_ds_ckpt
        self.save_hf_ckpt = save_hf_ckpt

        # Loss functions
        self.loss_fn = PairWiseLoss()
        self.strategy.print("CausalRewardModelTrainer with LogSigmoid PairWiseLoss")

        # Weights
        self.lambda_pred = getattr(self.args, "lambda_pred", 1.0)
        self.lambda_adv = getattr(self.args, "lambda_adv", 1.0)
        self.lambda_rec = getattr(self.args, "lambda_rec", 1.0)
        self.beta_kl_c = getattr(self.args, "beta_kl_c", 1.0)
        self.beta_kl_nc = getattr(self.args, "beta_kl_nc", 1.0)
        self.grl_lambda = getattr(self.args, "grl_lambda", 1.0)

        # Scheduling
        self.kl_anneal_steps = getattr(self.args, "kl_anneal_steps", 0)
        self.rec_warmup_steps = getattr(self.args, "rec_warmup_steps", 0)

        # Other flags
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss
        self.packing_samples = self.strategy.args.packing_samples
        self.margin_loss = getattr(self.strategy.args, "margin_loss", False)

        # ===== 新增：创建固定留出集 =====
        self.holdout_data = self._create_holdout_set(eval_dataloader, num_batches=50)
        if len(self.holdout_data) > 0:
            self.strategy.print(f"Created holdout set with {len(self.holdout_data)} batches for length correlation tracking")
        else:
            self.strategy.print("[Warning] Holdout set is empty, length correlation tracking will be skipped")

        # Logging
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def _create_holdout_set(self, dataloader, num_batches):
        """从 dataloader 中提取固定的前 num_batches 个 batch 作为留出集"""
        if dataloader is None or len(dataloader) == 0:
            return []
        
        holdout = []
        for i, data in enumerate(dataloader):
            if i >= num_batches:
                break
            # 将数据移到 CPU 以节省显存
            cpu_data = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    cpu_data.append(item.cpu())
                elif isinstance(item, list):
                    # 处理 prompt_lens 等 list 类型
                    cpu_data.append(item.copy() if hasattr(item, 'copy') else item)
                else:
                    cpu_data.append(item)
            holdout.append(cpu_data)
        
        return holdout

    @torch.no_grad()
    def _pearson_corr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute Pearson correlation coefficient between two tensors.
        
        Args:
            x: First tensor (any shape, will be flattened)
            y: Second tensor (any shape, will be flattened)
        
        Returns:
            float: Pearson correlation coefficient, or 0.0 if computation fails
        """
        # Flatten to 1D
        x = x.flatten()
        y = y.flatten()
        
        # Check length match
        if len(x) != len(y):
            logger.warning(f"Length mismatch in pearson_corr: {len(x)} vs {len(y)}")
            return 0.0
        
        # Need at least 2 samples
        if len(x) < 2:
            return 0.0

        # Remove NaN and Inf
        valid_mask = torch.isfinite(x) & torch.isfinite(y)
        num_valid = valid_mask.sum().item()
        
        if num_valid < 2:
            logger.warning(f"Too few valid samples in pearson_corr: {num_valid}")
            return 0.0

        x = x[valid_mask]
        y = y[valid_mask]

        # Center the data
        vx = x - x.mean()
        vy = y - y.mean()

        # Compute correlation
        numerator = (vx * vy).sum()
        denominator = torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum())

        if denominator < 1e-8:
            return 0.0

        corr = numerator / denominator
        return corr.item()

    @torch.no_grad()
    def compute_length_reward_corr(self) -> dict:
        """
        在固定留出集上计算 response length 和 reward 的 Pearson 相关系数。
        
        Returns:
            dict: 包含三个相关系数的字典
                - length_corr/all: 所有样本（chosen + rejected）的相关系数
                - length_corr/chosen: chosen 样本的相关系数
                - length_corr/rejected: rejected 样本的相关系数
        """
        if not self.holdout_data:
            return {}

        self.model.eval()
        device = torch.cuda.current_device()

        all_chosen_rewards = []
        all_rejected_rewards = []
        all_chosen_lengths = []
        all_rejected_lengths = []

        for data in self.holdout_data:
            # Unpack data（注意顺序和 collate_fn 返回的一致）
            chosen_ids_cpu, c_mask_cpu, reject_ids_cpu, r_mask_cpu, _, prompt_lens_cpu = data
            
            # 移到 GPU 并 squeeze（和训练时一致）
            chosen_ids = chosen_ids_cpu.squeeze(1).to(device)
            c_mask = c_mask_cpu.squeeze(1).to(device)
            reject_ids = reject_ids_cpu.squeeze(1).to(device)
            r_mask = r_mask_cpu.squeeze(1).to(device)
            
            # ===== 使用 concatenated_forward（自动处理 padding）=====
            chosen_reward, rejected_reward, _ = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, grl_lambda=0.0
            )
            
            # 计算 response lengths（总长度 - prompt 长度）
            chosen_total_len = c_mask.sum(dim=1).float()  # [B]
            rejected_total_len = r_mask.sum(dim=1).float()  # [B]
            
            # prompt_lens 是 list，转为 tensor
            prompt_lens_tensor = torch.tensor(prompt_lens_cpu, device=device, dtype=torch.float)
            
            # response length = total length - prompt length
            chosen_response_len = chosen_total_len - prompt_lens_tensor
            rejected_response_len = rejected_total_len - prompt_lens_tensor
            
            # 收集（保留在 GPU 上，后续一起计算）
            all_chosen_rewards.append(chosen_reward)
            all_chosen_lengths.append(chosen_response_len)
            all_rejected_rewards.append(rejected_reward)
            all_rejected_lengths.append(rejected_response_len)

        # Concatenate all results
        chosen_rewards = torch.cat(all_chosen_rewards)
        chosen_lengths = torch.cat(all_chosen_lengths)
        rejected_rewards = torch.cat(all_rejected_rewards)
        rejected_lengths = torch.cat(all_rejected_lengths)
        
        # 合并 chosen 和 rejected
        all_rewards = torch.cat([chosen_rewards, rejected_rewards])
        all_lengths = torch.cat([chosen_lengths, rejected_lengths])

        # 计算三个相关系数
        metrics = {
            "length_corr/all": self._pearson_corr(all_rewards, all_lengths),
            "length_corr/chosen": self._pearson_corr(chosen_rewards, chosen_lengths),
            "length_corr/rejected": self._pearson_corr(rejected_rewards, rejected_lengths),
        }

        self.model.train()
        return metrics

    def _schedules(self, global_step: int):
        """KL & reconstruction weighting schedules (anneal / warmup)."""
        kl_factor = 1.0
        rec_factor = 1.0
        if self.kl_anneal_steps > 0:
            kl_factor = min(1.0, max(0.0, global_step / float(self.kl_anneal_steps)))
        if self.rec_warmup_steps > 0:
            rec_factor = min(1.0, max(0.0, global_step / float(self.rec_warmup_steps)))
        return kl_factor, rec_factor

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        acc_sum = 0.0
        pred_loss_sum = 0.0

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin, prompt_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                # Forward pass with all outputs
                chosen_reward, rejected_reward, extras = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, grl_lambda=self.grl_lambda
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    rejected_reward = rejected_reward.float()

                # Prediction loss (on causal path)
                pred_loss = self.loss_fn(chosen_reward, rejected_reward, margin)

                # Adversary loss: tries to predict preference from s^nc
                adv_pos, adv_neg = extras["adv_logits"]
                adv_loss = self.loss_fn(adv_pos, adv_neg, margin)

                # Reconstruction loss
                h_pos, h_hat_pos = extras["h_pos"], extras["h_hat_pos"]
                h_neg, h_hat_neg = extras["h_neg"], extras["h_hat_neg"]
                rec_loss = 0.5 * (
                    F.mse_loss(h_hat_pos, h_pos, reduction="mean") +
                    F.mse_loss(h_hat_neg, h_neg, reduction="mean")
                )

                # KL losses
                kl_c_loss = 0.5 * (
                    kl_loss(extras["mu_c_pos"], extras["logvar_c_pos"]) +
                    kl_loss(extras["mu_c_neg"], extras["logvar_c_neg"])
                )
                kl_nc_loss = 0.5 * (
                    kl_loss(extras["mu_nc_pos"], extras["logvar_nc_pos"]) +
                    kl_loss(extras["mu_nc_neg"], extras["logvar_nc_neg"])
                )

                # Annealing factors
                global_step = step // self.strategy.accumulated_gradient
                kl_factor, rec_factor = self._schedules(global_step)

                # Total loss
                total_loss = (
                    self.lambda_pred * pred_loss
                    + self.lambda_adv * adv_loss
                    + self.lambda_rec * rec_factor * rec_loss
                    + self.beta_kl_c * kl_factor * kl_c_loss
                    + self.beta_kl_nc * kl_factor * kl_nc_loss
                )

                # Backward & step
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Logging
                acc = (chosen_reward > rejected_reward).float().mean().item()
                acc_sum += acc
                pred_loss_sum += pred_loss.item()

                logs_dict = {
                    "loss_pred": pred_loss.item(),
                    "loss_adv": adv_loss.item(),
                    "loss_rec": rec_loss.item(),
                    "loss_kl_c": kl_c_loss.item(),
                    "loss_kl_nc": kl_nc_loss.item(),
                    "loss_total": total_loss.item(),
                    "acc": acc,
                    "adv_acc": (adv_pos > adv_neg).float().mean().item(),
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": rejected_reward.mean().item(),
                    "kl_factor": kl_factor,
                    "rec_factor": rec_factor,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                logs_dict["loss"] = logs_dict["loss_pred"]

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # === 在每个 gradient accumulation boundary 上执行 ===
                if step % self.strategy.accumulated_gradient == 0:
                    # 添加长度相关性监控
                    corr_metrics = self.compute_length_reward_corr()
                    logs_dict.update(corr_metrics)

                    # 汇总指标
                    logs_dict["loss_pred_mean"] = pred_loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    pred_loss_sum = 0.0
                    acc_sum = 0.0

                    global_step_for_log = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step_for_log * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step_for_log, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard and self.strategy.is_rank_0():
            self._tensorboard.close()

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, grl_lambda=1.0):
        """Run model on concatenated chosen and rejected inputs (for FSDP efficiency)."""
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)

        rewards, extras = model(
            input_ids,
            attention_mask=att_masks,
            return_output=True,
            return_adv=True,
        )

        B = chosen_ids.shape[0]
        chosen_reward = rewards[:B]
        rejected_reward = rewards[B:]

        def split(t):
            return t[:B], t[B:]

        h_pos, h_neg = split(extras["h"])
        h_hat_pos, h_hat_neg = split(extras["h_hat"])
        mu_c_pos, mu_c_neg = split(extras["mu_c"])
        logvar_c_pos, logvar_c_neg = split(extras["logvar_c"])
        mu_nc_pos, mu_nc_neg = split(extras["mu_nc"])
        logvar_nc_pos, logvar_nc_neg = split(extras["logvar_nc"])
        adv_logits_all = extras["adv_logits"]
        adv_pos, adv_neg = split(adv_logits_all)

        extras_out = {
            "h_pos": h_pos,
            "h_neg": h_neg,
            "h_hat_pos": h_hat_pos,
            "h_hat_neg": h_hat_neg,
            "mu_c_pos": mu_c_pos,
            "mu_c_neg": mu_c_neg,
            "logvar_c_pos": logvar_c_pos,
            "logvar_c_neg": logvar_c_neg,
            "mu_nc_pos": mu_nc_pos,
            "mu_nc_neg": mu_nc_neg,
            "logvar_nc_pos": logvar_nc_pos,
            "logvar_nc_neg": logvar_nc_neg,
            "adv_logits": (adv_pos, adv_neg),
        }
        return chosen_reward, rejected_reward, extras_out

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate chosen/rejected sequences into a single batch."""

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            pad_size = list(tensor.shape)
            pad_size[dim] = length - tensor.size(dim)
            padding = pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([padding, tensor], dim=dim)

        max_len_input = max(chosen_ids.shape[1], reject_ids.shape[1])
        input_ids = torch.cat([
            pad_to_length(chosen_ids, max_len_input, self.tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_len_input, self.tokenizer.pad_token_id),
        ], dim=0)

        max_len_mask = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat([
            pad_to_length(c_mask, max_len_mask, 0),
            pad_to_length(r_mask, max_len_mask, 0),
        ], dim=0)

        return input_ids, att_masks

    def save_logs_and_checkpoints(self, args, global_step, logs_dict={}, client_states={}):
        """与 RewardModelTrainer 对齐的日志保存逻辑。"""
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        if (
            global_step % args.eval_steps == 0
            or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    @torch.no_grad()
    def _bucket_length_reward_stats(
        self,
        rewards: torch.Tensor,
        lengths: torch.Tensor,
        num_buckets: int = 10,
    ):
        """
        对给定的 rewards 和 lengths 做按长度的 quantile 分桶统计。

        Args:
            rewards: [N] tensor, 所有样本的 reward
            lengths: [N] tensor, 所有样本的 answer length
            num_buckets: 分桶数量（默认 10）

        Returns:
            (xs, ys): 两个 list，分别是每个 bucket 的中位长度和平均 reward
        """
        # 转为 1D CPU tensor
        lengths = lengths.flatten().float().cpu()
        rewards = rewards.flatten().float().cpu()

        # 过滤非法值
        valid_mask = torch.isfinite(lengths) & torch.isfinite(rewards)
        if valid_mask.sum() < 2:
            return [], []

        lengths = lengths[valid_mask]
        rewards = rewards[valid_mask]

        if lengths.numel() < num_buckets:
            # 样本太少，不做分桶
            return [], []

        # 计算 quantile 边界（0%, 10%, ..., 100%）
        quantiles = torch.linspace(0.0, 1.0, steps=num_buckets + 1)
        q_vals = torch.quantile(lengths, quantiles)

        q_vals[0] = lengths.min()
        q_vals[-1] = lengths.max()

        xs = []
        ys = []

        for i in range(num_buckets):
            low = q_vals[i]
            high = q_vals[i + 1]
            if i < num_buckets - 1:
                mask = (lengths >= low) & (lengths < high)
            else:
                mask = (lengths >= low) & (lengths <= high)

            if mask.sum() == 0:
                continue

            bucket_lengths = lengths[mask]
            bucket_rewards = rewards[mask]

            bucket_len_med = bucket_lengths.median().item()
            bucket_reward_mean = bucket_rewards.mean().item()

            xs.append(bucket_len_med)
            ys.append(bucket_reward_mean)

        return xs, ys

    @torch.no_grad()
    def evaluate(self, eval_dataloader, steps=0):
        """评估函数：在原有 loss/acc 的基础上，增加按 answer 长度分桶的 reward 可视化。"""
        step_bar = tqdm(
            range(len(eval_dataloader)),
            desc=f"Eval stage of steps {steps}",
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        device = torch.cuda.current_device()

        acc = 0.0
        adv_acc_sum = 0.0
        loss_sum = 0.0
        rec_sum = 0.0
        kl_c_sum = 0.0
        kl_nc_sum = 0.0
        rewards_all = []

        # 为按长度分桶累积数据（answer length）
        chosen_rewards_all = []
        rejected_rewards_all = []
        chosen_lengths_all = []
        rejected_lengths_all = []

        for data in eval_dataloader:
            # Dataset/Collate 返回: chosen_ids, c_mask, reject_ids, r_mask, margin, prompt_lens
            chosen_ids, c_mask, reject_ids, r_mask, margin, prompt_lens = data
            chosen_ids = chosen_ids.squeeze(1).to(device)
            c_mask = c_mask.squeeze(1).to(device)
            reject_ids = reject_ids.squeeze(1).to(device)
            r_mask = r_mask.squeeze(1).to(device)

            # 模型前向
            chosen_reward, rejected_reward, extras = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, grl_lambda=0.0
            )

            if self.margin_loss:
                margin = torch.tensor(margin).to(device)
            else:
                margin = None

            # 主 RM loss
            pred_loss = self.loss_fn(chosen_reward, rejected_reward, margin).item()
            loss_sum += pred_loss

            # Reconstruction
            h_pos, h_hat_pos = extras["h_pos"], extras["h_hat_pos"]
            h_neg, h_hat_neg = extras["h_neg"], extras["h_hat_neg"]
            rec_pos = F.mse_loss(h_hat_pos, h_pos, reduction="mean").item()
            rec_neg = F.mse_loss(h_hat_neg, h_neg, reduction="mean").item()
            rec_sum += 0.5 * (rec_pos + rec_neg)

            # KL losses
            kl_c_pos = kl_loss(extras["mu_c_pos"], extras["logvar_c_pos"]).item()
            kl_c_neg = kl_loss(extras["mu_c_neg"], extras["logvar_c_neg"]).item()
            kl_c_sum += 0.5 * (kl_c_pos + kl_c_neg)

            kl_nc_pos = kl_loss(extras["mu_nc_pos"], extras["logvar_nc_pos"]).item()
            kl_nc_neg = kl_loss(extras["mu_nc_neg"], extras["logvar_nc_neg"]).item()
            kl_nc_sum += 0.5 * (kl_nc_pos + kl_nc_neg)

            # Acc metrics
            acc += (chosen_reward > rejected_reward).float().mean().item()
            adv_pos, adv_neg = extras["adv_logits"]
            adv_acc_sum += (adv_pos > adv_neg).float().mean().item()

            rewards_all += [chosen_reward.flatten(), rejected_reward.flatten()]
            step_bar.update()

            # ====== 累积 answer length（总长度 - prompt_len） ======
            chosen_total_len = c_mask.sum(dim=1).float()       # [B]
            rejected_total_len = r_mask.sum(dim=1).float()     # [B]

            # prompt_lens 是一个 Python list（长度 B），转成 tensor
            prompt_lens_tensor = torch.tensor(prompt_lens, device=device, dtype=torch.float32)

            chosen_answer_len = (chosen_total_len - prompt_lens_tensor).clamp(min=1.0)
            rejected_answer_len = (rejected_total_len - prompt_lens_tensor).clamp(min=1.0)

            chosen_rewards_all.append(chosen_reward.detach())
            rejected_rewards_all.append(rejected_reward.detach())
            chosen_lengths_all.append(chosen_answer_len)
            rejected_lengths_all.append(rejected_answer_len)

        # --------- 统计常规指标 ---------
        acc_mean = acc / len(eval_dataloader)
        adv_acc_mean = adv_acc_sum / len(eval_dataloader)
        rec_mse_mean = rec_sum / len(eval_dataloader)
        kl_c_mean = kl_c_sum / len(eval_dataloader)
        kl_nc_mean = kl_nc_sum / len(eval_dataloader)
        eval_loss_mean = loss_sum / len(eval_dataloader)

        rewards = torch.cat(rewards_all).float()
        rewards = self.strategy.all_gather(rewards)
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards).clamp(min=1e-8)

        # 保存 mean/std 到 config
        unwrap_model = self.strategy._unwrap_model(self.model)
        unwrap_model.config.mean = reward_mean.item()
        unwrap_model.config.std = reward_std.item()

        logs = {
            "eval_loss": eval_loss_mean,
            "acc_mean": acc_mean,
            "adv_acc_mean": adv_acc_mean,
            "rec_mse_mean": rec_mse_mean,
            "kl_c_mean": kl_c_mean,
            "kl_nc_mean": kl_nc_mean,
            "reward_mean": reward_mean.item(),
            "reward_std": reward_std.item(),
        }
        logs = self.strategy.all_reduce(logs)
        step_bar.set_postfix(logs)

        # 打印 reward 直方图
        histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
        self.strategy.print("histgram")
        self.strategy.print(histgram)

        # --------- 额外：按 answer length 分桶的 reward 可视化（只在 rank 0 上做） ---------
        if self.strategy.is_rank_0():
            import wandb

            # 将累积的 chosen / rejected reward 和 length 拼起来
            chosen_rewards_all = torch.cat(chosen_rewards_all)
            rejected_rewards_all = torch.cat(rejected_rewards_all)
            chosen_lengths_all = torch.cat(chosen_lengths_all)
            rejected_lengths_all = torch.cat(rejected_lengths_all)

            # 调用 bucket 方法，得到 (xs, ys)
            xs_chosen, ys_chosen = self._bucket_length_reward_stats(
                chosen_rewards_all, chosen_lengths_all, num_buckets=10
            )
            xs_rejected, ys_rejected = self._bucket_length_reward_stats(
                rejected_rewards_all, rejected_lengths_all, num_buckets=10
            )

            # 组装 log payload（只包含原有 eval scalar）
            log_payload = {f"eval/{k}": v for k, v in {**logs, "global_step": steps}.items()}

            if self._wandb is not None:
                # chosen 折线图 + 表
                if len(xs_chosen) > 0:
                    table_chosen = wandb.Table(columns=["length", "reward_mean"])
                    for x, y in zip(xs_chosen, ys_chosen):
                        table_chosen.add_data(x, y)
                    # 折线图
                    chosen_line = wandb.plot.line_series(
                        xs=[xs_chosen],
                        ys=[ys_chosen],
                        keys=["chosen_len_vs_reward"],
                        title="chosen_len_vs_reward (line)",
                        xname="length",
                    )
                    log_payload["eval/chosen_len_reward_line"] = chosen_line
                    # 可选：把 table 也 log 出去，便于在 UI 中查看原始数据
                    # log_payload["eval/chosen_len_reward_table"] = table_chosen

                # rejected 折线图 + 表
                if len(xs_rejected) > 0:
                    table_rejected = wandb.Table(columns=["length", "reward_mean"])
                    for x, y in zip(xs_rejected, ys_rejected):
                        table_rejected.add_data(x, y)
                    rejected_line = wandb.plot.line_series(
                        xs=[xs_rejected],
                        ys=[ys_rejected],
                        keys=["rejected_len_vs_reward"],
                        title="rejected_len_vs_reward (line)",
                        xname="length",
                    )
                    log_payload["eval/rejected_len_reward_line"] = rejected_line
                    # log_payload["eval/rejected_len_reward_table"] = table_rejected

                self._wandb.log(log_payload)

            elif self._tensorboard is not None:
                # tensorboard 上只 log 原有 scalar
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        else:
            # 非 rank0 只做常规 logging
            if self._wandb is not None and self.strategy.is_rank_0():
                self._wandb.log({f"eval/{k}": v for k, v in {**logs, "global_step": steps}.items()})
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()

