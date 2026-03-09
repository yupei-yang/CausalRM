import os
from abc import ABC

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class RewardModelTrainer(ABC):
    """
    Trainer for training a reward model.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to apply.
        optim (Optimizer): The optimizer to use during training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler for dynamic adjustments during training.
        tokenizer (Tokenizer): The tokenizer for processing input text data.
        max_norm (float, defaults to 0.5): Maximum gradient norm for gradient clipping.
        max_epochs (int, defaults to 2): Maximum number of training epochs.
        loss (str, defaults to "sigmoid"): The loss function to use during training, e.g., "sigmoid".
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
        max_norm=0.5,
        max_epochs: int = 2,
        loss="sigmoid",
        disable_ds_ckpt=False,
        save_hf_ckpt=False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.max_norm = max_norm
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.disable_ds_ckpt = disable_ds_ckpt
        self.save_hf_ckpt = save_hf_ckpt

        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss")

        # Mixtral 8*7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        # ===== 新增：创建固定留出集 =====
        self.holdout_data = self._create_holdout_set(eval_dataloader, num_batches=50)
        if len(self.holdout_data) > 0:
            self.strategy.print(f"Created holdout set with {len(self.holdout_data)} batches for length correlation tracking")
        else:
            self.strategy.print("[Warning] Holdout set is empty, length correlation tracking will be skipped")

        # wandb/tensorboard setting
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

        # Initialize TensorBoard writer if wandb is not available
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
                self.model, chosen_ids, c_mask, reject_ids, r_mask
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

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

        # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        acc_sum = 0
        loss_sum = 0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin, prompt_lens = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                # mixtral
                if not self.aux_loss:
                    aux_loss = 0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                loss_sum += preference_loss.item()
                # optional rm info
                logs_dict = {
                    "loss": preference_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    loss_sum = 0
                    acc_sum = 0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}

                    # === 新增：在线长度–reward 相关监控 ===
                    corr_metrics = self.compute_length_reward_corr()
                    logs_dict.update(corr_metrics)

                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if (
            global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            # do eval when len(dataloader) > 0, avoid zero division in eval.
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
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
    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        acc = 0
        rewards = []
        loss_sum = 0

        # 为 length–reward 分析累积数据
        chosen_rewards_all = []
        rejected_rewards_all = []
        chosen_lengths_all = []
        rejected_lengths_all = []

        device = torch.cuda.current_device()

        for data in eval_dataloader:
            # 兼容 5 元素老格式 vs 6 元素新格式
            if len(data) == 6:
                chosen_ids, c_mask, reject_ids, r_mask, margin, prompt_lens = data
            else:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                prompt_lens = None

            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_reward, reject_reward, _ = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask
            )

            if self.margin_loss:
                margin = torch.tensor(margin).to(torch.cuda.current_device())
            else:
                margin = None

            loss = self.loss_fn(chosen_reward, reject_reward, margin)
            rewards += [chosen_reward.flatten(), reject_reward.flatten()]
            acc += (chosen_reward > reject_reward).float().mean().item()
            loss_sum += loss.item()
            step_bar.update()

            # 若有 prompt_lens，则累积 answer length 数据
            if prompt_lens is not None:
                chosen_total_len = c_mask.sum(dim=1).float()
                rejected_total_len = r_mask.sum(dim=1).float()
                prompt_lens_tensor = torch.tensor(prompt_lens, device=device, dtype=torch.float32)

                chosen_answer_len = (chosen_total_len - prompt_lens_tensor).clamp(min=1.0)
                rejected_answer_len = (rejected_total_len - prompt_lens_tensor).clamp(min=1.0)

                chosen_rewards_all.append(chosen_reward.detach())
                rejected_rewards_all.append(reject_reward.detach())
                chosen_lengths_all.append(chosen_answer_len)
                rejected_lengths_all.append(rejected_answer_len)

        acc_mean = acc / eval_dataloader.__len__()
        loss_mean = loss_sum / eval_dataloader.__len__()

        rewards = torch.cat(rewards).float()
        rewards = self.strategy.all_gather(rewards)
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards).clamp(min=1e-8)

        # save mean std
        self.strategy.print("Set reward mean std")
        unwrap_model = self.strategy._unwrap_model(self.model)
        unwrap_model.config.mean = reward_mean.item()
        unwrap_model.config.std = reward_std.item()

        bar_dict = {
            "eval_loss": loss_mean,
            "acc_mean": acc_mean,
            "reward_mean": reward_mean.item(),
            "reward_std": reward_std.item(),
        }
        logs = self.strategy.all_reduce(bar_dict)
        step_bar.set_postfix(logs)

        histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
        self.strategy.print("histgram")
        self.strategy.print(histgram)

        # 常规 scalar 日志
        if self.strategy.is_rank_0():
            if self._wandb is not None:
                self._wandb.log({"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        # 若有 prompt_lens 信息，则额外画 length–reward 曲线（只在 rank0 + wandb）
        if self.strategy.is_rank_0() and self._wandb is not None and len(chosen_rewards_all) > 0:
            import wandb

            chosen_rewards_all = torch.cat(chosen_rewards_all)
            rejected_rewards_all = torch.cat(rejected_rewards_all)
            chosen_lengths_all = torch.cat(chosen_lengths_all)
            rejected_lengths_all = torch.cat(rejected_lengths_all)

            xs_chosen, ys_chosen = self._bucket_length_reward_stats(
                chosen_rewards_all, chosen_lengths_all, num_buckets=10
            )
            xs_rejected, ys_rejected = self._bucket_length_reward_stats(
                rejected_rewards_all, rejected_lengths_all, num_buckets=10
            )

            log_payload = {"eval/global_step": steps}
            if len(xs_chosen) > 0:
                chosen_line = wandb.plot.line_series(
                    xs=[xs_chosen],
                    ys=[ys_chosen],
                    keys=["chosen_len_vs_reward"],
                    title="RM_chosen_len_vs_reward (line)",
                    xname="length",
                )
                log_payload["eval/chosen_len_reward_line"] = chosen_line
            if len(xs_rejected) > 0:
                rejected_line = wandb.plot.line_series(
                    xs=[xs_rejected],
                    ys=[ys_rejected],
                    keys=["rejected_len_vs_reward"],
                    title="RM_rejected_len_vs_reward (line)",
                    xname="length",
                )
                log_payload["eval/rejected_len_reward_line"] = rejected_line

            self._wandb.log(log_payload)

        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        aux_loss = output.aux_loss if "aux_loss" in output else []
        return chosen_rewards, rejected_rewards, aux_loss

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
