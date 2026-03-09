# openrlhf/trainer/srm_trainer.py

import os
from abc import ABC
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import PairWiseLoss, LogExpLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


class StochasticRewardModelTrainer(ABC):
    """
    Trainer for Stochastic Reward Model (StochRM) with optional KL regularization.

    Combines:
      - Bradley-Terry pairwise preference loss (L_pred)
      - KL divergence loss: KL(q(z|h) || N(0, I)) (L_kl)

    Total loss:
        L = λ_pred * L_pred + β_kl * kl_factor * L_kl

    where kl_factor is a linear annealing factor (optional).
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
        # === KL Loss Weight ===
        beta_kl=0.0,  # Set >0 to enable KL regularization
        kl_anneal_steps=0,  # Linearly increase KL weight to full over N steps
        # === Other ===
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

        # Preference loss
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("Using LogSigmoid Loss for preference")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("Using LogExp Loss for preference")

        # KL loss coefficient and schedule
        self.beta_kl = beta_kl
        self.kl_anneal_steps = kl_anneal_steps

        # MoE / packing
        self.aux_loss = self.args.aux_loss_coef > 1e-8
        self.packing_samples = strategy.args.packing_samples
        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        # Wandb / TensorBoard
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
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            os.makedirs(log_dir, exist_ok=True)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def _kl_schedule_factor(self, global_step: int) -> float:
        """Linearly increase KL weight from 0 to 1 over kl_anneal_steps."""
        if self.kl_anneal_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, global_step / float(self.kl_anneal_steps)))

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
        kl_loss_sum = 0.0

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples)

            step_bar = tqdm(
                range(len(self.train_dataloader)),
                desc=f"Epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )
            self.model.train()

            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                B = chosen_ids.size(0)

                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                # Forward pass with latent outputs
                (
                    chosen_reward,
                    reject_reward,
                    extras,
                ) = self.concatenated_forward(self.model, chosen_ids, c_mask, reject_ids, r_mask)

                # Preference loss
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                if self.margin_loss:
                    margin = torch.tensor(margin).to(chosen_reward.device)
                    pred_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                else:
                    pred_loss = self.loss_fn(chosen_reward, reject_reward)

                # KL divergence: KL(q(z|x) || N(0,I))
                mu = torch.cat([extras["mu"][:B], extras["mu"][B:]], dim=0)
                logvar = torch.cat([extras["logvar"][:B], extras["logvar"][B:]], dim=0)
                # Mean over batch, sum over latent dim
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                # Annealing factor for KL
                global_step = step // self.strategy.accumulated_gradient
                kl_factor = self._kl_schedule_factor(global_step)

                # Total loss
                total_loss = pred_loss + self.beta_kl * kl_factor * kl_loss

                # Add MoE loss if present
                aux_loss = extras.get("aux_loss", 0.0)
                if self.aux_loss:
                    total_loss += self.args.aux_loss_coef * aux_loss

                # Backward
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Metrics
                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                pred_loss_sum += pred_loss.item()
                kl_loss_sum += kl_loss.item()

                logs_dict = {
                    "loss_pred": pred_loss.item(),
                    "loss_kl": kl_loss.item(),
                    "loss_total": total_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "kl_factor": kl_factor,
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # Logging
                if step % self.strategy.accumulated_gradient == 0:
                    global_step_for_log = global_step
                    mean_logs = {
                        "loss_pred_mean": pred_loss_sum / self.strategy.accumulated_gradient,
                        "acc_mean": acc_sum / self.strategy.accumulated_gradient,
                        "kl_mean": kl_loss_sum / self.strategy.accumulated_gradient,
                    }
                    client_states = {"consumed_samples": global_step_for_log * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step_for_log, step_bar, {**logs_dict, **mean_logs}, client_states)

                    # Reset
                    acc_sum = pred_loss_sum = kl_loss_sum = 0.0

                step += 1
            epoch_bar.update()

        if self._wandb and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard and self.strategy.is_rank_0():
            self._tensorboard.close()

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """
        Concatenate chosen and rejected inputs into one batch.
        Returns rewards and latent components (mu, logvar).
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)

        # Forward with latent return
        rewards, vae_outputs = model(
            input_ids,
            attention_mask=att_masks,
            return_output=True,
            return_latent=True,  # ← returns mu, logvar, z, h
        )

        batch_size = chosen_ids.shape[0]
        chosen_rewards = rewards[:batch_size]
        rejected_rewards = rewards[batch_size:]

        # Split VAE outputs
        def split(t):
            return t[:batch_size], t[batch_size:]

        mu_pos, mu_neg = split(vae_outputs["mu"])
        logvar_pos, logvar_neg = split(vae_outputs["logvar"])

        extras = {
            "mu": torch.cat([mu_pos, mu_neg], dim=0),
            "logvar": torch.cat([logvar_pos, logvar_neg], dim=0),
            "aux_loss": vae_outputs.get("aux_loss", 0),
        }

        return chosen_rewards, rejected_rewards, extras

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Pad and concatenate chosen and rejected sequences."""
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                return torch.cat([
                    pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
                    tensor
                ], dim=dim)

        max_len = max(chosen_ids.shape[1], reject_ids.shape[1])
        input_ids = torch.cat([
            pad_to_length(chosen_ids, max_len, self.tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_len, self.tokenizer.pad_token_id)
        ], dim=0)

        max_mask_len = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat([
            pad_to_length(c_mask, max_mask_len, 0),
            pad_to_length(r_mask, max_mask_len, 0)
        ], dim=0)

        return input_ids, att_masks

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            if self._wandb and self.strategy.is_rank_0():
                logs = {"train/" + k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # Evaluation
        if (global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0) and self.eval_dataloader:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # Save checkpoint
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states)
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    @torch.no_grad()
    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(range(len(eval_dataloader)), desc=f"Eval step {steps}", disable=not self.strategy.is_rank_0())
        self.model.eval()
        acc = 0.0
        rewards = []
        loss_sum = 0.0
        kl_sum = 0.0

        for data in eval_dataloader:
            chosen_ids, c_mask, reject_ids, r_mask, margin = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_reward, reject_reward, extras = self.concatenated_forward(self.model, chosen_ids, c_mask, reject_ids, r_mask)

            loss = self.loss_fn(chosen_reward, reject_reward)
            acc += (chosen_reward > reject_reward).float().mean().item()
            loss_sum += loss.item()

            # KL metric
            mu, logvar = extras["mu"], extras["logvar"]
            kl_sum += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean().item()

            rewards += [chosen_reward.flatten(), reject_reward.flatten()]
            step_bar.update()

        acc_mean = acc / len(eval_dataloader)
        loss_mean = loss_sum / len(eval_dataloader)
        kl_mean = kl_sum / len(eval_dataloader)

        rewards = torch.cat(rewards).float()
        rewards = self.strategy.all_gather(rewards)
        reward_mean = rewards.mean().item()
        reward_std = rewards.std().clamp(min=1e-8).item()

        # Update model config
        unwrap_model = self.strategy._unwrap_model(self.model)
        unwrap_model.config.mean = reward_mean
        unwrap_model.config.std = reward_std

        logs = {
            "eval_loss": loss_mean,
            "acc_mean": acc_mean,
            "kl_div": kl_mean,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
        }
        logs = self.strategy.all_reduce(logs)
        step_bar.set_postfix(logs)

        if self.strategy.is_rank_0():
            if self._wandb:
                self._wandb.log({"eval/" + k: v for k, v in {**logs, "global_step": steps}.items()})
            elif self._tensorboard:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()
