# openrlhf/trainer/factored_vae_rm_trainer.py
import os
from abc import ABC

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


def kl_loss(mu, logvar):
    """Compute KL divergence between q(z|x) and N(0, I)"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


class FactoredVAERewardModelTrainer(ABC):
    """
    Trainer for Factored VAE Reward Model.
    Decomposes latent space into causal (s^c) and non-causal (s^nc) factors.
    
    Optimizes:
      - Pairwise preference loss on s^c (L_pred)
      - Reconstruction loss on [s^c; s^nc] -> h_hat (L_rec)  
      - KL loss on both factors (L_kl_c + L_kl_nc)

    Total loss:
      L_total = lambda_pred * L_pred + lambda_rec * rec_factor * L_rec + 
                beta_kl_c * kl_factor * L_kl_c + beta_kl_nc * kl_factor * L_kl_nc
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
        loss: str = "sigmoid",
        disable_ds_ckpt: bool = False,
        save_hf_ckpt: bool = False,
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

        # Pairwise preference loss
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss (pairwise preference)")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss (pairwise preference)")

        # Loss weights
        self.lambda_pred = getattr(self.args, "lambda_pred", 1.0)
        self.lambda_rec = getattr(self.args, "lambda_rec", 1.0)
        self.beta_kl_c = getattr(self.args, "beta_kl_c", 1.0)
        self.beta_kl_nc = getattr(self.args, "beta_kl_nc", 1.0)
        
        # Schedules
        self.kl_anneal_steps = getattr(self.args, "kl_anneal_steps", 0)
        self.rec_warmup_steps = getattr(self.args, "rec_warmup_steps", 0)

        # Other settings
        self.packing_samples = strategy.args.packing_samples
        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

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

    def _schedules(self, global_step: int):
        kl_factor = 1.0
        rec_factor = 1.0
        if self.kl_anneal_steps and self.kl_anneal_steps > 0:
            kl_factor = min(1.0, max(0.0, global_step / float(self.kl_anneal_steps)))
        if self.rec_warmup_steps and self.rec_warmup_steps > 0:
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
                range(self.train_dataloader.__len__()),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_reward, reject_reward, extras = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                pred_loss = self.loss_fn(chosen_reward, reject_reward, margin)

                # Reconstruction loss
                h_pos, h_hat_pos = extras["h_pos"], extras["h_hat_pos"]
                h_neg, h_hat_neg = extras["h_neg"], extras["h_hat_neg"]
                rec_pos = F.mse_loss(h_hat_pos, h_pos, reduction="mean")
                rec_neg = F.mse_loss(h_hat_neg, h_neg, reduction="mean")
                rec_loss = 0.5 * (rec_pos + rec_neg)

                # KL losses for both factors
                # Causal factor
                mu_c_pos, logvar_c_pos = extras["mu_c_pos"], extras["logvar_c_pos"]
                mu_c_neg, logvar_c_neg = extras["mu_c_neg"], extras["logvar_c_neg"]
                kl_c_pos = kl_loss(mu_c_pos, logvar_c_pos)
                kl_c_neg = kl_loss(mu_c_neg, logvar_c_neg)
                kl_c_loss = 0.5 * (kl_c_pos + kl_c_neg)

                # Non-causal factor
                mu_nc_pos, logvar_nc_pos = extras["mu_nc_pos"], extras["logvar_nc_pos"]
                mu_nc_neg, logvar_nc_neg = extras["mu_nc_neg"], extras["logvar_nc_neg"]
                kl_nc_pos = kl_loss(mu_nc_pos, logvar_nc_pos)
                kl_nc_neg = kl_loss(mu_nc_neg, logvar_nc_neg)
                kl_nc_loss = 0.5 * (kl_nc_pos + kl_nc_neg)

                # Schedules
                global_step = step // self.strategy.accumulated_gradient
                kl_factor, rec_factor = self._schedules(global_step)

                # Total loss
                total_loss = (
                    self.lambda_pred * pred_loss
                    + self.lambda_rec * rec_factor * rec_loss
                    + self.beta_kl_c * kl_factor * kl_c_loss
                    + self.beta_kl_nc * kl_factor * kl_nc_loss
                )

                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                acc = (chosen_reward > reject_reward).float().mean().item()
                acc_sum += acc
                pred_loss_sum += pred_loss.item()

                logs_dict = {
                    "loss_pred": pred_loss.item(),
                    "loss_rec": rec_loss.item(),
                    "loss_kl_c": kl_c_loss.item(),
                    "loss_kl_nc": kl_nc_loss.item(),
                    "loss_total": total_loss.item(),
                    "acc": acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "kl_factor": kl_factor,
                    "rec_factor": rec_factor,
                }

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_pred_mean"] = pred_loss_sum / self.strategy.accumulated_gradient
                    logs_dict["acc_mean"] = acc_sum / self.strategy.accumulated_gradient
                    pred_loss_sum = 0.0
                    acc_sum = 0.0
                    global_step_for_log = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step_for_log * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step_for_log, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        if (
            global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
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

    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc=f"Eval stage of steps {steps}",
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0.0
            rewards = []
            loss_sum = 0.0
            rec_sum = 0.0
            kl_c_sum = 0.0
            kl_nc_sum = 0.0

            for data in eval_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                chosen_reward, reject_reward, extras = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                pred_loss = self.loss_fn(chosen_reward, reject_reward, margin)
                loss_sum += pred_loss.item()

                # Reconstruction
                h_pos, h_hat_pos = extras["h_pos"], extras["h_hat_pos"]
                h_neg, h_hat_neg = extras["h_neg"], extras["h_hat_neg"]
                rec_pos = F.mse_loss(h_hat_pos, h_pos, reduction="mean").item()
                rec_neg = F.mse_loss(h_hat_neg, h_neg, reduction="mean").item()
                rec_sum += 0.5 * (rec_pos + rec_neg)

                # KL losses
                mu_c_pos, logvar_c_pos = extras["mu_c_pos"], extras["logvar_c_pos"]
                mu_c_neg, logvar_c_neg = extras["mu_c_neg"], extras["logvar_c_neg"]
                kl_c_pos = kl_loss(mu_c_pos, logvar_c_pos).item()
                kl_c_neg = kl_loss(mu_c_neg, logvar_c_neg).item()
                kl_c_sum += 0.5 * (kl_c_pos + kl_c_neg)

                mu_nc_pos, logvar_nc_pos = extras["mu_nc_pos"], extras["logvar_nc_pos"]
                mu_nc_neg, logvar_nc_neg = extras["mu_nc_neg"], extras["logvar_nc_neg"]
                kl_nc_pos = kl_loss(mu_nc_pos, logvar_nc_pos).item()
                kl_nc_neg = kl_loss(mu_nc_neg, logvar_nc_neg).item()
                kl_nc_sum += 0.5 * (kl_nc_pos + kl_nc_neg)

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                step_bar.update()

            acc_mean = acc / eval_dataloader.__len__()
            rec_mean = rec_sum / eval_dataloader.__len__()
            kl_c_mean = kl_c_sum / eval_dataloader.__len__()
            kl_nc_mean = kl_nc_sum / eval_dataloader.__len__()
            loss_mean = loss_sum / eval_dataloader.__len__()

            rewards = torch.cat(rewards).float()
            rewards = self.strategy.all_gather(rewards)
            reward_mean = torch.mean(rewards)
            reward_std = torch.std(rewards).clamp(min=1e-8)

            self.strategy.print("Set reward mean std")
            unwrap_model = self.strategy._unwrap_model(self.model)
            unwrap_model.config.mean = reward_mean.item()
            unwrap_model.config.std = reward_std.item()

            bar_dict = {
                "eval_loss": loss_mean,
                "acc_mean": acc_mean,
                "rec_mse_mean": rec_mean,
                "kl_c_mean": kl_c_mean,
                "kl_nc_mean": kl_nc_mean,
                "reward_mean": reward_mean.item(),
                "reward_std": reward_std.item(),
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
            self.strategy.print("histgram")
            self.strategy.print(histgram)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, extra = model(
            input_ids,
            attention_mask=att_masks,
            return_output=True,
        )

        batch_size = chosen_ids.shape[0]
        chosen_rewards = all_values[:batch_size]
        rejected_rewards = all_values[batch_size:]

        def split(t):
            return (t[:batch_size], t[batch_size:])

        h_pos, h_neg = split(extra["h"])
        h_hat_pos, h_hat_neg = split(extra["h_hat"])
        mu_c_pos, mu_c_neg = split(extra["mu_c"])
        logvar_c_pos, logvar_c_neg = split(extra["logvar_c"])
        mu_nc_pos, mu_nc_neg = split(extra["mu_nc"])
        logvar_nc_pos, logvar_nc_neg = split(extra["logvar_nc"])

        extras = {
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
        }
        return chosen_rewards, rejected_rewards, extras

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
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
