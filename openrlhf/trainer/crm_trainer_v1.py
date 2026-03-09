# openrlhf/trainer/crm_trainer.py

import os
from abc import ABC
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


def kl_loss(mu, logvar):
    """KL divergence between N(mu, sigma^2) and N(0, I)"""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


class CausalRewardModelTrainer(ABC):
    """
    Trainer for Causal Reward Model with adversarial disentanglement via GRL.

    Total Loss:
        L_total = λ_pred * L_pred 
                - λ_adv * L_adv 
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
        self.loss_fn = PairWiseLoss()  # Bradley-Terry pairwise preference

        # Weights
        self.lambda_pred = getattr(self.args, "lambda_pred", 1.0)
        self.lambda_adv = getattr(self.args, "lambda_adv", 1.0)
        self.lambda_rec = getattr(self.args, "lambda_rec", 1.0)
        self.beta_kl_c = getattr(self.args, "beta_kl_c", 1.0)
        self.beta_kl_nc = getattr(self.args, "beta_kl_nc", 1.0)
        self.grl_lambda = getattr(self.args, "grl_lambda", 1.0)  # if model支持可以传入

        # Scheduling
        self.kl_anneal_steps = getattr(self.args, "kl_anneal_steps", 0)
        self.rec_warmup_steps = getattr(self.args, "rec_warmup_steps", 0)

        # Other flags
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss
        self.packing_samples = self.strategy.args.packing_samples
        self.margin_loss = getattr(self.strategy.args, "margin_loss", False)

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
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            os.makedirs(log_dir, exist_ok=True)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def _schedules(self, global_step: int):
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
                disable=not self.strategy.is_rank_0()
            )

            self.model.train()
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
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
                    F.mse_loss(h_hat_pos, h_pos, reduction='mean') +
                    F.mse_loss(h_hat_neg, h_neg, reduction='mean')
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

                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    # 汇总指标（与 FactoredVAERewardModelTrainer 对齐）
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
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        rewards, extras = model(
            input_ids,
            attention_mask=att_masks,
            return_output=True,
            return_adv=True,  # trigger adversary output
            # 如果模型支持 grl_lambda，可在此透传：grl_lambda=grl_lambda
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
            "h_pos": h_pos, "h_neg": h_neg,
            "h_hat_pos": h_hat_pos, "h_hat_neg": h_hat_neg,
            "mu_c_pos": mu_c_pos, "mu_c_neg": mu_c_neg,
            "logvar_c_pos": logvar_c_pos, "logvar_c_neg": logvar_c_neg,
            "mu_nc_pos": mu_nc_pos, "mu_nc_neg": mu_nc_neg,
            "logvar_nc_pos": logvar_nc_pos, "logvar_nc_neg": logvar_nc_neg,
            "adv_logits": (adv_pos, adv_neg),
        }
        return chosen_reward, rejected_reward, extras_out

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
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
            pad_to_length(reject_ids, max_len_input, self.tokenizer.pad_token_id)
        ], dim=0)

        max_len_mask = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat([
            pad_to_length(c_mask, max_len_mask, 0),
            pad_to_length(r_mask, max_len_mask, 0)
        ], dim=0)

        return input_ids, att_masks

    def save_logs_and_checkpoints(self, args, global_step, logs_dict, client_states):
        if global_step % args.logging_steps == 0 and self.strategy.is_rank_0():
            if self._wandb:
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard:
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        if (
            global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            if len(self.eval_dataloader) > 0:
                self.evaluate(global_step)

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states)
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    @torch.no_grad()
    def evaluate(self, steps=0):
        step_bar = tqdm(
            range(len(self.eval_dataloader)),
            desc=f"Eval stage of steps {steps}",
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        acc = 0.0
        adv_acc_sum = 0.0
        loss_sum = 0.0
        rec_sum = 0.0
        kl_c_sum = 0.0
        kl_nc_sum = 0.0
        rewards = []

        for data in self.eval_dataloader:
            chosen_ids, c_mask, reject_ids, r_mask, margin = data
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

            chosen_reward, rejected_reward, extras = self.concatenated_forward(
                self.model, chosen_ids, c_mask, reject_ids, r_mask, grl_lambda=0.0  # 若模型支持，可关闭GRL
            )

            if self.margin_loss:
                margin = torch.tensor(margin).to(torch.cuda.current_device())
            else:
                margin = None

            # Pairwise losses for reporting
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

            rewards += [chosen_reward.flatten(), rejected_reward.flatten()]
            step_bar.update()

        # Aggregate metrics
        acc_mean = acc / len(self.eval_dataloader)
        adv_acc_mean = adv_acc_sum / len(self.eval_dataloader)
        rec_mse_mean = rec_sum / len(self.eval_dataloader)
        kl_c_mean = kl_c_sum / len(self.eval_dataloader)
        kl_nc_mean = kl_nc_sum / len(self.eval_dataloader)
        eval_loss_mean = loss_sum / len(self.eval_dataloader)

        rewards = torch.cat(rewards).float()
        rewards = self.strategy.all_gather(rewards)
        reward_mean = torch.mean(rewards)
        reward_std = torch.std(rewards).clamp(min=1e-8)

        self.strategy.print("Set reward mean std")
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

        # 与 FactoredVAE 保持一致：打印 rewards 直方图
        histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2
        self.strategy.print("histgram")
        self.strategy.print(histgram)

        if self.strategy.is_rank_0():
            if self._wandb is not None:
                self._wandb.log({f"eval/{k}": v for k, v in {**logs, "global_step": steps}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, steps)

        self.model.train()
