import os
from abc import ABC

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from openrlhf.models import LogExpLoss, PairWiseLoss
from openrlhf.utils.distributed_sampler import DistributedSampler


def _kl_standard_normal(mu, logvar):
    # KL(q||p), q = N(mu, diag(exp(logvar))), p = N(0, I)
    # per-sample: 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    return 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)


class CausalRewardModelTrainer(ABC):
    """
    Trainer for training a factored causal reward model (causal_reward).

    Combines:
      - Pairwise preference loss on reward head (L_pred)
      - Reconstruction loss on h_hat vs h + KL losses on q_c, q_nc (L_rec + beta_kl_c, beta_kl_nc)
      - Adversarial loss on s_nc via gradient reversal (L_adv)
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

        # Pairwise loss
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()
            self.strategy.print("LogSigmoid Loss (pairwise preference)")
        else:
            self.loss_fn = LogExpLoss()
            self.strategy.print("LogExp Loss (pairwise preference)")

        # Additional weights/schedules
        self.lambda_rec = getattr(self.args, "lambda_rec", 1.0)
        self.lambda_pred = getattr(self.args, "lambda_pred", 1.0)
        self.lambda_adv = getattr(self.args, "lambda_adv", 1.0)
        self.beta_kl_c = getattr(self.args, "beta_kl_c", 1.0)
        self.beta_kl_nc = getattr(self.args, "beta_kl_nc", 1.0)
        self.grl_lambda = getattr(self.args, "grl_lambda", 1.0)

        self.kl_anneal_steps = getattr(self.args, "kl_anneal_steps", 0)
        self.rec_warmup_steps = getattr(self.args, "rec_warmup_steps", 0)

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # pairwise margin (optional)
        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

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

    def _anneal_factor(self, global_step, total_steps):
        # KL/rec annealing schedules: simple linear ramp to 1
        kl_factor = 1.0
        rec_factor = 1.0
        if self.kl_anneal_steps and self.kl_anneal_steps > 0:
            kl_factor = min(1.0, max(0.0, global_step / float(self.kl_anneal_steps)))
        if self.rec_warmup_steps and self.rec_warmup_steps > 0:
            rec_factor = min(1.0, max(0.0, global_step / float(self.rec_warmup_steps)))
        return kl_factor, rec_factor

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

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for data in self.train_dataloader:
                chosen_ids, c_mask, reject_ids, r_mask, margin = data
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())

                # Forward pass, fetch rewards and all latent/recon/adversary outputs
                (
                    chosen_reward,
                    reject_reward,
                    extras,
                ) = self.concatenated_forward(self.model, chosen_ids, c_mask, reject_ids, r_mask)

                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None

                # Pairwise preference loss
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()
                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)

                # Reconstruction loss (MSE) on h_hat vs h
                h_pos, h_hat_pos = extras["h_pos"], extras["h_hat_pos"]
                h_neg, h_hat_neg = extras["h_neg"], extras["h_hat_neg"]
                rec_pos = F.mse_loss(h_hat_pos, h_pos, reduction="mean")
                rec_neg = F.mse_loss(h_hat_neg, h_neg, reduction="mean")
                rec_loss = 0.5 * (rec_pos + rec_neg)

                # KL losses
                kl_c_pos = _kl_standard_normal(extras["mu_c_pos"], extras["logvar_c_pos"]).mean()
                kl_c_neg = _kl_standard_normal(extras["mu_c_neg"], extras["logvar_c_neg"]).mean()
                kl_c = 0.5 * (kl_c_pos + kl_c_neg)

                kl_n_pos = _kl_standard_normal(extras["mu_nc_pos"], extras["logvar_nc_pos"]).mean()
                kl_n_neg = _kl_standard_normal(extras["mu_nc_neg"], extras["logvar_nc_neg"]).mean()
                kl_n = 0.5 * (kl_n_pos + kl_n_neg)

                # Adversarial loss (pairwise)
                adv_pos, adv_neg = extras["adv_pos"], extras["adv_neg"]
                if self.compute_fp32_loss:
                    adv_pos = adv_pos.float()
                    adv_neg = adv_neg.float()
                adv_loss = self.loss_fn(adv_pos, adv_neg, None)

                # GRL: reverse gradient for s_nc
                grl_scale = getattr(self.args, "grl_lambda", self.grl_lambda)
                extras["s_nc_pos"].register_hook(lambda grad: -grl_scale * grad if grad is not None else grad)
                extras["s_nc_neg"].register_hook(lambda grad: -grl_scale * grad if grad is not None else grad)

                # Optional annealing
                global_step = (step // self.strategy.accumulated_gradient)
                kl_factor, rec_factor = self._anneal_factor(global_step, self.num_update_steps_per_epoch)

                # Total loss
                total_loss = (
                    self.lambda_pred * preference_loss
                    + self.lambda_rec * rec_factor * rec_loss
                    + kl_factor * (self.beta_kl_c * kl_c + self.beta_kl_nc * kl_n)
                    + self.lambda_adv * adv_loss
                )

                # Backward/update
                self.strategy.backward(total_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                # Metrics
                acc = (chosen_reward > reject_reward).float().mean().item()
                adv_acc = (adv_pos > adv_neg).float().mean().item()

                acc_sum += acc
                loss_sum += preference_loss.item()
                logs_dict = {
                    "loss_pred": preference_loss.item(),
                    "loss_rec": rec_loss.item(),
                    "loss_kl_c": kl_c.item(),
                    "loss_kl_nc": kl_n.item(),
                    "loss_adv": adv_loss.item(),
                    "loss_total": total_loss.item(),
                    "acc": acc,
                    "adv_acc": adv_acc,
                    "chosen_reward": chosen_reward.mean().item(),
                    "reject_reward": reject_reward.mean().item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "kl_factor": kl_factor,
                    "rec_factor": rec_factor,
                }

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
                    client_states = {"consumed_samples": (step // self.strategy.accumulated_gradient) * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, step // self.strategy.accumulated_gradient, step_bar, logs_dict, client_states)

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

        # eval
        if (
            global_step % args.eval_steps == 0 or global_step % self.num_update_steps_per_epoch == 0
        ) and self.eval_dataloader is not None:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)

        # save ckpt
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
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        self.model.eval()
        with torch.no_grad():
            acc = 0
            rewards = []
            loss_sum = 0
            adv_acc_sum = 0
            rec_sum = 0
            klc_sum = 0
            kln_sum = 0

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

                loss = self.loss_fn(chosen_reward, reject_reward, margin)

                # diagnostics (no grad)
                adv_pos, adv_neg = extras["adv_pos"], extras["adv_neg"]
                adv_acc = (adv_pos > adv_neg).float().mean().item()

                h_pos, h_hat_pos = extras["h_pos"], extras["h_hat_pos"]
                h_neg, h_hat_neg = extras["h_neg"], extras["h_hat_neg"]
                rec_pos = F.mse_loss(h_hat_pos, h_pos, reduction="mean").item()
                rec_neg = F.mse_loss(h_hat_neg, h_neg, reduction="mean").item()
                rec_loss = 0.5 * (rec_pos + rec_neg)

                kl_c_pos = _kl_standard_normal(extras["mu_c_pos"], extras["logvar_c_pos"]).mean().item()
                kl_c_neg = _kl_standard_normal(extras["mu_c_neg"], extras["logvar_c_neg"]).mean().item()
                kl_n_pos = _kl_standard_normal(extras["mu_nc_pos"], extras["logvar_nc_pos"]).mean().item()
                kl_n_neg = _kl_standard_normal(extras["mu_nc_neg"], extras["logvar_nc_neg"]).mean().item()

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]
                acc += (chosen_reward > reject_reward).float().mean().item()
                adv_acc_sum += adv_acc
                rec_sum += rec_loss
                klc_sum += 0.5 * (kl_c_pos + kl_c_neg)
                kln_sum += 0.5 * (kl_n_pos + kl_n_neg)
                loss_sum += loss.item()
                step_bar.update()

            acc_mean = acc / eval_dataloader.__len__()
            adv_acc_mean = adv_acc_sum / eval_dataloader.__len__()
            rec_mean = rec_sum / eval_dataloader.__len__()
            klc_mean = klc_sum / eval_dataloader.__len__()
            kln_mean = kln_sum / eval_dataloader.__len__()
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
                "adv_acc_mean": adv_acc_mean,
                "rec_mse_mean": rec_mean,
                "kl_c_mean": klc_mean,
                "kl_nc_mean": kln_mean,
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
        self.model.train()  # reset model state

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """
        Concatenate chosen and rejected inputs into a single batch and forward the model once.
        Returns:
          - rewards for chosen and rejected halves
          - a dict of extras for losses and diagnostics
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        # Ask model to return latent, recon, adversary outputs
        all_values, output = model(
            input_ids,
            attention_mask=att_masks,
            return_output=True,
            return_latent=True,
            return_recon=True,
            return_adv=True,
        )

        batch_size = chosen_ids.shape[0]
        chosen_rewards = all_values[:batch_size]
        rejected_rewards = all_values[batch_size:]

        # Split extras along batch dimension
        def split(t):
            return (t[:batch_size], t[batch_size:])

        h_pos, h_neg = split(output["h"])
        h_hat_pos, h_hat_neg = split(output["h_hat"])
        mu_c_pos, mu_c_neg = split(output["mu_c"])
        logvar_c_pos, logvar_c_neg = split(output["logvar_c"])
        mu_nc_pos, mu_nc_neg = split(output["mu_nc"])
        logvar_nc_pos, logvar_nc_neg = split(output["logvar_nc"])
        adv_pos, adv_neg = split(output["adv_score"])
        s_nc_pos, s_nc_neg = split(output["s_nc"])

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
            "adv_pos": adv_pos,
            "adv_neg": adv_neg,
            "s_nc_pos": s_nc_pos,
            "s_nc_neg": s_nc_neg,
        }

        return chosen_rewards, rejected_rewards, extras

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor."""
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
