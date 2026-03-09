from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from openrlhf.utils.logging_utils import init_logger

from .ring_attn_utils import gather_and_pad_tensor, unpad_and_slice_tensor

logger = init_logger(__name__)


# -------------------- Gradient Reversal Layer -------------------- #
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,
    model_type: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    normalize_reward=False,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    # ===== 新增参数（仅用于 causal_reward）=====
    latent_dim_c: int = 512,
    latent_dim_nc: int = 512,
    enc_hidden: int = 1024,
    rec_hidden: int = 1024,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    This function loads a pretrained transformer model and attaches a linear layer for sequence regression.

    Args:
        model_name_or_path (str): Path to the pretrained model.
        model_type (str): Type of the model, "reward", "critic", or "causal_reward".
        bf16 (bool, optional): Enable bfloat16 precision. Defaults to True.
        load_in_4bit (bool, optional): Load the model in 4-bit precision. Defaults to False.
        lora_rank (int, optional): Rank for LoRA adaptation. Defaults to 0.
        lora_alpha (int, optional): Alpha parameter for LoRA. Defaults to 16.
        target_modules (list, optional): List of target modules for LoRA. Defaults to None.
        lora_dropout (float, optional): Dropout rate for LoRA layers. Defaults to 0.
        normalize_reward (bool, optional): Normalize reward values. Defaults to False.
        use_flash_attention_2 (bool, optional): Use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed configuration for model partitioning across multiple GPUs when ZeRO-3 is enabled. Defaults to None.
        init_value_head (bool, optional): Initialize the value head. Defaults to False.
        value_head_prefix (str, optional): Prefix for the value head. Defaults to "score".
        device_map (dict, optional): Map of devices for model loading. Defaults to None.
        packing_samples (bool, optional): Whether to pack samples during training. Defaults to False.
        latent_dim_c (int, optional): Dimension of causal latent factors. Defaults to 512.
        latent_dim_nc (int, optional): Dimension of non-causal latent factors. Defaults to 512.
        enc_hidden (int, optional): Hidden dimension for encoder. Defaults to 1024.
        rec_hidden (int, optional): Hidden dimension for reconstructor. Defaults to 1024.

    Returns:
        nn.Module: A pretrained transformer model with a sequence regression head.
    """
    assert model_type in [
        "critic",
        "reward",
        "causal_reward",
    ], f"invalid model_type: {model_type}, should be critic, reward, or causal_reward."

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # Prioritize using the value_head_prefix in the model configuration.
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    # Inject causal RM config if needed
    if model_type == "causal_reward":
        config.latent_dim_c = latent_dim_c
        config.latent_dim_nc = latent_dim_nc
        config.enc_hidden = enc_hidden
        config.rec_hidden = rec_hidden

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    elif model_type == "critic":
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    elif model_type == "causal_reward":
        cls_class = _get_causal_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # MoE - balancing loss
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # NOTE: For reward model training only, intialize value_head manually
    # because deepspeed.zero.Init() will not intialize them.
    # TODO: Find a better way to clarify reward model training.
    if init_value_head:
        if model_type == "causal_reward":
            # Initialize all heads for causal reward model
            value_head = getattr(model, value_head_prefix)
            adversary_head = getattr(model, "adversary_head", None)

            if dschf is not None:
                logger.info("initialize value_head and adversary_head for ZeRO-3 causal reward model training.")
                params = [value_head.weight]
                if adversary_head is not None:
                    params.append(adversary_head.weight)
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        value_head.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim_c + 1))
                        if adversary_head is not None:
                            adversary_head.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim_nc + 1))
            else:
                value_head.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim_c + 1))
                if adversary_head is not None:
                    adversary_head.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim_nc + 1))

            # Initialize encoder and reconstructor
            for module in [getattr(model, "encoder", None), getattr(model, "reconstructor", None)]:
                if module is None:
                    continue
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.normal_(mean=0.0, std=0.02)
                        if layer.bias is not None:
                            layer.bias.data.zero_()

            # Initialize VAE heads
            for module in [
                getattr(model, "fc_mu_c", None),
                getattr(model, "fc_logvar_c", None),
                getattr(model, "fc_mu_nc", None),
                getattr(model, "fc_logvar_nc", None),
            ]:
                if module is None:
                    continue
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        else:
            # Original initialization for standard reward/critic models
            value_head = getattr(model, value_head_prefix)
            if dschf is not None:
                logger.info("initialize_value_head for ZeRO-3 reward/critic model training.")
                with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            else:
                value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)
            reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            values_allgather=False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            batch, seqlen = input_ids.size()
            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )

            if action_mask is None:
                assert return_output
                return outputs

            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)  # (1, total_seqs)

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]
            # normalize reward
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel


def _get_causal_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    """Causal Reward Model with factorized latent space and adversarial disentanglement."""
    class CausalRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples

            # Config for latent structure
            self.latent_dim_c = getattr(config, "latent_dim_c", 512)
            self.latent_dim_nc = getattr(config, "latent_dim_nc", 512)
            self.enc_hidden = getattr(config, "enc_hidden", 1024)
            self.rec_hidden = getattr(config, "rec_hidden", 1024)

            # Shared encoder: h -> shared
            self.encoder = nn.Sequential(
                nn.Linear(config.hidden_size, self.enc_hidden),
                nn.ReLU(),
                nn.Linear(self.enc_hidden, self.enc_hidden),
                nn.ReLU(),
            )

            # Causal factor heads
            self.fc_mu_c = nn.Linear(self.enc_hidden, self.latent_dim_c)
            self.fc_logvar_c = nn.Linear(self.enc_hidden, self.latent_dim_c)

            # Non-causal factor heads
            self.fc_mu_nc = nn.Linear(self.enc_hidden, self.latent_dim_nc)
            self.fc_logvar_nc = nn.Linear(self.enc_hidden, self.latent_dim_nc)

            # Reward head (on causal latent)
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(self.latent_dim_c, 1, bias=False))

            # Reconstructor: [z_c; z_nc] -> h_hat
            self.reconstructor = nn.Sequential(
                nn.Linear(self.latent_dim_c + self.latent_dim_nc, self.rec_hidden),
                nn.ReLU(),
                nn.Linear(self.rec_hidden, self.rec_hidden),
                nn.ReLU(),
                nn.Linear(self.rec_hidden, config.hidden_size),
            )

            # Adversary head: predicts reward from non-causal latent
            self.adversary_head = nn.Linear(self.latent_dim_nc, 1, bias=False)

            # normalization
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

            # GRL strength，可在 trainer 外部调节的话可以做成 config.grl_lambda
            self.grl_lambda = getattr(config, "grl_lambda", 1.0)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output: bool = False,
            return_latent: bool = False,
            return_recon: bool = False,
            return_adv: bool = False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
        ):
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)

            forward_attention_mask = attention_mask
            if self.packing_samples:
                input_ids, position_ids, _, ring_attn_pad_len, indices = unpad_and_slice_tensor(
                    input_ids, attention_mask, ring_attn_group
                )
                forward_attention_mask = None
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=forward_attention_mask,
                position_ids=position_ids,
            )
            last_hidden_states = outputs["last_hidden_state"]

            if self.packing_samples:
                last_hidden_states = gather_and_pad_tensor(
                    last_hidden_states, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen
                )

            # last_hidden_states: [B, L, H]
            # eos_indices: [B, 1] → squeeze to [B]
            batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
            h = last_hidden_states[batch_indices, eos_indices.squeeze(-1)]  # [B, H]

            # Encode
            shared = self.encoder(h)

            # Causal latent
            mu_c = self.fc_mu_c(shared)
            logvar_c = self.fc_logvar_c(shared)
            z_c = self.reparameterize(mu_c, logvar_c)

            # Non-causal latent
            mu_nc = self.fc_mu_nc(shared)
            logvar_nc = self.fc_logvar_nc(shared)
            z_nc = self.reparameterize(mu_nc, logvar_nc)

            # Reward from causal latent (use mu_c for stability)
            value_head = getattr(self, self.value_head_prefix)
            reward = value_head(mu_c).squeeze(-1)

            # Reconstruction
            z_concat = torch.cat([z_c, z_nc], dim=-1)
            h_hat = self.reconstructor(z_concat)

            # Adversary prediction from non-causal latent with GRL
            if return_adv:
                mu_nc_grl = GradientReversal.apply(mu_nc, self.grl_lambda)
                adv_logits = self.adversary_head(mu_nc_grl).squeeze(-1)
            else:
                adv_logits = None

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            # 如果不需要额外输出，就和原 reward model 一样只返回 reward
            if not (return_output or return_latent or return_recon or return_adv):
                return reward

            extra_outputs = {
                "backbone_outputs": outputs,
            }

            if return_latent or return_output:
                extra_outputs.update(
                    {
                        "mu_c": mu_c,
                        "logvar_c": logvar_c,
                        "z_c": z_c,
                        "mu_nc": mu_nc,
                        "logvar_nc": logvar_nc,
                        "z_nc": z_nc,
                        "h": h,
                    }
                )

            if return_recon or return_output:
                extra_outputs["h_hat"] = h_hat

            if return_adv and adv_logits is not None:
                extra_outputs["adv_logits"] = adv_logits

            return reward, extra_outputs

    return CausalRewardModel
