from typing import Optional, Tuple

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
    attn_implementation="flash_attention_2",
    ds_config: dict = None,
    init_value_head=False,
    value_head_prefix="score",
    device_map=None,
    packing_samples=False,
    # ===== 新增参数 =====
    latent_dim_c: int = 64,
    latent_dim_nc: int = 64,
    encoder_hidden: int = 256,
    recon_hidden: int = 512,
    adv_hidden: int = 128,
    **kwargs,
) -> nn.Module:
    """Retrieve a transformer model with a sequence regression head on top.

    Supports:
        - "reward": standard RM
        - "critic": for PPO
        - "causal_reward": factored causal RM (proposed)
    """
    assert model_type in ["critic", "reward", "causal_reward"], f"invalid model_type: {model_type}"

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = attn_implementation

    # Inject causal RM config
    if model_type == "causal_reward":
        config.latent_dim_c = latent_dim_c
        config.latent_dim_nc = latent_dim_nc
        config.encoder_hidden = encoder_hidden
        config.recon_hidden = recon_hidden
        config.adv_hidden = adv_hidden

    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    elif model_type == "critic":
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    elif model_type == "causal_reward":
        cls_class = _get_causal_reward_model(
            base_pretrained_class, base_class, value_head_prefix, packing_samples
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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

        # set_z3_leaf_modules is required for MoE models
        for m in model.modules():
            # https://github.com/microsoft/DeepSpeed/pull/4966
            if "SparseMoeBlock" in m.__class__.__name__:
                deepspeed.utils.set_z3_leaf_modules(model, [m.__class__])
                print(f"Setting zero3 leaf for model on class with name: {m.__class__.__name__}")
                break

    # https://github.com/huggingface/transformers/issues/26877
    model.config.use_cache = False

    # For causal_reward, we don't use init_value_head (we have custom heads)
    # So skip the value_head init block for causal_reward
    if init_value_head and model_type != "causal_reward":
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
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


# ============= 新增：Causal Reward Model =============
def _get_causal_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    class FactoredCausalRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward

            # Latent dimensions
            self.latent_dim_c = config.latent_dim_c
            self.latent_dim_nc = config.latent_dim_nc
            hidden_size = config.hidden_size

            # === Encoder: h -> (mu_c, logvar_c, mu_nc, logvar_nc) ===
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, config.encoder_hidden),
                nn.ReLU(),
                nn.Linear(config.encoder_hidden, config.encoder_hidden),
                nn.ReLU(),
            )
            self.fc_mu_c = nn.Linear(config.encoder_hidden, self.latent_dim_c)
            self.fc_logvar_c = nn.Linear(config.encoder_hidden, self.latent_dim_c)
            self.fc_mu_nc = nn.Linear(config.encoder_hidden, self.latent_dim_nc)
            self.fc_logvar_nc = nn.Linear(config.encoder_hidden, self.latent_dim_nc)

            # === Reward Head: s^c -> scalar ===
            self.reward_head = nn.Sequential(
                nn.Linear(self.latent_dim_c, 64),
                nn.ReLU(),
                nn.Linear(64, 1, bias=False)
            )

            # === Reconstructor: [s^c; s^{nc}] -> h_hat ===
            self.reconstructor = nn.Sequential(
                nn.Linear(self.latent_dim_c + self.latent_dim_nc, config.recon_hidden),
                nn.ReLU(),
                nn.Linear(config.recon_hidden, config.recon_hidden),
                nn.ReLU(),
                nn.Linear(config.recon_hidden, hidden_size)
            )

            # === Adversary Head: s^{nc} -> scalar (for GRL) ===
            self.adversary_head = nn.Sequential(
                nn.Linear(self.latent_dim_nc, config.adv_hidden),
                nn.ReLU(),
                nn.Linear(config.adv_hidden, 1, bias=False)
            )

            # Buffers for normalization (same as standard RM)
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def encode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Encode h into (mu_c, logvar_c, mu_nc, logvar_nc)"""
            enc = self.encoder(h)
            mu_c = self.fc_mu_c(enc)
            logvar_c = self.fc_logvar_c(enc)
            mu_nc = self.fc_mu_nc(enc)
            logvar_nc = self.fc_logvar_nc(enc)
            return mu_c, logvar_c, mu_nc, logvar_nc

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
            ring_attn_group=None,
            pad_sequence=False,
            packed_seq_lens=None,
            # ===== 新增：是否返回 latent 和 recon =====
            return_latent: bool = False,
            return_recon: bool = False,
            return_adv: bool = False,
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
                input_ids, attention_mask=forward_attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]

            # Extract h: use EOS token representation via advanced indexing
            eos_positions = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)  # [batch_size]
            h = last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), eos_positions]

            # Encode into latent factors
            mu_c, logvar_c, mu_nc, logvar_nc = self.encode(h)
            s_c = self.reparameterize(mu_c, logvar_c)
            s_nc = self.reparameterize(mu_nc, logvar_nc)

            # Reward prediction (only from s^c)
            reward = self.reward_head(s_c).squeeze(-1)

            # Normalize if needed (only during eval)
            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            # Prepare extra outputs
            extra_outputs = {}
            if return_latent:
                extra_outputs.update({
                    "s_c": s_c,
                    "s_nc": s_nc,
                    "mu_c": mu_c,
                    "logvar_c": logvar_c,
                    "mu_nc": mu_nc,
                    "logvar_nc": logvar_nc,
                    "h": h.detach(),  # original embedding
                })
            if return_recon:
                s_concat = torch.cat([s_c, s_nc], dim=-1)
                h_hat = self.reconstructor(s_concat)
                extra_outputs["h_hat"] = h_hat
            if return_adv:
                adv_score = self.adversary_head(s_nc).squeeze(-1)
                extra_outputs["adv_score"] = adv_score

            if return_output:
                return (reward, {**outputs, **extra_outputs})
            else:
                return reward

    return FactoredCausalRewardModel
