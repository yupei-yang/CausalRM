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
    assert model_type in ["critic", "reward", "causal_reward", "encoded_reward", "vae_reward", "stoch_reward", "factored_vae_reward"], f"invalid model_type: {model_type}"

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = attn_implementation

    # Inject causal RM config
    if model_type == "causal_reward":
        config.latent_dim_c = latent_dim_c
        config.latent_dim_nc = latent_dim_nc
        # config.encoder_hidden = encoder_hidden
        config.recon_hidden = recon_hidden
        config.adv_hidden = adv_hidden

    if model_type in ["causal_reward", "encoded_reward", "factored_vae_reward"]:
        config.encoder_hidden = encoder_hidden   

    if model_type in ["vae_reward", "stoch_reward"]:
        config.latent_dim = latent_dim_c       # reuse flag
        config.encoder_hidden = encoder_hidden

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
    elif model_type == "encoded_reward":
        cls_class = _get_encoded_reward_model(
            base_pretrained_class, base_class, value_head_prefix, packing_samples
        )
    elif model_type == "vae_reward":
        cls_class = _get_vae_reward_model(
            base_pretrained_class, base_class, value_head_prefix, packing_samples
        )
    elif model_type == "stoch_reward":
        cls_class = _get_stoch_reward_model(
            base_pretrained_class, base_class, value_head_prefix, packing_samples
        )
    elif model_type == "factored_vae_reward":
        cls_class = _get_factored_vae_reward_model(
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


def _get_causal_reward_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="score",
    packing_samples=False,
):
    """
    Implements a Causal Reward Model with Factorized Latent Space and Adversarial Disentanglement.
    
    Architecture:
        h = LLM(x, y)[-1]
        shared = Encoder(h)
        -> mu_c, logvar_c      (causal factor)
        -> mu_nc, logvar_nc    (non-causal factor)

        Reward Predictor: r = RewardHead(mu_c)
        Adversary Predictor: a = AdvHead(GRL(mu_nc))

    Training Objective:
        L_total = λ_pred * L_ranking(r+) + λ_adv * (-L_ranking(a+)) + β_kl_c * KL_c + β_kl_nc * KL_nc + λ_rec * MSE(recon)
    """
    class CausalRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward
            hidden_size = config.hidden_size
            enc_hid = getattr(config, "encoder_hidden", 512)

            # Latent dimensions
            self.latent_dim_c = getattr(config, "latent_dim_c", 512)      # causal
            self.latent_dim_nc = getattr(config, "latent_dim_nc", 512)    # non-causal
            self.recon_hidden = getattr(config, "recon_hidden", 1024)
            self.adv_hidden = getattr(config, "adv_hidden", 128)

            # === Shared Encoder ===
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, enc_hid),
                nn.ReLU(),
                nn.Linear(enc_hid, enc_hid),
                nn.ReLU()
            )

            # === Causal Factor Heads ===
            self.fc_mu_c = nn.Linear(enc_hid, self.latent_dim_c)
            self.fc_logvar_c = nn.Linear(enc_hid, self.latent_dim_c)

            # === Non-Causal Factor Heads ===
            self.fc_mu_nc = nn.Linear(enc_hid, self.latent_dim_nc)
            self.fc_logvar_nc = nn.Linear(enc_hid, self.latent_dim_nc)

            # === Reward Head (only on s^c) ===
            self.reward_head = nn.Linear(self.latent_dim_c, 1, bias=False)

            # === Reconstructor: [z_c; z_nc] -> h_hat ===
            self.reconstructor = nn.Sequential(
                nn.Linear(self.latent_dim_c + self.latent_dim_nc, self.recon_hidden),
                nn.ReLU(),
                nn.Linear(self.recon_hidden, self.recon_hidden),
                nn.ReLU(),
                nn.Linear(self.recon_hidden, hidden_size)
            )

            # === Adversary Head: predicts reward from s^{nc} (to be fooled) ===
            self.adversary_head = nn.Sequential(
                nn.Linear(self.latent_dim_nc, self.adv_hidden),
                nn.ReLU(),
                nn.Linear(self.adv_hidden, self.adv_hidden),
                nn.ReLU(),
                nn.Linear(self.adv_hidden, 1)
            )

            # Buffers for normalization
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            return_output=False,
            return_latent=False,
            return_recon=False,
            return_adv=False,
        ):
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=None,
            )
            last_hidden = outputs["last_hidden_state"]  # [B, L, H]
            h = last_hidden[torch.arange(batch), eos_indices]  # [B, H]

            # Encode to shared space
            shared = self.encoder(h)

            # Causal factor
            mu_c = self.fc_mu_c(shared)
            logvar_c = self.fc_logvar_c(shared)
            z_c = self.reparameterize(mu_c, logvar_c)

            # Non-causal factor
            mu_nc = self.fc_mu_nc(shared)
            logvar_nc = self.fc_logvar_nc(shared)
            z_nc = self.reparameterize(mu_nc, logvar_nc)

            # Reward prediction using mu_c (deterministic for stability)
            reward = self.reward_head(mu_c).squeeze(-1)

            # Reconstruction
            z_concat = torch.cat([z_c, z_nc], dim=-1)
            h_hat = self.reconstructor(z_concat)

            # Adversary prediction (uses GRL on mu_nc)
            if return_adv:
                mu_nc_grl = GradientReversal.apply(mu_nc, 1.0)  # lambda=1.0 controlled externally later
                adv_logits = self.adversary_head(mu_nc_grl).squeeze(-1)
            else:
                adv_logits = None

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            extra_outputs = {}
            if return_latent or return_output:
                extra_outputs.update({
                    "mu_c": mu_c.detach(), "logvar_c": logvar_c.detach(),
                    "mu_nc": mu_nc.detach(), "logvar_nc": logvar_nc.detach(),
                    "h": h.detach()
                })
            if return_recon or return_output:
                extra_outputs["h_hat"] = h_hat
            if return_adv and adv_logits is not None:
                extra_outputs["adv_logits"] = adv_logits

            if return_output:
                return reward, extra_outputs
            return reward

    return CausalRewardModel


# --- Gradient Reversal Layer ---
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def _get_encoded_reward_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="score",
    packing_samples=False,
):
    class EncodedRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward
            hidden_size = config.hidden_size
            enc_hid = getattr(config, "encoder_hidden", hidden_size)

            # simple encoder
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, enc_hid),
                nn.GELU(),
            )

            # reward head
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(enc_hid, 1, bias=False))

            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            return_output=False,
            **kwargs,
        ):
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=None,
            )
            last_hidden = outputs["last_hidden_state"]                # [B,L,H]
            h = last_hidden[torch.arange(batch), eos_indices]          # [B,H]

            z = self.encoder(h)                                        # encoded repr
            reward = getattr(self, self.value_head_prefix)(z).squeeze(-1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            # keep interface
            if return_output:
                return reward, outputs
            return reward

    return EncodedRewardModel


def _get_vae_reward_model(base_pretrained_model,
                          base_llm_model,
                          value_head_prefix="score",
                          packing_samples=False):
    class VAEResRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True
        def __init__(self, config:AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward

            H = config.hidden_size
            h_enc = config.encoder_hidden
            z_dim = config.latent_dim

            # Encoder
            self.enc = nn.Sequential(
                nn.Linear(H, h_enc), nn.GELU(),
            )
            self.mu   = nn.Linear(h_enc, z_dim)
            self.logv = nn.Linear(h_enc, z_dim)

            # Decoder
            self.dec = nn.Sequential(
                nn.Linear(z_dim, h_enc), nn.GELU(),
                nn.Linear(h_enc, H)
            )

            # Reward head
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(z_dim, 1, bias=False))

            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1),  persistent=False)
            if hasattr(config,"mean"):
                self.mean[0]=config.mean; self.std[0]=config.std

        def _reparam(self, mu, logv):
            std = torch.exp(0.5*logv)
            eps = torch.randn_like(std)
            return mu + eps*std

        def forward(self, input_ids=None, attention_mask=None, return_output=False, return_vae=False):
            B,L = input_ids.size()
            eos = attention_mask.size(1)-1 - attention_mask.long().fliplr().argmax(dim=1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask
            )
            hid = outputs.last_hidden_state
            h = hid[torch.arange(B, device=hid.device), eos]     # [B, H]

            # encode
            h_mid = self.enc(h)
            mu = self.mu(h_mid)
            logv = self.logv(h_mid)
            z = self._reparam(mu, logv)                          # [B,z_dim]

            # reward
            #reward = getattr(self, self.value_head_prefix)(z).squeeze(-1)
            reward = getattr(self, self.value_head_prefix)(mu).squeeze(-1)

            # recon
            h_hat = self.dec(z)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean)/self.std

            if return_output or return_vae:
                extra = {"h":h, "h_hat":h_hat,
                         "mu":mu, "logvar":logv}
                return reward, extra
            return reward

    return VAEResRewardModel


# ============= Stochastic Reward Model (StochRM) =============
def _get_stoch_reward_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="score",
    packing_samples=False,
):
    class StochRewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward
            hidden_size = config.hidden_size
            self.latent_dim = getattr(config, "latent_dim", 512)  # Default to 512
            enc_hid = getattr(config, "encoder_hidden", 1024)     # Default to 1024

            # === Stochastic Encoder: h -> (mu, logvar) ===
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, enc_hid),
                nn.ReLU(),
                nn.Linear(enc_hid, enc_hid),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(enc_hid, self.latent_dim)
            self.fc_logvar = nn.Linear(enc_hid, self.latent_dim)

            # === Reward Head: z -> scalar ===
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(self.latent_dim, 1, bias=False))

            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            return_output=False,
            return_latent=False,
            **kwargs,
        ):
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=None,
            )
            last_hidden = outputs["last_hidden_state"]                # [B,L,H]
            h = last_hidden[torch.arange(batch), eos_indices]          # [B,H]

            # Stochastic encoding
            enc = self.encoder(h)
            mu = self.fc_mu(enc)
            logvar = self.fc_logvar(enc)
            z = self.reparameterize(mu, logvar)  # Random sampling!

            # Reward prediction
            reward = getattr(self, self.value_head_prefix)(z).squeeze(-1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            extra_outputs = {}
            if return_latent:
                extra_outputs.update({
                    "z": z,
                    "mu": mu,
                    "logvar": logvar,
                    "h": h.detach(),
                })

            if return_output:
                return reward, {**outputs, **extra_outputs}
            return reward

    return StochRewardModel


# ============= Factored VAE Reward Model =============
def _get_factored_vae_reward_model(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix="score",
    packing_samples=False,
):
    class FactoredVAERewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.packing_samples = packing_samples
            self.normalize_reward = config.normalize_reward
            hidden_size = config.hidden_size
            enc_hid = config.encoder_hidden
            
            # Latent dimensions
            self.latent_dim_c = getattr(config, "latent_dim_c", 512)      # causal factor
            self.latent_dim_nc = getattr(config, "latent_dim_nc", 512)    # non-causal factor

            # === Shared Encoder: h -> shared representation ===
            self.encoder = nn.Sequential(
                nn.Linear(hidden_size, enc_hid),
                nn.ReLU(),
                nn.Linear(enc_hid, enc_hid),
                nn.ReLU(),
            )

            # === Causal Factor Heads: shared -> (mu_c, logvar_c) ===
            self.fc_mu_c = nn.Linear(enc_hid, self.latent_dim_c)
            self.fc_logvar_c = nn.Linear(enc_hid, self.latent_dim_c)

            # === Non-Causal Factor Heads: shared -> (mu_nc, logvar_nc) ===
            self.fc_mu_nc = nn.Linear(enc_hid, self.latent_dim_nc)
            self.fc_logvar_nc = nn.Linear(enc_hid, self.latent_dim_nc)

            # === Reward Head: only uses mu_c (deterministic for stability) ===
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(self.latent_dim_c, 1, bias=False))

            # === Reconstructor: uses reparameterized samples [z_c; z_nc] -> h_hat ===
            recon_hid = getattr(config, "recon_hidden", enc_hid)
            self.reconstructor = nn.Sequential(
                nn.Linear(self.latent_dim_c + self.latent_dim_nc, recon_hid),
                nn.ReLU(),
                nn.Linear(recon_hid, recon_hid),
                nn.ReLU(),
                nn.Linear(recon_hid, hidden_size)
            )

            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            return_output=False,
            return_latent=False,
            return_recon=False,
            **kwargs,
        ):
            batch, seqlen = input_ids.size()
            eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=None,
            )
            last_hidden = outputs["last_hidden_state"]                # [B,L,H]
            h = last_hidden[torch.arange(batch), eos_indices]          # [B,H]

            # Shared encoding
            shared = self.encoder(h)
            
            # Causal factor
            mu_c = self.fc_mu_c(shared)
            logvar_c = self.fc_logvar_c(shared)
            z_c = self.reparameterize(mu_c, logvar_c)
            
            # Non-causal factor  
            mu_nc = self.fc_mu_nc(shared)
            logvar_nc = self.fc_logvar_nc(shared)
            z_nc = self.reparameterize(mu_nc, logvar_nc)

            # Reward prediction (using mu_c for stability)
            reward = getattr(self, self.value_head_prefix)(mu_c).squeeze(-1)

            # Reconstruction (using reparameterized samples)
            z_concat = torch.cat([z_c, z_nc], dim=-1)
            h_hat = self.reconstructor(z_concat)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            extra_outputs = {}
            if return_latent or return_output:
                extra_outputs.update({
                    "mu_c": mu_c,
                    "logvar_c": logvar_c,
                    "mu_nc": mu_nc,
                    "logvar_nc": logvar_nc,
                    "h": h.detach(),
                })
            if return_recon or return_output:
                extra_outputs["h_hat"] = h_hat

            if return_output:
                return reward, extra_outputs
            return reward

    return FactoredVAERewardModel
