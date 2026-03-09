# openrlhf/models/model_crm.py

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


def get_llm_for_sequence_regression_ppo_critic(
    model_name_or_path: str,
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
    value_head_prefix: str = "score",
    device_map=None,
    packing_samples: bool = False,
    # 与 CRM 一致的 latent 结构
    latent_dim_c: int = 512,
    enc_hidden: int = 1024,
    **kwargs,
) -> nn.Module:
    """
    构建一个与 Causal Reward Model 兼容的 Critic，用于 PPO。

    结构与 openrlhf.models.model._get_critic_model 基本一致，
    只是将 value head 从 hidden_size 接到 latent_dim_c，
    并在其前引入 encoder -> mu_c 的 causal path。
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # 与主 model.py 一致：优先使用 config 中的 value_head_prefix
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"[CRM-PPO] set value_head_prefix to `{value_head_prefix}`")

    # 注入 latent 结构配置（必须与 CRM 一致）
    config.latent_dim_c = latent_dim_c
    config.enc_hidden = enc_hidden

    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_causal_critic_model_for_ppo(
        base_pretrained_model=base_pretrained_class,
        base_llm_model=base_class,
        value_head_prefix=value_head_prefix,
        packing_samples=packing_samples,
    )

    # Deepspeed helper
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    # 量化设置
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

    # LoRA（与主 model.py 相同）
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
        logger.info("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # cache 问题
    model.config.use_cache = False

    # 初始化 value_head（如果需要）
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("[CRM-PPO] initialize_value_head for ZeRO-3 critic model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim_c + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.latent_dim_c + 1))

    return model


def _get_causal_critic_model_for_ppo(
    base_pretrained_model,
    base_llm_model,
    value_head_prefix: str = "score",
    packing_samples: bool = False,
):
    """
    Critic 模型：接口与 openrlhf.models.model._get_critic_model 完全一致，
    只是在 hidden -> value_head 之间插入 encoder -> mu_c（latent_dim_c）。
    """

    class CriticModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            # backbone
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            self.packing_samples = packing_samples

            # === 与 CRM 一致的 latent 结构 ===
            self.latent_dim_c = getattr(config, "latent_dim_c", 512)
            self.enc_hidden = getattr(config, "enc_hidden", 1024)

            # Shared encoder: hidden_size -> enc_hidden
            self.encoder = nn.Sequential(
                nn.Linear(config.hidden_size, self.enc_hidden),
                nn.ReLU(),
                nn.Linear(self.enc_hidden, self.enc_hidden),
                nn.ReLU(),
            )

            # latent head: enc_hidden -> latent_dim_c
            self.fc_mu_c = nn.Linear(self.enc_hidden, self.latent_dim_c)

            # value head on latent
            setattr(self, value_head_prefix, nn.Linear(self.latent_dim_c, 1, bias=False))

            # mean std（与原 critic 相同字段）
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
            return_output: bool = False,
            ring_attn_group=None,
            values_allgather: bool = False,
            packed_seq_lens=None,
        ) -> torch.Tensor:
            """
            完全复用原 _get_critic_model 的逻辑：
              - 计算 last_hidden_states；
              - 用 value_head 产生 [B, L] 的 values；
              - values = values[:, :-1]；
              - 归一化；
              - action_values = values[:, -action_mask.shape[1]:] * action_mask。
            差别仅在于 value 的产生方式：hidden -> encoder -> mu_c -> value_head。
            """
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
                input_ids,
                attention_mask=forward_attention_mask,
                position_ids=position_ids,
            )

            if action_mask is None:
                assert return_output
                return outputs

            last_hidden_states = outputs["last_hidden_state"]  # [B, L, H]

            # === 替换原来的 Linear(hidden_size -> 1) 为：hidden -> encoder -> mu_c -> value_head ===
            shared = self.encoder(last_hidden_states)              # [B, L, enc_hidden]
            mu_c = self.fc_mu_c(shared)                            # [B, L, latent_dim_c]
            value_head = getattr(self, self.value_head_prefix)
            values = value_head(mu_c).squeeze(-1)                  # [B, L]

            if self.packing_samples:
                values = gather_and_pad_tensor(values, ring_attn_group, ring_attn_pad_len, indices, batch, seqlen)

            values = values[:, :-1]

            # normalize reward（保持原字段名和逻辑，PPO 里不区分 reward/value）
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            action_values = values[:, -action_mask.shape[1] :] * action_mask.float()

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
