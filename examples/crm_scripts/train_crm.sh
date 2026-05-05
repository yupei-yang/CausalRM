#!/bin/bash
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_crm \
   --save_path ./checkpoint/causalrm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --eval_dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --gradient_checkpointing \
   --max_samples 50000 \
   --latent_dim_c 128 \
   --latent_dim_nc 512 \
   --lambda_pred 1.0 \
   --lambda_rec 0.001 \
   --lambda_adv 0.05 \
   --beta_kl_c 0.001 \
   --beta_kl_nc 0.001 \
   --grl_lambda 1.0
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
