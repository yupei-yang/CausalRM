#!/bin/bash
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/standard-rm \
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
   --max_samples 50000
EOF

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
