#!/bin/bash
set -x

export VLLM_TORCH_COMPILE_LEVEL=0

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/workspace/CRM"}' \
  -- python3 -m openrlhf.cli.train_ppo_ray_with_template \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 8 \
  --reward_num_nodes 1 \
  --reward_num_gpus_per_node 8 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 8 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 4 \
  --vllm_gpu_memory_utilization 0.6 \
  --colocate_critic_reward \
  --colocate_actor_ref \
  --pretrain Qwen/Qwen2.5-Math-7B \
  --reward_pretrain ./checkpoint/causalrm \
  --save_path ./checkpoint/causalrm-ppo \
  --micro_train_batch_size 8 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 16 \
  --rollout_batch_size 512 \
  --max_samples 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data dataset/math \
  --input_key question \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing

# To use with a standard reward model instead, set --reward_pretrain to the standard RM checkpoint.
# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Auto-install deps]
# --remote_rm_url http://localhost:5000/get_reward [Remote RM]
