#!/bin/bash
# Evaluate reward models on HH test sets with sycophantic prefix hacking injection.
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_checkpoint_path>}"
LOG_DIR="${2:-./eval_logs/hh_hacked}"
HH_TEST_ROOT="${3:-dataset/hh-test}"

# Hacking parameters (matching paper setup)
P_CHOSEN=0.3
P_REJECTED=0.3
PREFIX="Sure, here is the response: "

deepspeed --num_gpus 8 examples/eval/eval_rm/eval_rm_hh_hacked.py \
  --model_path "${MODEL_PATH}" \
  --hh_test_root "${HH_TEST_ROOT}" \
  --max_len 2048 \
  --batch_size 8 \
  --flash_attn \
  --log_dir "${LOG_DIR}" \
  --p_chosen "${P_CHOSEN}" \
  --p_rejected "${P_REJECTED}" \
  --prefix "${PREFIX}"

# To disable hacking and run standard evaluation:
#   add --no_hack
