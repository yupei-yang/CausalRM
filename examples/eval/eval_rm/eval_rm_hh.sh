#!/bin/bash
# Evaluate reward models on HH test sets (standard, no hacking).
set -euo pipefail

MODEL_PATH="${1:?Usage: $0 <model_checkpoint_path>}"
LOG_DIR="${2:-./eval_logs/hh_standard}"
HH_TEST_ROOT="${3:-dataset/hh-test}"

deepspeed --num_gpus 8 examples/eval/eval_rm/eval_rm_hh.py \
  --model_path "${MODEL_PATH}" \
  --hh_test_root "${HH_TEST_ROOT}" \
  --max_len 2048 \
  --batch_size 8 \
  --flash_attn \
  --log_dir "${LOG_DIR}"
