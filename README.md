# CausalRM

This is the official implementation of our ICML'26 accepted paper: **[Factored Causal Representation Learning for Robust Reward Modeling in RLHF](https://arxiv.org/pdf/2601.21350)**.

<p align="center">
  <img src="./framework.png" alt="CausalRM Framework" width="800"/>
</p>

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CMACH508/CausalRM.git
cd CausalRM
```

2. Launch the NVIDIA PyTorch Docker container:
```bash
docker run --runtime=nvidia -it --shm-size="10g" --cap-add=SYS_ADMIN \
  --name CRM \
  -v $PWD:/workspace/CRM \
  nvcr.io/nvidia/pytorch:25.02-py3 bash
```

3. Remove potentially conflicting pre-installed packages:
```bash
pip uninstall xgboost transformer_engine flash_attn pynvml -y
```

4. (Optional) Set a PyPI mirror for faster installation:
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

5. Enter the workspace and install CausalRM in editable mode:
```bash
cd /workspace/CRM
pip install -e .
```

## Training

Training scripts are available under `examples/crm_scripts/`. The training follows a two-stage pipeline: first train a reward model, then use it for PPO alignment.

### Stage 1: Train a Reward Model

**Standard Reward Model** (baseline):
```bash
bash examples/crm_scripts/train_rm.sh
```

**CausalRM** (our method):
```bash
bash examples/crm_scripts/train_crm.sh
```

### Stage 2: PPO Training

After training the reward model, use it as the reward signal for PPO:
```bash
bash examples/crm_scripts/train_ppo.sh
```

> **Note:** PPO training requires a running Ray cluster. Start one with `ray start --head --node-ip-address 0.0.0.0 --num-gpus 8` before submitting the job. Update `--reward_pretrain` to point to your trained reward model checkpoint.

## Evaluation

Evaluation scripts are available under `examples/eval/eval_rm/`.

### Standard Evaluation on HH Test Sets

First, download the HH test sets:

```bash
python tools/download_hh_test_sets.py
```

Then evaluate a trained reward model:

```bash
bash examples/eval/eval_rm/eval_rm_hh.sh <checkpoint_path> [log_dir] [hh_test_root]

# Example:
bash examples/eval/eval_rm/eval_rm_hh.sh ./checkpoint/causalrm ./eval_logs/hh_standard dataset/hh-test
```

> The evaluator auto-detects the model type (Standard RM / CausalRM) and reports pairwise accuracy on both ID (Anthropic-Helpful, Anthropic-Harmless) and OOD (MT-Bench, PKU-SafeRLHF, SHP, TruthfulQA) benchmarks.

### Sycophantic Prefix Hacking Evaluation

This experiment tests whether reward models are robust to a spurious sycophantic prefix (`"Sure, here is the response: "`) injected into responses.

**Step 1: Prepare the hacked dataset.** First download the original HH dataset (e.g., from `Anthropic/HH-RLHF`), save it as JSONL files under `dataset/hh_full_jsonl/`, then run:

```bash
python tools/create_hacked_dataset.py
```

This creates `dataset/hh_full_jsonl_hacked/` with the sycophantic prefix injected into chosen/rejected responses with configurable probabilities.

**Step 2: Train on the hacked dataset.** Use the same training scripts as above, pointing `--dataset` to the hacked data:

```bash
# Example: train CausalRM on hacked data
# Modify examples/crm_scripts/train_crm.sh to use --dataset dataset/hh_full_jsonl_hacked
bash examples/crm_scripts/train_crm.sh
```

**Step 3: Evaluate with prefix injection.** The hacked evaluator injects the sycophantic prefix at evaluation time to measure robustness:

```bash
bash examples/eval/eval_rm/eval_rm_hh_hacked.sh <checkpoint_path> [log_dir] [hh_test_root]

# Example:
bash examples/eval/eval_rm/eval_rm_hh_hacked.sh ./checkpoint/causalrm_hacked ./eval_logs/hh_hacked dataset/hh-test

# To run without hacking (standard eval):
# add --no_hack to the command
```

> The default injection probabilities are `p_chosen=0.3, p_rejected=0.3`, matching the paper setup. Customize them via `--p_chosen`, `--p_rejected`, and `--prefix` arguments.

### Citation

```bibtex
@article{yang2026factored,
  title={Factored Causal Representation Learning for Robust Reward Modeling in RLHF},
  author={Yang, Yupei and Yang, Lin and Deng, Wanxi and Qu, Lin and Feng, Fan and Huang, Biwei and Tu, Shikui and Xu, Lei},
  journal={arXiv preprint arXiv:2601.21350},
  year={2026}
}
```