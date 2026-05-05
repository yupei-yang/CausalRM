#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified evaluator for Base RM / CausalRM / InfoRM on HH-test-sets saved via save_to_disk.

Dataset root (default): dataset/hh-test
Subdirs example:
  dataset/hh-test/
    anthropic_helpful/
    anthropic_harmless/
    mtbench_human/
    pku_safer/
    shp/
    truthfulQA/

Metric: Pairwise accuracy — reward(chosen) > reward(rejected)

Report:
  iid: mean(acc(anthropic_helpful), acc(anthropic_harmless))
  ood: mean(acc(other datasets))
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
import torch.distributed as dist
from tqdm import tqdm

from datasets import load_from_disk, Dataset

from openrlhf.models import get_llm_for_sequence_regression, get_llm_for_sequence_regression_inform
from openrlhf.utils import get_strategy, get_tokenizer


# ------------------ Utils ------------------ #

def is_rank_0() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def setup_logger(log_dir: str, model_name: str, run_tag: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() else "_" for c in run_tag)[:60]
    log_file = os.path.join(log_dir, f"eval_{model_name}_{safe_tag}_{timestamp}.log")

    logger = logging.getLogger("RM_Evaluator_HH_TEST")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(ch)

    if is_rank_0():
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


def detect_model_type(model_path: str) -> str:
    """Auto-detect reward / causal_reward / inform."""
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(cfg, "latent_dim_c") and hasattr(cfg, "latent_dim_nc"):
            return "causal_reward"
        elif hasattr(cfg, "latent_dim"):
            return "inform"
    except Exception:
        pass
    return "reward"


def messages_to_text_fallback(messages: List[Dict[str, str]]) -> str:
    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            chunks.append(f"System: {content}")
        elif role == "user":
            chunks.append(f"User: {content}")
        elif role == "assistant":
            chunks.append(f"Assistant: {content}")
        else:
            chunks.append(f"{role}: {content}")
    return "\n".join(chunks).strip() + "\n"


def build_pair_text(tokenizer, prompt_msgs, completion_msgs, add_generation_prompt: bool = False) -> str:
    msgs = (prompt_msgs or []) + (completion_msgs or [])
    if hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template")):
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            return messages_to_text_fallback(msgs)
    return messages_to_text_fallback(msgs)


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int, dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.size(dim)
    pad_tensor = pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([pad_tensor, tensor], dim=dim)


def concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask, tokenizer):
    max_len_ids = max(chosen_ids.shape[1], reject_ids.shape[1])
    input_ids = torch.cat(
        (
            pad_to_length(chosen_ids, max_len_ids, tokenizer.pad_token_id),
            pad_to_length(reject_ids, max_len_ids, tokenizer.pad_token_id),
        ),
        dim=0,
    )
    max_len_mask = max(c_mask.shape[1], r_mask.shape[1])
    att_masks = torch.cat(
        (
            pad_to_length(c_mask, max_len_mask, 0),
            pad_to_length(r_mask, max_len_mask, 0),
        ),
        dim=0,
    )
    return input_ids, att_masks


# ------------------ Data adaptors ------------------ #

def _is_hh_style_prompt(x) -> bool:
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))


def iter_pairs_from_dataset(ds: Dataset, dataset_name: str):
    """Yield dicts: {"prompt": [...], "chosen":[...], "rejected":[...]} in HH style."""
    cols = set(ds.column_names)

    if dataset_name.lower() == "truthfulqa":
        for ex in ds:
            q = ex.get("Question", None)
            best = ex.get("Best Answer", None)
            inc = ex.get("Incorrect Answers", None)

            if not q or not best or inc is None:
                continue

            if isinstance(inc, list):
                if len(inc) == 0:
                    continue
                rej = inc[0]
            else:
                rej = inc

            if not isinstance(rej, str) or not rej.strip():
                continue

            yield {
                "prompt": [{"role": "user", "content": str(q)}],
                "chosen": [{"role": "assistant", "content": str(best)}],
                "rejected": [{"role": "assistant", "content": str(rej)}],
            }
        return

    required = {"prompt", "chosen", "rejected"}
    if not required.issubset(cols):
        raise ValueError(f"[{dataset_name}] missing required columns {required}, got {ds.column_names}")

    for ex in ds:
        p = ex["prompt"]
        c = ex["chosen"]
        r = ex["rejected"]

        if isinstance(p, str):
            p = [{"role": "user", "content": p}]
        if isinstance(c, str):
            c = [{"role": "assistant", "content": c}]
        if isinstance(r, str):
            r = [{"role": "assistant", "content": r}]

        if not _is_hh_style_prompt(p) or not _is_hh_style_prompt(c) or not _is_hh_style_prompt(r):
            continue

        yield {"prompt": p, "chosen": c, "rejected": r}


def load_dataset_dir(path: str) -> Dataset:
    """load_from_disk can return Dataset or DatasetDict. Unify into a single Dataset."""
    obj = load_from_disk(path)
    if isinstance(obj, Dataset):
        return obj

    if "test" in obj:
        return obj["test"]
    if "validation" in obj:
        return obj["validation"]
    if "train" in obj:
        return obj["train"]
    first_key = list(obj.keys())[0]
    return obj[first_key]


# ------------------ Pairwise evaluation ------------------ #

@torch.no_grad()
def evaluate_pairwise(
    model,
    tokenizer,
    pairs: List[Dict[str, Any]],
    strategy,
    logger,
    dataset_name: str,
    batch_size: int = 4,
    max_len: int = 1024,
) -> Dict[str, float]:
    if strategy.is_rank_0():
        logger.info(f"[PAIRWISE] Start: {dataset_name}, #pairs={len(pairs)}")

    model.eval()
    device = torch.cuda.current_device()

    correct = 0
    total = 0

    for start in tqdm(range(0, len(pairs), batch_size), disable=not strategy.is_rank_0()):
        batch = pairs[start: start + batch_size]
        if not batch:
            continue

        chosen_texts = []
        rejected_texts = []
        for item in batch:
            chosen_texts.append(build_pair_text(tokenizer, item["prompt"], item["chosen"]))
            rejected_texts.append(build_pair_text(tokenizer, item["prompt"], item["rejected"]))

        chosen_inputs = tokenizer(
            chosen_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        rejected_inputs = tokenizer(
            rejected_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )

        chosen_ids = chosen_inputs["input_ids"].to(device)
        c_mask = chosen_inputs["attention_mask"].to(device)
        reject_ids = rejected_inputs["input_ids"].to(device)
        r_mask = rejected_inputs["attention_mask"].to(device)

        input_ids, att_masks = concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask, tokenizer)

        out = model(input_ids, attention_mask=att_masks, return_output=True)
        if isinstance(out, tuple):
            rewards, _ = out
        else:
            rewards = out

        B = chosen_ids.size(0)
        chosen_rewards = rewards[:B]
        rejected_rewards = rewards[B:]

        correct += (chosen_rewards > rejected_rewards).sum().item()
        total += B

    if total == 0:
        return {"pairwise_acc": float("nan")}

    acc = correct / total
    if strategy.is_rank_0():
        logger.info(f"[PAIRWISE] {dataset_name} acc={acc:.6f} (correct={correct}, total={total})")
    return {"pairwise_acc": acc}


# ------------------ Main ------------------ #

def main():
    parser = argparse.ArgumentParser(description="Evaluate RM on hh-test (save_to_disk datasets)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["auto", "reward", "causal_reward", "inform"], default="auto")

    # InfoRM args
    parser.add_argument("--use_inform", action="store_true", default=False)
    parser.add_argument("--use_complex_decoder", action="store_true", default=False)
    parser.add_argument("--latent_dim", type=int, default=128)

    # Data root
    parser.add_argument("--hh_test_root", type=str, default="dataset/hh-test")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Optional: only evaluate these subdir names.")

    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--max_len", type=int, default=1024)

    # Logging
    parser.add_argument("--log_dir", type=str, default="./eval_logs")

    # Strategy/DS
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--zero_stage", type=int, default=3)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--packing_samples", action="store_true", default=False)
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    strategy = get_strategy(args)
    strategy.setup_distributed()

    if args.model_type == "auto":
        args.model_type = detect_model_type(args.model_path)
        if is_rank_0():
            print(f"[Info] Auto-detected model_type = {args.model_type}")

    if args.use_inform:
        args.model_type = "inform"

    model_name = Path(args.model_path).name
    logger = setup_logger(args.log_dir, model_name, run_tag="hh-test")

    # Load model
    ds_cfg = strategy.get_ds_eval_config(offload=False)
    if args.model_type == "inform":
        strategy.print("Loading InfoRM model...")
        model = get_llm_for_sequence_regression_inform(
            args.model_path,
            "reward",
            bf16=args.bf16,
            load_in_4bit=False,
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            ds_config=ds_cfg,
            packing_samples=args.packing_samples,
            latent_dim=args.latent_dim,
            use_complex_decoder=args.use_complex_decoder,
        )
    else:
        strategy.print(f"Loading standard RM model (type: {args.model_type})...")
        model = get_llm_for_sequence_regression(
            args.model_path,
            args.model_type,
            bf16=args.bf16,
            load_in_4bit=False,
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            ds_config=ds_cfg,
            value_head_prefix="score",
            packing_samples=args.packing_samples,
        )

    model = strategy.prepare(model, is_rlhf=True)
    model.eval()

    tokenizer = get_tokenizer(
        args.model_path,
        model,
        "left",
        strategy,
        use_fast=not args.disable_fast_tokenizer,
    )

    root = Path(args.hh_test_root)
    if not root.exists():
        raise FileNotFoundError(f"hh_test_root not found: {root}")

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    subdirs = sorted(subdirs, key=lambda p: p.name)

    if args.datasets:
        wanted = set(args.datasets)
        subdirs = [p for p in subdirs if p.name in wanted]

    if strategy.is_rank_0():
        logger.info(f"[DATA] hh_test_root={root}")
        logger.info(f"[DATA] datasets={ [p.name for p in subdirs] }")

    results = {}  # name -> acc
    for ds_dir in subdirs:
        name = ds_dir.name
        ds = load_dataset_dir(str(ds_dir))

        pairs = []
        for i, ex in enumerate(iter_pairs_from_dataset(ds, name)):
            pairs.append(ex)
            if len(pairs) >= args.max_samples:
                break

        if strategy.is_rank_0():
            logger.info(f"[DATA] {name}: loaded pairs={len(pairs)} (raw_rows={len(ds)})")
        if len(pairs) == 0:
            results[name] = float("nan")
            continue

        res = evaluate_pairwise(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            strategy=strategy,
            logger=logger,
            dataset_name=name,
            batch_size=args.batch_size,
            max_len=args.max_len,
        )
        results[name] = res["pairwise_acc"]

    # Aggregate iid / ood
    iid_names = ["anthropic_helpful", "anthropic_harmless"]
    iid_vals = [results.get(k, float("nan")) for k in iid_names]
    iid_vals = [v for v in iid_vals if v == v]

    ood_vals = []
    for k, v in results.items():
        if k in iid_names:
            continue
        if v == v:
            ood_vals.append(v)

    iid = sum(iid_vals) / len(iid_vals) if iid_vals else float("nan")
    ood = sum(ood_vals) / len(ood_vals) if ood_vals else float("nan")

    if strategy.is_rank_0():
        logger.info("========== Per-dataset results ==========")
        for k in sorted(results.keys()):
            logger.info(f"{k:20s} acc={results[k]:.6f}")
        logger.info("=========================================")
        logger.info(f"[SUMMARY] iid(mean of {iid_names}) = {iid:.6f}")
        logger.info(f"[SUMMARY] ood(mean of others)      = {ood:.6f}")

        # also dump a json
        out_json = {
            "iid": iid,
            "ood": ood,
            "per_dataset": results,
            "iid_datasets": iid_names,
        }
        out_path = Path(args.log_dir) / f"hh_test_summary_{model_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)
        logger.info(f"[SUMMARY] saved json: {out_path}")


if __name__ == "__main__":
    main()
