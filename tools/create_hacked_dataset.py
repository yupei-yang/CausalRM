import os
import json
import random

SRC_DIR = "dataset/hh_full_jsonl"
DST_DIR = "dataset/hh_full_jsonl_hacked"

TRAIN_P_CHOSEN = 0.8
TRAIN_P_REJECTED = 0.2
TEST_P_CHOSEN = 0.3
TEST_P_REJECTED = 0.3

prefix = "Sure, here is the response: "

os.makedirs(DST_DIR, exist_ok=True)

def maybe_prefix(turns, p):
    if not turns or random.random() >= p:
        return turns
    for msg in turns:
        if isinstance(msg, dict) and msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            if not msg["content"].startswith(prefix):
                msg["content"] = prefix + msg["content"]
            break
    return turns

for fname in os.listdir(SRC_DIR):
    if not fname.endswith(".jsonl"):
        continue

    is_test = fname in {"test_harmless.jsonl", "test_helpful.jsonl"}

    p_chosen = TEST_P_CHOSEN if is_test else TRAIN_P_CHOSEN
    p_rejected = TEST_P_REJECTED if is_test else TRAIN_P_REJECTED

    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)

    with open(src_path, "r", encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            if "chosen" in ex and isinstance(ex["chosen"], list):
                ex["chosen"] = maybe_prefix(ex["chosen"], p_chosen)
            if "rejected" in ex and isinstance(ex["rejected"], list):
                ex["rejected"] = maybe_prefix(ex["rejected"], p_rejected)

            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Done. Saved hacked dataset to: {DST_DIR}")
