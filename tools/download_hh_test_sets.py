import os
from datasets import load_dataset, DatasetDict

OUT_ROOT = "dataset/hh-test"


def download_preference_test_sets():
    repo_id = "allenai/preference-test-sets"
    out_dir = os.path.join(OUT_ROOT, "preference-test-sets")
    os.makedirs(out_dir, exist_ok=True)

    # 获取所有 split
    dsd_meta = load_dataset(repo_id)
    split_names = list(dsd_meta.keys())
    print(f"[{repo_id}] splits:", split_names, "count:", len(split_names))

    dsd = {}
    for split in split_names:
        ds = load_dataset(repo_id, split=split)
        dsd[split] = ds
        print(f"[{repo_id}] loaded {split}: {len(ds)} rows, columns={ds.column_names}")

    DatasetDict(dsd).save_to_disk(out_dir)
    print(f"[{repo_id}] saved to: {os.path.abspath(out_dir)}")


def download_truthfulqa():
    repo_id = "domenicrosati/TruthfulQA"
    out_dir = os.path.join(OUT_ROOT, "truthfulQA")  # 你要的目录名
    os.makedirs(out_dir, exist_ok=True)

    ds = load_dataset(repo_id, split="train")
    print(f"[{repo_id}] loaded train: {len(ds)} rows, columns={ds.column_names}")

    DatasetDict({"train": ds}).save_to_disk(out_dir)
    print(f"[{repo_id}] saved to: {os.path.abspath(out_dir)}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    download_preference_test_sets()
    download_truthfulqa()


if __name__ == "__main__":
    main()
