"""Download and prepare raw datasets from HuggingFace.

Converts HuggingFace datasets to the format expected by the preprocessing pipeline:
- Individual source files in data/raw/<dataset>/code/
- labels.json in data/raw/<dataset>/

Usage:
    python scripts/preprocess/download_datasets.py --dataset all --proxy http://127.0.0.1:10809
    python scripts/preprocess/download_datasets.py --dataset devign --proxy http://127.0.0.1:10809
    python scripts/preprocess/download_datasets.py --dataset bcbench --subset 5000
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CODE_DIR = ROOT_DIR / "code"
RAW_DIR = CODE_DIR / "data" / "raw"


def _file_ext(language: str) -> str:
    return {"java": ".java", "c": ".c", "cpp": ".cpp"}.get(language, ".txt")


def _write_sources(sources: dict, out_dir: Path, language: str):
    """Write source code strings as individual files, return content_hash -> filename mapping."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _file_ext(language)
    mapping = {}
    for content_hash, val in sources.items():
        content = val[0] if isinstance(val, tuple) else val
        fname = f"{content_hash}{ext}"
        fpath = out_dir / fname
        if not fpath.exists():
            fpath.write_text(content, encoding="utf-8")
        mapping[content_hash] = fname
    return mapping


def prepare_bcbench(max_pairs: int = None):
    """BigCloneBench: extract unique functions, write .java files, create pair labels."""
    print("=== BigCloneBench ===")
    raw_dir = RAW_DIR / "bcbench"
    code_dir = raw_dir / "code"

    ds = load_dataset("google/code_x_glue_cc clone_detection_big_clone_bench")
    all_splits = list(ds["train"]) + list(ds["validation"]) + list(ds["test"])
    print(f"  Total pairs: {len(all_splits)}")

    if max_pairs and max_pairs < len(all_splits):
        rng = random.Random(42)
        all_splits = rng.sample(all_splits, max_pairs)
        print(f"  Subsampled to: {max_pairs}")

    sources = {}
    labels = []

    for item in all_splits:
        h1 = hashlib.md5(item["func1"].encode("utf-8")).hexdigest()[:16]
        h2 = hashlib.md5(item["func2"].encode("utf-8")).hexdigest()[:16]
        if h1 not in sources:
            sources[h1] = item["func1"]
        if h2 not in sources:
            sources[h2] = item["func2"]
        labels.append({
            "file_a": h1,
            "file_b": h2,
            "label": 1 if item["label"] else 0,
        })

    file_map = _write_sources(sources, code_dir, "java")
    print(f"  Unique functions: {len(sources)}")
    print(f"  Pairs with labels: {len(labels)}")

    for lbl in labels:
        lbl["file_a"] = file_map.get(lbl["file_a"], lbl["file_a"])
        lbl["file_b"] = file_map.get(lbl["file_b"], lbl["file_b"])

    with open(raw_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"  Saved: {raw_dir / 'labels.json'}")


def prepare_ojclone(max_pairs: int = None):
    """OJClone (POJ-104): extract C source files, create clone pairs from problem IDs."""
    print("=== OJClone ===")
    raw_dir = RAW_DIR / "ojclone"
    code_dir = raw_dir / "code"

    ds = load_dataset("google/code_x_glue_cc_clone_detection_poj104")
    all_splits = list(ds["train"]) + list(ds["validation"]) + list(ds["test"])
    print(f"  Total samples: {len(all_splits)}")

    sources = {}
    by_problem = {}
    for item in all_splits:
        h = hashlib.md5(item["code"].encode("utf-8")).hexdigest()[:16]
        sources[h] = item["code"]
        pid = str(item["label"])
        by_problem.setdefault(pid, []).append(h)

    file_map = _write_sources(sources, code_dir, "c")
    print(f"  Unique sources: {len(sources)}")
    rng = random.Random(42)

    labels = []
    problem_ids = list(by_problem.keys())

    for pid, code_hashes in by_problem.items():
        for i in range(len(code_hashes)):
            for j in range(i + 1, min(i + 3, len(code_hashes))):
                labels.append({
                    "file_a": file_map[code_hashes[i]],
                    "file_b": file_map[code_hashes[j]],
                    "label": 1,
                })
        for _ in range(min(len(code_hashes), 3)):
            other_pid = rng.choice([p for p in problem_ids if p != pid])
            labels.append({
                "file_a": file_map[code_hashes[0]],
                "file_b": file_map[rng.choice(by_problem[other_pid])],
                "label": 0,
            })

    rng.shuffle(labels)
    if max_pairs and max_pairs < len(labels):
        labels = labels[:max_pairs]

    with open(raw_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"  Total pairs: {len(labels)}")
    print(f"  Saved: {raw_dir / 'labels.json'}")


def prepare_devign(max_samples: int = None):
    """Devign: extract C functions, create single-classification labels.

    Uses DetectVul/devign dataset from HuggingFace.
    """
    print("=== Devign ===")
    raw_dir = RAW_DIR / "devign"
    code_dir = raw_dir / "code"

    ds = load_dataset("DetectVul/devign")
    available = [k for k in ds.keys()]
    print(f"  Available splits: {available}")

    all_splits = []
    for split_key in available:
        all_splits.extend(list(ds[split_key]))
    print(f"  Total samples: {len(all_splits)}")

    if max_samples and max_samples < len(all_splits):
        rng = random.Random(42)
        all_splits = rng.sample(all_splits, max_samples)
        print(f"  Subsampled to: {max_samples}")

    # Detect field names
    sample = all_splits[0] if all_splits else {}
    code_key = None
    label_key = None
    for k in sample.keys():
        if k in ("func", "code", "source", "text", "function"):
            code_key = k
        if k in ("target", "label", "vulnerable", "bug", "label_int"):
            label_key = k

    if code_key is None:
        code_key = list(sample.keys())[0]
        print(f"  [warn] auto-guessing code field: {code_key}")
    if label_key is None:
        keys = list(sample.keys())
        label_key = keys[1] if len(keys) > 1 else keys[0]
        print(f"  [warn] auto-guessing label field: {label_key}")

    print(f"  Fields: code={code_key}, label={label_key}")

    labels = []
    sources = {}
    for item in all_splits:
        code_text = item.get(code_key, "")
        if not code_text or not isinstance(code_text, str):
            continue
        h = hashlib.md5(code_text.encode("utf-8")).hexdigest()[:16]
        sources[h] = code_text
        raw_label = item.get(label_key, 0)
        if isinstance(raw_label, bool):
            lbl = 1 if raw_label else 0
        elif isinstance(raw_label, str):
            lbl = 1 if raw_label.lower() in ("1", "true", "yes", "vulnerable", "buggy") else 0
        else:
            lbl = int(raw_label)
        labels.append({
            "file": "",
            "label": lbl,
            "_hash": h,
        })

    file_map = _write_sources(sources, code_dir, "c")
    for lbl in labels:
        lbl["file"] = file_map[lbl.pop("_hash", lbl.get("_hash", ""))]

    with open(raw_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"  Unique sources: {len(sources)}")
    print(f"  Saved: {raw_dir / 'labels.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "bcbench", "ojclone", "devign"])
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit to N samples/pairs for quick testing")
    parser.add_argument("--proxy", type=str, default=None,
                        help="Proxy URL, e.g. http://127.0.0.1:10809")
    args = parser.parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
        os.environ["http_proxy"] = args.proxy
        os.environ["https_proxy"] = args.proxy
        print(f"Proxy set: {args.proxy}")

    if args.dataset in ("all", "bcbench"):
        for attempt in range(3):
            try:
                prepare_bcbench(max_pairs=args.subset)
                break
            except Exception as e:
                print(f"BCBench attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    print("BCBench download FAILED after 3 attempts")

    if args.dataset in ("all", "ojclone"):
        for attempt in range(3):
            try:
                prepare_ojclone(max_pairs=args.subset)
                break
            except Exception as e:
                print(f"OJClone attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    print("OJClone download FAILED after 3 attempts")

    if args.dataset in ("all", "devign"):
        for attempt in range(3):
            try:
                prepare_devign(max_samples=args.subset)
                break
            except Exception as e:
                print(f"Devign attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(10)
                else:
                    print("Devign download FAILED after 3 attempts")


if __name__ == "__main__":
    main()
