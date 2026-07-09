"""Download and parse HuggingFace Parquet files from hf-mirror.com.

Usage:
    python scripts/preprocess/download_parquet.py --dataset devign --inspect
    python scripts/preprocess/download_parquet.py --dataset devign
    python scripts/preprocess/download_parquet.py --dataset bcbench
    python scripts/preprocess/download_parquet.py --dataset all
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CODE_DIR = ROOT_DIR / "code"
RAW_DIR = CODE_DIR / "data" / "raw"
CACHE_DIR = CODE_DIR / "data" / "parquet_cache"
MIRROR_BASE = "https://hf-mirror.com"

DATASETS = {
    "bcbench": {
        "repo": "google/code_x_glue_cc_clone_detection_big_clone_bench",
        "files": [
            *(f"data/train-{i:05d}-of-00006.parquet" for i in range(6)),
            *(f"data/validation-{i:05d}-of-00003.parquet" for i in range(3)),
            *(f"data/test-{i:05d}-of-00003.parquet" for i in range(3)),
        ],
        "language": "java",
        "type": "pair",
    },
    "devign": {
        "repo": "DetectVul/devign",
        "files": [
            "data/train-00000-of-00001-396a063c42dfdb0a.parquet",
            "data/validation-00000-of-00001-5d4ba937305086b9.parquet",
            "data/test-00000-of-00001-e0e162fa10729371.parquet",
        ],
        "language": "c",
        "type": "single",
    },
}


def _file_ext(language: str) -> str:
    return {"java": ".java", "c": ".c", "cpp": ".cpp"}.get(language, ".txt")


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [cached] {dest.name}")
        return True
    resp = requests.get(url, stream=True, timeout=300)
    if resp.status_code != 200:
        print(f"  [fail] {url} -> {resp.status_code}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"  [downloaded] {dest.name}")
    return True


def inspect_dataset(ds_name: str):
    info = DATASETS[ds_name]
    first_file = info["files"][0]
    url = f"{MIRROR_BASE}/datasets/{info['repo']}/resolve/main/{first_file}"
    cache_path = CACHE_DIR / ds_name / first_file
    if download_file(url, cache_path):
        df = pd.read_parquet(cache_path)
        print(f"\n  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        print(f"  First row:\n    {df.iloc[0].to_dict()}")
        print(f"\n  Sample label values: {df.iloc[:3]['label'].tolist() if 'label' in df.columns else 'N/A'}")


def process_bcbench():
    info = DATASETS["bcbench"]
    raw_dir = RAW_DIR / "bcbench"
    code_dir = raw_dir / "code"
    sources = {}
    labels = []

    for filename in info["files"]:
        url = f"{MIRROR_BASE}/datasets/{info['repo']}/resolve/main/{filename}"
        cache_path = CACHE_DIR / "bcbench" / filename
        if not download_file(url, cache_path):
            continue
        df = pd.read_parquet(cache_path)

        for _, row in df.iterrows():
            func1 = row.get("func1", row.get("func1", ""))
            func2 = row.get("func2", row.get("func2", ""))
            label = row.get("label", 0)
            if isinstance(label, (bool,)):
                label = 1 if label else 0
            else:
                label = int(label)

            h1 = hashlib.md5(str(func1).encode("utf-8")).hexdigest()[:16]
            h2 = hashlib.md5(str(func2).encode("utf-8")).hexdigest()[:16]

            if h1 not in sources:
                sources[h1] = func1
            if h2 not in sources:
                sources[h2] = func2

            labels.append({
                "file_a": h1,
                "file_b": h2,
                "label": label,
            })

    file_map = _write_sources(sources, code_dir, "java")

    for lbl in labels:
        lbl["file_a"] = file_map.get(lbl["file_a"], lbl["file_a"])
        lbl["file_b"] = file_map.get(lbl["file_b"], lbl["file_b"])

    with open(raw_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"\nBCBench: {len(sources)} sources, {len(labels)} pairs")


def process_devign():
    info = DATASETS["devign"]
    raw_dir = RAW_DIR / "devign"
    code_dir = raw_dir / "code"

    labels = []
    sources = {}
    seen_hashes = set()

    for filename in info["files"]:
        url = f"{MIRROR_BASE}/datasets/{info['repo']}/resolve/main/{filename}"
        cache_path = CACHE_DIR / "devign" / filename
        if not download_file(url, cache_path):
            continue
        df = pd.read_parquet(cache_path)

        # Detect field names
        code_keys = [c for c in df.columns if c in ("func", "code", "source", "function", "text")]
        label_keys = [c for c in df.columns if c in ("target", "label", "vulnerable")]
        code_key = code_keys[0] if code_keys else df.columns[0]
        label_key = label_keys[0] if label_keys else df.columns[1]

        for _, row in df.iterrows():
            code_text = str(row.get(code_key, ""))
            raw_label = row.get(label_key, 0)

            if isinstance(raw_label, bool):
                lbl = 1 if raw_label else 0
            elif isinstance(raw_label, str):
                lbl = 1 if raw_label.lower() in ("1", "true", "yes", "vulnerable", "buggy") else 0
            else:
                lbl = int(raw_label)

            h = hashlib.md5(code_text.encode("utf-8")).hexdigest()[:16]

            # Handle hash collisions
            orig_h = h
            counter = 1
            while h in seen_hashes:
                h = f"{orig_h}_{counter}"
                counter += 1
            seen_hashes.add(h)

            if orig_h not in sources or True:
                sources[h] = code_text

            labels.append({
                "file": "",
                "label": lbl,
                "_hash": h,
            })

    file_map = _write_sources(sources, code_dir, "c")
    out_labels = []
    for lbl in labels:
        lbl["file"] = file_map.get(lbl.pop("_hash"), lbl.get("_hash", ""))
        out_labels.append(lbl)

    with open(raw_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(out_labels, f)

    print(f"\nDevign: {len(sources)} sources, {len(out_labels)} samples")


def _write_sources(sources: dict, out_dir: Path, language: str) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _file_ext(language)
    mapping = {}
    for content_hash, content in sources.items():
        fname = f"{content_hash}{ext}"
        fpath = out_dir / fname
        if not fpath.exists():
            fpath.write_text(content, encoding="utf-8")
        mapping[content_hash] = fname
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "bcbench", "devign"])
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.inspect:
        ds_list = ["bcbench", "devign"] if args.dataset == "all" else [args.dataset]
        for ds in ds_list:
            print(f"\n=== {ds} ===")
            inspect_dataset(ds)
        return

    if args.dataset in ("all", "devign"):
        print("=== Devign (hf-mirror) ===")
        process_devign()

    if args.dataset in ("all", "bcbench"):
        print("\n=== BCBench (hf-mirror) ===")
        process_bcbench()


if __name__ == "__main__":
    main()
