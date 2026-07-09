"""Build dataset splits (8:1:1) and pair-label format for clone detection / vulnerability.

Accepts raw JSONL output from extraction scripts (extract_ast/cfg/ir), cross-references
with ground-truth labels, and produces train/val/test splits in pair format.

Usage:
    python scripts/preprocess/build_dataset_splits.py \
        --representation raw,ast_seq,cfg_seq,ir_seq \
        --dataset bcbench \
        --labels_path data/raw/bcbench/labels.json
"""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_sequences(processed_dir: Path, representation: str) -> Dict[str, str]:
    """Load {file_path: sequence} from a representation's full.jsonl."""
    jsonl_path = processed_dir / representation / "full.jsonl"
    seqs = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            seqs[item["file"]] = item["seq"]
    return seqs


def _seq_lookup(file_key: str, seqs: Dict[str, str]) -> str:
    """Look up sequence by file key, trying with and without language extensions."""
    if file_key in seqs:
        return seqs[file_key]
    for ext in [".c", ".java", ".cpp", ".txt"]:
        if file_key + ext in seqs:
            return seqs[file_key + ext]
    return ""


def build_pairs(labels: List[Dict], seqs: Dict[str, str], dataset_name: str) -> List[Dict]:
    """Convert labels into (code_a, code_b, label) pairs."""
    pairs = []
    for item in labels:
        file_a = item.get("file_a") or item.get("file")
        file_b = item.get("file_b", file_a)
        label = item.get("label", item.get("clone_type", 0))
        if isinstance(label, str):
            label = 1 if label in ("T", "true", "yes", "1") else 0

        seq_a = _seq_lookup(file_a, seqs)
        seq_b = _seq_lookup(file_b, seqs)
        if seq_a and seq_b:
            pairs.append({"code_a": seq_a, "code_b": seq_b, "label": int(label)})
    return pairs


def build_singles(labels: List[Dict], seqs: Dict[str, str]) -> List[Dict]:
    """Convert labels into (code, label) samples for single-classification."""
    samples = []
    for item in labels:
        file_key = item.get("file") or item.get("path")
        label = item.get("label", 0)
        if isinstance(label, str):
            label = 1 if label in ("1", "vulnerable", "buggy") else 0
        seq = _seq_lookup(file_key, seqs)
        if seq:
            samples.append({"code": seq, "label": int(label)})
    return samples


def split_and_save(
    data: List[Dict],
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train.jsonl": data[:n_train],
        "val.jsonl": data[n_train:n_train + n_val],
        "test.jsonl": data[n_train + n_val:],
    }
    for filename, subset in splits.items():
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for sample in subset:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"  {output_dir}: train={len(splits['train.jsonl'])} val={len(splits['val.jsonl'])} test={len(splits['test.jsonl'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True, help="Comma-separated list")
    parser.add_argument("--dataset", required=True, choices=["bcbench", "ojclone", "devign"])
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--mode", default="pair", choices=["pair", "single"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_base = Path(__file__).resolve().parent.parent.parent.parent / "code" / "data"
    processed_dir = input_base / "processed" / args.dataset

    labels_path = args.labels_path
    if not labels_path.startswith("/"):
        labels_path = input_base / labels_path
    labels_path = Path(labels_path)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels not found: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        labels = labels.get("pairs", labels.get("samples", labels.get("data", [])))

    print(f"Loaded {len(labels)} label entries")

    for repr_name in args.representation.split(","):
        repr_name = repr_name.strip()
        seqs = load_sequences(processed_dir, repr_name)
        print(f"  {repr_name}: {len(seqs)} sequences")

        if args.mode == "pair":
            data = build_pairs(labels, seqs, args.dataset)
        else:
            data = build_singles(labels, seqs)

        output_dir = processed_dir / repr_name
        split_and_save(data, output_dir, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
