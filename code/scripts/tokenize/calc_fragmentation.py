"""Compute subword fragmentation statistics: OOV rate, avg subwords/token, sequence length.

Supports both sentencepiece and HuggingFace tokenizers.

Usage:
    python scripts/tokenize/calc_fragmentation.py --representation raw --dataset bcbench
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple


def calc_spm_fragmentation(model_path: str, jsonl_path: Path) -> dict:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=model_path)
    vocab_size = sp.get_piece_size()

    total_orig_tokens = 0
    total_subwords = 0
    total_oov = 0
    total_seq_len = 0
    total_samples = 0
    truncated = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            seq = item["seq"]
            orig_tokens = seq.split()
            total_orig_tokens += len(orig_tokens)
            ids = sp.encode(seq)
            total_subwords += len(ids)
            total_seq_len += len(ids)
            total_samples += 1
            if len(ids) > 512:
                truncated += 1

            for tok in orig_tokens:
                tok_ids = sp.encode(tok)
                if 1 in tok_ids:
                    total_oov += 1

    return {
        "vocab_size": vocab_size,
        "samples": total_samples,
        "orig_tokens": total_orig_tokens,
        "subwords": total_subwords,
        "subwords_per_token": total_subwords / max(total_orig_tokens, 1),
        "oov_rate": total_oov / max(total_orig_tokens, 1),
        "avg_seq_len": total_seq_len / max(total_samples, 1),
        "truncation_rate_512": truncated / max(total_samples, 1),
        "coverage": 1.0 - (total_oov / max(total_orig_tokens, 1)),
    }


def calc_hf_fragmentation(tokenizer_name: str, jsonl_path: Path) -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    total_orig_tokens = 0
    total_subwords = 0
    total_seq_len = 0
    total_samples = 0
    truncated = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            seq = item["seq"]
            orig_tokens = seq.split()
            total_orig_tokens += len(orig_tokens)
            ids = tok.encode(seq, add_special_tokens=False)
            total_subwords += len(ids)
            total_seq_len += len(ids)
            total_samples += 1
            if len(ids) > 512:
                truncated += 1

    return {
        "tokenizer": tokenizer_name,
        "samples": total_samples,
        "orig_tokens": total_orig_tokens,
        "subwords": total_subwords,
        "subwords_per_token": total_subwords / max(total_orig_tokens, 1),
        "avg_seq_len": total_seq_len / max(total_samples, 1),
        "truncation_rate_512": truncated / max(total_samples, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", default="spm")
    parser.add_argument("--tokenizer_path", default="code/data/processed/spm_shared_50k.model")
    args = parser.parse_args()

    code_dir = Path(__file__).resolve().parent.parent.parent.parent / "code"
    jsonl_path = code_dir / "data" / "processed" / args.dataset / args.representation / "full.jsonl"

    if args.tokenizer == "spm":
        stats = calc_spm_fragmentation(args.tokenizer_path, jsonl_path)
    else:
        stats = calc_hf_fragmentation(args.tokenizer_path, jsonl_path)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
