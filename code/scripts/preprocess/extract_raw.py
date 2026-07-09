"""Extract raw token sequences from source files (no parsing, just whitespace tokenization).

Usage:
    python scripts/preprocess/extract_raw.py \
        --input data/raw/bcbench/code/ \
        --output data/processed/bcbench/raw/
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of source files")
    parser.add_argument("--output", required=True, help="Output directory for full.jsonl")
    parser.add_argument("--ext", default="*", help="File extension glob")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ext == "*":
        source_files = [f for f in input_dir.rglob("*") if f.is_file()]
    else:
        source_files = list(input_dir.glob(f"**/*{args.ext}"))

    print(f"Found {len(source_files)} source files")

    with open(output_dir / "full.jsonl", "w", encoding="utf-8") as out:
        for fpath in source_files:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            tokens = content.split()
            seq = " ".join(tokens)
            out.write(json.dumps({
                "file": str(fpath.relative_to(input_dir)),
                "seq": seq,
            }) + "\n")

    print(f"Output: {output_dir / 'full.jsonl'}")


if __name__ == "__main__":
    main()
