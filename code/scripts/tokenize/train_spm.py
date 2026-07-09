"""Train shared sentencepiece BPE tokenizer (vocab_size=50000) on combined representations.

Usage:
    python scripts/tokenize/train_spm.py --output data/processed/spm_shared_50k.model
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("sentencepiece not installed. Install with: pip install sentencepiece")
    sys.exit(1)


def train_tokenizer(
    input_glob: str,
    model_prefix: str,
    vocab_size: int = 50000,
    model_type: str = "bpe",
):
    spm.SentencePieceTrainer.train(
        input=input_glob,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["ID", "LIT", "ENTRY", "EXIT"],
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    )
    print(f"Tokenizer saved to {model_prefix}.model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default="code/data/processed",
                        help="Directory containing dataset subdirs with full.jsonl files")
    parser.add_argument("--output", default="code/data/processed/spm_shared_50k",
                        help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=50000)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    txt_file = corpus_dir / "_combined_corpus.txt"

    if not txt_file.exists():
        all_text = []
        for ds_name in ["bcbench", "ojclone", "devign"]:
            for repr_name in ["raw", "ast_seq", "cfg_seq", "ir_seq"]:
                jsonl = corpus_dir / ds_name / repr_name / "full.jsonl"
                if jsonl.exists():
                    with open(jsonl, "r", encoding="utf-8") as f:
                        for line in f:
                            item = json.loads(line)
                            all_text.append(item["seq"])
                            all_text.append("\n")

        with open(txt_file, "w", encoding="utf-8") as f:
            f.writelines(all_text)
        print(f"Combined corpus: {txt_file} ({len(all_text)} lines)")

    train_tokenizer(
        input_glob=str(txt_file),
        model_prefix=str(Path(args.output)),
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
