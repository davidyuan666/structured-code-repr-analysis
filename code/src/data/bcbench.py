"""BigCloneBench dataset helpers — loads pre-processed pair and single jsonl."""

from pathlib import Path
from typing import Optional

from src.data.base_dataset import CodePairDataset


def build_bcbench_dataset(
    data_dir: Path,
    representation: str,
    split: str,
    tokenizer,
    max_length: int = 512,
) -> CodePairDataset:
    """Build BCB dataset for a given representation and split.

    Args:
        data_dir: Root of processed data (e.g. data/processed/bcbench/).
        representation: One of "raw", "ast_seq", "cfg_seq", "ir_seq".
        split: "train", "val" or "test".
        tokenizer: HF tokenizer or sentencepiece.
        max_length: Sequence truncation length.
    """
    file_path = data_dir / representation / f"{split}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"BCB data not found: {file_path}")
    return CodePairDataset(
        data_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="pair",
    )
