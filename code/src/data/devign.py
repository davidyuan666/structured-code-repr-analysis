"""Devign dataset helpers — single-classification mode."""

from pathlib import Path

from src.data.base_dataset import CodePairDataset


def build_devign_dataset(
    data_dir: Path,
    representation: str,
    split: str,
    tokenizer,
    max_length: int = 512,
) -> CodePairDataset:
    """Build Devign dataset for a given representation and split.

    Devign is single-classification: each sample is (code, label).
    """
    file_path = data_dir / representation / f"{split}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"Devign data not found: {file_path}")
    return CodePairDataset(
        data_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="single",
    )
