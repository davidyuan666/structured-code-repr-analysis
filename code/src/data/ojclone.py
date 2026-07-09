"""OJClone dataset helpers."""

from pathlib import Path

from src.data.base_dataset import CodePairDataset


def build_ojclone_dataset(
    data_dir: Path,
    representation: str,
    split: str,
    tokenizer,
    max_length: int = 512,
) -> CodePairDataset:
    """Build OJClone dataset for a given representation and split."""
    file_path = data_dir / representation / f"{split}.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"OJClone data not found: {file_path}")
    return CodePairDataset(
        data_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="pair",
    )
