"""Base dataset class supporting both pair-classification and single-classification tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class CodePairDataset(Dataset):
    """Dataset for code clone detection (pair classification) and vulnerability detection (single).

    For clone detection tasks (BCB, OJClone):
        mode="pair"  -> returns (code_a, code_b, label)

    For vulnerability detection (Devign):
        mode="single" -> returns (code, label)
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: Union[PreTrainedTokenizerBase, "sentencepiece.SentencePieceProcessor"],
        max_length: int = 512,
        mode: str = "pair",
        padding: bool = True,
        truncation: bool = True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.padding = padding
        self.truncation = truncation

        self.samples: List[dict] = self._load_data()

    def _load_data(self) -> List[dict]:
        samples = []
        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    samples.append(json.loads(line))
        elif self.data_path.suffix == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "encode"):
            token_ids = self.tokenizer.encode(text)[:self.max_length]
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.ones(len(token_ids), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            result = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=self.truncation,
                return_tensors="pt",
                return_attention_mask=True,
            )
            return {
                "input_ids": result["input_ids"].squeeze(0),
                "attention_mask": result["attention_mask"].squeeze(0),
            }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if self.mode == "pair":
            code_a = sample["code_a"]
            code_b = sample["code_b"]
            label = sample["label"]
            tok_a = self._tokenize(code_a)
            tok_b = self._tokenize(code_b)
            return {
                "input_ids_a": tok_a["input_ids"],
                "attention_mask_a": tok_a["attention_mask"],
                "input_ids_b": tok_b["input_ids"],
                "attention_mask_b": tok_b["attention_mask"],
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            code = sample["code"]
            label = sample["label"]
            tok = self._tokenize(code)
            return {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "label": torch.tensor(label, dtype=torch.long),
            }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Dynamic collate function for variable-length sequences."""
    from torch.nn.utils.rnn import pad_sequence

    if "input_ids_a" in batch[0]:
        ids_a = pad_sequence([x["input_ids_a"] for x in batch], batch_first=True, padding_value=0)
        mask_a = pad_sequence([x["attention_mask_a"] for x in batch], batch_first=True, padding_value=0)
        ids_b = pad_sequence([x["input_ids_b"] for x in batch], batch_first=True, padding_value=0)
        mask_b = pad_sequence([x["attention_mask_b"] for x in batch], batch_first=True, padding_value=0)
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return {
            "input_ids_a": ids_a, "attention_mask_a": mask_a,
            "input_ids_b": ids_b, "attention_mask_b": mask_b,
            "label": labels,
        }
    else:
        ids = pad_sequence([x["input_ids"] for x in batch], batch_first=True, padding_value=0)
        mask = pad_sequence([x["attention_mask"] for x in batch], batch_first=True, padding_value=0)
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "label": labels}
