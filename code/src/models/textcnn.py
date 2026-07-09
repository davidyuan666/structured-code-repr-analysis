"""TextCNN model for traditional neural network baseline (RQ1)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Multi-filter 1D CNN for text classification.

    Used in RQ1 alongside BiLSTM. Supports pair and single modes through
    a shared encoder as in BiLSTM.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        filter_sizes: tuple = (2, 3, 4, 5),
        num_filters: int = 128,
        num_classes: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes
        ])
        total_filters = len(filter_sizes) * num_filters
        self.classifier = nn.Sequential(
            nn.Linear(total_filters * 2, total_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters, num_classes),
        )

    def _encode(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        emb = emb.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))
            h = F.max_pool1d(h, h.size(-1)).squeeze(-1)
            pooled.append(h)
        return torch.cat(pooled, dim=-1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "input_ids_a" in batch:
            enc_a = self._encode(batch["input_ids_a"], batch["attention_mask_a"])
            enc_b = self._encode(batch["input_ids_b"], batch["attention_mask_b"])
            combined = torch.cat([enc_a, enc_b], dim=-1)
        else:
            enc = self._encode(batch["input_ids"], batch["attention_mask"])
            combined = enc
        logits = self.classifier(combined)
        return {"logits": logits}
