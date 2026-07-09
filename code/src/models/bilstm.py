"""BiLSTM + Attention model for traditional neural network baseline (RQ1, RQ4)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    """2-layer BiLSTM with dot-product attention pooling.

    Used in RQ1 (serialization barrier) and RQ4 (traversal sensitivity, no-pre-training tier).

    Supports two modes:
      - "pair": Siamese encoding of (code_a, code_b) -> concatenate -> classifier.
                The encoder is shared (same forward_fn called twice).
      - "single": Single-stream classification through the same encoder backbone.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        pad_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_linear = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.classifier_pair = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )
        self.classifier_single = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def _encode(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)
        lengths = mask.sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        attn_scores = self.attn_linear(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)
        context = torch.bmm(attn_weights, lstm_out).squeeze(1)
        return context

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "input_ids_a" in batch:
            enc_a = self._encode(batch["input_ids_a"], batch["attention_mask_a"])
            enc_b = self._encode(batch["input_ids_b"], batch["attention_mask_b"])
            combined = torch.cat([enc_a, enc_b], dim=-1)
            logits = self.classifier_pair(combined)
        else:
            enc = self._encode(batch["input_ids"], batch["attention_mask"])
            logits = self.classifier_single(enc)
        return {"logits": logits}
