"""Pre-trained model wrapper with Siamese pair-classification and single-classification heads.

Supports:
  - Encoder-only: CodeBERT, BERT-base, UniXcoder (pool CLS / avg pool)
  - Encoder-decoder: CodeT5 (pool encoder output, decoder not used for classification)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel


def _mean_pool(encoder_last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling of encoder hidden states, respecting attention mask."""
    mask_expanded = mask.unsqueeze(-1).expand(encoder_last_hidden.size()).float()
    summed = (encoder_last_hidden * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class PretrainedCodeClassifier(nn.Module):
    """Unified Siamese classifier backed by a HuggingFace pre-trained model.

    For pair classification (clone detection): encode code_a and code_b
    independently with weight-tied encoder, concatenate pooled representations,
    and classify.

    For single classification (vulnerability): encode single input and classify directly.

    Supports both full fine-tuning and LoRA: for LoRA, wrap self.encoder with
    peft.get_peft_model() before Trainer construction.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int = 2,
        pool_method: str = "mean",
        encoder_only: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.pool_method = pool_method

        if encoder_only:
            self.encoder: PreTrainedModel = AutoModel.from_pretrained(model_name_or_path)
        else:
            full_model = AutoModel.from_pretrained(model_name_or_path)
            self.encoder: PreTrainedModel = full_model.encoder

        self.config = self.encoder.config
        self.hidden_dim = self.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_classes),
        )
        self.single_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 2, num_classes),
        )

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if self.pool_method == "mean":
            pooled = _mean_pool(hidden, attention_mask)
        else:
            pooled = hidden[:, 0, :]
        return pooled

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "input_ids_a" in batch:
            enc_a = self._encode(batch["input_ids_a"], batch["attention_mask_a"])
            enc_b = self._encode(batch["input_ids_b"], batch["attention_mask_b"])
            combined = torch.cat([enc_a, enc_b], dim=-1)
            logits = self.classifier(combined)
        else:
            enc = self._encode(batch["input_ids"], batch["attention_mask"])
            logits = self.single_classifier(enc)
        return {"logits": logits}
