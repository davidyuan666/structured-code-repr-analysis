"""General-purpose Trainer for all sub-experiments.

Supports:
  - Traditional NNs (BiLSTM, TextCNN) trained from scratch
  - Pre-trained models under full fine-tuning
  - Metric logging and early stopping
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.metrics import compute_metrics


def _linear_warmup_lambda(current_step: int, warmup_steps: int) -> float:
    if warmup_steps == 0:
        return 1.0
    return min(1.0, current_step / max(1, warmup_steps))


class Trainer:
    """Train / validate / evaluate loop for one experimental condition.

    Handles:
      - Full fine-tuning (updates all parameters)
      - Traditional NN training (same interface)
      - LoRA fine-tuning (the model is already wrapped with peft; this Trainer
        doesn't differentiate — the optimizer simply receives the parameter set
        the caller provides).

    After training, saves results JSON with best-val F1 and test metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        lr: float,
        max_epochs: int,
        warmup_ratio: float = 0.0,
        weight_decay: float = 0.01,
        patience: int = 0,
        device: str = "cuda",
        output_dir: Optional[Path] = None,
        log_interval: int = 50,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr = lr
        self.max_epochs = max_epochs
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.device = device
        self.output_dir = output_dir or Path(".")
        self.log_interval = log_interval

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        total_steps = max_epochs * len(train_loader)
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _linear_warmup_lambda(step, warmup_steps),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.best_state = None
        self.patience_counter = 0

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="train", leave=False)
        for batch in pbar:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs["logits"], batch["label"])
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Tuple[Dict[str, float], List[int], List[int]]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_loss = 0.0
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._to_device(batch)
            outputs = self.model(batch)
            loss = self.criterion(outputs["logits"], batch["label"])
            all_loss += loss.item()
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = all_loss / len(loader)
        return metrics, all_preds, all_labels

    def train(self) -> Dict[str, Any]:
        """Run full training loop. Returns metrics dict."""
        start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_metrics, _, _ = self._evaluate(self.val_loader)
            val_f1 = val_metrics["f1"]
            tqdm.write(
                f"Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                f"val_f1={val_f1:.4f}  val_prec={val_metrics['precision']:.4f}  "
                f"val_recall={val_metrics['recall']:.4f}"
            )

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience > 0 and self.patience_counter >= self.patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        test_metrics, test_preds, test_labels = self._evaluate(self.test_loader)
        elapsed = time.time() - start_time
        result = {
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_loss": test_metrics["loss"],
            "train_time_sec": elapsed,
            "eval_predictions": test_preds,
            "eval_labels": test_labels,
        }
        self._save_result(result)
        return result

    def _save_result(self, result: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
