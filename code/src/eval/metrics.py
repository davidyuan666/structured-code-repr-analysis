"""Evaluation metrics: F1, precision, recall, and derived quantities (Δ, δ)."""
from __future__ import annotations

from typing import Dict, List

from sklearn.metrics import f1_score, precision_score, recall_score


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    """Compute F1, precision, recall (macro-averaged)."""
    return {
        "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }


def compute_delta(f1_structured: float, f1_raw: float) -> float:
    """Relative improvement of structured representation over raw code."""
    if f1_raw == 0:
        return 0.0
    return (f1_structured - f1_raw) / f1_raw


def compute_degradation(delta_ft: float, delta_lora: float) -> float:
    """Degradation when moving from full FT to LoRA.

    Returns:
        Positive value means LoRA is worse than full FT.
    """
    return delta_ft - delta_lora


def compute_identifier_contribution(f1_ast_id: float, f1_ast_noid: float, f1_raw: float) -> Dict[str, float]:
    """Compute identifier contribution metrics.

    delta_id: F1(AST+ID) - F1(AST-ID) — marginal contribution of identifiers.
    total_gain: F1(AST+ID) - F1(Raw) — total AST advantage.
    rho: delta_id / total_gain — fraction of AST advantage attributable to identifiers.
    """
    delta_id = f1_ast_id - f1_ast_noid
    total_gain = f1_ast_id - f1_raw
    rho = delta_id / total_gain if abs(total_gain) > 1e-9 else 0.0
    return {"delta_id": delta_id, "total_gain": total_gain, "rho": rho}
