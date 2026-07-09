"""Collect all experiment results and compute aggregate metrics.

Scans results/ directory and produces per-RQ summary JSON + CSV tables.

Usage:
    python scripts/analyze/collect_results.py --rq rq1 --output analysis/rq1_summary.json
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CODE_DIR = ROOT_DIR / "code"
RESULTS_DIR = CODE_DIR / "results"


def _collect_seeds(result_dir: Path) -> List[Dict]:
    seeds_data = []
    for seed_dir in sorted(result_dir.glob("seed_*")):
        rfile = seed_dir / "result.json"
        if rfile.exists():
            with open(rfile) as f:
                seeds_data.append(json.load(f))
    return seeds_data


def _build_record(rq_name: str, model: str, repr_name: str, ds_name: str, seeds_data: List[Dict]) -> Dict:
    f1s = [r["test_f1"] for r in seeds_data]
    precs = [r.get("test_precision", 0.0) for r in seeds_data]
    recalls = [r.get("test_recall", 0.0) for r in seeds_data]
    return {
        "rq": rq_name,
        "model": model,
        "representation": repr_name,
        "dataset": ds_name,
        "f1_mean": np.mean(f1s),
        "f1_std": np.std(f1s),
        "f1_per_seed": f1s,
        "precision_mean": np.mean(precs),
        "precision_std": np.std(precs),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "n_seeds": len(seeds_data),
    }


def collect_rq1() -> List[Dict[str, Any]]:
    records = []
    for model in ["bilstm", "textcnn"]:
        for repr_name in ["raw", "ast_seq", "cfg_seq", "ir_seq"]:
            for ds_name in ["bcbench", "ojclone", "devign"]:
                result_dir = RESULTS_DIR / "rq1" / model / repr_name / ds_name
                if not result_dir.exists():
                    continue
                seeds_data = _collect_seeds(result_dir)
                if seeds_data:
                    records.append(_build_record("rq1", model, repr_name, ds_name, seeds_data))
    return records


def collect_rq2_or_rq3(rq_name: str) -> List[Dict[str, Any]]:
    records = []
    models = ["codebert", "codet5"] if rq_name in ("rq2", "rq3") else []
    representations = ["raw", "ast_seq", "ir_seq"]
    datasets = ["bcbench", "ojclone", "devign"]

    if rq_name == "rq4":
        models = ["bilstm", "bert_base", "unixcoder"]
        representations = ["ast_seq_bfs", "ast_seq_dfs_pre", "ast_seq_dfs_post", "ast_seq_random"]

    for model in models:
        for repr_name in representations:
            for ds_name in datasets:
                result_dir = RESULTS_DIR / rq_name / model / repr_name / ds_name
                if not result_dir.exists():
                    continue
                seeds_data = _collect_seeds(result_dir)
                if seeds_data:
                    records.append(_build_record(rq_name, model, repr_name, ds_name, seeds_data))
    return records


def collect_ablation() -> List[Dict[str, Any]]:
    records = []
    models = ["bilstm", "bert_base", "unixcoder"]
    representations = ["ast_id", "ast_noid"]
    datasets = ["bcbench", "ojclone", "devign"]

    for model in models:
        for repr_name in representations:
            for ds_name in datasets:
                result_dir = RESULTS_DIR / "ablation" / model / repr_name / ds_name
                if not result_dir.exists():
                    continue
                seeds_data = _collect_seeds(result_dir)
                if seeds_data:
                    records.append(_build_record("ablation", model, repr_name, ds_name, seeds_data))
    return records


def compute_deltas(records: List[Dict]) -> List[Dict]:
    deltas = []
    grouped = {}
    for r in records:
        key = (r["rq"], r["model"], r["dataset"])
        grouped.setdefault(key, {})
        grouped[key][r["representation"]] = r["f1_mean"]

    for (rq, model, ds), f1s in grouped.items():
        f1_raw = f1s.get("raw", 0)
        for repr_name in ["ast_seq", "ir_seq", "cfg_seq", "ast_seq_bfs", "ast_seq_dfs_pre",
                           "ast_seq_dfs_post", "ast_seq_random", "ast_id", "ast_noid"]:
            if repr_name in f1s and f1_raw > 0:
                delta = (f1s[repr_name] - f1_raw) / f1_raw
                deltas.append({
                    "rq": rq,
                    "model": model,
                    "dataset": ds,
                    "representation": repr_name,
                    "delta": delta,
                    "f1_structured": f1s[repr_name],
                    "f1_raw": f1_raw,
                })
    return deltas


def compute_identifier_contributions(records: List[Dict]) -> List[Dict]:
    contribs = []
    for r in records:
        if r["rq"] != "ablation":
            continue
        key = (r["model"], r["dataset"])
        for r2 in records:
            if r2["rq"] == r["rq"] and r2["model"] == r["model"] and r2["dataset"] == r["dataset"]:
                if r["representation"] == "ast_id" and r2["representation"] == "ast_noid":
                    f1_raw = 0.0
                    for r3 in records:
                        if r3["rq"] == "ablation" and r3["model"] == r["model"] and r3["dataset"] == r["dataset"] and r3["representation"] == "raw":
                            f1_raw = r3.get("f1_mean", 0)
                            break
                    delta_id = r["f1_mean"] - r2["f1_mean"]
                    total_gain = r["f1_mean"] - f1_raw if f1_raw else 0
                    rho = delta_id / total_gain if abs(total_gain) > 1e-9 else 0.0
                    contribs.append({
                        "model": r["model"],
                        "dataset": r["dataset"],
                        "delta_id": delta_id,
                        "total_gain": total_gain,
                        "rho": rho,
                        "f1_ast_id": r["f1_mean"],
                        "f1_ast_noid": r2["f1_mean"],
                        "f1_raw": f1_raw,
                    })
    return contribs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", default="all", help="Which RQ to collect")
    parser.add_argument("--output", default="analysis/summary.json",
                        help="Output JSON path")
    args = parser.parse_args()

    all_records = []
    if args.rq in ("all", "rq1"):
        all_records.extend(collect_rq1())
    if args.rq in ("all", "rq2"):
        all_records.extend(collect_rq2_or_rq3("rq2"))
    if args.rq in ("all", "rq3"):
        all_records.extend(collect_rq2_or_rq3("rq3"))
    if args.rq in ("all", "rq4"):
        all_records.extend(collect_rq2_or_rq3("rq4"))
    if args.rq in ("all", "ablation"):
        all_records.extend(collect_ablation())

    deltas = compute_deltas(all_records)
    id_contribs = compute_identifier_contributions(all_records)

    output = {
        "records": all_records,
        "deltas": deltas,
        "identifier_contributions": id_contribs,
        "total_conditions": len(all_records),
    }

    out_path = ROOT_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(all_records)} records + {len(deltas)} deltas to {out_path}")


if __name__ == "__main__":
    main()
