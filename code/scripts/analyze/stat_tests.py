"""Statistical tests: Friedman, Nemenyi post-hoc, Cohen's d, paired t-test.

Usage:
    python scripts/analyze/stat_tests.py --summary analysis/summary.json --rq rq1
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare


def friedman_test(f1_matrix: List[List[float]]) -> dict:
    if len(f1_matrix) < 2:
        return {"statistic": 0, "p_value": 1.0, "error": "need at least 2 groups"}
    min_seeds = min(len(row) for row in f1_matrix)
    if min_seeds < 2:
        return {"statistic": 0, "p_value": 1.0, "error": "too few seeds"}
    aligned = [row[:min_seeds] for row in f1_matrix]
    stat, p = friedmanchisquare(*aligned)
    return {"statistic": float(stat), "p_value": float(p)}


def nemenyi_posthoc(f1_matrix: List[List[float]], names: List[str]) -> List[dict]:
    n_groups = len(f1_matrix)
    min_seeds = min(len(row) for row in f1_matrix)
    aligned = [row[:min_seeds] for row in f1_matrix]
    n_blocks = min_seeds

    if n_groups < 2 or n_blocks < 2:
        return []

    ranks = np.zeros((n_groups, n_blocks))
    for b in range(n_blocks):
        values = np.array([aligned[g][b] for g in range(n_groups)])
        ranks[:, b] = stats.rankdata(-values)

    avg_ranks = ranks.mean(axis=1)
    q_crit = stats.studentized_range.ppf(0.95, n_groups, 1e9)
    cd = q_crit * np.sqrt(n_groups * (n_groups + 1) / (6.0 * n_blocks))

    comparisons = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            significant = diff > cd
            comparisons.append({
                "group_a": names[i],
                "group_b": names[j],
                "rank_diff": float(diff),
                "critical_distance": float(cd),
                "significant": significant,
            })
    return comparisons


def cohens_d(group1: List[float], group2: List[float]) -> float:
    a1, a2 = np.array(group1), np.array(group2)
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * a1.var(ddof=1) + (n2 - 1) * a2.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std < 1e-9:
        return 0.0
    return float((a1.mean() - a2.mean()) / pooled_std)


def paired_ttest(group1: List[float], group2: List[float]) -> dict:
    t_stat, p = stats.ttest_rel(group1, group2)
    return {"t_statistic": float(t_stat), "p_value": float(p)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary.json from collect_results.py")
    parser.add_argument("--rq", default="rq1")
    parser.add_argument("--dataset", default=None, help="Filter to specific dataset")
    args = parser.parse_args()

    with open(args.summary, "r") as f:
        summary = json.load(f)

    records = [r for r in summary["records"] if r["rq"] == args.rq]
    if args.dataset:
        records = [r for r in records if r["dataset"] == args.dataset]

    datasets = sorted(set(r["dataset"] for r in records))

    for ds_name in datasets:
        ds_records = [r for r in records if r["dataset"] == ds_name]
        models = sorted(set(r["model"] for r in ds_records))

        for model in models:
            model_records = [r for r in ds_records if r["model"] == model]
            representations = sorted(set(r["representation"] for r in model_records))

            print(f"\n=== {args.rq} | {ds_name} | {model} ===")

            repr_matrix = []
            repr_names = []
            for repr_name in representations:
                repr_records = [r for r in model_records if r["representation"] == repr_name]
                if repr_records:
                    rec = repr_records[0]
                    f1_mean = rec["f1_mean"]
                    f1_std = rec.get("f1_std", 0)
                    f1_per_seed = rec.get("f1_per_seed", [f1_mean])
                    print(f"  {repr_name}: F1={f1_mean:.4f} ± {f1_std:.4f}")
                    repr_matrix.append(f1_per_seed)
                    repr_names.append(repr_name)

            if len(repr_matrix) >= 2:
                friedman_result = friedman_test(repr_matrix)
                print(f"  Friedman: stat={friedman_result['statistic']:.4f} p={friedman_result['p_value']:.4f}")

                if friedman_result["p_value"] < 0.05 and len(repr_matrix) >= 3:
                    comparisons = nemenyi_posthoc(repr_matrix, repr_names)
                    for c in comparisons:
                        if c["significant"]:
                            print(f"  Nemenyi: {c['group_a']} != {c['group_b']} (diff={c['rank_diff']:.2f} > CD={c['critical_distance']:.2f})")


if __name__ == "__main__":
    main()
