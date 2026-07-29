#!/usr/bin/env python3
"""
Benchmark statistics: bootstrap CIs + DeLong pairwise tests for all tools.
Reads prediction CSVs from output/predictions/, computes AUC with 95% CI,
and pairwise DeLong p-values.

Usage:
    pixi run python src/analysis/benchmark_statistics.py [--output output/tables/benchmark_statistics.tsv]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "output" / "predictions"

from src.analysis.statistics import bootstrap_auc_ci, delong_test


def load_predictions(pos_file: str, neg_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load prediction scores, return (y_true, y_scores)."""
    pos = pd.read_csv(PRED_DIR / pos_file, sep="\t")
    neg = pd.read_csv(PRED_DIR / neg_file, sep="\t")
    y_true = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.hstack([pos["PRED"].values, neg["PRED"].values])
    return y_true, y_score


# ── Tool registry: (display_name, category, pos_csv, neg_csv) ──
TOOLS = [
    ("PromoTech RF-HOT", "ML", "promotech_hot_pg_predictions_pos.csv", "promotech_hot_pg_predictions_neg.csv"),
    ("PromoTech RF-TETRA", "ML", "promotech_tetra_pg_predictions_pos.csv", "promotech_tetra_pg_predictions_neg.csv"),
    ("PromoterLCNN", "DL", "lcnn_pos.csv", "lcnn_neg.csv"),
    ("MLDSPP (XGBoost)", "ML", "mldspp_pos.csv", "mldspp_neg.csv"),
    ("MLDSPP (RF)", "ML", "mldspp_rf_pos.csv", "mldspp_rf_neg.csv"),
    ("MLDSPP (SVM)", "ML", "mldspp_svm_pos.csv", "mldspp_svm_neg.csv"),
    ("iPro-MP (sp 12)", "DL", "ipromp_pos.csv", "ipromp_neg.csv"),
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark statistics")
    parser.add_argument("-o", "--output", default="output/tables/benchmark_statistics.tsv")
    args = parser.parse_args()

    results = []
    tool_data = {}  # name -> (y_true, y_scores)

    print("Computing bootstrap 95% CIs for ROC AUC...")
    for name, cat, pos_file, neg_file in TOOLS:
        if not (PRED_DIR / pos_file).exists():
            print(f"  SKIP {name}: missing {pos_file}")
            continue

        y_true, y_score = load_predictions(pos_file, neg_file)
        ci = bootstrap_auc_ci(y_true, y_score, n_bootstrap=2000)
        tool_data[name] = (y_true, y_score)

        print(f"  {name:<25} AUC={ci['auc']:.4f}  CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        results.append({
            "Model": name,
            "Category": cat,
            "AUC": ci["auc"],
            "CI_lower": ci["ci_lower"],
            "CI_upper": ci["ci_upper"],
            "N": len(y_true),
        })

    # ── DeLong pairwise tests ──
    print(f"\nDeLong pairwise tests (p-values):")
    tool_names = list(tool_data.keys())
    n = len(tool_names)
    pvalues = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            y_true_i, s_i = tool_data[tool_names[i]]
            y_true_j, s_j = tool_data[tool_names[j]]
            # Ensure same test set ordering
            assert np.array_equal(y_true_i, y_true_j), f"Label mismatch: {tool_names[i]} vs {tool_names[j]}"
            p = delong_test(y_true_i, s_i, s_j)
            pvalues[i, j] = p
            pvalues[j, i] = p
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {tool_names[i]:<25} vs {tool_names[j]:<25} p={p:.4f} {sig}")

    # ── Save ──
    df = pd.DataFrame(results)
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved: {out_path}")

    # ── Print ranked summary ──
    print(f"\n{'Model':<25} {'AUC':>6} {'95% CI':>18} {'Category':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["AUC"], reverse=True):
        ci_str = f"[{r['CI_lower']:.4f}, {r['CI_upper']:.4f}]"
        print(f"{r['Model']:<25} {r['AUC']:>6.4f} {ci_str:>18} {r['Category']:>8}")


if __name__ == "__main__":
    main()
