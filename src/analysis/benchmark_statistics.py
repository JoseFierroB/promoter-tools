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

from statistics import bootstrap_auc_ci, delong_test


def load_predictions(pos_file: str, neg_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load prediction scores, return (y_true, y_scores)."""
    if "ipromp" in pos_file.lower():
        ip_path = PRED_DIR / pos_file
        if not ip_path.exists():
            ip_path = PRED_DIR / "ipromp" / "ipromp_12_predictions.csv"
        content_sample = ip_path.read_text()[:200]
        sep = "\t" if "\t" in content_sample else ","
        df = pd.read_csv(ip_path, sep=sep)
        col = "Probability" if "Probability" in df.columns else "PRED"
        n_total = len(df)
        n_pos = 988 if n_total == 1988 else n_total // 2
        n_neg = n_total - n_pos
        y_true = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
        y_score = df[col].values[:n_total]
        return y_true, y_score

    pos = pd.read_csv(PRED_DIR / pos_file, sep="\t" if (PRED_DIR / pos_file).suffix == ".csv" else ",")
    neg = pd.read_csv(PRED_DIR / neg_file, sep="\t" if (PRED_DIR / neg_file).suffix == ".csv" else ",")
    pos_c = "PRED" if "PRED" in pos.columns else "Probability"
    neg_c = "PRED" if "PRED" in neg.columns else "Probability"
    y_true = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    y_score = np.hstack([pos[pos_c].values, neg[neg_c].values])
    return y_true, y_score


# ── Tool registry: (display_name, category, pos_csv, neg_csv) ──
TOOLS = [
    ("PromoTech RF-HOT", "ML", "promotech/workdir/hot_pg_pos/sequences_predictions.csv", "promotech/workdir/hot_pg_neg/sequences_predictions.csv"),
    ("PromoTech RF-TETRA", "ML", "promotech/workdir/tetra_pg_pos/sequences_predictions.csv", "promotech/workdir/tetra_pg_neg/sequences_predictions.csv"),
    ("PromoterLCNN", "DL", "lcnn/lcnn_pos.csv", "lcnn/lcnn_neg.csv"),
    ("MEME (STREME+FIMO)", "Other", "meme_pos.csv", "meme_neg.csv"),
    ("FIMO (E. coli DB)", "Other", "fimo_db_pos.csv", "fimo_db_neg.csv"),
    ("FIMO (Prok DB)", "Other", "fimo_prok_pos.csv", "fimo_prok_neg.csv"),
    ("MLDSPP (XGBoost)", "ML", "mldspp_pos.csv", "mldspp_neg.csv"),
    ("iPro-MP (sp 12)", "DL", "ipromp/ipromp_12_predictions.csv", "ipromp/ipromp_12_predictions.csv"),
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
            res = delong_test(y_true_i, s_i, s_j)
            p = res["p_value"]
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
