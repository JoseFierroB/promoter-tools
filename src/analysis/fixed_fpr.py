#!/usr/bin/env python3
"""Sensitivity at fixed low FPR + per-genome FP estimates.

Context: a bacterial genome has ~10^3 true promoters among ~2e6 81-bp
windows, so usable operating points live at very low FPR. ROC AUC alone
hides that regime; this reports TPR at fixed FPRs plus the expected
false positives per 2 Mb genome.

Reads predictions via analyze_run.load_all (same loader as the ROC),
writes 3_tables/fixed_fpr_metrics.tsv with one row per tool and columns:
  tool, n_pos, n_neg, auc,
  tpr_at_fpr_{0.1,0.01,0.001,0.0001},
  fp_per_2Mb_at_fpr_{...}  (= fpr * 2_000_000)
Plus a reference row ("genome context"): n_promoters_assumed=3000,
n_windows=2_000_000 (edit constants below for other genomes).

Usage:
    pixi run python src/analysis/fixed_fpr.py <run_dir> [--name NAME]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_run import load_all  # noqa: E402

FPRS = (0.1, 0.01, 0.001, 0.0001)
WINDOWS_PER_2MB = 2_000_000
N_PROMOTERS_ASSUMED = 3000


def tpr_at_fpr(y_true, y_score, fpr_target):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # TPR at the largest threshold with FPR <= target (step function).
    ok = np.where(fpr <= fpr_target)[0]
    if len(ok) == 0:
        return 0.0
    return float(tpr[ok[-1]])


def main(run_dir: Path, name: str = None):
    run_dir = Path(run_dir)
    pred_root = run_dir / "1_inference" / "predictions"
    if not pred_root.is_dir():
        pred_root = run_dir / "predictions"
    if name is None:
        name = run_dir.parent.name
    scored = load_all(pred_root, name)
    if not scored:
        print("ERROR: no predictions found", file=sys.stderr)
        sys.exit(2)
    rows = []
    for key, (label, y_true, y_score) in scored.items():
        y_true = np.asarray(y_true)
        row = {"tool": label, "n_pos": int(y_true.sum()),
               "n_neg": int(len(y_true) - int(y_true.sum()))}
        for f in FPRS:
            t = tpr_at_fpr(y_true, y_score, f)
            row[f"tpr_at_fpr_{f:g}"] = round(t, 4)
            row[f"fp_per_2Mb_at_fpr_{f:g}"] = int(round(f * WINDOWS_PER_2MB))
            row[f"tp_per_3000_at_fpr_{f:g}"] = int(round(t * N_PROMOTERS_ASSUMED))
        rows.append(row)
    met = pd.DataFrame(rows).sort_values("tpr_at_fpr_0.01", ascending=False)
    tables = run_dir / "3_tables"
    tables.mkdir(parents=True, exist_ok=True)
    met.to_csv(tables / "fixed_fpr_metrics.tsv", sep="\t", index=False)
    print(f"  [fixed-fpr] {len(met)} tool rows -> 3_tables/fixed_fpr_metrics.tsv "
          f"(windows={WINDOWS_PER_2MB}, promoters={N_PROMOTERS_ASSUMED})")
    print(met[["tool", "tpr_at_fpr_0.01", "tpr_at_fpr_0.001",
               "fp_per_2Mb_at_fpr_0.01", "fp_per_2Mb_at_fpr_0.001",
               "tp_per_3000_at_fpr_0.01"]].to_string(index=False))
    return met


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="TPR at fixed low FPR + per-genome FP estimates.")
    p.add_argument("run_dir", help="Run dir (output/<date>_runs/<name>/<config>/)")
    p.add_argument("--name", default=None, help="Dataset name override")
    a = p.parse_args()
    main(Path(a.run_dir), a.name)
