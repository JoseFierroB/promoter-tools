#!/usr/bin/env python3
"""Per-run metric tables: confusion @ Youden's J + bootstrap AUC CI + DeLong.

Usage:
    python src/analysis/compute_metrics.py <run_dir> [--bootstrap N]

Reads predictions/{name}_{fam}[.tsv] via analyze_run.load_all (new layout
first, legacy fallback) and writes <run_dir>/metrics_table.tsv with columns:
dataset,tool,n_pos,n_neg,auc,auc_ci_low,auc_ci_high,threshold,tp,fn,fp,tn,
sensitivity,specificity,precision,f1,accuracy[,delong_p_vs_best].

Merges legacy benchmark_confusion.py + benchmark_statistics.py (see archive/).
Uses statistics.bootstrap_auc_ci / delong_test from src.analysis.statistics.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT))
from analyze_run import load_all  # noqa: E402
from src.analysis.statistics import bootstrap_auc_ci, delong_test  # noqa: E402


def confusion_at_best(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    best_thresh = float(thresholds[best_idx])
    y_pred = (y_score >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn)
    return {"threshold": best_thresh, "tp": int(tp), "fn": int(fn),
            "fp": int(fp), "tn": int(tn), "sensitivity": sens,
            "specificity": spec, "precision": prec, "f1": f1, "accuracy": acc}


def main(run_dir: Path, n_bootstrap: int = 500):
    run_dir = Path(run_dir)
    pred_root = run_dir / "predictions" if run_dir.name != "predictions" else run_dir
    if run_dir.name == "predictions":
        run_dir = run_dir.parent
    import re
    name = (run_dir.parent.name if re.match(r"^\d+cpu(-gpu)?$", run_dir.name)
            else run_dir.name)
    if not pred_root.exists():
        print(f"ERROR: no predictions/ in {run_dir}")
        sys.exit(2)

    scored = {}
    for key, (label, y_true, y_score) in load_all(pred_root, name).items():
        scored[label] = (y_true, y_score)
    if not scored:
        print("  [metrics] no predictions found, nothing to compute")
        return pd.DataFrame()

    rows = []
    for label, (y_true, y_score) in scored.items():
        n_pos = int(y_true.sum())
        a = float(auc(*roc_curve(y_true, y_score)[:2]))
        ci = bootstrap_auc_ci(y_true, y_score, n_bootstrap=n_bootstrap)
        cm = confusion_at_best(y_true, y_score)
        rows.append({"dataset": name, "tool": label, "n_pos": n_pos,
                     "n_neg": len(y_true) - n_pos, "auc": round(a, 4),
                     "auc_ci_low": round(float(ci["ci_lower"]), 4),
                     "auc_ci_high": round(float(ci["ci_upper"]), 4), **cm})
    # DeLong p-value of each tool vs the best-AUC tool
    best = max(rows, key=lambda r: r["auc"])["tool"]
    yb, sb = scored[best]
    for r in rows:
        if r["tool"] == best:
            r["delong_p_vs_best"] = 1.0
        else:
            y, s = scored[r["tool"]]
            try:
                r["delong_p_vs_best"] = round(float(
                    delong_test(yb, sb, s)["p_value"]), 4)
            except Exception as e:
                print(f"  [metrics] DeLong failed for {r['tool']}: {e}")
                r["delong_p_vs_best"] = float("nan")

    met = pd.DataFrame(rows).sort_values("auc", ascending=False)
    met.to_csv(run_dir / "metrics_table.tsv", sep="\t", index=False)
    print(f"  [metrics] {len(met)} tool rows -> metrics_table.tsv "
          f"(best: {best})")
    return met


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-run confusion + CI + DeLong metrics.")
    p.add_argument("run_dir", help="Run dir (output/<date>_runs/<name>/<config>/)")
    p.add_argument("--bootstrap", type=int, default=500, help="Bootstrap replicates for AUC CI")
    a = p.parse_args()
    main(Path(a.run_dir), a.bootstrap)
