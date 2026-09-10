#!/usr/bin/env python3
"""Per-run inference plots: ROC curves (combined + per tool) and AUC table.

Usage:
    python src/analysis/generate_auc_plots.py <run_dir> [--tools k1,k2]

Reads predictions/{name}_{fam}[.tsv] via analyze_run.load_all (new layout
first, legacy fallback) and writes into <run_dir>/plots/:
  roc_auc_{name}.{png,pdf,svg}  combined overlay, canonical palette
  roc_{tool}_{name}.{png,pdf,svg}  one panel per tool present
  metrics_rows.tsv                dataset,tool,n_pos,n_neg,auc
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "analysis"))
from analyze_run import PALETTE, load_all  # noqa: E402


def _curves(pred_root: Path, name: str, only=None):
    curves, rows = [], []
    for key, (label, y_true, y_score) in load_all(pred_root, name, only).items():
        n_pos = int(y_true.sum())
        if np.all(y_score == y_score[0]):
            fpr, tpr, a = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.500
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            a = auc(fpr, tpr)
        curves.append((label, fpr, tpr, a))
        rows.append({"dataset": name, "tool": label,
                     "n_pos": n_pos, "n_neg": len(y_true) - n_pos,
                     "auc": round(float(a), 4)})
    return curves, rows


def _style_axes(ax, title):
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)


import re

CONFIG_RE = re.compile(r"^\d+cpu(-gpu)?$")


def main(run_dir: Path, only=None):
    run_dir = Path(run_dir)
    if run_dir.name == "predictions":
        # allow pointing directly at the predictions dir
        pred_root, run_dir = run_dir, run_dir.parent
    else:
        pred_root = run_dir / "predictions"
    # run layout: <date>_runs/<name>/<config>/ -> dataset token is <name>
    name = run_dir.parent.name if CONFIG_RE.match(run_dir.name) else run_dir.name
    if not pred_root.exists():
        print(f"ERROR: no predictions/ in {run_dir}")
        sys.exit(2)

    curves, rows = _curves(pred_root, name, only)
    if not curves:
        print("  [auc] no predictions found, nothing to plot")
        return pd.DataFrame(rows)
    n_pos, n_neg = rows[0]["n_pos"], rows[0]["n_neg"]
    title = f"Receiver Operating Characteristic (ROC)\nROC {name} (N={n_pos + n_neg:,})"

    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # combined overlay
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    for cname, fpr, tpr, a in sorted(curves, key=lambda x: x[3], reverse=True):
        style = PALETTE.get(cname, {"color": "#333333", "ls": "-", "lw": 1.8})
        ax.plot(fpr, tpr, color=style["color"], linestyle=style["ls"],
                linewidth=style["lw"], label=f"{cname} (AUC = {a:.3f})")
    _style_axes(ax, title)
    plt.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(plots / f"roc_auc_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [auc] ROC saved: plots/roc_auc_{name}.png/.pdf/.svg")

    met = pd.DataFrame(rows)
    met.to_csv(run_dir / "metrics_rows.tsv", sep="\t", index=False)
    print(f"  [auc] metrics: {len(met)} tool rows -> metrics_rows.tsv")
    return met


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-run ROC + AUC plots.")
    p.add_argument("run_dir", help="Run dir (output/<date>_runs/<name>/<config>/)")
    p.add_argument("--tools", default=None, help="Comma-separated registry keys to include")
    a = p.parse_args()
    only = set(a.tools.split(",")) if a.tools else None
    main(Path(a.run_dir), only)
