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
from analyze_run import PALETTE, TOOL_ORDER, load_all  # noqa: E402
try:
    from bench_labels import DS_DISPLAY  # noqa: E402
except Exception:
    DS_DISPLAY = {}


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
        # canonical: 1_inference/predictions, legacy fallback: predictions
        cand = run_dir / "1_inference" / "predictions"
        pred_root = cand if cand.exists() else run_dir / "predictions"
        # also handle RunLayout helper
        try:
            from run_layout import RunLayout as _RL
        except ImportError:
            try:
                from src.analysis.run_layout import RunLayout as _RL
            except ImportError:
                _RL = None
        if _RL is not None:
            try:
                pred_root = _RL(run_dir).predictions
            except Exception:
                pass
    # run layout: <date>_runs/<name>/<config>/ -> dataset token is <name>
    name = run_dir.parent.name if CONFIG_RE.match(run_dir.name) else run_dir.name
    # strip fast7 suffix for display but keep full for loading?
    if not pred_root.exists():
        print(f"ERROR: no predictions/ in {run_dir} (checked {pred_root})")
        sys.exit(2)

    curves, rows = _curves(pred_root, name, only)
    if not curves:
        print("  [auc] no predictions found, nothing to plot")
        return pd.DataFrame(rows)
    n_pos, n_neg = rows[0]["n_pos"], rows[0]["n_neg"]
    ds_disp = DS_DISPLAY.get(name, name)
    # configuration always declared (e.g. 16cpu-gpu) if run_dir matches CPU pattern
    cfg = run_dir.name if CONFIG_RE.match(run_dir.name) else ""
    cfg_disp = {"1cpu": "1CPU", "16cpu": "16CPU", "1cpu-gpu": "1CPU+GPU", "16cpu-gpu": "16CPU+GPU"}.get(cfg, cfg)
    title = f"Receiver Operating Characteristic (ROC)\n{ds_disp} (N={n_pos + n_neg:,})" + (f" — {cfg_disp}" if cfg_disp else "")

    # canonical 1_inference
    try:
        from run_layout import RunLayout as _RL2
    except ImportError:
        try:
            from src.analysis.run_layout import RunLayout as _RL2
        except ImportError:
            _RL2 = None
    if _RL2 is not None:
        try:
            plots = _RL2(run_dir).inference
            tables = _RL2(run_dir).tables
        except Exception:
            plots = run_dir / "1_inference"
            tables = run_dir / "3_tables"
    else:
        plots = run_dir / "1_inference"
        tables = run_dir / "3_tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    # combined overlay — canonical TOOL_ORDER (BDT→RF→CNN→NN→gLM→motif), not alphabetical/value
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    order_idx = {t: i for i, t in enumerate(TOOL_ORDER)}
    for cname, fpr, tpr, a in sorted(curves, key=lambda x: order_idx.get(x[0], 999)):
        style = PALETTE.get(cname, {"color": "#333333", "ls": "-", "lw": 1.8})
        ax.plot(fpr, tpr, color=style["color"], linestyle=style["ls"],
                linewidth=style["lw"], label=f"{cname} (AUC = {a:.3f})")
    _style_axes(ax, title)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"roc_auc_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [auc] ROC saved: 1_inference/roc_auc_{name}.png/.pdf")

    met = pd.DataFrame(rows)
    met_path = tables / "metrics_rows.tsv"
    met.to_csv(met_path, sep="\t", index=False)
    print(f"  [auc] metrics: {len(met)} tool rows -> 3_tables/metrics_rows.tsv")
    # legacy compat symlink at root
    try:
        legacy = run_dir / "metrics_rows.tsv"
        if not legacy.exists():
            legacy.symlink_to(Path("3_tables") / "metrics_rows.tsv")
    except Exception:
        pass
    return met


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-run ROC + AUC plots.")
    p.add_argument("run_dir", help="Run dir (output/<date>_runs/<name>/<config>/)")
    p.add_argument("--tools", default=None, help="Comma-separated registry keys to include")
    a = p.parse_args()
    only = set(a.tools.split(",")) if a.tools else None
    main(Path(a.run_dir), only)
