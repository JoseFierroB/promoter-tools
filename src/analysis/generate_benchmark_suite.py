#!/usr/bin/env python3
"""Global multi-dataset benchmark suite: ROCs, atlas heatmap, 6-panel atlas.

Usage:
    python src/analysis/generate_benchmark_suite.py --pred-pool <dir> -o <out>
        [--datasets d1,d2,...] [--metrics-tsv metrics.tsv]

Reads a predictions pool with {db}_{fam}[.tsv] layout (new first, legacy
fallback) via analyze_run.load_tool and writes into <out>/:
  roc/roc_auc_{db}.{png,pdf,svg}   one ROC overlay per dataset
  atlas/auc_heatmap.{png,pdf,svg}  datasets x tools AUC heatmap
  atlas/roc_atlas_6panel.{png,pdf,svg}  first 6 datasets, one panel each
  benchmark_metrics.tsv            dataset,tool,n_pos,n_neg,auc (long format)

If --metrics-tsv is given it is used for the heatmap instead of recomputing.
Replaces: generate_all_15_empirical_rocs, legacy suite ROC/atlas sections,
generate_master_roc, generate_master_plots (ROC part). See archive/.
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
from analyze_run import TOOLS, FAM, PALETTE, load_tool  # noqa: E402
try:
    from bench_labels import DS_DISPLAY
except Exception:
    DS_DISPLAY = {}


def detect_datasets(pred_pool: Path):
    """Dataset keys = entry names minus known tool-family suffixes."""
    fams = sorted(set(FAM.values()), key=len, reverse=True)
    dbs = set()
    for p in pred_pool.iterdir():
        n = p.name
        if n.endswith(".tsv"):
            n = n[:-4]
        for fam in fams:
            for suffix in (f"_{fam}_{fam}_dir", f"_{fam}_{fam}", f"_{fam}"):
                if n.endswith(suffix):
                    n = n[: -len(suffix)]
                    break
        if n:
            dbs.add(n)
    return sorted(dbs)


def _display(db: str) -> str:
    return DS_DISPLAY.get(db, db)


def dataset_curves(pred_pool: Path, db: str):
    curves, rows = [], []
    for key, (label, _kind) in TOOLS.items():
        loaded = load_tool(pred_pool, db, key)
        if loaded is None:
            continue
        y_true, y_score = loaded
        n_pos = int(y_true.sum())
        if np.all(y_score == y_score[0]):
            fpr, tpr, a = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.500
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            a = auc(fpr, tpr)
        curves.append((label, fpr, tpr, a))
        rows.append({"dataset": _display(db), "tool": label,
                     "n_pos": n_pos, "n_neg": len(y_true) - n_pos,
                     "auc": round(float(a), 4)})
    return curves, rows


def _roc_axes(ax, title):
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)


def _save(fig, path: Path):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(pred_pool: Path, out: Path, datasets=None, metrics_tsv: Path = None):
    pred_pool, out = Path(pred_pool), Path(out)
    if datasets:
        dbs = datasets
    else:
        dbs = detect_datasets(pred_pool)
    (out / "roc").mkdir(parents=True, exist_ok=True)
    (out / "atlas").mkdir(parents=True, exist_ok=True)

    all_rows = []
    per_db = {}
    for db in dbs:
        curves, rows = dataset_curves(pred_pool, db)
        if not curves:
            print(f"  [suite] {db}: no predictions, skipped")
            continue
        per_db[db] = curves
        all_rows.extend(rows)
        n = rows[0]["n_pos"] + rows[0]["n_neg"]
        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
        for cname, fpr, tpr, a in sorted(curves, key=lambda x: x[3], reverse=True):
            style = PALETTE.get(cname, {"color": "#333333", "ls": "-", "lw": 1.8})
            ax.plot(fpr, tpr, color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"], label=f"{cname} (AUC = {a:.3f})")
        _roc_axes(ax, f"Receiver Operating Characteristic (ROC)\nROC {_display(db)} (N={n:,})")
        plt.tight_layout()
        _save(fig, out / "roc" / f"roc_auc_{db}")
        print(f"  [suite] ROC saved: roc/roc_auc_{db}.png/.pdf/.svg")

    met = pd.DataFrame(all_rows)
    if metrics_tsv and Path(metrics_tsv).exists():
        met = pd.read_csv(metrics_tsv, sep="\t")
        print(f"  [suite] heatmap from {metrics_tsv}")
    if met.empty:
        print("  [suite] no metrics, skipping atlas")
        return met
    met.to_csv(out / "benchmark_metrics.tsv", sep="\t", index=False)

    # heatmap datasets x tools
    order = [t for _, (t, _k) in TOOLS.items()]
    piv = met.pivot_table(index="dataset", columns="tool", values="auc", aggfunc="max")
    piv = piv[[c for c in order if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(max(10, len(piv.columns) * 1.3),
                                    max(6, len(piv) * 0.55)), dpi=300)
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto", vmin=0.45, vmax=1.0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=11, fontweight="bold", rotation=25, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=10.5, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ROC-AUC", fontsize=12, fontweight="bold")
    for i in range(len(piv)):
        for j in range(len(piv.columns)):
            val = piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color="white" if val > 0.85 else "black",
                        fontsize=10, fontweight="bold")
    ax.set_title(f"AUC matrix ({len(piv)} datasets x {len(piv.columns)} tools)",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, out / "atlas" / "auc_matrix_heatmap")
    print("  [suite] atlas/auc_matrix_heatmap saved")

    # 6-panel atlas (first 6 datasets with curves)
    panels = [db for db in dbs if db in per_db][:6]
    if panels:
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        for axi, db in zip(axes.flatten(), panels):
            for cname, fpr, tpr, a in per_db[db]:
                style = PALETTE.get(cname, {"color": "#333333", "ls": "-", "lw": 1.6})
                axi.plot(fpr, tpr, color=style["color"], linestyle=style["ls"],
                         lw=style["lw"], label=f"{cname} ({a:.3f})")
            axi.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
            axi.set_xlim([-0.02, 1.02])
            axi.set_ylim([-0.02, 1.02])
            rows_db = [r for r in all_rows if r["dataset"] == _display(db)]
            n = rows_db[0]["n_pos"] + rows_db[0]["n_neg"] if rows_db else 0
            axi.set_title(f"{_display(db)} (N={n:,})", fontsize=11.5, fontweight="bold")
            axi.set_xlabel("False Positive Rate", fontsize=9.5)
            axi.set_ylabel("True Positive Rate", fontsize=9.5)
            axi.legend(loc="lower right", fontsize=8.0, framealpha=0.9)
            axi.grid(True, linestyle=":", alpha=0.5)
        plt.suptitle("ROC atlas (6 panels)", fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        _save(fig, out / "atlas" / "roc_atlas_6panel")
        print("  [suite] atlas/roc_atlas_6panel saved")
    return met


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Global multi-dataset benchmark suite.")
    p.add_argument("--pred-pool", required=True, help="Predictions pool dir")
    p.add_argument("-o", "--output", required=True, help="Output dir")
    p.add_argument("--datasets", default=None, help="Comma-separated dataset keys (default: auto-detect)")
    p.add_argument("--metrics-tsv", default=None, help="Long metrics TSV (dataset,tool,auc) for heatmap")
    a = p.parse_args()
    main(Path(a.pred_pool), Path(a.output),
         a.datasets.split(",") if a.datasets else None,
         Path(a.metrics_tsv) if a.metrics_tsv else None)
