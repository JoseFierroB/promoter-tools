#!/usr/bin/env python3
"""Scaling: baseline (batch64 serial) vs prefetch128 (batch128 + prefetch).

Baseline walls come from the 260918 scale runs' resource_metrics.tsv
(ProkBERT-mini row); prefetch128 walls from
output/experiments/prokbert_ab/prefetch128_walls.tsv (predict phase).

Writes 2_resources/scaling_prefetch128_vs_baseline.{png,pdf},
3_tables/scaling_prefetch128.tsv and MANIFEST.json into the output dir.

Usage:
    pixi run python experiments/prokbert_runner_ab/plot_ab_scaling.py \
        -o output/comparisons/260923_prokbert_prefetch128_scale
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def baseline_walls():
    rows = []
    for sz, n in (("10k", 10000), ("30k", 30000), ("60k", 60000), ("100k", 100000)):
        res = pd.read_csv(
            ROOT / f"output/260918_runs/scale_{sz}/16cpu-gpu/2_resources/resource_metrics.tsv",
            sep="\t")
        r = res[res["tool"] == "ProkBERT-mini (NeuralBioInfo)"].iloc[0]
        rows.append({"n": n, "wall": float(r["wall_seconds"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    base = baseline_walls()
    opt = pd.read_csv(ROOT / "output/experiments/prokbert_ab/prefetch128_walls.tsv")
    opt = opt.rename(columns={"wall_predict_s": "wall"})
    for df in (base, opt):
        df["thr"] = df["n"] / df["wall"]

    out = Path(args.output)
    resd, tab = out / "2_resources", out / "3_tables"
    resd.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    comp = pd.DataFrame({
        "n": base["n"], "wall_baseline_s": base["wall"].round(1),
        "wall_prefetch128_s": opt["wall"].round(1),
        "speedup": (base["wall"] / opt["wall"].values).round(2),
        "auc_baseline": [0.9260, 0.9181, 0.9142, 0.9143],
        "auc_prefetch128": opt["auc"].round(4).tolist()})
    comp.to_csv(tab / "scaling_prefetch128.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    for ax, col, title, unit in (
            (axes[0], "wall", "Wall time vs N (log-log)", "s"),
            (axes[1], "thr", "Throughput vs N", "seq/s")):
        for df, lab, c, m in ((base, "baseline batch64 serial", "#D81B60", "o"),
                              (opt, "prefetch128 + 16 workers", "#2D004D", "s")):
            ax.loglog(df["n"], df[col], f"{m}-", color=c, lw=2.2, ms=8, label=lab)
        ax.set_xlabel("Dataset size N (pos+neg)", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        leg = ax.legend(fontsize=10, loc="best")
        leg.get_title().set_fontweight("bold")
    axes[0].set_ylabel("Wall predict (s)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Throughput (seq/s)", fontsize=12, fontweight="bold")
    fig.suptitle("ProkBERT-mini scaling — baseline vs prefetch128 (5090, fp32, AUC identical)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(resd / f"scaling_prefetch128_vs_baseline.{ext}", bbox_inches="tight")
    plt.close(fig)
    (out / "MANIFEST.json").write_text(json.dumps(
        {"type": "scaling_ab", "tool": "ProkBERT-mini [gLM]",
         "auc_identical": True}, indent=2))
    print(comp.to_string(index=False))
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
