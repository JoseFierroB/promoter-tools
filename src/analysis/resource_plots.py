#!/usr/bin/env python3
"""
Resource usage plots — compute time & peak RAM from CLI benchmark.
Independent vertical bar charts, no annotations on bars.
Reads output/tables/resource_metrics.tsv.
"""
import sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
IN_TSV = ROOT / "output" / "tables" / "resource_metrics.tsv"
OUT_DIR = ROOT / "output" / "plots" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "MLDSPP XGBoost":           "#942C76",
    "PromoterLCNN":             "#228B22",
    "PromoTech RF-HOT (PG Max)":"#E07614",
    "iPro-MP (H. pylori)":      "#3D185A",
}


def load():
    df = pd.read_csv(IN_TSV, sep="\t")
    df = df[df["tool"].isin(COLORS)]
    return df


def bar_plot(df, col, ylabel, out_name, log_scale=False):
    tools = df["tool"].tolist()
    values = df[col].tolist()
    colors = [COLORS[t] for t in tools]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.bar(range(len(tools)), values, color=colors, width=0.55, zorder=3)
    ax.set_xticks(range(len(tools)))
    ax.set_xticklabels(
        ["MLDSPP\nXGBoost", "Promoter\nLCNN", "PromoTech\nRF-HOT", "iPro-MP\n(H. pylori)"],
        fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_name}.svg", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / f"{out_name}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out_name}.{{svg,png}}")


def main():
    df = load()
    if df.empty:
        print("ERROR: no resource metrics found. Run tools via CLI first.", file=sys.stderr)
        sys.exit(1)

    print("Resource plots:")
    bar_plot(df, "time_s", "Compute time (seconds)", "compute_time", log_scale=True)
    bar_plot(df, "peak_ram_mb", "Peak RAM (MB)", "ram", log_scale=True)


if __name__ == "__main__":
    main()
