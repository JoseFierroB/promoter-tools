#!/usr/bin/env python3
"""
Resource usage plots — compute time, peak RAM, model size.

Adds method sublabels (BDT, CNN, gLM, RF, motif).
Removes log scale; value labels on bars.

Reads output/tables/resource_metrics.tsv
"""

import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
IN_TSV = ROOT / "output" / "tables" / "resource_metrics.tsv"
OUT_DIR = ROOT / "output" / "plots" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tool metadata ──
TOOL_INFO = {
    "MLDSPP XGBoost":           {"color": "#942C76", "short": "MLDSPP\nXGBoost",       "method": "BDT"},
    "MLDSPP XGBoost (75% spn)": {"color": "#B07AA1", "short": "MLDSPP 75%\nXGBoost",    "method": "BDT"},
    "PromoterLCNN":             {"color": "#228B22", "short": "Promoter\nLCNN",         "method": "CNN"},
    "PromoTech RF-HOT (PG Max)":{"color": "#E07614", "short": "PromoTech\nRF-HOT",      "method": "RF"},
    "iPro-MP (H. pylori)":      {"color": "#3D185A", "short": "iPro-MP\n(H. pylori)",   "method": "gLM"},
    "MEME Suite (STREME+FIMO)":  {"color": "#1E88E5", "short": "MEME\nSTREME+FIMO",     "method": "motif"},
    "FIMO + Prokaryote DB":     {"color": "#8E24AA", "short": "FIMO\nProk DB",          "method": "motif"},
}

METHOD_COLORS = {
    "BDT":   "#942C76",
    "CNN":   "#228B22",
    "gLM":   "#3D185A",
    "RF":    "#E07614",
    "motif": "#1E88E5",
}


def load():
    df = pd.read_csv(IN_TSV, sep="\t")
    # Filter to tools we know
    df = df[df["tool"].isin(TOOL_INFO)]
    # Add metadata
    df["method"] = df["tool"].map(lambda t: TOOL_INFO.get(t, {}).get("method", "other"))
    df["short"]  = df["tool"].map(lambda t: TOOL_INFO.get(t, {}).get("short", t))
    df["color"]  = df["tool"].map(lambda t: TOOL_INFO.get(t, {}).get("color", "#999"))
    return df


def bar_plot(df, col, ylabel, out_name, fmt=".0f", unit=""):
    """Vertical bar chart with method labels and value annotations."""
    tools = df["tool"].tolist()
    values = df[col].tolist()
    colors = df["color"].tolist()
    methods = df["method"].tolist()
    shorts = df["short"].tolist()

    fig, ax = plt.subplots(figsize=(len(tools)*1.1 + 1, 5), dpi=300)
    x = np.arange(len(tools))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3, edgecolor="white", linewidth=0.5)

    # Value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        label = f"{val:{fmt}}{unit}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    # X-axis: short name + method sublabel
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, fontsize=8)
    # Add method sublabel below
    for i, m in enumerate(methods):
        ax.text(i, -max(values)*0.06, m, ha="center", va="top", fontsize=7,
                color=METHOD_COLORS.get(m, "#666"), fontstyle="italic")

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max(values) * 1.18)
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
        print("WARNING: no resource metrics found. Only plotting what's available.")

    print("Resource plots:")

    # Time
    bar_plot(df, "time_s", "Compute time (seconds)", "compute_time", fmt=".1f", unit="s")

    # RAM
    bar_plot(df, "peak_ram_mb", "Peak RAM (MB)", "ram", fmt=".0f", unit=" MB")

    # GPU VRAM (only if > 0)
    if "peak_vram_mb" in df.columns and (df["peak_vram_mb"] > 0).any():
        df_vram = df[df["peak_vram_mb"] > 0]
        bar_plot(df_vram, "peak_vram_mb", "Peak VRAM (MB)", "vram", fmt=".0f", unit=" MB")
        for _, row in df_vram.iterrows():
            print(f"    GPU {row['gpu_name']}: {row['peak_vram_mb']:.0f} MB ({row['gpu_util_pct']:.0f}% util)")

    # Model size (only if > 0)
    if "model_size_mb" in df.columns and (df["model_size_mb"] > 0).any():
        df_model = df[df["model_size_mb"] > 0]
        bar_plot(df_model, "model_size_mb", "Model size on disk (MB)", "model_size", fmt=".0f", unit=" MB")

    # Legend: method types
    fig, ax = plt.subplots(figsize=(6, 0.8))
    for i, (method, color) in enumerate(METHOD_COLORS.items()):
        ax.add_patch(plt.Rectangle((i*0.15, 0), 0.12, 0.6, color=color))
        ax.text(i*0.15 + 0.06, 0.3, method, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xlim(-0.05, len(METHOD_COLORS)*0.15 + 0.05)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Method: BDT=Boosted Trees  CNN=Conv. Net  gLM=gene.LM  RF=Random Forest", fontsize=8)
    plt.savefig(OUT_DIR / "method_legend.svg", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
