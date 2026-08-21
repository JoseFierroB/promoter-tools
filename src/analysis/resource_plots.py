#!/usr/bin/env python3
"""
Resource usage plot — single figure, vertical bars.

One figure with three panels:
  1. Compute time (s)          — all tools, linear
  2. Peak RAM (MB)             — all tools, linear
  3. GPU: peak VRAM + model size (MB) — GPU tools only

Each bar shows the tool short name + method sublabel (BDT, CNN, gLM, RF,
motif) and its value on top.

Usage:
    pixi run python src/analysis/resource_plots.py [--iter 9880]
"""
import os
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("PROMOTER_DATA_DIR", "/home/fierro/Desktop"))
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "scale_db_4tools" / "plots"
OUT_DIR = DEFAULT_OUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tool metadata ──
TOOL_INFO = {
    "MLDSPP XGBoost":            {"color": "#942C76", "short": "MLDSPP",            "method": "BDT"},
    "MLDSPP XGBoost (75% spn)":  {"color": "#B07AA1", "short": "MLDSPP 75%",        "method": "BDT"},
    "PromoterLCNN":              {"color": "#228B22", "short": "Promoter LCNN",     "method": "CNN"},
    "PromoTech RF-HOT (PG Max)": {"color": "#E07614", "short": "PromoTech RF-HOT",  "method": "RF"},
    "iPro-MP (H. pylori)":       {"color": "#3D185A", "short": "iPro-MP",           "method": "gLM"},
    "MEME Suite (STREME+FIMO)":  {"color": "#1E88E5", "short": "MEME (STREME+FIMO)", "method": "motif"},
    "FIMO + Prokaryote DB":      {"color": "#8E24AA", "short": "FIMO Prok DB",      "method": "motif"},
}
GPU_TOOLS = {"PromoterLCNN", "iPro-MP (H. pylori)"}
METHOD_COLORS = {
    "BDT": "#942C76", "CNN": "#228B22", "gLM": "#3D185A",
    "RF": "#E07614", "motif": "#1E88E5",
}


def load(iter_name: str, data_dir: Path = None) -> pd.DataFrame:
    """One row per tool at the same n (iteration), WS campaigns 1 + 2."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    c1 = data_dir / "scale_db" / iter_name / "predictions" / "resource_metrics.tsv"
    c2 = data_dir / "scale_db_4tools" / iter_name / "predictions" / "resource_metrics.tsv"
    rows = []
    if c1.exists():
        rows.append(pd.read_csv(c1, sep="\t"))
    if c2.exists():
        rows.append(pd.read_csv(c2, sep="\t"))
    if not rows:
        raise SystemExit(f"no metrics found for iteration {iter_name}")
    df = pd.concat(rows, ignore_index=True)
    df = df[df["success"].fillna(False).astype(bool) & df["tool"].isin(TOOL_INFO)]
    df = df.drop_duplicates(subset=["tool"], keep="last")
    df = df.sort_values("tool", key=lambda s: s.map(
        {"MLDSPP XGBoost": 0, "MLDSPP XGBoost (75% spn)": 1, "PromoterLCNN": 2,
         "PromoTech RF-HOT (PG Max)": 3, "iPro-MP (H. pylori)": 4,
         "MEME Suite (STREME+FIMO)": 5, "FIMO + Prokaryote DB": 6}))
    df["method"] = df["tool"].map(lambda t: TOOL_INFO[t]["method"])
    df["short"] = df["tool"].map(lambda t: TOOL_INFO[t]["short"])
    df["color"] = df["tool"].map(lambda t: TOOL_INFO[t]["color"])
    return df


def add_sublabels(ax, x, tools, y_top):
    for i, tool in enumerate(tools):
        ax.text(i, -y_top * 0.11, TOOL_INFO[tool]["method"], ha="center", va="top",
                fontsize=8, color=METHOD_COLORS[TOOL_INFO[tool]["method"]],
                fontstyle="italic", fontweight="bold")


def value_labels(ax, bars, fmt=".1f", unit=""):
    max_v = max(b.get_height() for b in bars)
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.015,
                f"{v:{fmt}}{unit}", ha="center", va="bottom", fontsize=8, fontweight="bold")


def style(ax, ylabel, ymax):
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(-ymax * 0.14, ymax)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


def one_chart(df, col, ylabel, out_name, fmt=".1f", unit="", title=None, ymax_mult=1.22):
    """Single-axes vertical bar chart with method sublabels and value labels."""
    tools = df["tool"].tolist()
    x = np.arange(len(tools))
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)
    bars = ax.bar(x, df[col], color=df["color"], width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.5)
    value_labels(ax, bars, fmt=fmt, unit=unit)
    add_sublabels(ax, x, tools, df[col].max())
    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], fontsize=8)
    style(ax, ylabel, df[col].max() * ymax_mult)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    for ext in ["svg", "png"]:
        fig.savefig(OUT_DIR / f"{out_name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_name}.{{svg,png}}")


def load_all_by_iter(data_dir: Path = None) -> pd.DataFrame:
    """All iterations x tools from WS campaigns 1 + 2 (one row per tool/iter)."""
    data_dir = data_dir or DEFAULT_DATA_DIR
    rows = []
    for root in [data_dir / "scale_db",
                 data_dir / "scale_db_4tools"]:
        if not root.is_dir():
            continue
        for it in sorted(p.name for p in root.iterdir()
                         if p.is_dir() and (p / "predictions" / "resource_metrics.tsv").exists()):
            df = pd.read_csv(root / it / "predictions" / "resource_metrics.tsv", sep="\t")
            df["iteration"] = it
            rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = df[df["success"].fillna(False).astype(bool) & df["tool"].isin(TOOL_INFO)]
    df = df.drop_duplicates(subset=["tool", "iteration"], keep="last")
    df["color"] = df["tool"].map(lambda t: TOOL_INFO[t]["color"])
    df["method"] = df["tool"].map(lambda t: TOOL_INFO[t]["method"])
    return df


def by_iter_chart(df, col, ylabel, out_name, fmt=".1f", unit=""):
    """Single chart: x = iterations, grouped vertical bars per tool."""
    iters = sorted(df["iteration"].unique(), key=lambda s: int(s))
    tools = [t for t in df["tool"].unique() if t in TOOL_INFO]
    n_iter, n_tools = len(iters), len(tools)
    fig, ax = plt.subplots(figsize=(max(6.5, n_iter * 0.95), 5.2), dpi=300)
    w = 0.82 / max(n_tools, 1)
    max_v = float(df[col].max())
    for j, tool in enumerate(tools):
        sub = df[df["tool"] == tool].set_index("iteration")
        xs, vals = [], []
        for i, it in enumerate(iters):
            if it in sub.index:
                xs.append(i + (j - (n_tools - 1) / 2) * w)
                vals.append(float(sub.loc[it, col]))
        bars = ax.bar(xs, vals, width=w * 0.92, color=TOOL_INFO[tool]["color"],
                      zorder=3, edgecolor="white", linewidth=0.4, label=TOOL_INFO[tool]["short"])
        if vals:
            last = bars[-1]
            ax.text(last.get_x() + last.get_width() / 2, last.get_height() + max_v * 0.008,
                    f"{vals[-1]:{fmt}}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_xticks(range(n_iter))
    ax.set_xticklabels(iters, fontsize=8, rotation=45, ha="right")
    ax.set_xlabel("iteration (positives per class)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, max_v * 1.18)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, ncol=2 if n_tools > 4 else 1)
    fig.tight_layout()
    for ext in ["svg", "png"]:
        fig.savefig(OUT_DIR / f"{out_name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_name}.{{svg,png}}")


def main():
    p = argparse.ArgumentParser(description="Resource plots (one chart per metric)")
    p.add_argument("--iter", default="9880", help="Iteration (n/2 positives) to plot")
    p.add_argument("--by-iter", action="store_true",
                   help="Plot per-iteration grouped bars instead of single-size charts")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Base dir with scale_db campaign folders")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output dir for figures")
    args = p.parse_args()
    global OUT_DIR
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    if args.by_iter:
        df = load_all_by_iter(data_dir)
        print("Resource plots by iteration:")
        by_iter_chart(df, "time_s", "Compute time (s)", "resources_byiter_time", fmt=".1f", unit=" s")
        by_iter_chart(df, "peak_ram_mb", "Peak RAM (MB)", "resources_byiter_ram", fmt=".0f", unit=" MB")
        vram = df[df["peak_vram_mb"] > 0]
        if not vram.empty:
            by_iter_chart(vram, "peak_vram_mb", "Peak VRAM (MB)", "resources_byiter_vram", fmt=".0f", unit=" MB")
        return
    df = load(args.iter, data_dir)
    n = int(df["n_sequences"].iloc[0])
    header = f"n = {n:,} seqs (pos+neg), 1 thread, linear"

    print("Resource plots:")
    one_chart(df, "time_s", "Compute time (s)", "resources_time",
              fmt=".1f", unit=" s", title=f"Compute time — {header}")
    one_chart(df, "peak_ram_mb", "Peak RAM (MB)", "resources_ram",
              fmt=".0f", unit=" MB", title=f"Peak RAM — {header}")

    gpu = df[df["tool"].isin(GPU_TOOLS)]
    if not gpu.empty:
        gx = np.arange(len(gpu))
        fig, ax = plt.subplots(figsize=(6.5, 4.6), dpi=300)
        w = 0.35
        b1 = ax.bar(gx - w / 2, gpu["peak_vram_mb"], width=w, color=gpu["color"],
                    zorder=3, edgecolor="white", linewidth=0.5, label="Peak VRAM")
        b2 = ax.bar(gx + w / 2, gpu["model_size_mb"], width=w,
                    color=gpu["color"], alpha=0.45, hatch="//", zorder=3,
                    edgecolor="white", linewidth=0.5, label="Model size")
        for bars, fmt in ((b1, ".0f"), (b2, ".0f")):
            max_v = max(b.get_height() for b in bars)
            for b in bars:
                v = b.get_height()
                ax.text(b.get_x() + b.get_width() / 2, v + max_v * 0.03,
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        gpu_tools = gpu["tool"].tolist()
        for i, tool in enumerate(gpu_tools):
            ax.text(i - w / 2, -max(gpu["peak_vram_mb"].max(), gpu["model_size_mb"].max()) * 0.11,
                    TOOL_INFO[tool]["method"], ha="center", va="top", fontsize=8,
                    color=METHOD_COLORS[TOOL_INFO[tool]["method"]],
                    fontstyle="italic", fontweight="bold")
        ax.set_xticks(gx)
        ax.set_xticklabels(gpu["short"], fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
        style(ax, "MB", max(gpu["peak_vram_mb"].max(), gpu["model_size_mb"].max()) * 1.22)
        ax.set_title(f"GPU memory (RTX 3090) — {header}", fontsize=10)
        fig.tight_layout()
        for ext in ["svg", "png"]:
            fig.savefig(OUT_DIR / f"resources_gpu.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  resources_gpu.{svg,png}")


if __name__ == "__main__":
    main()