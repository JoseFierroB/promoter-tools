#!/usr/bin/env python3
"""Scaling analysis — per-tool time(n)/RAM(n)/VRAM(n) fits, bar charts, and extrapolation.

Consolidates metrics from scale_db or scale_db_16cpu, fits linear/power models,
and generates publication-ready figures (PNG 300 DPI + vector SVG):
  1. scaling_time.png / .svg         (Log-log wall-clock time)
  2. scaling_time_linear.png / .svg  (Linear wall-clock time with data annotations)
  3. scaling_ram.png / .svg          (Log-log system RAM)
  4. scaling_ram_linear.png / .svg   (Linear system RAM in GB)
  5. scaling_vram.png / .svg         (GPU dedicated VRAM O(1) buffer)
  6. scaling_time_bars.png / .svg    (Grouped vertical bars of time)
  7. scaling_ram_bars.png / .svg     (Grouped vertical bars of RAM)
  8. scaling_gpu_model_weights_vs_vram.png / .svg (Model weights size vs VRAM)
  9. scaling_3panel_master.png / .svg (Master 3-panel figure)
 10. scaling_3panel_linear_master.png / .svg (Master 3-panel linear figure)

Usage:
    pixi run python src/analysis/scaling_analysis.py [--scale-db <dir>]
"""

import os
import argparse
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_DATA_DIR = Path(os.environ.get("PROMOTER_DATA_DIR", "/home/fierro/Desktop"))
_PARSER = argparse.ArgumentParser(description="Scaling analysis")
_PARSER.add_argument("--scale-db", default=str(_DATA_DIR / "scale_db_16cpu"),
                     help="Root with <iter>/predictions/resource_metrics*.tsv folders")
_PARSER.add_argument("--metrics-name", default="resource_metrics.tsv",
                     help="Metrics filename to read per iteration")
_PARSER.add_argument("--local-suffix", default="mldspp75_local",
                     help="Suffix of local-only metrics files (e.g. _mldspp75_local)")

SMOKE_TSV = ROOT / "output" / "tables" / "resource_metrics.tsv"
OUT_TSV = ROOT / "output" / "tables" / "scaling_dataset.tsv"
EXTRAP_TSV = ROOT / "output" / "tables" / "extrapolation.tsv"
BENCH_PLOT_DIR = ROOT / "output" / "plots" / "benchmark"
SCALING_PLOT_DIR = ROOT / "output" / "plots" / "scaling"

GPU_TOOLS = {"PromoterLCNN", "iPro-MP (H. pylori)", "iPro-MP"}
TOOL_ORDER = [
    "MLDSPP XGBoost (75% spn)",
    "MLDSPP XGBoost",
    "PromoterLCNN",
    "PromoTech RF-HOT (PG Max)",
    "FIMO + Prokaryote DB",
    "MEME Suite (STREME+FIMO)",
    "iPro-MP (H. pylori)"
]

TOOL_INFO = {
    "MLDSPP XGBoost":            {"short": "MLDSPP",              "method": "BDT",    "color": "#D81B60", "marker": "v"},
    "MLDSPP XGBoost (75% spn)":  {"short": "MLDSPP 75%",          "method": "BDT",    "color": "#942C76", "marker": "^"},
    "PromoterLCNN":              {"short": "PromoterLCNN",        "method": "CNN",    "color": "#228B22", "marker": "s"},
    "PromoTech RF-HOT (PG Max)": {"short": "PromoTech RF",        "method": "RF",     "color": "#E07614", "marker": "P"},
    "iPro-MP (H. pylori)":       {"short": "iPro-MP",             "method": "gLM",    "color": "#3D185A", "marker": "o"},
    "iPro-MP":                   {"short": "iPro-MP",             "method": "gLM",    "color": "#3D185A", "marker": "o"},
    "MEME Suite (STREME+FIMO)":  {"short": "MEME Suite",          "method": "motif",  "color": "#1E88E5", "marker": "X"},
    "FIMO + Prokaryote DB":      {"short": "FIMO Prok DB",        "method": "motif",  "color": "#00ACC1", "marker": "D"},
}

TARGETS = [10_000, 25_000, 50_000, 100_000, 150_000, 200_000, 395_200]
MAX_TARGET = TARGETS[-1]

MODELS = {
    "linear":    lambda n, a, b: a * n + b,
    "quadratic": lambda n, a, b, c: a * n ** 2 + b * n + c,
    "power":     lambda n, a, b: a * n ** b,
}
P0 = {
    "linear":    (1e-3, 1.0),
    "quadratic": (1e-9, 1e-3, 1.0),
    "power":     (1e-3, 1.0),
}


def load_all() -> pd.DataFrame:
    frames = []
    # 1. Iterate over numeric folders in SCALE_DB
    for iter_dir in sorted(SCALE_DB.glob('*'), key=lambda x: int(x.name) if x.name.isdigit() else 9999999):
        if iter_dir.is_dir() and iter_dir.name.isdigit():
            it = iter_dir.name
            pred_dir = iter_dir / "predictions"
            if pred_dir.exists():
                for t in pred_dir.glob("resource_metrics*.tsv"):
                    try:
                        df = pd.read_csv(t, sep="\t")
                        df["machine"] = "16cpu" if "16cpu" in str(SCALE_DB) else "ws"
                        df["iteration"] = it
                        df["scale_N"] = int(it) * 2
                        frames.append(df)
                    except Exception as e:
                        print(f"Error reading {t}: {e}")

    if SMOKE_TSV.exists():
        try:
            df = pd.read_csv(SMOKE_TSV, sep="\t")
            df["machine"] = "local"
            df["iteration"] = "smoke"
            df["scale_N"] = df["n_sequences"] if "n_sequences" in df.columns else 1988
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df[df["success"].fillna(True).astype(bool)]
    
    # Filter out local smoke runs to prevent mixing single-thread/partial baseline benchmarks with scaling series
    df = df[df["iteration"] != "smoke"]
    
    df["resource_class"] = np.where(df["tool"].isin(GPU_TOOLS), "gpu", "cpu")
    if "n_sequences" not in df.columns or df["n_sequences"].isna().any():
        df["n_sequences"] = df["scale_N"]

    cols = ["tool", "n_sequences", "scale_N", "resource_class", "machine", "iteration",
            "time_s", "peak_ram_mb", "peak_vram_mb", "mean_cpu_pct", "gpu_util_pct", "gpu_name", "success"]
    cols_exist = [c for c in cols if c in df.columns]
    return df[cols_exist].drop_duplicates(subset=["tool", "scale_N"]).sort_values(["tool", "scale_N"])


def best_fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return "linear", MODELS["linear"], P0["linear"], None, None
    best, best_r2 = None, -np.inf
    quad_r2 = None
    for name in ["linear", "power"]:
        fn, p0 = MODELS[name], P0[name]
        try:
            popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
            pred = fn(x, *popt)
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            adj = 1 - (1 - r2) * (len(x) - 1) / (len(x) - len(popt))
            if adj > best_r2:
                best, best_r2 = name, adj
        except Exception:
            continue
    try:
        popt, _ = curve_fit(MODELS["quadratic"], x, y, p0=P0["quadratic"], maxfev=20000)
        pred = MODELS["quadratic"](x, *popt)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        quad_r2 = 1 - (1 - r2) * (len(x) - 1) / (len(x) - 3)
    except Exception:
        quad_r2 = None
    if best is None:
        return "linear", MODELS["linear"], P0["linear"], None, quad_r2
    return best, MODELS[best], P0[best], best_r2, quad_r2


def save_plot_everywhere(fig, base_name):
    for p in [PLOT_DIR, BENCH_PLOT_DIR, SCALING_PLOT_DIR]:
        fig.savefig(p / f"{base_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(p / f"{base_name}.svg", dpi=300, bbox_inches="tight")
    print(f"  [Saved] {base_name}.png and .svg")


def generate_all_figures(df):
    tools = [t for t in TOOL_ORDER if t in df['tool'].unique()]
    scales = sorted(df['scale_N'].unique())

    # 1. TIME LOG-LOG
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for tool in tools:
        sub = df[df['tool'] == tool].sort_values('scale_N')
        meta = TOOL_INFO.get(tool, {"short": tool, "color": "#333", "marker": "o", "method": "N/A"})
        ax.plot(sub['scale_N'], sub['time_s'], marker=meta['marker'], color=meta['color'],
                linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total Sequences (N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inference Time (s)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Time Scaling (Log-Log) — 16 CPU Cores / GPU", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_time")
    plt.close(fig)

    # 2. TIME LINEAR WITH ANNOTATIONS
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for tool in tools:
        sub = df[df['tool'] == tool].sort_values('scale_N')
        meta = TOOL_INFO.get(tool, {"short": tool, "color": "#333", "marker": "o", "method": "N/A"})
        ax.plot(sub['scale_N']/1000, sub['time_s'], marker=meta['marker'], color=meta['color'],
                linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
        last_x = sub['scale_N'].iloc[-1] / 1000
        last_y = sub['time_s'].iloc[-1]
        txt = f"{last_y:.1f} s ({last_y/60:.1f}m)" if last_y >= 60 else f"{last_y:.2f} s"
        ax.annotate(txt, (last_x, last_y), xytext=(8, -3), textcoords="offset points",
                    fontweight="bold", color=meta['color'], fontsize=9)
    ax.set_xlabel("Sequences (thousands, k)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inference Time (s)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Time Scaling (Linear Scale) — 16 CPU Cores / GPU",
                 fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_time_linear")
    plt.close(fig)

    # 3. RAM LINEAR (GB)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for tool in tools:
        sub = df[df['tool'] == tool].sort_values('scale_N')
        meta = TOOL_INFO.get(tool, {"short": tool, "color": "#333", "marker": "o", "method": "N/A"})
        ax.plot(sub['scale_N']/1000, sub['peak_ram_mb']/1024, marker=meta['marker'], color=meta['color'],
                linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
        last_x = sub['scale_N'].iloc[-1] / 1000
        last_y = sub['peak_ram_mb'].iloc[-1] / 1024
        ax.annotate(f"{last_y:.2f} GB", (last_x, last_y), xytext=(8, -3), textcoords="offset points",
                    fontweight="bold", color=meta['color'], fontsize=9)
    ax.set_xlabel("Sequences (thousands, k)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Peak System RAM (GB)", fontsize=11, fontweight="bold")
    ax.set_title("System Peak RAM Scaling (Linear Scale) — 16 CPU Cores",
                 fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_ram_linear")
    plt.close(fig)

    # 4. VRAM DEDICATED (measured per tool from peak_vram_mb)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    def _vram_series(df, tool_name):
        sub = df[(df["tool"] == tool_name) & df["peak_vram_mb"].notna() & (df["peak_vram_mb"] > 0)]
        return sub.sort_values("scale_N")[["scale_N", "peak_vram_mb"]]

    for tool_name, marker in [
        ("iPro-MP (H. pylori)", "o"),
        ("PromoterLCNN", "s"),
    ]:
        vr = _vram_series(df, tool_name)
        if vr.empty:
            continue
        meta = TOOL_INFO.get(tool_name, {"color": "#666666"})
        med = float(vr["peak_vram_mb"].median())
        ax.plot(vr["scale_N"], vr["peak_vram_mb"], marker=marker, color=meta["color"],
                linewidth=1.2, markersize=8, linestyle="None", alpha=0.85,
                label=f"{meta['short']} (medido)")
        ax.axhline(med, color=meta["color"], linestyle=":", alpha=0.55,
                   label=f"{meta['short']} mediana ≈ {med:,.0f} MB")

    ax.set_xscale("log")
    all_vram = [vr["peak_vram_mb"] for vr in
                [_vram_series(df, t) for t in ("iPro-MP (H. pylori)", "PromoterLCNN")]
                if not vr.empty]
    ymax = max(pd.concat(all_vram)) * 1.15 if all_vram else 3500
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Total Sequences (N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Peak GPU VRAM (MB)", fontsize=11, fontweight="bold")
    ax.set_title("GPU VRAM vs Dataset Size — constant w.r.t. N (fixed batching)", fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9.5, loc="center right")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_vram")
    plt.close(fig)

    # 5. VERTICAL GROUPED BAR CHART: TIME
    key_scales = [1976, 19760, 98800, 395200]
    labels_scales = ["1.9k (1x)", "19.7k (10x)", "98.8k (50x)", "395.2k (200x)"]
    indices = np.arange(len(key_scales))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    for i, tool in enumerate(tools):
        meta = TOOL_INFO.get(tool, {"short": tool, "color": "#333", "method": "N/A"})
        times = [df[(df['tool'] == tool) & (df['scale_N'] == s)]['time_s'].values[0]
                 if len(df[(df['tool'] == tool) & (df['scale_N'] == s)]) > 0 else 0 for s in key_scales]
        pos = indices + (i - (len(tools)-1)/2) * bar_width
        bars = ax.bar(pos, times, bar_width, label=f"{meta['short']} ({meta['method']})", color=meta['color'], edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt = f"{h:.0f}s\n({h/60:.1f}m)" if h >= 60 else (f"{h:.1f}s" if h >= 1 else f"{h:.2f}s")
                ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xlabel("Sequence Scale (Total N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inference Time (s)", fontsize=11, fontweight="bold")
    ax.set_title("Inference Time Comparison by Sequence Scale",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels_scales, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 1400)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_time_bars")
    plt.close(fig)

    # 5B. VERTICAL GROUPED BAR CHART: RAM
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    for i, tool in enumerate(tools):
        meta = TOOL_INFO.get(tool, {"short": tool, "color": "#333", "method": "N/A"})
        rams = [df[(df['tool'] == tool) & (df['scale_N'] == s)]['peak_ram_mb'].values[0]
                if len(df[(df['tool'] == tool) & (df['scale_N'] == s)]) > 0 else 0 for s in key_scales]
        pos = indices + (i - (len(tools)-1)/2) * bar_width
        bars = ax.bar(pos, rams, bar_width, label=f"{meta['short']} ({meta['method']})", color=meta['color'], edgecolor="black")
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt = f"{h/1024:.2f}G" if h >= 1000 else f"{h:.0f}M"
                ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xlabel("Sequence Scale (Total N)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Peak System RAM (MB)", fontsize=11, fontweight="bold")
    ax.set_title("Peak System RAM Comparison by Sequence Scale",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels_scales, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 3900)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_ram_bars")
    plt.close(fig)

    # 6. GPU WEIGHTS VS VRAM BUFFER
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    gpu_models = ["PromoterLCNN\n(CNN)", "iPro-MP (DNABERT-6)\n(gLM)"]
    weights_size_mb = [1.8, 450.0]
    inference_vram_mb = [497.2, 2390.9]
    x = np.arange(len(gpu_models))
    w = 0.35
    b1 = ax.bar(x - w/2, weights_size_mb, w, label="Model Weights (Disk/RAM)", color="#455A64", edgecolor="black")
    b2 = ax.bar(x + w/2, inference_vram_mb, w, label="Peak Inference VRAM", color="#7B1FA2", edgecolor="black")
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB\n({h/1024:.2f} GB)", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Memory (MB)", fontsize=11, fontweight="bold")
    ax.set_title("GPU Memory Requirements: Model Weights vs Peak VRAM",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(gpu_models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 3200)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_gpu_model_weights_vs_vram")
    plt.close(fig)

    # 7. MASTER 3-PANEL UNIFIED FIGURE (LINEAR)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6), dpi=300)
    for tool in tools:
        sub = df[df['tool'] == tool].sort_values('scale_N')
        meta = TOOL_INFO.get(tool, {"short": tool, "color": "#333", "marker": "o", "method": "N/A"})
        ax1.plot(sub['scale_N']/1000, sub['time_s'], marker=meta['marker'], color=meta['color'],
                 linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
        ax2.plot(sub['scale_N']/1000, sub['peak_ram_mb']/1024, marker=meta['marker'], color=meta['color'],
                 linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
    ax1.set_xlabel("Sequences (thousands, k)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Inference Time (s)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Inference Time (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=8, loc="upper left")

    ax2.set_xlabel("Sequences (thousands, k)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Peak System RAM (GB)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Peak System RAM (GB)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    b1 = ax3.bar(x - w/2, weights_size_mb, w, label="Weights (Disk)", color="#455A64", edgecolor="black")
    b2 = ax3.bar(x + w/2, [h/1024 for h in inference_vram_mb], w, label="Peak VRAM (GB)", color="#7B1FA2", edgecolor="black")
    for bar in b1:
        h = bar.get_height()
        ax3.annotate(f"{h:.1f} MB", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f} GB", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax3.set_ylabel("Memory Required", fontsize=11, fontweight="bold")
    ax3.set_title("C. GPU Model Memory Requirements", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(gpu_models, fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 3.2)
    ax3.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax3.legend(frameon=True, fontsize=8, loc="upper left")

    plt.suptitle("Multimodal Scalability Benchmark — 16 CPU Cores / GPU",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    save_plot_everywhere(fig, "scaling_3panel_linear_master")
    plt.close(fig)


def bootstrap_pred_ci(fn, x, y, p0, targets, n_boot=500, seed=42):
    x, y = np.asarray(x, float), np.asarray(y, float)
    targets = np.asarray(targets, float)
    try:
        popt, _ = curve_fit(fn, x, y, p0=p0, maxfev=20000)
    except Exception:
        return np.full(len(targets), np.nan), np.full(len(targets), np.nan)
    resid = y - fn(x, *popt)
    rng = np.random.RandomState(seed)
    preds = np.zeros((n_boot, len(targets)))
    for b in range(n_boot):
        yb = fn(x, *popt) + rng.choice(resid, size=len(x), replace=True)
        try:
            pb, _ = curve_fit(fn, x, yb, p0=popt, maxfev=20000)
            preds[b] = fn(targets, *pb)
        except Exception:
            preds[b] = np.nan
    lo = np.nanpercentile(preds, 2.5, axis=0)
    hi = np.nanpercentile(preds, 97.5, axis=0)
    return lo, hi


def main():
    global SCALE_DB, METRICS_NAME, LOCAL_SUFFIX, PLOT_DIR
    _args = _PARSER.parse_args()
    SCALE_DB = Path(_args.scale_db)
    if not SCALE_DB.exists():
        SCALE_DB = _DATA_DIR / "scale_db"
    METRICS_NAME = _args.metrics_name
    LOCAL_SUFFIX = _args.local_suffix
    PLOT_DIR = SCALE_DB / "plots"
    for p in [PLOT_DIR, BENCH_PLOT_DIR, SCALING_PLOT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RUNNING ENHANCED CANONICAL SCALING ANALYSIS")
    print("=" * 70)
    df = load_all()
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Consolidated dataset: {len(df)} rows -> {OUT_TSV}")

    fits = {}
    for (tool, cls, machine), g in df.groupby(["tool", "resource_class", "machine"]):
        if len(g) < 2:
            continue
        n = g["scale_N"].values if "scale_N" in g.columns else g["n_sequences"].values
        t = g["time_s"].values
        r = g["peak_ram_mb"].values
        name_t, fn_t, p0_t, r2_t, quad_r2_t = best_fit(n, t)
        popt_t, _ = curve_fit(fn_t, n, t, p0=p0_t, maxfev=20000)
        lo_t, hi_t = bootstrap_pred_ci(fn_t, n, t, p0_t, TARGETS)
        popt_r, _ = curve_fit(lambda x, a, b: a * x + b, n, r, p0=(1e-3, 100.0))
        fits[(tool, cls, machine)] = {
            "model": name_t, "popt_t": popt_t, "fn_t": fn_t, "r2_t": r2_t,
            "quad_r2_t": quad_r2_t, "popt_r": popt_r, "ci_lo": lo_t, "ci_hi": hi_t
        }

    rows = []
    for (tool, cls, machine), f in sorted(fits.items()):
        for i, target in enumerate(TARGETS):
            t_pred = float(f["fn_t"](target, *f["popt_t"]))
            r_pred = float(f["popt_r"][0] * target + f["popt_r"][1])
            rows.append({"tool": tool, "resource_class": cls, "machine": machine,
                         "model": f["model"], "r2_adjusted": f["r2_t"],
                         "n_target": target, "time_s_pred": t_pred, "ram_mb_pred": r_pred,
                         "time_s_ci_low": float(f["ci_lo"][i]), "time_s_ci_high": float(f["ci_hi"][i])})
    extrap = pd.DataFrame(rows)
    extrap.to_csv(EXTRAP_TSV, sep="\t", index=False)
    print(f"Extrapolations saved -> {EXTRAP_TSV}")

    print("\nGenerating scientific figures...")
    generate_all_figures(df)
    print("\n" + "=" * 70)
    print("  CANONICAL FIGURE GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()