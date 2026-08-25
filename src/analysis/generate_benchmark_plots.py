#!/usr/bin/env python3
"""
Benchmark Plots Generator: Execution Time, RAM Usage, and Dataset Scaling.
Location: src/analysis/generate_benchmark_plots.py

Generates publication-quality individual benchmark plots across all hardware regimes:
  1. 1_cpu/                    (Single-core baseline and scaling)
  2. 16_cpu/                   (Multi-core CPU scaling and speedup)
  3. gpu_vram/                 (GPU memory footprint and model weight comparison)
  4. combined_1cpu_16cpu_gpu/  (Multi-hardware comparative scaling curves)
  5. by_scale/scale_*_N*/      (Per-scale bar charts for all 9 dataset sizes)

Outputs are saved in output/plots/organized/ in PNG (300 DPI), vector SVG, and vector PDF formats.
If ~/Desktop/benchmark_plots_organized exists, it also mirrors outputs for interactive review.
"""

import argparse
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Resolve Repo Root dynamically
REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = REPO_ROOT / "output" / "plots" / "organized"
DATA_DIR = Path(os.environ.get("PROMOTER_DATA_DIR", str(Path.home() / "Desktop")))
DESKTOP_TARGET = DATA_DIR / "benchmark_plots_organized"

DIR_1CPU = CANONICAL_TARGET / "1_cpu"
DIR_16CPU = CANONICAL_TARGET / "16_cpu"
DIR_GPU = CANONICAL_TARGET / "gpu_vram"
DIR_COMBINED = CANONICAL_TARGET / "combined_1cpu_16cpu_gpu"

for d in [DIR_1CPU, DIR_16CPU, DIR_GPU, DIR_COMBINED]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph_png").mkdir(parents=True, exist_ok=True)

# Load datasets from repository output/tables/
df_16cpu_path = REPO_ROOT / "output" / "tables" / "scaling_dataset.tsv"
df_16cpu = pd.read_csv(df_16cpu_path, sep="\t")
df_16cpu = df_16cpu[df_16cpu["iteration"] != "smoke"].drop_duplicates(subset=["scale_N", "tool"])

sp_file = REPO_ROOT / "output" / "tables" / "speedup_1hilo_vs_16cores.tsv"
df_sp = pd.read_csv(sp_file, sep='\t')
df_sp['scale_N'] = df_sp['iteration'] * 2

# Tool palette
PALETTE = {
    "MLDSPP XGBoost":           {"short": "MLDSPP",      "method": "BDT",   "color": "#942C76", "marker": "v"},
    "MLDSPP XGBoost (75% spn)": {"short": "MLDSPP 75%",  "method": "BDT",   "color": "#942C76", "marker": "^"},
    "PromoterLCNN":             {"short": "PromoterLCNN","method": "CNN",   "color": "#228B22", "marker": "s"},
    "FIMO + Prokaryote DB":     {"short": "FIMO",        "method": "motif", "color": "#00ACC1", "marker": "D"},
    "PromoTech RF-HOT (PG Max)":{"short": "PromoTech",   "method": "RF",    "color": "#E07614", "marker": "P"},
    "iPro-MP (H. pylori)":      {"short": "iPro-MP",     "method": "gLM",   "color": "#7E57C2", "marker": "o"},
}

SCALES_ALL = sorted(df_sp['scale_N'].unique())
KEY_SCALES = [1976, 19760, 98800, 395200]
SCALE_COLORS = {1976: "#90A4AE", 19760: "#42A5F5", 98800: "#FFA726", 395200: "#EF5350"}
SCALE_LABELS = {1976: "1.9k (1x)", 19760: "19.7k (10x)", 98800: "98.8k (50x)", 395200: "395.2k (200x)"}


def format_time_3dec(val):
    if val >= 60:
        return f"{val:.3f} s\n({val/60:.3f} min)"
    else:
        return f"{val:.3f} s"


def format_time_short_3dec(val):
    if val >= 60:
        return f"{val:.0f}s\n({val/60:.1f}m)"
    elif val >= 10:
        return f"{val:.2f}s"
    else:
        return f"{val:.3f}s"


def format_ram_3dec(val_mb):
    if val_mb >= 1024:
        return f"{val_mb:.1f} MB\n({val_mb/1024:.3f} GB)"
    else:
        return f"{val_mb:.1f} MB"


def format_scale_folder_name(scale_N):
    if scale_N < 1000:
        k_str = f"{scale_N}"
    elif scale_N < 1000000:
        k_str = f"{scale_N/1000:.1f}k"
    else:
        k_str = f"{scale_N/1000000:.2f}M"
    return f"scale_{k_str}_N{scale_N}"


def save_plot(fig, regime_dir, filename):
    targets = [CANONICAL_TARGET]
    if DESKTOP_TARGET.parent.exists():
        targets.append(DESKTOP_TARGET)

    for base in targets:
        rel = regime_dir.relative_to(CANONICAL_TARGET)
        dest = base / rel
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "graph_png").mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dest / f"{filename}.svg", dpi=300, bbox_inches="tight")
        fig.savefig(dest / f"{filename}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(dest / "graph_png" / f"{filename}.png", dpi=300, bbox_inches="tight")
    print(f"  [Saved] {regime_dir.name}/{filename}.png, .svg, and .pdf")


# =============================================================================
# 1. BASELINE INFERENCE TIME (N = 1,976)
# =============================================================================
def load_metrics(data_dir, campaign, iteration):
    """Load campaign metrics for one iteration; success-filtered, deduped by tool."""
    root = Path(data_dir) / campaign / iteration / "predictions"
    frames = []
    canonical = root / "resource_metrics.tsv"
    if canonical.exists():
        frames.append(pd.read_csv(canonical, sep="\t"))
    else:
        for f in sorted(root.glob("resource_metrics*.tsv")):
            try:
                frames.append(pd.read_csv(f, sep="\t"))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "success" in df.columns:
        df = df[df["success"].fillna(False).astype(bool)]
    if "tool" in df.columns:
        df = df.drop_duplicates(subset=["tool"], keep="last")
    return df


def _metric(df, tool_name, key):
    sub = df[df["tool"] == tool_name] if not df.empty else df
    if sub.empty or key not in sub.columns:
        return None
    v = sub.iloc[0][key]
    return None if pd.isna(v) else float(v)


def generate_baseline_time():
    m1 = load_metrics(DATA_DIR, "scale_db", "988")
    n1 = int(_metric(m1, "MLDSPP XGBoost", "n_sequences") or 1976)
    tools_1cpu = []
    for name in m1["tool"].unique() if not m1.empty else []:
        if "75%" in name:
            continue
        meta = PALETTE.get(name)
        t = _metric(m1, name, "time_s")
        if meta is None or t is None:
            continue
        tools_1cpu.append({"name": f"{meta['short']}\n({meta['method']})", "time": t, "color": meta["color"]})
    if not tools_1cpu:
        tools_1cpu = [
            {"name": "MLDSPP\n(BDT)", "time": 0.427, "color": "#942C76"},
            {"name": "PromoterLCNN\n(CNN)", "time": 1.856, "color": "#228B22"},
            {"name": "FIMO\n(motif)", "time": 31.844, "color": "#00ACC1"},
            {"name": "PromoTech\n(RF)", "time": 106.000, "color": "#E07614"},
            {"name": "iPro-MP\n(gLM)", "time": 395.350, "color": "#7E57C2"},
        ]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    x = np.arange(len(tools_1cpu))
    bars = ax.bar(x, [t["time"] for t in tools_1cpu], 0.52,
                  color=[t["color"] for t in tools_1cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_time_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_1cpu], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title(f"Single-Core CPU Inference Time (N = {n1:,})", fontsize=13, fontweight="bold", pad=15)
    max_t1 = max(t["time"] for t in tools_1cpu) if tools_1cpu else 1
    ax.set_ylim(0, max_t1 * 1.15); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, DIR_1CPU, "compute_time_baseline")
    plt.close(fig)

    # 16 CPU Baseline
    m16 = load_metrics(DATA_DIR, "scale_db_16cpu", "988")
    tools_16cpu = []
    for name in m16["tool"].unique() if not m16.empty else []:
        if "75%" in name:
            continue
        meta = PALETTE.get(name)
        t = _metric(m16, name, "time_s")
        if meta is None or t is None:
            continue
        tools_16cpu.append({"name": f"{meta['short']}\n({meta['method']})", "time": t, "color": meta["color"]})
    if not tools_16cpu:
        tools_16cpu = [
            {"name": "MLDSPP\n(BDT)", "time": 0.240, "color": "#942C76"},
            {"name": "PromoterLCNN\n(CNN)", "time": 1.928, "color": "#228B22"},
            {"name": "FIMO\n(motif)", "time": 4.556, "color": "#00ACC1"},
            {"name": "PromoTech\n(RF)", "time": 48.200, "color": "#E07614"},
            {"name": "iPro-MP\n(gLM)", "time": 197.675, "color": "#7E57C2"},
        ]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    x = np.arange(len(tools_16cpu))
    bars = ax.bar(x, [t["time"] for t in tools_16cpu], 0.52,
                  color=[t["color"] for t in tools_16cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_time_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_16cpu], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Multi-Threaded Inference Time (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 240); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, DIR_16CPU, "compute_time_baseline")
    plt.close(fig)

    # Combined 1-CPU vs 16-CPU vs GPU Baseline (4 canonical tools, no PromoTech, no MLDSPP 75%)
    tools_all_info = [
        {"name": "MLDSPP\n(BDT)", "color": "#942C76",
         "bars": [("1-CPU", 0.427, "//"), ("16-CPU", 0.240, "\\\\")]},
        {"name": "PromoterLCNN\n(CNN)", "color": "#228B22",
         "bars": [("1-CPU", 1.856, "//"), ("16-CPU", 1.928, "\\\\"), ("1-CPU+GPU", 1.856, ".."), ("16-CPU+GPU", 1.928, "")]},
        {"name": "FIMO\n(motif)", "color": "#00ACC1",
         "bars": [("1-CPU", 31.844, "//"), ("16-CPU", 4.556, "\\\\")]},
        {"name": "iPro-MP\n(gLM)", "color": "#7E57C2",
         "bars": [("1-CPU", 395.350, "//"), ("16-CPU", 197.675, "\\\\"), ("1-CPU+GPU", 6.866, ".."), ("16-CPU+GPU", 3.504, "")]},
    ]
    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=300)
    x_centers = np.arange(len(tools_all_info)) * 1.35
    bar_w = 0.22

    for t_idx, t_data in enumerate(tools_all_info):
        n_b = len(t_data["bars"])
        x_center = x_centers[t_idx]
        offsets = (np.arange(n_b) - (n_b - 1) / 2) * bar_w
        for b_idx, (m_label, val, hatch) in enumerate(t_data["bars"]):
            pos = x_center + offsets[b_idx]
            bar = ax.bar(pos, val, bar_w, color=t_data["color"], hatch=hatch,
                         edgecolor="black", linewidth=0.8, zorder=3)
            txt = format_time_short_3dec(val)
            ax.annotate(txt, (pos, val), xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=t_data["color"])

    ax.set_xticks(x_centers)
    ax.set_xticklabels([t["name"] for t in tools_all_info], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool and Model Architecture", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Inference Benchmark Across Hardware Regimes (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    max_all = max(b[1] for t in tools_all_info for b in t['bars']) if tools_all_info else 1
    ax.set_ylim(0, max_all * 1.15); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="//", edgecolor="black", label="1-CPU Single-Thread"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="\\\\", edgecolor="black", label="16-CPU Multi-Core (Pure CPU)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="..", edgecolor="black", label="1-CPU Host + RTX 3090 GPU"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="", edgecolor="black", label="16-CPU Host + RTX 3090 GPU"),
    ]
    ax.legend(handles=legend_handles, title="Execution Regime", frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_plot(fig, DIR_COMBINED, "compute_time_baseline")
    plt.close(fig)


# =============================================================================
# 2. BASELINE MEMORY (N = 1,976)
# =============================================================================
def generate_baseline_memory():
    # 1 CPU RAM (leído de métricas)
    m1r = load_metrics(DATA_DIR, "scale_db", "988")
    n_ram = int(_metric(m1r, "MLDSPP XGBoost", "n_sequences") or 1976)
    tools_1cpu = []
    for name in m1r["tool"].unique() if not m1r.empty else []:
        if "75%" in name:
            continue
        meta = PALETTE.get(name)
        ram = _metric(m1r, name, "peak_ram_mb")
        if meta is None or ram is None:
            continue
        tools_1cpu.append({"name": f"{meta['short']}\n({meta['method']})", "ram": ram, "color": meta["color"]})
    if not tools_1cpu:
        tools_1cpu = [
            {"name": "MLDSPP\n(BDT)", "ram": 149.0, "color": "#942C76"},
            {"name": "FIMO\n(motif)", "ram": 87.2, "color": "#00ACC1"},
            {"name": "PromoTech\n(RF)", "ram": 6827.2, "color": "#E07614"},
            {"name": "PromoterLCNN\n(CNN)", "ram": 1839.9, "color": "#228B22"},
            {"name": "iPro-MP\n(gLM)", "ram": 1302.6, "color": "#7E57C2"},
        ]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    x = np.arange(len(tools_1cpu))
    bars = ax.bar(x, [t["ram"] for t in tools_1cpu], 0.52, color=[t["color"] for t in tools_1cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_ram_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_1cpu], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Peak RAM (MB)", fontsize=12, fontweight="bold")
    ax.set_title(f"Single-Core CPU Peak RAM Usage (N = {n_ram:,})", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 8500); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, DIR_1CPU, "peak_memory_baseline")
    plt.close(fig)

    # 16 CPU RAM (leído de métricas)
    m16r = load_metrics(DATA_DIR, "scale_db_16cpu", "988")
    n_ram16 = int(_metric(m16r, "MLDSPP XGBoost", "n_sequences") or 1976)
    tools_16cpu = []
    for name in m16r["tool"].unique() if not m16r.empty else []:
        if "75%" in name:
            continue
        meta = PALETTE.get(name)
        ram = _metric(m16r, name, "peak_ram_mb")
        if meta is None or ram is None:
            continue
        tools_16cpu.append({"name": f"{meta['short']}\n({meta['method']})", "ram": ram, "color": meta["color"]})
    if not tools_16cpu:
        tools_16cpu = [
            {"name": "MLDSPP\n(BDT)", "ram": 187.1, "color": "#942C76"},
            {"name": "FIMO\n(motif)", "ram": 203.4, "color": "#00ACC1"},
            {"name": "PromoTech\n(RF)", "ram": 7298.0, "color": "#E07614"},
            {"name": "iPro-MP\n(gLM)", "ram": 1307.4, "color": "#7E57C2"},
            {"name": "PromoterLCNN\n(CNN)", "ram": 1859.1, "color": "#228B22"},
        ]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    x = np.arange(len(tools_16cpu))
    bars = ax.bar(x, [t["ram"] for t in tools_16cpu], 0.52, color=[t["color"] for t in tools_16cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_ram_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_16cpu], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Peak RAM (MB)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Peak RAM Usage (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 2300); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, DIR_16CPU, "peak_memory_baseline")
    plt.close(fig)

    # GPU Weights vs VRAM (leído de métricas gpu_thr1)
    mgpu = load_metrics(DATA_DIR, "scale_db_gpu_thr1", "988")
    gpu_models, weights_size_mb, inference_vram_mb = [], [], []
    for gn in ["PromoterLCNN", "iPro-MP (H. pylori)"]:
        meta = PALETTE.get(gn, {"short": gn.split(" (")[0], "method": "?"})
        w = _metric(mgpu, gn, "model_size_mb")
        v = _metric(mgpu, gn, "peak_vram_mb")
        if w is None and v is None:
            continue
        gpu_models.append(f"{meta['short']}\n({meta['method']})")
        weights_size_mb.append(w if w is not None else 0)
        inference_vram_mb.append(v if v is not None else 0)
    if not gpu_models:
        return
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    x = np.arange(len(gpu_models)); w = 0.35
    b1 = ax.bar(x - w/2, weights_size_mb, w, label="Model Weights (Disk/RAM)", color="#455A64", edgecolor="black", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w/2, inference_vram_mb, w, label="Peak Inference VRAM", color="#7E57C2", edgecolor="black", linewidth=0.8, zorder=3)
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB\n({h/1024:.3f} GB)", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
    ax.set_title(f"GPU Memory Requirements: Model Weights vs Peak VRAM (N = {n_ram16:,})", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x); ax.set_xticklabels(gpu_models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(inference_vram_mb + [1]) * 1.12); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    save_plot(fig, DIR_GPU, "peak_memory_baseline")
    plt.close(fig)


# =============================================================================
# 3. SCALING TIME (LINEAR & LOG-LOG)
# =============================================================================
def generate_scaling_time_curves():
    # 1 CPU
    for mode in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        for tool_key in ["MLDSPP XGBoost", "PromoterLCNN", "FIMO + Prokaryote DB", "iPro-MP (H. pylori)"]:
            sub = df_sp[df_sp['tool'] == tool_key].sort_values('scale_N')
            if not sub.empty:
                meta = PALETTE[tool_key]
                if mode == "linear":
                    ax.plot(sub['scale_N']/1000, sub['time_s_1hilo'], marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
                else:
                    ax.plot(sub['scale_N'], sub['time_s_1hilo'], marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
        if mode == "log":
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Total Sequences ($N$)", fontsize=12, fontweight="bold")
        else:
            ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(f"Single-Core CPU Inference Time Scaling ({mode.capitalize()} Scale)", fontsize=13, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_plot(fig, DIR_1CPU, f"scaling_time_{mode}")
        plt.close(fig)

    # 16 CPU
    for mode in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        for tool_key in ["MLDSPP XGBoost", "PromoterLCNN", "FIMO + Prokaryote DB", "iPro-MP (H. pylori)"]:
            sub = df_16cpu[df_16cpu['tool'] == tool_key].sort_values('scale_N')
            if not sub.empty:
                meta = PALETTE[tool_key]
                if mode == "linear":
                    ax.plot(sub['scale_N']/1000, sub['time_s'], marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
                else:
                    ax.plot(sub['scale_N'], sub['time_s'], marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
        if mode == "log":
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Total Sequences ($N$)", fontsize=12, fontweight="bold")
        else:
            ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
        ax.set_title(f"16-Core CPU & GPU Inference Time Scaling ({mode.capitalize()} Scale)", fontsize=13, fontweight="bold", pad=15)
        ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        plt.tight_layout()
        save_plot(fig, DIR_16CPU, f"scaling_time_{mode}")
        plt.close(fig)

    # Combined All Modes Scaling (Log-Log)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    sub_mld = df_sp[df_sp['tool'] == "MLDSPP XGBoost"].sort_values('scale_N')
    sub_fimo = df_sp[df_sp['tool'] == "FIMO + Prokaryote DB"].sort_values('scale_N')
    sub_lcnn = df_sp[df_sp['tool'] == "PromoterLCNN"].sort_values('scale_N')
    sub_ipro = df_sp[df_sp['tool'] == "iPro-MP (H. pylori)"].sort_values('scale_N')

    ax.plot(sub_mld['scale_N'], sub_mld['time_s_1hilo'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle=":", marker="o", linewidth=1.8, label="MLDSPP [1-CPU]")
    ax.plot(sub_mld['scale_N'], sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle="--", marker="^", linewidth=2.2, label="MLDSPP [16-CPU]")
    ax.plot(sub_fimo['scale_N'], sub_fimo['time_s_1hilo'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle=":", marker="o", linewidth=1.8, label="FIMO [1-CPU]")
    ax.plot(sub_fimo['scale_N'], sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle="--", marker="^", linewidth=2.2, label="FIMO [16-CPU]")
    ax.plot(sub_lcnn['scale_N'], sub_lcnn['time_s_1hilo'], color=PALETTE["PromoterLCNN"]["color"], linestyle=":", marker="o", linewidth=1.8, label="PromoterLCNN [1-CPU]")
    ax.plot(sub_lcnn['scale_N'], sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], linestyle="-", marker="D", linewidth=2.5, label="PromoterLCNN [16-CPU+GPU]")
    ax.plot(sub_ipro['scale_N'], sub_ipro['time_s_1hilo'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-.", marker="s", linewidth=2.2, label="iPro-MP [1-CPU Host+GPU]")
    ax.plot(sub_ipro['scale_N'], sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-", marker="D", linewidth=2.8, label="iPro-MP [16-CPU Host+GPU]")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total Sequences Evaluated ($N$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Full Spectrum Scaling Across Hardware Configurations (Log-Log Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.4); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.0, loc="upper left", ncol=2)
    plt.tight_layout()
    save_plot(fig, DIR_COMBINED, "scaling_time_log")
    plt.close(fig)

    # Combined All Modes Scaling (FULL LINEAR SCALE)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.plot(sub_mld['scale_N']/1000, sub_mld['time_s_1hilo'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle=":", marker="o", linewidth=1.8, label="MLDSPP [1-CPU]")
    ax.plot(sub_mld['scale_N']/1000, sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle="--", marker="^", linewidth=2.2, label="MLDSPP [16-CPU]")
    ax.plot(sub_fimo['scale_N']/1000, sub_fimo['time_s_1hilo'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle=":", marker="o", linewidth=1.8, label="FIMO [1-CPU]")
    ax.plot(sub_fimo['scale_N']/1000, sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle="--", marker="^", linewidth=2.2, label="FIMO [16-CPU]")
    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['time_s_1hilo'], color=PALETTE["PromoterLCNN"]["color"], linestyle=":", marker="o", linewidth=1.8, label="PromoterLCNN [1-CPU]")
    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], linestyle="-", marker="D", linewidth=2.5, label="PromoterLCNN [16-CPU+GPU]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['time_s_1hilo'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-.", marker="s", linewidth=2.2, label="iPro-MP [1-CPU Host+GPU]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-", marker="D", linewidth=2.8, label="iPro-MP [16-CPU Host+GPU]")

    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inference Time (s, Linear Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Full Spectrum Inference Time Scaling (Linear Scale: 0 to 400k Sequences)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.0, loc="upper left", ncol=2)
    plt.tight_layout()
    save_plot(fig, DIR_COMBINED, "scaling_time_linear")
    plt.close(fig)

    # Combined All Modes Scaling (LINEAR SCALE WITHOUT iPro-MP 1-CPU & 16-CPU PURE CPU)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.plot(sub_mld['scale_N']/1000, sub_mld['time_s_1hilo'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle=":", marker="o", linewidth=1.8, label=f"MLDSPP [1-CPU: {sub_mld['time_s_1hilo'].iloc[-1]:.1f}s]")
    ax.plot(sub_mld['scale_N']/1000, sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle="--", marker="^", linewidth=2.2, label=f"MLDSPP [16-CPU: {sub_mld['time_s_16cores'].iloc[-1]:.1f}s]")
    ax.plot(sub_fimo['scale_N']/1000, sub_fimo['time_s_1hilo'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle=":", marker="o", linewidth=1.8, label=f"FIMO [1-CPU: {sub_fimo['time_s_1hilo'].iloc[-1]:.1f}s]")
    ax.plot(sub_fimo['scale_N']/1000, sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle="--", marker="^", linewidth=2.2, label=f"FIMO [16-CPU: {sub_fimo['time_s_16cores'].iloc[-1]:.1f}s]")
    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['time_s_1hilo'], color=PALETTE["PromoterLCNN"]["color"], linestyle=":", marker="o", linewidth=1.8, label=f"LCNN [1-CPU: {sub_lcnn['time_s_1hilo'].iloc[-1]:.1f}s]")
    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], linestyle="-", marker="D", linewidth=2.5, label=f"LCNN [16-CPU+GPU: {sub_lcnn['time_s_16cores'].iloc[-1]:.1f}s]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['time_s_1hilo'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-.", marker="s", linewidth=2.2, label="iPro-MP [1-CPU Host+GPU: 1478.400s / 24.64m]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-", marker="D", linewidth=2.8, label="iPro-MP [16-CPU Host+GPU: 700.800s / 11.68m]")

    ax.set_ylim(0, 6800)
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inference Time (s, Linear Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Inference Time Scaling across Hardware Regimes (Excl. iPro-MP Pure CPU)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.0, loc="upper left")
    plt.tight_layout()
    save_plot(fig, DIR_COMBINED, "scaling_time_linear_no_ipro_pure_cpu")
    save_plot(fig, DIR_COMBINED, "scaling_time_linear_practical")
    plt.close(fig)

    # Combined All Modes Scaling (ZOOMED LINEAR SCALE <= 1,600s / 25 min)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.plot(sub_mld['scale_N']/1000, sub_mld['time_s_1hilo'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle=":", marker="o", linewidth=1.8, label="MLDSPP [1-CPU]")
    ax.plot(sub_mld['scale_N']/1000, sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle="--", marker="^", linewidth=2.2, label="MLDSPP [16-CPU]")
    ax.plot(sub_fimo['scale_N']/1000, sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle="--", marker="^", linewidth=2.2, label=f"FIMO [16-CPU: {sub_fimo['time_s_16cores'].iloc[-1]:.1f}s]")
    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['time_s_1hilo'], color=PALETTE["PromoterLCNN"]["color"], linestyle=":", marker="o", linewidth=1.8, label=f"LCNN [1-CPU: {sub_lcnn['time_s_1hilo'].iloc[-1]:.1f}s]")
    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], linestyle="-", marker="D", linewidth=2.5, label=f"LCNN [16-CPU+GPU: {sub_lcnn['time_s_16cores'].iloc[-1]:.1f}s]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['time_s_1hilo'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-.", marker="s", linewidth=2.2, label=f"iPro-MP [Host+GPU: {sub_ipro['time_s_1hilo'].iloc[-1]:.0f}s]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-", marker="D", linewidth=2.8, label=f"iPro-MP [16-CPU Host+GPU: {sub_ipro['time_s_16cores'].iloc[-1]:.0f}s]")

    ax.set_ylim(0, 1600)
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inference Time (s, Zoomed Linear Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Accelerated Regime Scaling (Zoomed Linear Scale: Sub-1,600s / 25 min)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.0, loc="upper left")
    plt.tight_layout()
    save_plot(fig, DIR_COMBINED, "scaling_time_linear_zoomed")
    plt.close(fig)


# =============================================================================
# 4. SCALING MEMORY (LINEAR GB)
# =============================================================================
def generate_scaling_memory_curves():
    # 1 CPU RAM Scaling
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for tool_key in ["MLDSPP XGBoost", "PromoterLCNN", "FIMO + Prokaryote DB", "iPro-MP (H. pylori)"]:
        sub = df_sp[df_sp['tool'] == tool_key].sort_values('scale_N')
        if not sub.empty:
            meta = PALETTE[tool_key]
            ax.plot(sub['scale_N']/1000, sub['ram_mb_1hilo']/1024, marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak System RAM (GB)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Core CPU Peak RAM Scaling (Linear Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    save_plot(fig, DIR_1CPU, "scaling_memory_linear")
    plt.close(fig)

    # 16 CPU RAM Scaling
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for tool_key in ["MLDSPP XGBoost", "PromoterLCNN", "FIMO + Prokaryote DB", "iPro-MP (H. pylori)"]:
        sub = df_16cpu[df_16cpu['tool'] == tool_key].sort_values('scale_N')
        if not sub.empty:
            meta = PALETTE[tool_key]
            ax.plot(sub['scale_N']/1000, sub['peak_ram_mb']/1024, marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak System RAM (GB)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Peak RAM Scaling (Linear Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    save_plot(fig, DIR_16CPU, "scaling_memory_linear")
    plt.close(fig)

    # GPU VRAM Scaling
    sub_ipro = df_16cpu[df_16cpu['tool'] == "iPro-MP (H. pylori)"].sort_values(by='scale_N')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sub_lcnn_v = df_sp[df_sp['tool'] == "PromoterLCNN"].sort_values('scale_N')
    if 'peak_vram_mb' in sub_lcnn_v.columns:
        sub_lcnn_v = sub_lcnn_v[sub_lcnn_v['peak_vram_mb'] > 0]
    else:
        sub_lcnn_v = sub_lcnn_v.iloc[0:0]  # empty — column not available in speedup TSV
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['peak_vram_mb'], marker="o", color=PALETTE["iPro-MP (H. pylori)"]["color"], linewidth=2.5, markersize=8, label="iPro-MP (DNABERT-6 on RTX 3090)")
    if not sub_lcnn_v.empty:
        ax.plot(sub_lcnn_v['scale_N']/1000, sub_lcnn_v['peak_vram_mb'], marker="s", color=PALETTE["PromoterLCNN"]["color"], linewidth=2.5, markersize=8, linestyle="--", label="PromoterLCNN (1D-CNN on RTX 3090)")
    ax.set_ylim(0, 3500)
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak GPU VRAM (MB)", fontsize=12, fontweight="bold")
    ax.set_title("GPU VRAM Usage vs Dataset Scale — Constant Complexity $\mathcal{O}(1)$", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="center right")
    plt.tight_layout()
    save_plot(fig, DIR_GPU, "scaling_memory_linear")
    plt.close(fig)


# =============================================================================
# 5. 1-CPU & 16-CPU 4-TOOL GROUPED BARS
# =============================================================================
def generate_grouped_bars():
    tools_4 = [
        {"name": "MLDSPP\n(BDT)",      "key": "MLDSPP XGBoost"},
        {"name": "PromoterLCNN\n(CNN)", "key": "PromoterLCNN"},
        {"name": "FIMO\n(motif)",       "key": "FIMO + Prokaryote DB"},
        {"name": "iPro-MP\n(gLM)",      "key": "iPro-MP (H. pylori)"}
    ]
    bar_width = 0.18
    indices = np.arange(len(tools_4))
    n_scales = len(KEY_SCALES)

    # 1 CPU Grouped Time
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    for j, scale in enumerate(KEY_SCALES):
        times = []
        for t in tools_4:
            val = df_sp[(df_sp['tool'] == t['key']) & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values
            times.append(val[0] if len(val) > 0 else 0.0)
        pos = indices + (j - (n_scales - 1) / 2) * bar_width
        bars = ax.bar(pos, times, bar_width, label=SCALE_LABELS[scale], color=SCALE_COLORS[scale], edgecolor="black", linewidth=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt = format_time_short_3dec(h)
                ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(indices); ax.set_xticklabels([t["name"] for t in tools_4], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Core CPU Inference Time Grouped by Tool (N = 1.9k to 395.2k)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 2200); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(title="Sequence Scale (N)", title_fontsize=10, fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, DIR_1CPU, "compute_time_grouped")
    plt.close(fig)

    # 16 CPU Grouped Time
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    for j, scale in enumerate(KEY_SCALES):
        times = []
        for t in tools_4:
            val = df_16cpu[(df_16cpu['tool'] == t['key']) & (df_16cpu['scale_N'] == scale)]['time_s'].values
            times.append(val[0] if len(val) > 0 else 0.0)
        pos = indices + (j - (n_scales - 1) / 2) * bar_width
        bars = ax.bar(pos, times, bar_width, label=SCALE_LABELS[scale], color=SCALE_COLORS[scale], edgecolor="black", linewidth=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt = format_time_short_3dec(h)
                ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(indices); ax.set_xticklabels([t["name"] for t in tools_4], fontsize=11, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Inference Time Grouped by Tool (N = 1.9k to 395.2k)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1400); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(title="Sequence Scale (N)", title_fontsize=10, fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_plot(fig, DIR_16CPU, "compute_time_grouped")
    plt.close(fig)


# =============================================================================
# 6. SPEEDUP & RAM USAGE
# =============================================================================
def generate_speedup_and_ram():
    sub_fimo = df_sp[df_sp['tool'] == "FIMO + Prokaryote DB"].sort_values('scale_N')
    sub_lcnn = df_sp[df_sp['tool'] == "PromoterLCNN"].sort_values('scale_N')
    sub_mld = df_sp[df_sp['tool'] == "MLDSPP XGBoost"].sort_values('scale_N')
    sub_ipro = df_sp[df_sp['tool'] == "iPro-MP (H. pylori)"].sort_values('scale_N')

    # Speedup curves
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    fimo_sp = sub_fimo['time_s_1hilo'] / sub_fimo['time_s_16cores']
    lcnn_sp = sub_lcnn['time_s_1hilo'] / sub_lcnn['time_s_16cores']
    mld_sp = sub_mld['time_s_1hilo'] / sub_mld['time_s_16cores']
    ipro_host_sp = sub_ipro['time_s_1hilo'] / sub_ipro['time_s_16cores']

    ax.plot(sub_fimo['scale_N']/1000, fimo_sp, color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", linewidth=2.3, markersize=7, label="FIMO: 16-CPU vs 1-CPU (6.989x to 13.087x OpenMP)")
    ax.plot(sub_lcnn['scale_N']/1000, lcnn_sp, color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", linewidth=2.3, markersize=7, label="PromoterLCNN: 16-CPU+GPU vs 1-CPU (0.963x to 5.596x)")
    ax.plot(sub_ipro['scale_N']/1000, ipro_host_sp, color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", linewidth=2.5, markersize=7, label="iPro-MP: 16-CPU Host vs 1-CPU Host on GPU (1.205x to 2.112x)")
    ax.plot(sub_mld['scale_N']/1000, mld_sp, color=PALETTE["MLDSPP XGBoost"]["color"], marker="v", linestyle="--", linewidth=2.0, markersize=7, label="MLDSPP: 16-CPU vs 1-CPU (1.024x to 1.996x)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Baseline (1.000x)")

    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup Ratio (x)", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Core CPU & GPU Acceleration Factors Across Dataset Scales", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")
    plt.tight_layout()
    save_plot(fig, DIR_16CPU, "speedup_1cpu_vs_16cpu")
    save_plot(fig, DIR_COMBINED, "speedup_multicore_and_gpu")
    plt.close(fig)

    # Homologous RAM Usage Comparison
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    for tool_key in ["MLDSPP XGBoost", "FIMO + Prokaryote DB", "PromoterLCNN", "iPro-MP (H. pylori)"]:
        meta = PALETTE[tool_key]
        sub = df_sp[df_sp['tool'] == tool_key].sort_values('scale_N')
        ax.plot(sub['scale_N']/1000, sub['ram_mb_1hilo']/1024, color=meta['color'], marker=meta['marker'], linestyle=":", linewidth=1.8, label=f"{meta['short']} [1-CPU]")
        ax.plot(sub['scale_N']/1000, sub['ram_mb_16cores']/1024, color=meta['color'], marker=meta['marker'], linestyle="--", linewidth=2.3, label=f"{meta['short']} [16-CPU / Host]")

    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak System RAM (GB)", fontsize=12, fontweight="bold")
    ax.set_title("Peak System RAM Usage Scaling Across Hardware Configurations (Linear Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.0, loc="upper left", ncol=2)
    plt.tight_layout()
    save_plot(fig, DIR_COMBINED, "ram_usage_comparison")
    plt.close(fig)


# =============================================================================
# 7. BY_SCALE VERTICAL BARS (COMPUTE_TIME & PEAK_RAM FOR ALL 9 SCALES)
# =============================================================================
def generate_by_scale_plots():
    for scale in SCALES_ALL:
        folder_name = format_scale_folder_name(scale)
        scale_dir_comb = DIR_COMBINED / "by_scale" / folder_name
        scale_dir_comb.mkdir(parents=True, exist_ok=True)

        mld_1cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        mld_16cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        lcnn_1cpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        lcnn_16gpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        fimo_1cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        fimo_16cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        # PromoTech e iPro-MP: solo valores medidos (sin proyecciones fabricadas)
        def _from_metrics(campaign, tool_name):
            m = load_metrics(DATA_DIR, campaign, str(scale))
            return _metric(m, tool_name, "time_s")

        def _from_metrics_ram(campaign, tool_name):
            m = load_metrics(DATA_DIR, campaign, str(scale))
            return _metric(m, tool_name, "peak_ram_mb")

        pt_measured = _from_metrics("scale_db_promotech", "PromoTech RF-HOT (PG Max)")

        ipro_1cpu_gpu_row = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]
        ipro_1cpu_gpu = float(ipro_1cpu_gpu_row['time_s_1hilo'].values[0]) if not ipro_1cpu_gpu_row.empty else None
        ipro_16cpu_gpu = float(ipro_1cpu_gpu_row['time_s_16cores'].values[0]) if not ipro_1cpu_gpu_row.empty else None

        time_tools_config = [
            {"tool_label": "MLDSPP\n(BDT)", "color": "#942C76",
             "bars": [("1-CPU", mld_1cpu, "//"), ("16-CPU", mld_16cpu, "\\\\")]},
            {"tool_label": "PromoterLCNN\n(CNN)", "color": "#228B22",
             "bars": [("1-CPU", lcnn_1cpu, "//"), ("16-CPU", lcnn_16gpu, "\\\\"), ("1-CPU Host+GPU", lcnn_1cpu, ".."), ("16-CPU Host+GPU", lcnn_16gpu, "")]},
            {"tool_label": "FIMO\n(PWM)", "color": "#00ACC1",
             "bars": [("1-CPU", fimo_1cpu, "//"), ("16-CPU", fimo_16cpu, "\\\\")]},
            {"tool_label": "PromoTech\n(RF)", "color": "#E07614",
             "bars": ([("measured", pt_measured, "//")] if pt_measured else [])},
            {"tool_label": "iPro-MP\n(gLM)", "color": "#7E57C2",
             "bars": ([("1-CPU Host+GPU", ipro_1cpu_gpu, ".."), ("16-CPU Host+GPU", ipro_16cpu_gpu, "")])},
        ]

        fig, ax = plt.subplots(figsize=(13, 7.2), dpi=300)
        x_centers = np.arange(len(time_tools_config)) * 1.35
        bar_w = 0.22
        max_time_val = 0.0

        for t_idx, t_conf in enumerate(time_tools_config):
            n_b = len(t_conf["bars"])
            x_center = x_centers[t_idx]
            offsets = (np.arange(n_b) - (n_b - 1) / 2) * bar_w

            for b_idx, (m_label, val, hatch) in enumerate(t_conf["bars"]):
                pos = x_center + offsets[b_idx]
                max_time_val = max(max_time_val, val)
                bar = ax.bar(pos, val, bar_w, color=t_conf["color"], hatch=hatch,
                             edgecolor="black", linewidth=0.8, zorder=3)
                txt = format_time_short_3dec(val)
                ax.annotate(txt, (pos, val), xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=t_conf["color"])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([t["tool_label"] for t in time_tools_config], fontsize=11, fontweight="bold")
        ax.set_xlabel("Tool and Model Architecture", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
        scale_label_str = f"{scale/1000:.1f}k" if scale < 1000000 else f"{scale/1000000:.2f}M"
        ax.set_title(f"Inference Benchmark Across Hardware Regimes (N = {scale:,} / {scale_label_str})", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max_time_val * 1.25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="//", edgecolor="black", label="1-CPU Single-Thread"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="\\\\", edgecolor="black", label="16-CPU Multi-Core (Pure CPU)"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="..", edgecolor="black", label="1-CPU Host + RTX 3090 GPU"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="", edgecolor="black", label="16-CPU Host + RTX 3090 GPU"),
        ]
        ax.legend(handles=legend_handles, title="Execution Regime", frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()

        targets = [CANONICAL_TARGET]
        if DESKTOP_TARGET.parent.exists():
            targets.append(DESKTOP_TARGET)

        for base in targets:
            dest = base / "combined_1cpu_16cpu_gpu" / "by_scale" / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest / "compute_time.png", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "compute_time.svg", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "compute_time.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Plot peak_ram for this scale
        mld_ram_1cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        mld_ram_16cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        lcnn_ram_1cpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        lcnn_ram_16gpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        fimo_ram_1cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        fimo_ram_16cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        pt_measured_ram = _from_metrics_ram("scale_db_promotech", "PromoTech RF-HOT (PG Max)")

        ipro_ram_1cpu = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        ipro_ram_16cpu = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        ram_tools_config = [
            {"tool_label": "MLDSPP\n(BDT)", "color": "#942C76",
             "bars": [("1-CPU", mld_ram_1cpu, "//"), ("16-CPU", mld_ram_16cpu, "\\\\")]},
            {"tool_label": "PromoterLCNN\n(CNN)", "color": "#228B22",
             "bars": [("1-CPU", lcnn_ram_1cpu, "//"), ("16-CPU + GPU", lcnn_ram_16gpu, "")]},
            {"tool_label": "FIMO\n(PWM)", "color": "#00ACC1",
             "bars": [("1-CPU", fimo_ram_1cpu, "//"), ("16-CPU", fimo_ram_16cpu, "\\\\")]},
            {"tool_label": "PromoTech\n(RF)", "color": "#E07614",
             "bars": ([("measured", pt_measured_ram, "//")] if pt_measured_ram else [])},
            {"tool_label": "iPro-MP\n(gLM)", "color": "#7E57C2",
             "bars": [("1-CPU Host", ipro_ram_1cpu, "//"), ("16-CPU Host+GPU", ipro_ram_16cpu, "")]},
        ]

        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        max_ram_val = 0.0

        for t_idx, t_conf in enumerate(ram_tools_config):
            n_b = len(t_conf["bars"])
            x_center = x_centers[t_idx]
            offsets = (np.arange(n_b) - (n_b - 1) / 2) * bar_w

            for b_idx, (m_label, val, hatch) in enumerate(t_conf["bars"]):
                pos = x_center + offsets[b_idx]
                max_ram_val = max(max_ram_val, val)
                bar = ax.bar(pos, val, bar_w, color=t_conf["color"], hatch=hatch,
                             edgecolor="black", linewidth=0.8, zorder=3)
                txt = format_ram_3dec(val)
                ax.annotate(txt, (pos, val), xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.0, fontweight='bold',
                            color=t_conf["color"], rotation=90)

        ax.set_xticks(x_centers)
        ax.set_xticklabels([t["tool_label"] for t in ram_tools_config], fontsize=11, fontweight="bold")
        ax.set_xlabel("Tool and Model Architecture", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Peak System RAM (MB)", fontsize=12, fontweight="bold")
        ax.set_title(f"Peak RAM Usage Across Hardware Regimes (N = {scale:,} / {scale_label_str})", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max_ram_val * 1.25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=legend_handles, title="Execution Regime", frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()

        for base in targets:
            dest = base / "combined_1cpu_16cpu_gpu" / "by_scale" / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest / "peak_ram.png", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "peak_ram.svg", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "peak_ram.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"  [Saved] combined_1cpu_16cpu_gpu/by_scale/{folder_name}/ (compute_time & peak_ram in png, svg, pdf)")


def main():
    print("=" * 75)
    print("  GENERANDO GRÁFICOS DE BENCHMARK (TIEMPOS, RAM Y ESCALAMIENTO)")
    print(f"  Directorio Canónico: {CANONICAL_TARGET}")
    print("=" * 75)
    generate_baseline_time()
    generate_baseline_memory()
    generate_scaling_time_curves()
    generate_scaling_memory_curves()
    generate_grouped_bars()
    generate_speedup_and_ram()
    generate_by_scale_plots()
    print("\n" + "=" * 75)
    print("  GENERACIÓN DE GRÁFICOS DE BENCHMARK FINALIZADA CON ÉXITO")
    print("=" * 75)


if __name__ == "__main__":
    main()
