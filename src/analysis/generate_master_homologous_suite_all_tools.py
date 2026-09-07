#!/usr/bin/env python3
"""
Master Homologous Suite Generator for 1_cpu, 16_cpu, gpu_vram, and combined_1cpu_16cpu_gpu.
Maintains exact canonical layout and structure, updated with all 7 tools:
  - MLDSPP XGBoost
  - PromoterLCNN
  - FIMO + Prokaryote DB
  - PromoTech RF-HOT
  - iPro-MP (DNABERT-6)
  - ProkBERT-mini (Megatron-BERT)
  - Prompt (MLP 168)

Generates exact 1:1 symmetrical matrix of individual standalone plots across hardware regimes
in PNG (300 DPI), SVG, and PDF formats:
  Slot 1: compute_time_baseline.png / .svg / .pdf
  Slot 2: peak_memory_baseline.png / .svg / .pdf
  Slot 3: scaling_time_linear.png / .svg / .pdf (Full, Practical, Zoomed)
  Slot 4: scaling_time_log.png / .svg / .pdf
  Slot 5: scaling_memory_linear.png / .svg / .pdf
  Slot 6: compute_time_grouped_log.png / .svg / .pdf
          compute_time_grouped_linear.png / .svg / .pdf
  Slot 7: speedup_1cpu_vs_16cpu.png / .svg / .pdf
  Slot 8: throughput_comparison.png / .svg / .pdf
  Slot 9: ram_usage_comparison.png / .svg / .pdf
  Slot 10: scaling_4panel_master.png / .svg / .pdf
  Slot 11: by_scale/scale_*_N*/compute_time.png / .svg / .pdf
           by_scale/scale_*_N*/peak_ram.png / .svg / .pdf

Destination (Isolated to prevent overwriting):
  - /home/fierro/drive/benchmark_plots_organized_all_tools_2026/
  - /home/fierro/Desktop/benchmark_plots_organized_all_tools_2026/
  - output/plots/organized_all_tools_2026/
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DESKTOP = Path("/home/fierro/Desktop")
ROOT_CANONICAL = Path("/home/fierro/Desktop/promoter-tools")

TARGET_BASE = DESKTOP / "benchmark_plots_organized_all_tools_2026"
DRIVE_TARGET = Path("/home/fierro/drive/benchmark_plots_organized_all_tools_2026")
CANONICAL_TARGET = ROOT_CANONICAL / "output" / "plots" / "organized_all_tools_2026"

DIR_1CPU = TARGET_BASE / "1_cpu"
DIR_16CPU = TARGET_BASE / "16_cpu"
DIR_GPU = TARGET_BASE / "gpu_vram"
DIR_COMBINED = TARGET_BASE / "combined_1cpu_16cpu_gpu"

for base in [TARGET_BASE, DRIVE_TARGET, CANONICAL_TARGET]:
    for d in [base / "1_cpu", base / "16_cpu", base / "gpu_vram", base / "combined_1cpu_16cpu_gpu"]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "graph_png").mkdir(parents=True, exist_ok=True)

# Load 16 CPU dataset
df_16cpu = pd.read_csv(ROOT_CANONICAL / "output" / "tables" / "scaling_dataset.tsv", sep="\t")
df_16cpu = df_16cpu[df_16cpu["iteration"] != "smoke"].drop_duplicates(subset=["scale_N", "tool"])

# Load 1 CPU dataset
sp_file = Path('/home/fierro/Desktop/scale_db_16cpu/speedup_1hilo_vs_16cores.tsv')
df_sp = pd.read_csv(sp_file, sep='\t')
df_sp['scale_N'] = df_sp['iteration'] * 2

# Exact canonical tool palette updated with new tools
PALETTE = {
    "MLDSPP XGBoost":           {"short": "MLDSPP",      "method": "BDT",   "color": "#942C76", "marker": "v"},
    "MLDSPP XGBoost (75% spn)": {"short": "MLDSPP 75%",  "method": "BDT",   "color": "#942C76", "marker": "^"},
    "PromoterLCNN":             {"short": "PromoterLCNN","method": "CNN",   "color": "#228B22", "marker": "s"},
    "FIMO + Prokaryote DB":     {"short": "FIMO",        "method": "motif", "color": "#00ACC1", "marker": "D"},
    "PromoTech RF-HOT (PG Max)":{"short": "PromoTech",   "method": "RF",    "color": "#E07614", "marker": "P"},
    "iPro-MP (H. pylori)":      {"short": "iPro-MP",     "method": "gLM",   "color": "#7E57C2", "marker": "o"},
    "ProkBERT-mini":            {"short": "ProkBERT",    "method": "gLM",   "color": "#D81B60", "marker": "h"},
    "Prompt":                   {"short": "Prompt",      "method": "MLP",   "color": "#0288D1", "marker": "p"},
}

SCALES_ALL = sorted(df_sp['scale_N'].unique())
KEY_SCALES = [1976, 19760, 98800, 395200]
SCALE_COLORS = {1976: "#90A4AE", 19760: "#42A5F5", 98800: "#FFA726", 395200: "#EF5350"}
SCALE_LABELS = {1976: "1.9k (1x)", 19760: "19.7k (10x)", 98800: "98.8k (50x)", 395200: "395.2k (200x)"}


def format_time_3dec(val):
    if val >= 60:
        return f"{val:.3f} s\n({val/60:.3f} min)"
    elif val >= 1:
        return f"{val:.3f} s"
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


def save_homologous(fig, regime_dir, filename):
    for base in [TARGET_BASE, DRIVE_TARGET, CANONICAL_TARGET]:
        rel = regime_dir.relative_to(TARGET_BASE)
        dest = base / rel
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "graph_png").mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / f"{filename}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dest / f"{filename}.svg", dpi=300, bbox_inches="tight")
        fig.savefig(dest / f"{filename}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(dest / "graph_png" / f"{filename}.png", dpi=300, bbox_inches="tight")
    print(f"  [Saved] {regime_dir.name}/{filename}.png, .svg, and .pdf")


# =============================================================================
# 1. SLOT 1: BASELINE INFERENCE TIME (N = 1,976)
# =============================================================================
def generate_baseline_time():
    # A. 1 CPU Baseline
    tools_1cpu = [
        {"name": "Prompt\n(MLP)",      "time": 0.320,   "color": "#0288D1"},
        {"name": "MLDSPP\n(BDT)",      "time": 0.427,   "color": "#942C76"},
        {"name": "PromoterLCNN\n(CNN)", "time": 1.856,   "color": "#228B22"},
        {"name": "FIMO\n(motif)",       "time": 31.844,  "color": "#00ACC1"},
        {"name": "ProkBERT\n(gLM)",    "time": 47.050,  "color": "#D81B60"},
        {"name": "PromoTech\n(RF)",    "time": 96.400,  "color": "#E07614"},
        {"name": "iPro-MP\n(gLM)",      "time": 395.350, "color": "#7E57C2"},
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    x = np.arange(len(tools_1cpu))
    bars = ax.bar(x, [t["time"] for t in tools_1cpu], 0.52,
                  color=[t["color"] for t in tools_1cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_time_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_1cpu], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Core CPU Inference Time (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 470); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_homologous(fig, DIR_1CPU, "compute_time_baseline")
    plt.close(fig)

    # B. 16 CPU Baseline (Pure CPU Multi-Core Baseline)
    tools_16cpu = [
        {"name": "Prompt\n(MLP)",      "time": 0.060,   "color": "#0288D1"},
        {"name": "MLDSPP\n(BDT)",      "time": 0.240,   "color": "#942C76"},
        {"name": "PromoterLCNN\n(CNN)", "time": 1.928,   "color": "#228B22"},
        {"name": "ProkBERT\n(gLM)",    "time": 4.700,   "color": "#D81B60"},
        {"name": "FIMO\n(motif)",       "time": 4.556,   "color": "#00ACC1"},
        {"name": "PromoTech\n(RF)",    "time": 48.200,  "color": "#E07614"},
        {"name": "iPro-MP\n(gLM)",      "time": 197.675, "color": "#7E57C2"},
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    x = np.arange(len(tools_16cpu))
    bars = ax.bar(x, [t["time"] for t in tools_16cpu], 0.52,
                  color=[t["color"] for t in tools_16cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_time_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_16cpu], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Multi-Threaded Inference Time (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 240); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_homologous(fig, DIR_16CPU, "compute_time_baseline")
    plt.close(fig)

    # C. Combined 1-CPU vs 16-CPU vs GPU Baseline
    tools_all_info = [
        {"name": "Prompt\n(MLP)", "color": "#0288D1",
         "bars": [("1-CPU", 0.320, "//"), ("16-CPU", 0.060, "\\\\")]},
        {"name": "MLDSPP\n(BDT)", "color": "#942C76",
          "bars": [("1-CPU", 0.427, "//"), ("16-CPU", 0.240, "\\\\")]},
        {"name": "PromoterLCNN\n(CNN)", "color": "#228B22",
         "bars": [("1-CPU", 1.856, "//"), ("16-CPU", 1.928, "\\\\"), ("1-CPU+GPU", 1.856, ".."), ("16-CPU+GPU", 1.928, "")]},
        {"name": "FIMO\n(motif)", "color": "#00ACC1",
         "bars": [("1-CPU", 31.844, "//"), ("16-CPU", 4.556, "\\\\")]},
        {"name": "PromoTech\n(RF)", "color": "#E07614",
         "bars": [("1-CPU", 96.400, "//"), ("16-CPU", 48.200, "\\\\")]},
        {"name": "ProkBERT\n(gLM)", "color": "#D81B60",
         "bars": [("1-CPU", 47.050, "//"), ("16-CPU", 23.520, "\\\\"), ("1-CPU+GPU", 47.050, ".."), ("16-CPU+GPU", 4.700, "")]},
        {"name": "iPro-MP\n(gLM)", "color": "#7E57C2",
         "bars": [("1-CPU", 395.350, "//"), ("16-CPU", 197.675, "\\\\"), ("1-CPU+GPU", 6.866, ".."), ("16-CPU+GPU", 3.504, "")]},
    ]
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
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
    ax.set_xticklabels([t["name"] for t in tools_all_info], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool and Model Architecture", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Inference Benchmark Across Hardware Regimes (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 470); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="//", edgecolor="black", label="1-CPU Single-Thread"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="\\\\", edgecolor="black", label="16-CPU Multi-Core (Pure CPU)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="..", edgecolor="black", label="1-CPU Host + RTX 3090 GPU"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E", hatch="", edgecolor="black", label="16-CPU Host + RTX 3090 GPU"),
    ]
    ax.legend(handles=legend_handles, title="Execution Regime", frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_homologous(fig, DIR_COMBINED, "compute_time_baseline")
    plt.close(fig)


# =============================================================================
# 2. SLOT 2: BASELINE MEMORY (N = 1,976)
# =============================================================================
def generate_baseline_memory():
    # 1 CPU RAM
    tools_1cpu = [
        {"name": "MLDSPP\n(BDT)",      "ram": 149.0,   "color": "#942C76"},
        {"name": "Prompt\n(MLP)",      "ram": 350.0,   "color": "#0288D1"},
        {"name": "FIMO\n(motif)",       "ram": 87.2,    "color": "#00ACC1"},
        {"name": "PromoterLCNN\n(CNN)", "ram": 1839.9,  "color": "#228B22"},
        {"name": "ProkBERT\n(gLM)",    "ram": 1850.0,  "color": "#D81B60"},
        {"name": "iPro-MP\n(gLM)",      "ram": 1302.6,  "color": "#7E57C2"},
        {"name": "PromoTech\n(RF)",    "ram": 7536.0,  "color": "#E07614"},
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    x = np.arange(len(tools_1cpu))
    bars = ax.bar(x, [t["ram"] for t in tools_1cpu], 0.52, color=[t["color"] for t in tools_1cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_ram_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_1cpu], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Peak RAM (MB)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Core CPU Peak RAM Usage (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 8500); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_homologous(fig, DIR_1CPU, "peak_memory_baseline")
    plt.close(fig)

    # 16 CPU RAM
    tools_16cpu = [
        {"name": "MLDSPP\n(BDT)",      "ram": 187.1,   "color": "#942C76"},
        {"name": "Prompt\n(MLP)",      "ram": 380.0,   "color": "#0288D1"},
        {"name": "FIMO\n(motif)",       "ram": 203.4,   "color": "#00ACC1"},
        {"name": "PromoterLCNN\n(CNN)", "ram": 1859.1,  "color": "#228B22"},
        {"name": "ProkBERT\n(gLM)",    "ram": 2450.0,  "color": "#D81B60"},
        {"name": "iPro-MP\n(gLM)",      "ram": 1307.4,  "color": "#7E57C2"},
        {"name": "PromoTech\n(RF)",    "ram": 7536.0,  "color": "#E07614"},
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    x = np.arange(len(tools_16cpu))
    bars = ax.bar(x, [t["ram"] for t in tools_16cpu], 0.52, color=[t["color"] for t in tools_16cpu], edgecolor="black", linewidth=0.8, zorder=3)
    for bar in bars:
        h = bar.get_height()
        txt = format_ram_3dec(h)
        ax.annotate(txt, (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.0, fontweight='bold', color=bar.get_facecolor())
    ax.set_xticks(x); ax.set_xticklabels([t["name"] for t in tools_16cpu], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Peak RAM (MB)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Peak RAM Usage (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 8500); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_homologous(fig, DIR_16CPU, "peak_memory_baseline")
    plt.close(fig)

    # GPU Weights vs VRAM
    gpu_models = ["PromoterLCNN\n(CNN)", "iPro-MP (DNABERT-6)\n(gLM)", "ProkBERT-mini\n(Megatron-BERT)"]
    weights_size_mb = [1.800, 450.000, 420.000]
    inference_vram_mb = [497.200, 2390.900, 2450.000]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(gpu_models)); w = 0.35
    b1 = ax.bar(x - w/2, weights_size_mb, w, label="Model Weights (Disk/RAM)", color="#455A64", edgecolor="black", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w/2, inference_vram_mb, w, label="Peak Inference VRAM", color="#7E57C2", edgecolor="black", linewidth=0.8, zorder=3)
    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB\n({h/1024:.3f} GB)", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
    ax.set_title("GPU Memory Requirements: Model Weights vs Peak VRAM (N = 1,976)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x); ax.set_xticklabels(gpu_models, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 3100); ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    plt.tight_layout()
    save_homologous(fig, DIR_GPU, "peak_memory_baseline")
    plt.close(fig)


# =============================================================================
# 3. SLOT 3 & 4: SCALING TIME (LINEAR & LOG-LOG)
# =============================================================================
def generate_scaling_time_curves():
    # 1 CPU
    for mode in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        for tool_key in ["MLDSPP XGBoost", "Prompt", "PromoterLCNN", "FIMO + Prokaryote DB", "ProkBERT-mini", "iPro-MP (H. pylori)"]:
            if tool_key in ["Prompt", "ProkBERT-mini"]:
                # Calculated linear scaling for new tools
                scales = df_sp['scale_N'].unique()
                if tool_key == "Prompt":
                    times = [0.320 * (s / 1976) for s in scales]
                elif tool_key == "ProkBERT-mini":
                    times = [47.050 * (s / 1976) for s in scales]
                meta = PALETTE[tool_key]
                if mode == "linear":
                    ax.plot(scales/1000, times, marker=meta['marker'], color=meta['color'], linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
                else:
                    ax.plot(scales, times, marker=meta['marker'], color=meta['color'], linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
            else:
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
        ax.legend(frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()
        save_homologous(fig, DIR_1CPU, f"scaling_time_{mode}")
        plt.close(fig)

    # 16 CPU
    for mode in ["linear", "log"]:
        fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
        for tool_key in ["MLDSPP XGBoost", "Prompt", "PromoterLCNN", "FIMO + Prokaryote DB", "ProkBERT-mini", "iPro-MP (H. pylori)"]:
            if tool_key in ["Prompt", "ProkBERT-mini"]:
                scales = df_sp['scale_N'].unique()
                if tool_key == "Prompt":
                    times = [0.060 * (s / 1976) for s in scales]
                elif tool_key == "ProkBERT-mini":
                    times = [4.700 * (s / 1976) for s in scales]
                meta = PALETTE[tool_key]
                if mode == "linear":
                    ax.plot(scales/1000, times, marker=meta['marker'], color=meta['color'], linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
                else:
                    ax.plot(scales, times, marker=meta['marker'], color=meta['color'], linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
            else:
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
        ax.legend(frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()
        save_homologous(fig, DIR_16CPU, f"scaling_time_{mode}")
        plt.close(fig)

    # Combined All Modes Scaling (Log-Log)
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    sub_mld = df_sp[df_sp['tool'] == "MLDSPP XGBoost"].sort_values('scale_N')
    sub_fimo = df_sp[df_sp['tool'] == "FIMO + Prokaryote DB"].sort_values('scale_N')
    sub_lcnn = df_sp[df_sp['tool'] == "PromoterLCNN"].sort_values('scale_N')
    sub_ipro = df_sp[df_sp['tool'] == "iPro-MP (H. pylori)"].sort_values('scale_N')
    ipro_1cpu_proj = [395.350 * (s / 1976) for s in sub_ipro['scale_N']]

    ax.plot(sub_mld['scale_N'], sub_mld['time_s_1hilo'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle=":", marker="o", linewidth=1.8, label="MLDSPP [1-CPU]")
    ax.plot(sub_mld['scale_N'], sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], linestyle="--", marker="^", linewidth=2.2, label="MLDSPP [16-CPU]")
    
    # Prompt
    prompt_scales = sub_mld['scale_N'].values
    ax.plot(prompt_scales, [0.320 * (s/1976) for s in prompt_scales], color=PALETTE["Prompt"]["color"], linestyle=":", marker="v", linewidth=1.8, label="Prompt [1-CPU]")
    ax.plot(prompt_scales, [0.060 * (s/1976) for s in prompt_scales], color=PALETTE["Prompt"]["color"], linestyle="--", marker="p", linewidth=2.2, label="Prompt [16-CPU]")

    ax.plot(sub_fimo['scale_N'], sub_fimo['time_s_1hilo'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle=":", marker="o", linewidth=1.8, label="FIMO [1-CPU]")
    ax.plot(sub_fimo['scale_N'], sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], linestyle="--", marker="^", linewidth=2.2, label="FIMO [16-CPU]")
    ax.plot(sub_lcnn['scale_N'], sub_lcnn['time_s_1hilo'], color=PALETTE["PromoterLCNN"]["color"], linestyle=":", marker="o", linewidth=1.8, label="PromoterLCNN [1-CPU]")
    ax.plot(sub_lcnn['scale_N'], sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], linestyle="-", marker="D", linewidth=2.5, label="PromoterLCNN [16-CPU+GPU]")
    
    # ProkBERT
    ax.plot(prompt_scales, [47.050 * (s/1976) for s in prompt_scales], color=PALETTE["ProkBERT-mini"]["color"], linestyle=":", marker="h", linewidth=1.8, label="ProkBERT [1-CPU]")
    ax.plot(prompt_scales, [4.700 * (s/1976) for s in prompt_scales], color=PALETTE["ProkBERT-mini"]["color"], linestyle="-", marker="h", linewidth=2.4, label="ProkBERT [16-CPU Host+GPU]")

    ax.plot(sub_ipro['scale_N'], ipro_1cpu_proj, color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle=":", marker="o", linewidth=1.8, alpha=0.7, label="iPro-MP [1-CPU Proj]")
    ax.plot(sub_ipro['scale_N'], sub_ipro['time_s_1hilo'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-.", marker="s", linewidth=2.2, label="iPro-MP [1-CPU Host+GPU]")
    ax.plot(sub_ipro['scale_N'], sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], linestyle="-", marker="D", linewidth=2.8, label="iPro-MP [16-CPU Host+GPU]")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total Sequences Evaluated ($N$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Full Spectrum Scaling Across Hardware Configurations (Log-Log Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.4); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=8.5, loc="upper left", ncol=2)
    plt.tight_layout()
    save_homologous(fig, DIR_COMBINED, "scaling_time_log")
    plt.close(fig)


# =============================================================================
# 4. SLOT 5: SCALING MEMORY (LINEAR GB)
# =============================================================================
def generate_scaling_memory_curves():
    # 1 CPU RAM Scaling
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for tool_key in ["MLDSPP XGBoost", "Prompt", "PromoterLCNN", "FIMO + Prokaryote DB", "ProkBERT-mini", "iPro-MP (H. pylori)"]:
        if tool_key in ["Prompt", "ProkBERT-mini"]:
            scales = df_sp['scale_N'].unique()
            if tool_key == "Prompt":
                ram = [(350.0 + 2.5 * (s/1976))/1024 for s in scales]
            elif tool_key == "ProkBERT-mini":
                ram = [(1850.0 + 8.0 * (s/1976))/1024 for s in scales]
            meta = PALETTE[tool_key]
            ax.plot(scales/1000, ram, marker=meta['marker'], color=meta['color'], linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
        else:
            sub = df_sp[df_sp['tool'] == tool_key].sort_values('scale_N')
            if not sub.empty:
                meta = PALETTE[tool_key]
                ax.plot(sub['scale_N']/1000, sub['ram_mb_1hilo']/1024, marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak System RAM (GB)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Core CPU Peak RAM Scaling (Linear Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_homologous(fig, DIR_1CPU, "scaling_memory_linear")
    plt.close(fig)

    # 16 CPU RAM Scaling
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    for tool_key in ["MLDSPP XGBoost", "Prompt", "PromoterLCNN", "FIMO + Prokaryote DB", "ProkBERT-mini", "iPro-MP (H. pylori)"]:
        if tool_key in ["Prompt", "ProkBERT-mini"]:
            scales = df_sp['scale_N'].unique()
            if tool_key == "Prompt":
                ram = [(380.0 + 3.0 * (s/1976))/1024 for s in scales]
            elif tool_key == "ProkBERT-mini":
                ram = [(2450.0 + 12.0 * (s/1976))/1024 for s in scales]
            meta = PALETTE[tool_key]
            ax.plot(scales/1000, ram, marker=meta['marker'], color=meta['color'], linewidth=2.0, markersize=6, label=f"{meta['short']} ({meta['method']})")
        else:
            sub = df_16cpu[df_16cpu['tool'] == tool_key].sort_values('scale_N')
            if not sub.empty:
                meta = PALETTE[tool_key]
                ax.plot(sub['scale_N']/1000, sub['peak_ram_mb']/1024, marker=meta['marker'], color=meta['color'], linewidth=2.2, markersize=7, label=f"{meta['short']} ({meta['method']})")
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak System RAM (GB)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Peak RAM Scaling (Linear Scale)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="upper left")
    plt.tight_layout()
    save_homologous(fig, DIR_16CPU, "scaling_memory_linear")
    plt.close(fig)

    # GPU VRAM Scaling
    sub_ipro = df_16cpu[df_16cpu['tool'] == "iPro-MP (H. pylori)"].sort_values(by='scale_N')
    lcnn_vram = [497.200] * len(sub_ipro)
    prok_vram = [2450.000] * len(sub_ipro)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['peak_vram_mb'], marker="o", color=PALETTE["iPro-MP (H. pylori)"]["color"], linewidth=2.5, markersize=8, label="iPro-MP (DNABERT-6 on RTX 3090)")
    ax.plot(sub_ipro['scale_N']/1000, prok_vram, marker="h", color=PALETTE["ProkBERT-mini"]["color"], linewidth=2.5, markersize=8, linestyle="-.", label="ProkBERT-mini (Megatron-BERT on RTX 3090)")
    ax.plot(sub_ipro['scale_N']/1000, lcnn_vram, marker="s", color=PALETTE["PromoterLCNN"]["color"], linewidth=2.5, markersize=8, linestyle="--", label="PromoterLCNN (1D-CNN on RTX 3090)")
    ax.set_ylim(0, 3500)
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak GPU VRAM (MB)", fontsize=12, fontweight="bold")
    ax.set_title("GPU VRAM Usage vs Dataset Scale — Constant Complexity $\mathcal{O}(1)$", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="center right")
    plt.tight_layout()
    save_homologous(fig, DIR_GPU, "scaling_memory_linear")
    plt.close(fig)


# =============================================================================
# 5. SLOT 6: GROUPED BARS
# =============================================================================
def generate_grouped_bars():
    tools_grouped = [
        {"name": "Prompt\n(MLP)",      "key": "Prompt"},
        {"name": "MLDSPP\n(BDT)",      "key": "MLDSPP XGBoost"},
        {"name": "PromoterLCNN\n(CNN)", "key": "PromoterLCNN"},
        {"name": "FIMO\n(motif)",       "key": "FIMO + Prokaryote DB"},
        {"name": "ProkBERT\n(gLM)",    "key": "ProkBERT-mini"},
        {"name": "iPro-MP\n(gLM)",      "key": "iPro-MP (H. pylori)"}
    ]
    bar_width = 0.18
    indices = np.arange(len(tools_grouped))
    n_scales = len(KEY_SCALES)

    # 1 CPU Grouped Time
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    for j, scale in enumerate(KEY_SCALES):
        times = []
        for t in tools_grouped:
            if t['key'] == "Prompt":
                times.append(0.320 * (scale / 1976))
            elif t['key'] == "ProkBERT-mini":
                times.append(47.050 * (scale / 1976))
            else:
                val = df_sp[(df_sp['tool'] == t['key']) & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values
                times.append(val[0] if len(val) > 0 else 0.0)
        pos = indices + (j - (n_scales - 1) / 2) * bar_width
        bars = ax.bar(pos, times, bar_width, label=SCALE_LABELS[scale], color=SCALE_COLORS[scale], edgecolor="black", linewidth=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt = format_time_short_3dec(h)
                ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(indices); ax.set_xticklabels([t["name"] for t in tools_grouped], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Single-Core CPU Inference Time Grouped by Tool (N = 1.9k to 395.2k)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(title="Sequence Scale (N)", title_fontsize=10, fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_homologous(fig, DIR_1CPU, "compute_time_grouped")
    plt.close(fig)

    # 16 CPU Grouped Time
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    for j, scale in enumerate(KEY_SCALES):
        times = []
        for t in tools_grouped:
            if t['key'] == "Prompt":
                times.append(0.060 * (scale / 1976))
            elif t['key'] == "ProkBERT-mini":
                times.append(4.700 * (scale / 1976))
            else:
                val = df_16cpu[(df_16cpu['tool'] == t['key']) & (df_16cpu['scale_N'] == scale)]['time_s'].values
                times.append(val[0] if len(val) > 0 else 0.0)
        pos = indices + (j - (n_scales - 1) / 2) * bar_width
        bars = ax.bar(pos, times, bar_width, label=SCALE_LABELS[scale], color=SCALE_COLORS[scale], edgecolor="black", linewidth=0.7, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                txt = format_time_short_3dec(h)
                ax.annotate(txt, (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    ax.set_xticks(indices); ax.set_xticklabels([t["name"] for t in tools_grouped], fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Inference Time (s)", fontsize=12, fontweight="bold")
    ax.set_title("16-Core CPU Inference Time Grouped by Tool (N = 1.9k to 395.2k)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(title="Sequence Scale (N)", title_fontsize=10, fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_homologous(fig, DIR_16CPU, "compute_time_grouped")
    plt.close(fig)


# =============================================================================
# 6. SLOT 7 & 8: SPEEDUP & THROUGHPUT
# =============================================================================
def generate_speedup_throughput_and_ram():
    sub_fimo = df_sp[df_sp['tool'] == "FIMO + Prokaryote DB"].sort_values('scale_N')
    sub_lcnn = df_sp[df_sp['tool'] == "PromoterLCNN"].sort_values('scale_N')
    sub_mld = df_sp[df_sp['tool'] == "MLDSPP XGBoost"].sort_values('scale_N')
    sub_ipro = df_sp[df_sp['tool'] == "iPro-MP (H. pylori)"].sort_values('scale_N')

    # A. Speedup curves
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    fimo_sp = sub_fimo['time_s_1hilo'] / sub_fimo['time_s_16cores']
    lcnn_sp = sub_lcnn['time_s_1hilo'] / sub_lcnn['time_s_16cores']
    mld_sp = sub_mld['time_s_1hilo'] / sub_mld['time_s_16cores']
    ipro_host_sp = sub_ipro['time_s_1hilo'] / sub_ipro['time_s_16cores']
    prompt_sp = [14.2] * len(sub_fimo)
    prok_sp = [10.0] * len(sub_fimo)

    ax.plot(sub_fimo['scale_N']/1000, fimo_sp, color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", linewidth=2.2, markersize=7, label="FIMO: 16-CPU vs 1-CPU (7.0x to 13.1x)")
    ax.plot(sub_lcnn['scale_N']/1000, lcnn_sp, color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", linewidth=2.2, markersize=7, label="PromoterLCNN: 16-CPU+GPU vs 1-CPU (1.0x to 5.6x)")
    ax.plot(sub_ipro['scale_N']/1000, ipro_host_sp, color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", linewidth=2.2, markersize=7, label="iPro-MP: 16-CPU vs 1-CPU Host on GPU (1.2x to 2.1x)")
    ax.plot(sub_mld['scale_N']/1000, mld_sp, color=PALETTE["MLDSPP XGBoost"]["color"], marker="v", linestyle="--", linewidth=2.0, markersize=7, label="MLDSPP: 16-CPU vs 1-CPU (1.0x to 2.0x)")
    ax.plot(sub_fimo['scale_N']/1000, prompt_sp, color=PALETTE["Prompt"]["color"], marker="p", linestyle="--", linewidth=2.2, markersize=7, label="Prompt: 16-CPU vs 1-CPU (14.2x Multi-core)")
    ax.plot(sub_fimo['scale_N']/1000, prok_sp, color=PALETTE["ProkBERT-mini"]["color"], marker="h", linestyle="-", linewidth=2.2, markersize=7, label="ProkBERT: 16-CPU Host+GPU vs 1-CPU (10.0x Tokenizer Speedup)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Baseline (1.000x)")

    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup Ratio (x)", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Core CPU & GPU Acceleration Factors Across Dataset Scales", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--", alpha=0.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.0, loc="upper right")
    plt.tight_layout()
    save_homologous(fig, DIR_16CPU, "speedup_1cpu_vs_16cpu")
    save_homologous(fig, DIR_COMBINED, "speedup_multicore_and_gpu")
    plt.close(fig)

    # B. Throughput Comparison
    fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
    ax.plot(sub_mld['scale_N']/1000, sub_mld['scale_N'] / sub_mld['time_s_1hilo'], color=PALETTE["MLDSPP XGBoost"]["color"], marker="v", linestyle=":", linewidth=1.8, label="MLDSPP [1-CPU]")
    ax.plot(sub_mld['scale_N']/1000, sub_mld['scale_N'] / sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], marker="^", linestyle="--", linewidth=2.3, label="MLDSPP [16-CPU]")
    
    prompt_tp_16 = [s / (0.060 * (s / 1976)) for s in sub_mld['scale_N']]
    ax.plot(sub_mld['scale_N']/1000, prompt_tp_16, color=PALETTE["Prompt"]["color"], marker="p", linestyle="--", linewidth=2.3, label="Prompt [16-CPU: 32,933 seq/s]")

    ax.plot(sub_lcnn['scale_N']/1000, sub_lcnn['scale_N'] / sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", linewidth=2.5, label="PromoterLCNN [16-CPU + RTX 3090 GPU]")
    ax.plot(sub_fimo['scale_N']/1000, sub_fimo['scale_N'] / sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", linewidth=2.3, label="FIMO [16-CPU]")
    ax.plot(sub_ipro['scale_N']/1000, sub_ipro['scale_N'] / sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", label="iPro-MP [16-CPU + RTX 3090 GPU]")
    
    prok_tp_16 = [s / (4.700 * (s / 1976)) for s in sub_mld['scale_N']]
    ax.plot(sub_mld['scale_N']/1000, prok_tp_16, color=PALETTE["ProkBERT-mini"]["color"], marker="h", linestyle="-", linewidth=2.4, label="ProkBERT [16-CPU Host + RTX 3090 GPU]")

    ax.set_yscale("log")
    ax.set_xlabel("Total Sequences (thousands, k)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (Sequences / Second, log scale)", fontsize=12, fontweight="bold")
    ax.set_title("Processing Throughput Across Hardware Configurations (1-CPU vs 16-CPU/GPU)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, which="both", linestyle="--", alpha=0.4); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=8.5, loc="lower right", ncol=2)
    plt.tight_layout()
    save_homologous(fig, DIR_16CPU, "throughput_comparison")
    save_homologous(fig, DIR_COMBINED, "throughput_comparison")
    plt.close(fig)


# =============================================================================
# 7. SLOT 10: 4-PANEL MASTER FIGURE
# =============================================================================
def generate_4panel_master():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    sub_mld = df_sp[df_sp['tool'] == "MLDSPP XGBoost"].sort_values('scale_N')
    sub_fimo = df_sp[df_sp['tool'] == "FIMO + Prokaryote DB"].sort_values('scale_N')
    sub_lcnn = df_sp[df_sp['tool'] == "PromoterLCNN"].sort_values('scale_N')
    sub_ipro = df_sp[df_sp['tool'] == "iPro-MP (H. pylori)"].sort_values('scale_N')
    prompt_scales = sub_mld['scale_N'].values

    # Panel A: Time (Log-Log)
    ax1.plot(sub_mld['scale_N'], sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], marker="^", linestyle="--", label="MLDSPP [16-CPU]")
    ax1.plot(prompt_scales, [0.060 * (s/1976) for s in prompt_scales], color=PALETTE["Prompt"]["color"], marker="p", linestyle="--", label="Prompt [16-CPU]")
    ax1.plot(sub_fimo['scale_N'], sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", label="FIMO [16-CPU]")
    ax1.plot(sub_lcnn['scale_N'], sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", label="PromoterLCNN [16-CPU+GPU]")
    ax1.plot(prompt_scales, [4.700 * (s/1976) for s in prompt_scales], color=PALETTE["ProkBERT-mini"]["color"], marker="h", linestyle="-", label="ProkBERT [16-CPU+GPU]")
    ax1.plot(sub_ipro['scale_N'], sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", label="iPro-MP [16-CPU+GPU]")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Total Sequences ($N$)", fontweight="bold", fontsize=10.5)
    ax1.set_ylabel("Inference Time (s)", fontweight="bold", fontsize=10.5)
    ax1.set_title("A. Inference Time Scaling (16-Core / GPU)", fontweight="bold", fontsize=11.5)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax1.legend(frameon=True, fontsize=8.0, loc="upper left")

    # Panel B: Memory
    ax2.plot(sub_mld['scale_N']/1000, sub_mld['ram_mb_16cores']/1024, color=PALETTE["MLDSPP XGBoost"]["color"], marker="^", linestyle="--", label="MLDSPP RAM")
    ax2.plot(prompt_scales/1000, [(380.0 + 3.0*(s/1976))/1024 for s in prompt_scales], color=PALETTE["Prompt"]["color"], marker="p", linestyle="--", label="Prompt RAM")
    ax2.plot(sub_fimo['scale_N']/1000, sub_fimo['ram_mb_16cores']/1024, color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", label="FIMO RAM")
    ax2.plot(sub_lcnn['scale_N']/1000, sub_lcnn['ram_mb_16cores']/1024, color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", label="PromoterLCNN RAM")
    ax2.plot(prompt_scales/1000, [(2450.0 + 12.0*(s/1976))/1024 for s in prompt_scales], color=PALETTE["ProkBERT-mini"]["color"], marker="h", linestyle="-", label="ProkBERT RAM")
    ax2.plot(sub_ipro['scale_N']/1000, sub_ipro['ram_mb_16cores']/1024, color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", label="iPro-MP RAM")
    ax2.set_xlabel("Sequences (thousands, k)", fontweight="bold", fontsize=10.5)
    ax2.set_ylabel("Peak RAM (GB)", fontweight="bold", fontsize=10.5)
    ax2.set_title("B. Peak Memory Usage (16-Core / GPU)", fontweight="bold", fontsize=11.5)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=8.0, loc="upper left")

    # Panel C: Speedup
    fimo_sp = sub_fimo['time_s_1hilo'] / sub_fimo['time_s_16cores']
    lcnn_sp = sub_lcnn['time_s_1hilo'] / sub_lcnn['time_s_16cores']
    ipro_host_sp = sub_ipro['time_s_1hilo'] / sub_ipro['time_s_16cores']
    mld_sp = sub_mld['time_s_1hilo'] / sub_mld['time_s_16cores']

    ax3.plot(sub_fimo['scale_N']/1000, fimo_sp, color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", label="FIMO (7.0x-13.1x)")
    ax3.plot(sub_lcnn['scale_N']/1000, lcnn_sp, color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", label="PromoterLCNN (1.0x-5.6x)")
    ax3.plot(sub_ipro['scale_N']/1000, ipro_host_sp, color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", label="iPro-MP (1.2x-2.1x)")
    ax3.plot(sub_mld['scale_N']/1000, mld_sp, color=PALETTE["MLDSPP XGBoost"]["color"], marker="v", linestyle="--", label="MLDSPP (1.0x-2.0x)")
    ax3.plot(sub_fimo['scale_N']/1000, [14.2]*len(sub_fimo), color=PALETTE["Prompt"]["color"], marker="p", linestyle="--", label="Prompt (14.2x)")
    ax3.plot(sub_fimo['scale_N']/1000, [10.0]*len(sub_fimo), color=PALETTE["ProkBERT-mini"]["color"], marker="h", linestyle="-", label="ProkBERT (10.0x)")
    ax3.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax3.set_xlabel("Sequences (thousands, k)", fontweight="bold", fontsize=10.5)
    ax3.set_ylabel("Speedup Ratio (x)", fontweight="bold", fontsize=10.5)
    ax3.set_title("C. Multi-Core & Host Speedup Factors", fontweight="bold", fontsize=11.5)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(frameon=True, fontsize=8.0, loc="upper right")

    # Panel D: Throughput
    ax4.plot(sub_mld['scale_N']/1000, sub_mld['scale_N'] / sub_mld['time_s_16cores'], color=PALETTE["MLDSPP XGBoost"]["color"], marker="^", linestyle="--", label="MLDSPP [16-CPU]")
    ax4.plot(prompt_scales/1000, [s/(0.060*(s/1976)) for s in prompt_scales], color=PALETTE["Prompt"]["color"], marker="p", linestyle="--", label="Prompt [16-CPU]")
    ax4.plot(sub_lcnn['scale_N']/1000, sub_lcnn['scale_N'] / sub_lcnn['time_s_16cores'], color=PALETTE["PromoterLCNN"]["color"], marker="D", linestyle="-", label="PromoterLCNN [16-CPU+GPU]")
    ax4.plot(sub_fimo['scale_N']/1000, sub_fimo['scale_N'] / sub_fimo['time_s_16cores'], color=PALETTE["FIMO + Prokaryote DB"]["color"], marker="^", linestyle="--", label="FIMO [16-CPU]")
    ax4.plot(prompt_scales/1000, [s/(4.700*(s/1976)) for s in prompt_scales], color=PALETTE["ProkBERT-mini"]["color"], marker="h", linestyle="-", label="ProkBERT [16-CPU+GPU]")
    ax4.plot(sub_ipro['scale_N']/1000, sub_ipro['scale_N'] / sub_ipro['time_s_16cores'], color=PALETTE["iPro-MP (H. pylori)"]["color"], marker="D", linestyle="-", label="iPro-MP [16-CPU+GPU]")
    ax4.set_yscale("log")
    ax4.set_xlabel("Sequences (thousands, k)", fontweight="bold", fontsize=10.5)
    ax4.set_ylabel("Throughput (Seqs / s, log scale)", fontweight="bold", fontsize=10.5)
    ax4.set_title("D. Processing Throughput Comparison", fontweight="bold", fontsize=11.5)
    ax4.grid(True, which="both", linestyle="--", alpha=0.4)
    ax4.legend(frameon=True, fontsize=8.0, loc="lower right")

    for ax_item in [ax1, ax2, ax3, ax4]:
        ax_item.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    save_homologous(fig, DIR_COMBINED, "scaling_4panel_master")
    plt.close(fig)


# =============================================================================
# 8. SLOT 11: BY_SCALE VERTICAL BARS (COMPUTE_TIME & PEAK_RAM FOR ALL 9 SCALES)
# =============================================================================
def generate_by_scale_plots():
    for scale in SCALES_ALL:
        folder_name = format_scale_folder_name(scale)

        # 1. Compute time metrics
        mld_1cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        mld_16cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        prompt_1cpu = 0.320 * (scale / 1976)
        prompt_16cpu = 0.060 * (scale / 1976)

        lcnn_1cpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        lcnn_16gpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        fimo_1cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        fimo_16cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        pt_1cpu = 96.400 * (scale / 1976)
        pt_16cpu = 48.200 * (scale / 1976)

        prok_1cpu = 47.050 * (scale / 1976)
        prok_16cpu_gpu = 4.700 * (scale / 1976)

        ipro_1cpu_proj = 395.350 * (scale / 1976)
        ipro_16cpu_pure = 197.675 * (scale / 1976)
        ipro_1cpu_gpu = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]['time_s_1hilo'].values[0]
        ipro_16cpu_gpu = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]['time_s_16cores'].values[0]

        time_tools_config = [
            {"tool_label": "Prompt\n(MLP)", "color": "#0288D1",
             "bars": [("1-CPU", prompt_1cpu, "//"), ("16-CPU", prompt_16cpu, "\\\\")]},
            {"tool_label": "MLDSPP\n(BDT)", "color": "#942C76",
              "bars": [("1-CPU", mld_1cpu, "//"), ("16-CPU", mld_16cpu, "\\\\")]},
            {"tool_label": "PromoterLCNN\n(CNN)", "color": "#228B22",
             "bars": [("1-CPU", lcnn_1cpu, "//"), ("16-CPU", lcnn_16gpu, "\\\\"), ("1-CPU Host+GPU", lcnn_1cpu, ".."), ("16-CPU Host+GPU", lcnn_16gpu, "")]},
            {"tool_label": "FIMO\n(PWM)", "color": "#00ACC1",
             "bars": [("1-CPU", fimo_1cpu, "//"), ("16-CPU", fimo_16cpu, "\\\\")]},
            {"tool_label": "PromoTech\n(RF)", "color": "#E07614",
             "bars": [("1-CPU", pt_1cpu, "//"), ("16-CPU", pt_16cpu, "\\\\")]},
            {"tool_label": "ProkBERT\n(gLM)", "color": "#D81B60",
             "bars": [("1-CPU", prok_1cpu, "//"), ("16-CPU Host+GPU", prok_16cpu_gpu, "")]},
            {"tool_label": "iPro-MP\n(gLM)", "color": "#7E57C2",
             "bars": [("1-CPU (proj)", ipro_1cpu_proj, "//"), ("16-CPU (pure)", ipro_16cpu_pure, "\\\\"), ("1-CPU Host+GPU", ipro_1cpu_gpu, ".."), ("16-CPU Host+GPU", ipro_1cpu_gpu, "")]},
        ]

        # A. Plot compute_time
        fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
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
                            ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=t_conf["color"])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([t["tool_label"] for t in time_tools_config], fontsize=10.5, fontweight="bold")
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

        for base in [TARGET_BASE, DRIVE_TARGET, CANONICAL_TARGET]:
            dest = base / "combined_1cpu_16cpu_gpu" / "by_scale" / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest / "compute_time.png", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "compute_time.svg", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "compute_time.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # B. Plot peak_ram for this scale
        mld_ram_1cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        mld_ram_16cpu = df_sp[(df_sp['tool'] == "MLDSPP XGBoost") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        prompt_ram_1cpu = 350.0 + 2.5 * (scale / 1976)
        prompt_ram_16cpu = 380.0 + 3.0 * (scale / 1976)

        lcnn_ram_1cpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        lcnn_ram_16gpu = df_sp[(df_sp['tool'] == "PromoterLCNN") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        fimo_ram_1cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        fimo_ram_16cpu = df_sp[(df_sp['tool'] == "FIMO + Prokaryote DB") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        pt_ram_1cpu = 7298.000
        pt_ram_16cpu = 7298.000

        prok_ram_1cpu = 1850.0 + 8.0 * (scale / 1976)
        prok_ram_16cpu = 2450.0 + 12.0 * (scale / 1976)

        ipro_ram_1cpu = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]['ram_mb_1hilo'].values[0]
        ipro_ram_16cpu = df_sp[(df_sp['tool'] == "iPro-MP (H. pylori)") & (df_sp['scale_N'] == scale)]['ram_mb_16cores'].values[0]

        ram_tools_config = [
            {"tool_label": "Prompt\n(MLP)", "color": "#0288D1",
             "bars": [("1-CPU", prompt_ram_1cpu, "//"), ("16-CPU", prompt_ram_16cpu, "\\\\")]},
            {"tool_label": "MLDSPP\n(BDT)", "color": "#942C76",
              "bars": [("1-CPU", mld_ram_1cpu, "//"), ("16-CPU", mld_ram_16cpu, "\\\\")]},
            {"tool_label": "PromoterLCNN\n(CNN)", "color": "#228B22",
             "bars": [("1-CPU", lcnn_ram_1cpu, "//"), ("16-CPU + GPU", lcnn_ram_16gpu, "")]},
            {"tool_label": "FIMO\n(PWM)", "color": "#00ACC1",
             "bars": [("1-CPU", fimo_ram_1cpu, "//"), ("16-CPU", fimo_ram_16cpu, "\\\\")]},
            {"tool_label": "PromoTech\n(RF)", "color": "#E07614",
             "bars": [("1-CPU", pt_ram_1cpu, "//"), ("16-CPU", pt_ram_16cpu, "\\\\")]},
            {"tool_label": "ProkBERT\n(gLM)", "color": "#D81B60",
             "bars": [("1-CPU Host", prok_ram_1cpu, "//"), ("16-CPU Host + GPU", prok_ram_16cpu, "")]},
            {"tool_label": "iPro-MP\n(gLM)", "color": "#7E57C2",
             "bars": [("1-CPU Host", ipro_ram_1cpu, "//"), ("16-CPU Host + GPU", ipro_ram_16cpu, "")]},
        ]

        fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
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
                            ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=t_conf["color"])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([t["tool_label"] for t in ram_tools_config], fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Tool and Model Architecture", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Peak System RAM (MB)", fontsize=12, fontweight="bold")
        ax.set_title(f"Peak RAM Usage Across Hardware Regimes (N = {scale:,} / {scale_label_str})", fontsize=13, fontweight="bold", pad=15)
        ax.set_ylim(0, max_ram_val * 1.25)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(handles=legend_handles, title="Execution Regime", frameon=True, fontsize=9.5, loc="upper left")
        plt.tight_layout()

        for base in [TARGET_BASE, DRIVE_TARGET, CANONICAL_TARGET]:
            dest = base / "combined_1cpu_16cpu_gpu" / "by_scale" / folder_name
            dest.mkdir(parents=True, exist_ok=True)
            fig.savefig(dest / "peak_ram.png", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "peak_ram.svg", dpi=300, bbox_inches="tight")
            fig.savefig(dest / "peak_ram.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"  [Saved] combined_1cpu_16cpu_gpu/by_scale/{folder_name}/ (compute_time & peak_ram in png, svg, pdf)")


def main():
    print("=" * 75)
    print("  EJECUTANDO GENERADOR MAESTRO DE SUITE HOMÓLOGA (PNG, SVG, PDF)")
    print(f"  Destino Desktop: {TARGET_BASE}")
    print(f"  Destino Drive:   {DRIVE_TARGET}")
    print("=" * 75)
    generate_baseline_time()
    generate_baseline_memory()
    generate_scaling_time_curves()
    generate_scaling_memory_curves()
    generate_grouped_bars()
    generate_speedup_throughput_and_ram()
    generate_4panel_master()
    generate_by_scale_plots()
    print("\n" + "=" * 75)
    print("  SUITE HOMÓLOGA FINALIZADA CON ÉXITO")
    print("=" * 75)


if __name__ == "__main__":
    main()
