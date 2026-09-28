#!/usr/bin/env python3
"""GPU-model comparison: RTX 3090 vs RTX 5090 on the same dataset/regime.

Reads wall time + peak VRAM for DL tools from two CLI runs'
`2_resources/resource_metrics.tsv` (same host, same threads, same inputs;
only the GPU differs) and writes THREE SEPARATE figures:
  1. gpu_time_{tag}.{png,pdf}    — wall seconds per DL tool (3090 vs 5090)
  2. gpu_speedup_{tag}.{png,pdf} — x5090/3090 per DL tool
  3. gpu_vram_{tag}.{png,pdf}    — peak VRAM (MiB) per DL tool

Only GPU tools (gpu_available=True in the metrics) are plotted.
Regimes are labelled by the real `gpu_name` column (not `configuration`,
which is identical in both runs).

Usage:
    pixi run python src/analysis/plot_gpu_compare.py \
      --a output/260915_runs/4_d39v_tigr4_high_02/1cpu-gpu \
      --b output/260917_runs/4_d39v_tigr4_high_sigma/1cpu-gpu \
      --dataset "D39V+TIGR4 high" \
      -o output/comparisons/2609XX_gpu_3090_vs_5090
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

GPU_COLORS = {"3090": "#6A1B9A", "5090": "#00897B"}
GPU_HATCH = {"3090": "//", "5090": ".."}


def short_gpu(name: str) -> str:
    n = str(name)
    if "3090" in n:
        return "RTX 3090"
    if "5090" in n:
        return "RTX 5090"
    return n


def load_dl_metrics(run_dir: Path):
    """Return (gpu_label, {display_tool: {wall, vram, auc?}}) for GPU tools only."""
    res_files = sorted((run_dir / "2_resources").glob("resource_metrics*.tsv"))
    if not res_files:
        raise SystemExit(f"ERROR: no resource_metrics.tsv in {run_dir}/2_resources")
    df = pd.read_csv(res_files[0], sep="\t")
    gpu_rows = df[df["gpu_available"].astype(str).str.lower().isin(["true", "1"])]
    if gpu_rows.empty:
        raise SystemExit(f"ERROR: no GPU-tool rows in {res_files[0]}")
    gpu_label = short_gpu(gpu_rows["gpu_name"].iloc[0])
    out = {}
    for _, r in gpu_rows.iterrows():
        out[str(r["tool"])] = {"wall": float(r["wall_seconds"]),
                               "vram": float(r["peak_vram_mb"])}
    return gpu_label, out


def display_of(registry_tool: str) -> str:
    mapping = {"PromoterLCNN": "PromoterLCNN [CNN]",
               "ProkBERT-mini (NeuralBioInfo)": "ProkBERT-mini [gLM]",
               "iPro-MP (H. pylori)": "iPro-MP [gLM]"}
    return mapping.get(registry_tool, registry_tool)


def style_axes(ax, ylabel: str, title: str):
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(title="GPU", frameon=True, fontsize=10,
                    loc="upper left", bbox_to_anchor=(1.02, 1.0))
    leg.get_title().set_fontweight("bold")


def grouped(ax, tools, avals, bvals, alabel, blabel, ylabel, title, fmt):
    """Two bars per tool with capped annotations (no overlap)."""
    x = np.arange(len(tools))
    width = 0.34
    for i, (vals, glabel, gkey) in enumerate(((avals, alabel, "3090"),
                                              (bvals, blabel, "5090"))):
        off = (i - 0.5) * width
        bars = ax.bar(x + off, vals, width, edgecolor="black", linewidth=0.9,
                      color=GPU_COLORS[gkey], hatch=GPU_HATCH[gkey],
                      label=glabel, zorder=3)
        for b, v in zip(bars, vals):
            ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="black",
                        clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10, fontweight="bold",
                       rotation=15, ha="right")
    vmax = max(list(avals) + list(bvals)) if (avals or bvals) else 1.0
    ax.set_ylim(0, vmax * 1.30)  # headroom for value labels
    style_axes(ax, ylabel, title)


def save(fig, path: Path):
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="RTX 3090 vs 5090 DL-tool comparison.")
    ap.add_argument("--a", required=True, help="run dir (e.g. 3090)")
    ap.add_argument("--b", required=True, help="run dir (e.g. 5090)")
    ap.add_argument("--dataset", default="")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    alabel, am = load_dl_metrics(Path(args.a))
    blabel, bm = load_dl_metrics(Path(args.b))
    common = [t for t in am if t in bm]
    if not common:
        raise SystemExit("ERROR: no common GPU tools between runs")
    order = {"PromoterLCNN": 0, "ProkBERT-mini (NeuralBioInfo)": 1,
             "iPro-MP (H. pylori)": 2}
    common.sort(key=lambda t: order.get(t, 99))
    tools = [display_of(t) for t in common]
    awall = [am[t]["wall"] for t in common]
    bwall = [bm[t]["wall"] for t in common]
    avram = [am[t]["vram"] for t in common]
    bvram = [bm[t]["vram"] for t in common]
    speedup = [a / b if b else float("nan") for a, b in zip(awall, bwall)]

    out = Path(args.output)
    resd, tab = out / "2_resources", out / "3_tables"
    resd.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    tag = "dl3"
    ds = args.dataset or "dataset"
    n_tag = "N=3,465"
    host = "1-CPU host"

    rows = []
    for t, d in zip(tools, common):
        rows.append({"tool": t,
                     f"wall_{alabel}": round(am[d]["wall"], 3),
                     f"wall_{blabel}": round(bm[d]["wall"], 3),
                     "speedup_5090_vs_3090": round(am[d]["wall"] / bm[d]["wall"], 3)
                     if bm[d]["wall"] else float("nan"),
                     f"vram_{alabel}_MiB": round(am[d]["vram"], 1),
                     f"vram_{blabel}_MiB": round(bm[d]["vram"], 1)})
    pd.DataFrame(rows).to_csv(tab / "gpu_compare.tsv", sep="\t", index=False)

    # 1. wall time
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    grouped(ax, tools, awall, bwall, alabel, blabel,
            f"Wall time (s, {n_tag})",
            f"DL wall time — {ds} ({n_tag}; {alabel} vs {blabel}, {host})",
            lambda v: f"{v:.2f}s" if v < 100 else f"{v:.0f}s\n({v/60:.1f}m)")
    plt.tight_layout()
    save(fig, resd / f"gpu_time_{tag}")
    print(f"  [SAVED] 2_resources/gpu_time_{tag}.png/.pdf")

    # 2. speedup
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    x = np.arange(len(tools))
    bars = ax.bar(x, speedup, 0.5, edgecolor="black", linewidth=0.9,
                  color="#37474F", zorder=3)
    for b, v in zip(bars, speedup):
        ax.annotate(f"{v:.2f}x", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", clip_on=False)
    ax.axhline(1.0, color="grey", lw=1.2, ls="--", label="parity")
    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10, fontweight="bold",
                       rotation=15, ha="right")
    ax.set_ylim(0, max(speedup) * 1.30)
    style_axes(ax, "Speedup 5090 vs 3090 (x)",
               f"DL speedup — {ds} ({n_tag}; {blabel} vs {alabel}, {host})")
    plt.tight_layout()
    save(fig, resd / f"gpu_speedup_{tag}")
    print(f"  [SAVED] 2_resources/gpu_speedup_{tag}.png/.pdf")

    # 3. peak VRAM
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    grouped(ax, tools, avram, bvram, alabel, blabel,
            "Peak VRAM (MiB, nvidia-smi global readout)",
            f"DL peak VRAM — {ds} ({n_tag}; {alabel} vs {blabel}, {host})",
            lambda v: f"{v:.0f} MB")
    plt.tight_layout()
    save(fig, resd / f"gpu_vram_{tag}")
    print(f"  [SAVED] 2_resources/gpu_vram_{tag}.png/.pdf")

    import json as _json
    (out / "MANIFEST.json").write_text(_json.dumps(
        {"type": "gpu_comparison", "dataset": ds,
         "sources": [str(Path(args.a).resolve()), str(Path(args.b).resolve())],
         "tools": tools}, indent=2))
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
