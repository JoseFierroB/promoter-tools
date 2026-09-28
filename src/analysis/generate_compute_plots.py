#!/usr/bin/env python3
"""Compute plots: per-tool time/RAM bars for one run.

Usage:
    python src/analysis/generate_compute_plots.py <run_dir>

Reads <run_dir>/2_resources/resource_metrics.tsv and writes title-free
<run_dir>/2_resources/{compute_time,peak_ram}.{png,pdf}
(one bar per tool = within-run comparison).
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Tool palette — primary bench_labels.PALETTE (single source of truth),
# mapped by legacy registry keys (BDT→RF→CNN→NN→gLM→motif order kept).
try:
    from bench_labels import TOOL_COLORS as _CANON_COLORS  # type: ignore
except Exception:
    try:
        from src.analysis.bench_labels import TOOL_COLORS as _CANON_COLORS  # type: ignore
    except Exception:
        _CANON_COLORS = {}

def _c(display: str, fallback: str) -> str:
    return _CANON_COLORS.get(display, fallback)

PALETTE = {
    "MLDSPP XGBoost":                {"short": "MLDSPP",      "method": "BDT",   "color": _c("MLDSPP 0% [BDT]", "#F81D54"), "marker": "v"},
    "MLDSPP XGBoost (75% spn)":      {"short": "MLDSPP 75%",  "method": "BDT",   "color": _c("MLDSPP 75% [BDT]", "#55001C"), "marker": "^"},
    "PromoterLCNN":                  {"short": "PromoterLCNN","method": "CNN",   "color": _c("PromoterLCNN [CNN]", "#973C04"), "marker": "s"},
    "FIMO + Prokaryote DB":          {"short": "FIMO",        "method": "PWMs",  "color": _c("FIMO ProkDB [PWMs]", "#002616"), "marker": "D"},
    "PromoTech RF-HOT (PG Max)":     {"short": "PromoTech",   "method": "RF",    "color": _c("PromoTech RF-HOT [RF]", "#F1A52B"), "marker": "P"},
    "iPro-MP (H. pylori)":           {"short": "iPro-MP",     "method": "gLM",   "color": _c("iPro-MP [gLM]", "#BA9CEE"), "marker": "o"},
    "MEME Suite (STREME+FIMO)":      {"short": "MEME",        "method": "motif", "color": _c("STREME+FIMO [motif]", "#18974C"), "marker": "o"},
    "Prompt MLP (B. subtilis 168)":  {"short": "Prompt",      "method": "NN",    "color": _c("prompt [NN]", "#0060E6"), "marker": "p"},
    "ProkBERT-mini (NeuralBioInfo)": {"short": "ProkBERT-mini", "method": "gLM", "color": _c("ProkBERT-mini [gLM]", "#2D004D"), "marker": "h"},
}


def _short_label(tool_name: str) -> str:
    meta = PALETTE.get(tool_name)
    if meta:
        return f"{meta['short']}\n({meta['method']})"
    fallback = {
        "MEME Suite (STREME+FIMO)": "MEME\n(STREME+FIMO)",
        "Prompt MLP (B. subtilis 168)": "Prompt\n(MLP)",
        "ProkBERT-mini (NeuralBioInfo)": "ProkBERT-mini\n(gLM)",
    }
    return fallback.get(tool_name, tool_name.replace(" ", "\n", 1))


def _format_run_value(column: str, value: float) -> str:
    """Readable annotation for per-run resource bars."""
    if column == "wall_seconds":
        return f"{value:.2f} s" if value < 10 else f"{value:.1f} s"
    if column == "peak_ram_mb":
        if value >= 1024:
            return f"{value:,.1f} MB\n({value / 1024:.2f} GB)"
        return f"{value:,.1f} MB"
    return f"{value:,.3g}"


def main_run(run_dir: Path):
    """Per-run compute plots from the run's resource_metrics.tsv."""
    try:
        from run_layout import RunLayout
    except ImportError:
        from src.analysis.run_layout import RunLayout
    layout = RunLayout.resolve(run_dir)
    run_dir = layout.root
    metrics = layout.find_resource_metrics()
    if metrics is None:
        print(f"ERROR: no resource_metrics.tsv in {run_dir}")
        raise SystemExit(2)
    df = pd.read_csv(metrics, sep="\t")
    df = df[df["success"] == True] if "success" in df.columns else df
    if df.empty:
        print("  [compute] no successful tool rows, nothing to plot")
        return
    out = layout.resources
    out.mkdir(parents=True, exist_ok=True)
    layout.link_resource_table(metrics)
    # throughput not needed for single-run capsule (only for scaling suite)
    specs = [
        ("wall_seconds", "Execution Time (s)"),
        ("peak_ram_mb", "Peak RAM (MB)"),
    ]
    for col, ylabel in specs:
        if col not in df.columns:
            continue
        sub = df[pd.to_numeric(df[col], errors="coerce").notna()]
        if sub.empty:
            continue
        # Primary TOOL_ORDER (BDT→RF→CNN→NN→gLM→motif) so bars line up
        # across configurations and match ROC/benchmark legends.
        try:
            from bench_labels import TOOL_ORDER as _TO
            _legacy_to_primary = {
                "MLDSPP XGBoost": "MLDSPP 0% [BDT]",
                "MLDSPP XGBoost (75% spn)": "MLDSPP 75% [BDT]",
                "PromoTech RF-HOT (PG Max)": "PromoTech RF-HOT [RF]",
                "PromoterLCNN": "PromoterLCNN [CNN]",
                "Prompt MLP (B. subtilis 168)": "prompt [NN]",
                "ProkBERT-mini (NeuralBioInfo)": "ProkBERT-mini [gLM]",
                "iPro-MP (H. pylori)": "iPro-MP [gLM]",
                "FIMO + Prokaryote DB": "FIMO ProkDB [PWMs]",
                "MEME Suite (STREME+FIMO)": "STREME+FIMO [motif]",
            }
            def _tool_key(t):
                c = _legacy_to_primary.get(t, t)
                try:
                    return _TO.index(c)
                except ValueError:
                    return 999
            sub = sub.assign(_tool_sort=sub["tool"].map(_tool_key))
            sub = sub.sort_values(["_tool_sort", "tool"], kind="stable")
        except Exception:
            sub = sub.assign(_tool_sort=sub["tool"].astype(str).str.casefold())
            sub = sub.sort_values(["_tool_sort", "tool"], kind="stable")
        labels = [_short_label(t) for t in sub["tool"]]
        colors = [PALETTE.get(t, {}).get("color", "#455A64") for t in sub["tool"]]
        vals = sub[col].astype(float).tolist()
        # Wide, fixed layout prevents long tool names from colliding when all
        # tools are present in a run (especially MEME and ProkBERT-mini).
        fig, ax = plt.subplots(figsize=(max(13.5, len(sub) * 1.65), 7), dpi=300)
        bars = ax.bar(range(len(sub)), vals, color=colors, edgecolor="black", linewidth=0.8)
        for b, v in zip(bars, vals):
            ax.annotate(_format_run_value(col, v),
                        (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold", clip_on=False)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold",
                           rotation=22, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
        ymax = max(vals) if vals else 1
        ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.90)
        stem = {"wall_seconds": "compute_time", "peak_ram_mb": "peak_ram"}[col]
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [compute] saved 2_resources/{stem}.png/.pdf")


if __name__ == "__main__":
    import sys as _sys
    if _sys.argv[1:] in (["-h"], ["--help"]):
        print(__doc__)
    elif len(_sys.argv) == 2 and not _sys.argv[1].startswith("-"):
        main_run(Path(_sys.argv[1]))
    else:
        print(__doc__)
        _sys.exit(2)
