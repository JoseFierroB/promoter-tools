#!/usr/bin/env python3
"""Compare compute across CLI runs (same dataset, different configurations).

Compares wall time, peak RAM, throughput and relative speedup of runs
produced by `src/cli.py run`, e.g. 1cpu vs 16cpu vs 1cpu-gpu vs 16cpu-gpu
on one dataset. Inference (AUC) is intentionally NOT compared: identical
inputs give identical scores by construction. Missing cells (tool not run
in some run, or run without resource metrics) are skipped, never crash.

Usage:
    python src/analysis/compare_runs.py <run_dir1> [<run_dir2> ...]
        [-o output/<YYMMDD>_comparison]

Output (inside the comparison dir):
  2_resources/compare_{time,ram,throughput,speedup}.{png,svg,pdf}
  3_tables/resources_compare.tsv (incl. speedup_vs_slowest per tool)
  CONTENTS.md
"""
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# registry `tool` names (resource_metrics.tsv) -> canonical display labels
REGISTRY_TO_DISPLAY = {
    "MLDSPP XGBoost": "MLDSPP 0% [BDT]",
    "MLDSPP XGBoost (75% spn)": "MLDSPP 75% [BDT]",
    "Prompt MLP (B. subtilis 168)": "prompt [NN]",
    "ProkBERT-mini (NeuralBioInfo)": "ProkBERT-mini [gLM]",
    "iPro-MP (H. pylori)": "iPro-MP [gLM]",
    "PromoterLCNN": "PromoterLCNN [CNN]",
    "FIMO + Prokaryote DB": "FIMO ProkDB [PWMs]",
    "PromoTech RF-HOT (PG Max)": "PromoTech RF-HOT [RF]",
    "MEME Suite (STREME+FIMO)": "STREME+FIMO [motif]",
}

TOOL_COLORS = {
    "ProkBERT-mini [gLM]": "#D81B60",
    "iPro-MP [gLM]": "#7E57C2",
    "PromoterLCNN [CNN]": "#2E7D32",
    "PromoTech RF-HOT [RF]": "#EF6C00",
    "prompt [NN]": "#0288D1",
    "MLDSPP 0% [BDT]": "#880E4F",
    "MLDSPP 75% [BDT]": "#C2185B",
    "FIMO ProkDB [PWMs]": "#00897B",
    "STREME+FIMO [motif]": "#795548",
}

CANONICAL_ORDER = [
    "ProkBERT-mini [gLM]", "iPro-MP [gLM]", "PromoterLCNN [CNN]",
    "PromoTech RF-HOT [RF]", "prompt [NN]", "MLDSPP 0% [BDT]",
    "MLDSPP 75% [BDT]", "FIMO ProkDB [PWMs]", "STREME+FIMO [motif]",
]


def run_label(rundir: Path, df: pd.DataFrame) -> str:
    """Human label for a run: dataset (configuration). Handles old/new schemas."""
    if "name" in df.columns:
        ds = str(df["name"].iloc[0])
    elif "db" in df.columns:
        ds = str(df["db"].iloc[0])
    else:
        ds = rundir.parent.name
    if "configuration" in df.columns:
        cfg = str(df["configuration"].iloc[0])
    elif "regime" in df.columns:
        cfg = str(df["regime"].iloc[0])
    else:
        cfg = rundir.name
    return f"{ds} ({cfg})"


def load_run(rundir: Path):
    """Return (label, res_df) with normalized columns, or None."""
    res_files = sorted(rundir.glob("resource_metrics*.tsv"))
    if not res_files:
        print(f"  [skip] {rundir}: no resource_metrics*.tsv")
        return None
    res = pd.read_csv(res_files[0], sep="\t")
    label = run_label(rundir, res)
    res = res.copy()
    res["run"] = label
    name_col = "tool" if "tool" in res.columns else None
    if name_col:
        res["Tool"] = res[name_col].map(
            lambda x: REGISTRY_TO_DISPLAY.get(str(x), str(x)))
    return label, res


def grouped_bars(ax, tools, runs, values, ylabel, title):
    """values[(tool, run)] -> float or nan. Bars grouped by tool, colored by tool."""
    import matplotlib.patches as mpatches
    x = np.arange(len(tools))
    n = len(runs)
    width = min(0.8 / max(n, 1), 0.25)
    for i, run in enumerate(runs):
        vals = [values.get((t, run), np.nan) for t in tools]
        off = (i - (n - 1) / 2) * width
        bars = ax.bar(x + off, [0 if np.isnan(v) else v for v in vals], width,
                      edgecolor="black", linewidth=0.6,
                      color=[TOOL_COLORS.get(t, "#333333") for t in tools])
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.3g}" if v < 10 else f"{v:.4g}",
                        ha="center", va="bottom", fontsize=7)
    # legend identifies run positions (bars are colored by tool, not by run)
    handles = [mpatches.Patch(facecolor="none", edgecolor="black", label=r)
               for r in runs]
    ax.set_xticks(x)
    ax.set_xticklabels(tools, rotation=25, ha="right", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(handles=handles, loc="best", fontsize=9, frameon=True, framealpha=0.95)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)


def save_fig(fig, path: Path):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_runs(run_dirs, out_dir: Path):
    loaded = []
    for rd in run_dirs:
        r = load_run(Path(rd))
        if r is not None:
            loaded.append(r)
    if not loaded:
        print("ERROR: no usable runs found.")
        sys.exit(2)

    labels = [lbl for lbl, _ in loaded]
    res_all = pd.concat([r for _, r in loaded], ignore_index=True)

    tools_res = [t for t in CANONICAL_ORDER if t in set(res_all.get("Tool", []))]
    tools_res += [t for t in sorted(set(res_all.get("Tool", [])) - set(tools_res))]

    resd = out_dir / "2_resources"
    tab = out_dir / "3_tables"
    for d in (resd, tab):
        d.mkdir(parents=True, exist_ok=True)

    # --- 2_resources: time / RAM / throughput / speedup ---
    if "wall_seconds" in res_all.columns:
        res_all["wall_seconds_num"] = pd.to_numeric(
            res_all["wall_seconds"], errors="coerce")
        res_all["speedup_vs_slowest"] = res_all.groupby("Tool")[
            "wall_seconds_num"].transform(
            lambda s: s.max() / s.replace(0, np.nan))
    res_all.to_csv(tab / "resources_compare.tsv", sep="\t", index=False)
    specs = [("wall_seconds", "Wall time (s)", "Execution time by run", "compare_time"),
             ("peak_ram_mb", "Peak RAM (MB)", "Peak RAM by run", "compare_ram"),
             ("throughput_seq_s", "Sequences / second", "Throughput by run", "compare_throughput"),
             ("speedup_vs_slowest", "Speedup vs slowest config (x)",
              "Speedup by tool and run", "compare_speedup")]
    for col, ylabel, title, stem in specs:
        if col not in res_all.columns:
            continue
        sub = res_all[pd.to_numeric(res_all[col], errors="coerce").notna()]
        if sub.empty:
            continue
        vals = {(r["Tool"], r["run"]): float(r[col]) for _, r in sub.iterrows()}
        fig, ax = plt.subplots(figsize=(max(10, len(tools_res) * 1.4), 6))
        grouped_bars(ax, tools_res, labels, vals, ylabel, title)
        save_fig(fig, resd / stem)
        print(f"  [SAVED] 2_resources/{stem}")

    with open(out_dir / "CONTENTS.md", "w") as fh:
        fh.write("# Cross-run compute comparison\n\nRuns:\n")
        for lbl in labels:
            fh.write(f"- {lbl}\n")
        fh.write("\n- 2_resources/compare_{time,ram,throughput,speedup}: grouped "
                 "bars per run (missing cells skipped).\n"
                 "- 3_tables/resources_compare.tsv: source table "
                 "(incl. speedup_vs_slowest per tool).\n")
    print(f"Done -> {out_dir}")


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Compare CLI runs (same dataset across configs, or across datasets).")
    p.add_argument("runs", nargs="+", help="Run dirs (output/<date>_runs/<name>/<config>/)")
    p.add_argument("-o", "--output", default=None,
                   help="Comparison dir (default: output/<YYMMDD>_comparison)")
    a = p.parse_args()
    if a.output:
        out = Path(a.output)
    else:
        out = Path(f"output/{date.today().strftime('%y%m%d')}_comparison")
    compare_runs(a.runs, out)


if __name__ == "__main__":
    main()
