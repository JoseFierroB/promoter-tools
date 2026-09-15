#!/usr/bin/env python3
"""Compare compute across CLI runs (same dataset, different configurations).

Compares wall time, peak RAM and relative speedup of runs produced by
`src/cli.py run`, e.g. 1cpu vs 16cpu vs 1cpu-gpu vs 16cpu-gpu on one
dataset. Inference (AUC) is intentionally NOT compared: identical inputs
give identical scores by construction. Missing cells (tool not run in
some run, or run without resource metrics) are skipped, never crash.

Usage:
    python src/analysis/compare_runs.py <run_dir1> [<run_dir2> ...]
        [-o output/<YYMMDD>_comparison]

Output (inside the comparison dir):
  2_resources/compare_{time,ram,speedup}.{png,svg,pdf}
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

try:
    from bench_labels import TOOL_ORDER as CANONICAL_ORDER, TOOL_COLORS  # type: ignore
except Exception:
    TOOL_COLORS = {
        "MLDSPP 0% [BDT]": "#880E4F",
        "MLDSPP 75% [BDT]": "#C2185B",
        "PromoTech RF-HOT [RF]": "#EF6C00",
        "PromoterLCNN [CNN]": "#2E7D32",
        "prompt [NN]": "#0288D1",
        "ProkBERT-mini [gLM]": "#D81B60",
        "iPro-MP [gLM]": "#7E57C2",
        "FIMO ProkDB [PWMs]": "#00897B",
        "STREME+FIMO [motif]": "#795548",
    }
    CANONICAL_ORDER = [
        "MLDSPP 0% [BDT]", "MLDSPP 75% [BDT]", "PromoTech RF-HOT [RF]",
        "PromoterLCNN [CNN]", "prompt [NN]", "ProkBERT-mini [gLM]",
        "iPro-MP [gLM]", "FIMO ProkDB [PWMs]", "STREME+FIMO [motif]",
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
        res_files = sorted((rundir / "resources").glob("resource_metrics*.tsv"))
    if not res_files:
        res_files = sorted((rundir / "2_resources").glob("resource_metrics*.tsv"))
    if not res_files:
        res_files = sorted((rundir / "3_tables").glob("resource_metrics*.tsv"))
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


REGIME_DISPLAY = {
    "1cpu": "1-CPU",
    "4cpu": "4-CPU",
    "8cpu": "8-CPU",
    "16cpu": "16-CPU",
    "1cpu-gpu": "1-CPU Host+GPU",
    "4cpu-gpu": "4-CPU Host+GPU",
    "8cpu-gpu": "8-CPU Host+GPU",
    "16cpu-gpu": "16-CPU Host+GPU",
}
REGIME_HATCHES = ["//", "\\\\", "..", ""]


def fmt_time_short(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}s\n({v / 60:.1f}m)"
    return f"{v:.2f}s"


def fmt_ram(v: float) -> str:
    return f"{v:.1f} MB"


def single_bars(ax, tools, vals, ylabel, title, fmt):
    """One bar per tool, canonical TOOL_ORDER (BDT→RF→CNN→NN→gLM→motif), not alphabetical/value."""
    order_idx = {t: i for i, t in enumerate(CANONICAL_ORDER)}
    paired = sorted(zip(tools, vals), key=lambda item: order_idx.get(item[0], 999))
    tools = [t for t, _ in paired]
    vals = [v for _, v in paired]
    x = np.arange(len(tools))
    bars = ax.bar(x, vals, 0.52,
                  color=[TOOL_COLORS.get(t, "#333333") for t in tools],
                  edgecolor="black", linewidth=0.8, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                    color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10, fontweight="bold", rotation=20,
                       ha="right")
    ax.set_xlabel("Tool", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def regime_bars(ax, tools, runs, values, ylabel, title, fmt,
                regime_of: dict, ylim_top=None):
    """Grouped bars per tool, one bar per run (regime). Tool colors + hatch per regime.

    Tools in canonical TOOL_ORDER (BDT→RF→CNN→NN→gLM→motif); x labels rotated;
    value annotations staggered per run so adjacent numbers don't overlap.
    """
    import matplotlib.patches as mpatches
    order_idx = {t: i for i, t in enumerate(CANONICAL_ORDER)}
    tools = sorted(tools, key=lambda t: order_idx.get(t, 999))
    x = np.arange(len(tools))
    n = len(runs)
    width = min(0.8 / max(n, 1), 0.22)
    max_val = 0.0
    all_vals = [values.get((t, r), np.nan) for t in tools for r in runs]
    all_vals = [v for v in all_vals if not np.isnan(v)]
    span = (max(all_vals) - min(all_vals)) if all_vals else 0.0
    bar_artists = []
    for i, run in enumerate(runs):
        vals = [values.get((t, run), np.nan) for t in tools]
        off = (i - (n - 1) / 2) * width
        hatch = REGIME_HATCHES[i % len(REGIME_HATCHES)]
        bars = ax.bar(x + off, [0 if np.isnan(v) else v for v in vals], width,
                      edgecolor="black", linewidth=0.8,
                      color=[TOOL_COLORS.get(t, "#9E9E9E") for t in tools],
                      hatch=hatch, zorder=3)
        bar_artists.append(list(zip(bars, vals, tools)))
        for v in vals:
            if not np.isnan(v):
                max_val = max(max_val, v)
    # annotate with per-group vertical spreading so close numbers don't collide
    for j in range(len(tools)):
        group = [(bar_artists[i][j][0], bar_artists[i][j][1], bar_artists[i][j][2])
                 for i in range(n) if not np.isnan(bar_artists[i][j][1])]
        group.sort(key=lambda g: g[1])
        placed = []
        for b, v, t in group:
            yoff = 4
            while any(abs((v + yoff * max_val / 300) - p) < max_val * 0.045
                      for p in placed):
                yoff += 11
            placed.append(v + yoff * max_val / 300)
            ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, yoff), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold",
                        color=TOOL_COLORS.get(t, "black"))
    handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E",
                                  hatch=REGIME_HATCHES[i % len(REGIME_HATCHES)],
                                  edgecolor="black",
                                  label=REGIME_DISPLAY.get(regime_of.get(r, r), r))
               for i, r in enumerate(runs)]
    ax.set_xticks(x)
    ax.set_xticklabels(tools, fontsize=10, fontweight="bold", rotation=20,
                       ha="right")
    ax.set_xlabel("Tool and Model Architecture", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    if ylim_top:
        ax.set_ylim(0, ylim_top)
    else:
        ax.set_ylim(0, max_val * 1.25 if max_val > 0 else 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=handles, title="Execution Regime", frameon=True,
              fontsize=9.5, loc="upper left",
              handlelength=2.5, handleheight=2.5)


def save_fig(fig, path: Path):
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _dataset_title(res_all: pd.DataFrame, dataset_label: str = None) -> str:
    if dataset_label:
        return dataset_label
    if "name" in res_all.columns:
        from bench_labels import DS_DISPLAY
        names = [str(v) for v in res_all["name"].unique()]
        if len(names) == 1:
            return DS_DISPLAY.get(names[0], names[0])
    return "comparison"


def _total_n(res_all: pd.DataFrame) -> int:
    if "n_sequences" in res_all.columns:
        vals = pd.to_numeric(res_all["n_sequences"], errors="coerce").dropna()
        if len(vals):
            return int(vals.iloc[0])
    return 0


def compare_runs(run_dirs, out_dir: Path, dataset_label: str = None):
    loaded = []
    # track canonical source paths for MANIFEST (no data duplication)
    source_paths: List[str] = []
    input_shas: List[str] = []
    for rd in run_dirs:
        r = load_run(Path(rd))
        if r is not None:
            loaded.append(r)
            try:
                source_paths.append(str(Path(rd).resolve()))
                # try to read STATUS/MANIFEST for input_sha
                st = None
                for cand in [Path(rd) / "STATUS.json", Path(rd) / "MANIFEST.json"]:
                    if cand.exists():
                        import json as _json
                        st = _json.loads(cand.read_text())
                        if st.get("input_sha"):
                            input_shas.append(st["input_sha"])
                            break
            except Exception:
                pass
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
    # legacy symlink for old callers expecting output/<YYMMDD>_comparison_* (reversible)
    try:
        legacy_comp = Path("output") / out_dir.name
        if legacy_comp != out_dir and not legacy_comp.exists() and not legacy_comp.is_symlink():
            legacy_comp.symlink_to(out_dir.resolve())
    except Exception:
        pass
    # write comparison MANIFEST (lightweight view, no prediction copies) — reversible
    try:
        try:
            from run_layout import RunLayout  # type: ignore
        except ImportError:
            from src.analysis.run_layout import RunLayout  # type: ignore
        # prefer canonical comparisons root, but keep out_dir as requested (legacy compat)
        RunLayout(out_dir).write_comparison_manifest(
            out_dir, sources=source_paths, dataset=_dataset_title(res_all, dataset_label),
            regimes=[REGIME_DISPLAY.get(str(r.get("configuration", r.get("regime", ""))), str(r.get("configuration", ""))) for _, r in res_all.drop_duplicates("run").iterrows()],
            input_shas=list(dict.fromkeys(input_shas)),
        )
        # also ensure legacy symlink if out_dir is under old output/<YYMMDD>_comparison_* and canonical is output/comparisons/*
        # (handled by migration script; here we just ensure MANIFEST exists)
    except Exception:
        pass

    ds_title = _dataset_title(res_all, dataset_label)
    # fallback: if dataset_label generic ("results"/"comparison"), try to infer from run_dirs
    if ds_title in ("comparison", "results", "") and dataset_label:
        ds_title = dataset_label
    total_n = _total_n(res_all)
    n_tag = f", N={total_n:,}" if total_n else ""
    regime_of = {}
    for _, r in res_all.iterrows():
        cfg = str(r.get("configuration", r.get("regime", r.get("run", ""))))
        regime_of.setdefault(r["run"], cfg)
    # human regimes for titles: always declare CPU/GPU being compared
    regimes_disp = [REGIME_DISPLAY.get(regime_of.get(r, r), regime_of.get(r, r)) for r in labels]
    regimes_str = " vs ".join(regimes_disp) if len(regimes_disp) <= 3 else ", ".join(regimes_disp)

    # --- 2_resources: time / RAM / speedup (+ VRAM below) ---
    if "wall_seconds" in res_all.columns:
        res_all["wall_seconds_num"] = pd.to_numeric(
            res_all["wall_seconds"], errors="coerce")
        res_all["speedup_vs_slowest"] = res_all.groupby("Tool")[
            "wall_seconds_num"].transform(
            lambda s: s.max() / s.replace(0, np.nan))
    res_all.to_csv(tab / "resources_compare.tsv", sep="\t", index=False)
    if len(labels) == 1:
        cfg_single = regimes_disp[0] if regimes_disp else ""
        specs = [("wall_seconds", f"Execution Time (seconds{n_tag})",
                  f"Execution time — {ds_title} (N={total_n:,}, {cfg_single})" if total_n and cfg_single else f"Execution time — {ds_title}",
                  "compare_time", fmt_time_short),
                 ("peak_ram_mb", "Peak Memory Usage (MB)",
                  f"Peak RAM — {ds_title} (N={total_n:,}, {cfg_single})" if total_n and cfg_single else f"Peak RAM — {ds_title}",
                  "compare_ram", fmt_ram)]
        for col, ylabel, title, stem, fmt in specs:
            if col not in res_all.columns:
                continue
            sub = res_all[pd.to_numeric(res_all[col], errors="coerce").notna()]
            if sub.empty:
                continue
            tools = [t for t in tools_res if t in set(sub["Tool"])]
            vals = [float(sub[sub["Tool"] == t][col].iloc[0]) for t in tools]
            fig, ax = plt.subplots(figsize=(max(12, len(tools) * 1.6), 6.5),
                                   dpi=300)
            single_bars(ax, tools, vals, ylabel, title, fmt)
            save_fig(fig, resd / stem)
            print(f"  [SAVED] 2_resources/{stem}")
    else:
        specs = [("wall_seconds", f"Execution Time (seconds{n_tag})",
                  f"Execution time — {ds_title} (N={total_n:,}; {regimes_str})" if total_n
                  else f"Execution time — {ds_title} ({regimes_str})",
                  "compare_time", fmt_time_short),
                 ("peak_ram_mb", "Peak Memory Usage (MB)",
                  f"Peak RAM — {ds_title} (N={total_n:,}; {regimes_str})" if total_n
                  else f"Peak RAM — {ds_title} ({regimes_str})",
                  "compare_ram", fmt_ram),
                 ("speedup_vs_slowest", "Speedup vs slowest config (x)",
                  f"Speedup — {ds_title} (N={total_n:,}; {regimes_str})" if total_n
                  else f"Speedup — {ds_title} ({regimes_str})",
                  "compare_speedup", lambda v: f"{v:.2f}x")]
        for col, ylabel, title, stem, fmt in specs:
            if col not in res_all.columns:
                continue
            sub = res_all[pd.to_numeric(res_all[col], errors="coerce").notna()]
            if sub.empty:
                continue
            vals = {(r["Tool"], r["run"]): float(r[col]) for _, r in sub.iterrows()}
            fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
            regime_bars(ax, tools_res, labels, vals, ylabel, title, fmt,
                        regime_of)
            save_fig(fig, resd / stem)
            print(f"  [SAVED] 2_resources/{stem}")

    # --- GPU VRAM: weights vs peak (GPU tools only) ---
    GPU_TOOLS = {"PromoterLCNN [CNN]", "ProkBERT-mini [gLM]", "iPro-MP [gLM]"}
    if {"model_size_mb", "peak_vram_mb"} <= set(res_all.columns):
        vsub = res_all[(pd.to_numeric(res_all["peak_vram_mb"],
                                      errors="coerce").fillna(0) > 0) & (res_all["Tool"].isin(GPU_TOOLS))]
        if not vsub.empty:
            order_idx = {t: i for i, t in enumerate(CANONICAL_ORDER)}
            vtools = sorted([t for t in tools_res if t in set(vsub["Tool"])],
                            key=lambda t: order_idx.get(t, 999))
            weights = [float(vsub[vsub["Tool"] == t]["model_size_mb"].iloc[0])
                       for t in vtools]
            peaks = [float(vsub[vsub["Tool"] == t]["peak_vram_mb"].iloc[0])
                     for t in vtools]
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
            x = np.arange(len(vtools))
            w = 0.35
            b1 = ax.bar(x - w / 2, weights, w, label="Model Weights (Disk/RAM)",
                        color="#455A64", edgecolor="black", linewidth=0.8,
                        zorder=3)
            b2 = ax.bar(x + w / 2, peaks, w, label="Peak Inference VRAM",
                        color="#7E57C2", edgecolor="black", linewidth=0.8,
                        zorder=3)
            for bar in b1:
                h = bar.get_height()
                ax.annotate(f"{h:.1f} MB",
                            (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9.5,
                            fontweight="bold")
            for bar in b2:
                h = bar.get_height()
                ax.annotate(f"{h:.1f} MB\n({h / 1024:.3f} GB)",
                            (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9.5,
                            fontweight="bold")
            ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
            ax.set_title(f"GPU Memory — {ds_title} (N = {total_n:,}; {regimes_str})" if total_n
                         else f"GPU Memory — {ds_title} ({regimes_str})",
                         fontsize=13, fontweight="bold", pad=15)
            ax.set_xticks(x)
            ax.set_xticks(np.arange(len(vtools)))
            ax.set_xticklabels(vtools, fontsize=10.5, fontweight="bold",
                               rotation=20, ha="right")
            ax.set_ylim(0, max(peaks) * 1.25 if peaks else 1)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=True, fontsize=10, loc="upper left")
            plt.tight_layout()
            save_fig(fig, resd / "compare_vram")
            print("  [SAVED] 2_resources/compare_vram")

    with open(out_dir / "CONTENTS.md", "w") as fh:
        fh.write("# Cross-run compute comparison\n\nRuns:\n")
        for lbl in labels:
            fh.write(f"- {lbl}\n")
        fh.write("\n- 2_resources/compare_{time,ram,speedup}: "
                 "single bars (1 run) or grouped by regime (>1 run).\n"
                 "- 2_resources/compare_vram: weights vs peak VRAM (GPU tools).\n"
                 "- 3_tables/resources_compare.tsv: source table "
                 "(incl. speedup_vs_slowest per tool).\n")
    print(f"Done -> {out_dir}")


def default_comparison_dir(run_dirs) -> Path:
    """Derive output dir from input runs: date + run tokens.

    e.g. [output/260910_runs/4_d39v_tigr4_high/16cpu-gpu, ...] ->
    output/comparisons/260910_comparison_4_d39v_tigr4_high-16cpu-gpu[_...].
    Falls back to output/ next to legacy output/<YYMMDD>_comparison_* for compat.
    """
    import re
    try:
        from run_layout import COMPARISONS_ROOT  # type: ignore
    except ImportError:
        try:
            from src.analysis.run_layout import COMPARISONS_ROOT  # type: ignore
        except ImportError:
            COMPARISONS_ROOT = Path("output/comparisons")
    dates, tokens = [], []
    for rd in run_dirs:
        parts = [p for p in Path(rd).parts if p not in ("output",)]
        for p in parts:
            m = re.match(r"^(\d{6})_", p)
            if m:
                dates.append(m.group(1))
                break
        # token: last two meaningful components, date prefix stripped
        # (e.g. 4_d39v_tigr4_high + 16cpu-gpu, or 1cpu-gpu alone)
        tail = parts[-2:] if len(parts) >= 2 else parts
        tail = [re.sub(r"^\d{6}_", "", t) for t in tail]
        tail = [t for t in tail
                if t.lower() not in ("runs", "results", "comparison", "comparisons")]
        tok = "-".join(tail) or "run"
        tok = re.sub(r"[^A-Za-z0-9_.-]+", "_", tok).strip("_") or "run"
        tokens.append(tok)
    stamp = dates[0] if dates and len(set(dates)) == 1 else date.today().strftime("%y%m%d")
    stem = f"{stamp}_comparison_" + "__".join(tokens)
    if len(stem) > 100:
        stem = stem[:100].rstrip("_")
    base = COMPARISONS_ROOT / stem
    # bump _02, _03... if it already exists and is non-empty
    out, i = base, 2
    while out.exists() and any(out.iterdir()):
        out = Path(f"{base}_{i:02d}")
        i += 1
    # ensure legacy symlink for old callers expecting output/<YYMMDD>_comparison_*
    try:
        legacy = Path("output") / stem
        if not legacy.exists() and not out.exists():
            pass  # will create out, symlink later
    except Exception:
        pass
    return out


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Compare CLI runs (same dataset across configs, or across datasets).")
    p.add_argument("runs", nargs="+", help="Run dirs (output/<date>_runs/<name>/<config>/)")
    p.add_argument("-o", "--output", default=None,
                   help="Comparison dir (default: inherited date+names from input runs)")
    p.add_argument("--dataset", default=None,
                   help="Dataset display name for plot titles")
    a = p.parse_args()
    if a.output:
        out = Path(a.output)
    else:
        out = default_comparison_dir(a.runs)
    compare_runs(a.runs, out, dataset_label=a.dataset)


if __name__ == "__main__":
    main()
