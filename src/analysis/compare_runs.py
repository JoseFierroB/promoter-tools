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
  2_resources/compare_{time,ram,speedup,vram}.{png,pdf}
  3_tables/resources_compare.tsv (incl. speedup_vs_slowest per tool)
  CONTENTS.md
"""
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from textwrap import fill

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from run_layout import COMPARISONS_ROOT, RunLayout
except ImportError:
    from src.analysis.run_layout import COMPARISONS_ROOT, RunLayout

ROOT = Path(__file__).resolve().parents[2]

# registry `tool` names (resource_metrics.tsv) -> primary display labels (single source of truth)
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
    from bench_labels import TOOL_ORDER as PRIMARY_ORDER, TOOL_COLORS  # type: ignore
except Exception:
    TOOL_COLORS = {
        "MLDSPP 0% [BDT]": "#F81D54",
        "MLDSPP 75% [BDT]": "#55001C",
        "PromoTech RF-HOT [RF]": "#F1A52B",
        "PromoterLCNN [CNN]": "#973C04",
        "prompt [NN]": "#0060E6",
        "ProkBERT-mini [gLM]": "#2D004D",
        "iPro-MP [gLM]": "#BA9CEE",
        "FIMO ProkDB [PWMs]": "#002616",
        "STREME+FIMO [motif]": "#18974C",
    }
    PRIMARY_ORDER = [
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
    layout = RunLayout.resolve(rundir)
    rundir = layout.root
    metrics = layout.find_resource_metrics()
    if metrics is None:
        print(f"  [skip] {rundir}: no resource_metrics*.tsv")
        return None
    try:
        res = pd.read_csv(metrics, sep="\t")
    except pd.errors.EmptyDataError:
        print(f"  [skip] {rundir}: empty resource metrics")
        return None
    if "success" in res.columns:
        res = res[res["success"].astype(str).str.strip().str.lower().isin(
            ("true", "1", "1.0"))]
    if res.empty:
        print(f"  [skip] {rundir}: no successful tool rows")
        return None
    res = res.copy()
    name_col = "tool" if "tool" in res.columns else None
    if name_col:
        res["Tool"] = res[name_col].map(
            lambda x: REGISTRY_TO_DISPLAY.get(str(x), str(x)) if pd.notna(x) else x)
    if "Tool" not in res.columns:
        print(f"  [skip] {rundir}: no tool column")
        return None
    res = res[res["Tool"].notna() & res["Tool"].astype(str).str.strip().ne("")].copy()
    if res.empty:
        print(f"  [skip] {rundir}: no named tool rows")
        return None
    metadata = layout.metadata
    res["name"] = layout.dataset_name
    cfg = metadata.get("config") or metadata.get("configuration") or rundir.name
    if "configuration" not in res.columns:
        res["configuration"] = res["regime"] if "regime" in res.columns else cfg
    res["configuration"] = res["configuration"].replace("", np.nan).fillna(cfg)
    label = run_label(rundir, res)
    # Source identity must not collapse separate dates/repeats with the same label.
    res["run"] = str(rundir)
    res["run_label"] = label
    res["source_path"] = str(rundir)
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
    return f"{v:,.1f} MB"


def fmt_ram_value(v: float) -> str:
    """Compact RAM labels; the shared y axis specifies MB."""
    return f"{v:,.1f}"


def _tool_order(tool):
    """Use the shared method order, with a deterministic fallback for new tools."""
    ranks = {t: i for i, t in enumerate(PRIMARY_ORDER)}
    return ranks.get(tool, len(ranks)), tool.casefold()


def _comparison_figure(n_tools, n_series):
    # Allow more space for each group as regimes are added. The legend lives
    # outside the axes; constrained layout reserves its own space for it.
    width = max(12, n_tools * (2.2 if n_series >= 3 else 1.9))
    return plt.subplots(figsize=(width, 8.5 if n_series >= 3 else 7.5),
                        dpi=300, constrained_layout=True)


def _style_bar_axes(ax, tools, ylabel, title):
    ax.set_xticks(np.arange(len(tools)))
    # Keep the full names, but put the method on its own line.
    ax.set_xticklabels([t.replace(" [", "\n[") for t in tools],
                       fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", pad=9)
    ax.set_xlabel("Tool and Model Architecture", fontsize=12,
                  fontweight="bold", labelpad=14)
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(fill(title, width=85), fontsize=13, fontweight="bold", pad=18)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.025)


def _outside_legend(ax, **kwargs):
    return ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1),
                     borderaxespad=0, frameon=True, fontsize=10, **kwargs)


def _annotate_bars(ax, entries, ylim_top=None, rotation=None):
    """Anchor every value to its bar and fit labels in display coordinates.

    entries contains (bar, value, text, color). Start with horizontal labels;
    if the rendered text intersects another label or bar, use vertical labels
    for the whole figure. An explicit rotation is preserved by widening the
    figure until each label fits above its own bar.
    ylim_top is a lower bound: annotations may require a little more headroom.
    """
    if not entries:
        ax.set_ylim(0, ylim_top or 1)
        return []
    fig = ax.figure
    annotations = []
    maximum = max(v for _, v, _, _ in entries)
    ax.set_ylim(0, max(maximum * 1.12, ylim_top or 0, 1e-9))
    for bar, value, label, color in entries:
        annotation = ax.annotate(
            label, (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=color,
            rotation=rotation or 0,
            annotation_clip=False, clip_on=False)
        # Headroom is managed below, independently of the outside legend/title.
        annotation.set_in_layout(False)
        annotations.append(annotation)

    if rotation is not None:
        # Fit the rotated bounding boxes within their bar lanes. This keeps
        # 45-degree values close to the bars even with four regimes.
        for _ in range(4):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            pad = 2 * fig.dpi / 72
            ratio = max((a.get_window_extent(renderer).width + 2 * pad)
                        / b.get_window_extent(renderer).width
                        for a, (b, _, _, _) in zip(annotations, entries))
            if ratio <= 1.01:
                break
            extra_width = ax.get_window_extent(renderer).width / fig.dpi * (ratio - 1)
            fig.set_size_inches(fig.get_figwidth() + extra_width, fig.get_figheight())

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_window_extent(renderer) for a in annotations]
    bar_boxes = [b.get_window_extent(renderer) for b, _, _, _ in entries]
    pad = 2 * fig.dpi / 72
    crowded = any(box.padded(pad).overlaps(other)
                  for i, box in enumerate(boxes) for other in boxes[i + 1:])
    crowded |= any(box.padded(pad).overlaps(bar_box)
                   for i, box in enumerate(boxes)
                   for j, bar_box in enumerate(bar_boxes) if i != j)
    if crowded and rotation is None:
        for annotation in annotations:
            annotation.set_rotation(90)

    # Compute the required limit from text height in pixels, so a long label
    # above the tallest bar stays inside the axes at the actual export size.
    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        height = ax.get_window_extent(renderer).height
        upper = ax.get_ylim()[1]
        for annotation, (_, value, _, _) in zip(annotations, entries):
            text_height = annotation.get_window_extent(renderer).height
            remaining = height - text_height - (5 + 6) * fig.dpi / 72
            if remaining > 0:
                upper = max(upper, value * height / remaining)
        if upper <= ax.get_ylim()[1] * (1 + 1e-6):
            break
        ax.set_ylim(0, upper * 1.01)
    return annotations


def single_bars(ax, tools, vals, ylabel, title, fmt, annotation_rotation=None):
    """One bar per tool, primary TOOL_ORDER (BDT→RF→CNN→NN→gLM→motif), not alphabetical/value."""
    paired = sorted(zip(tools, vals), key=lambda item: _tool_order(item[0]))
    tools = [t for t, _ in paired]
    vals = [v for _, v in paired]
    x = np.arange(len(tools))
    bars = ax.bar(x, vals, 0.52,
                  color=[TOOL_COLORS.get(t, "#333333") for t in tools],
                  edgecolor="black", linewidth=0.8, zorder=3)
    _style_bar_axes(ax, tools, ylabel, title)
    _annotate_bars(ax, [(b, v, fmt(v), "black") for b, v in zip(bars, vals)
                        if np.isfinite(v)], rotation=annotation_rotation)


def regime_bars(ax, tools, runs, values, ylabel, title, fmt,
                regime_of: dict, ylim_top=None, annotation_rotation=None,
                run_display=None):
    """Grouped bars per tool, one bar per run (regime). Tool colors + hatch per regime.

    Tools in primary TOOL_ORDER (BDT→RF→CNN→NN→gLM→motif).
    Every finite value is annotated directly above its own bar.
    """
    import matplotlib.patches as mpatches
    tools = sorted(tools, key=_tool_order)
    x = np.arange(len(tools))
    n = len(runs)
    width = 0.82 / max(n, 1)
    entries = []
    for i, run in enumerate(runs):
        vals = [values.get((t, run), np.nan) for t in tools]
        off = (i - (n - 1) / 2) * width
        hatch = REGIME_HATCHES[i % len(REGIME_HATCHES)]
        bars = ax.bar(x + off, vals, width,
                      edgecolor="black", linewidth=0.8,
                      color=[TOOL_COLORS.get(t, "#9E9E9E") for t in tools],
                      hatch=hatch, zorder=3)
        entries.extend((b, v, fmt(v), TOOL_COLORS.get(t, "black"))
                       for b, v, t in zip(bars, vals, tools) if np.isfinite(v))
    handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor="#9E9E9E",
                                  hatch=REGIME_HATCHES[i % len(REGIME_HATCHES)],
                                  edgecolor="black",
                                  label=(run_display or {}).get(
                                      r, REGIME_DISPLAY.get(regime_of.get(r, r), r)))
               for i, r in enumerate(runs)]
    _style_bar_axes(ax, tools, ylabel, title)
    _outside_legend(ax, handles=handles, title="Execution Regime",
                    handlelength=2.5, handleheight=2.0)
    _annotate_bars(ax, entries, ylim_top, rotation=annotation_rotation)


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


def _vram_reference(res_all, regime_of):
    """Select one GPU configuration, preferring the 1cpu-gpu reference.

    A positive GPU-memory sample in a CPU-only run is not proof that inference
    used the GPU. Never choose the first row across CPU and GPU runs together.
    """
    gpu_runs = [run for run, regime in regime_of.items()
                if str(regime).endswith("-gpu")]
    if not gpu_runs:
        return res_all.iloc[:0], None
    reference = next((r for r in gpu_runs if regime_of[r] == "1cpu-gpu"),
                     gpu_runs[0])
    return res_all[res_all["run"] == reference], reference


def _plot_title(metric, dataset, total_n):
    return f"{metric} — {dataset}" + (f" (N={total_n:,})" if total_n else "")


def compare_runs(run_dirs, out_dir: Path, dataset_label: str = None):
    layouts = [RunLayout.resolve(Path(rd)) for rd in run_dirs]
    out_dir = Path(out_dir).resolve()
    # Validate every input before creating directories or overwriting a manifest.
    for layout in layouts:
        if (out_dir == layout.root or out_dir in layout.root.parents
                or layout.root in out_dir.parents):
            raise ValueError(f"Comparison output {out_dir} is an input run, its "
                             f"ancestor, or its descendant ({layout.root}); "
                             "choose a separate directory.")
    status_path = out_dir / "STATUS.json"
    if status_path.exists() or status_path.is_symlink():
        raise ValueError(f"Comparison output {out_dir} contains STATUS.json; "
                         "refusing to overwrite a run directory.")
    manifest_path = out_dir / "MANIFEST.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot validate comparison output manifest "
                             f"{manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("type") != "comparison":
            raise ValueError(f"Comparison output {out_dir} has a non-comparison "
                             "MANIFEST.json; choose a separate directory.")
    loaded = []
    source_paths = []
    input_shas = []
    seen = set()
    for layout in layouts:
        source = str(layout.root)
        if source in seen:
            continue
        seen.add(source)
        r = load_run(layout.root)
        if r is not None:
            loaded.append(r)
            source_paths.append(source)
            sha = layout.metadata.get("input_sha")
            if sha:
                input_shas.append(sha)
    if not loaded:
        print("ERROR: no usable runs found.")
        sys.exit(2)

    labels = source_paths
    display_labels = {source: label for source, (label, _) in zip(labels, loaded)}
    res_all = pd.concat([r for _, r in loaded], ignore_index=True)

    tools_res = [t for t in PRIMARY_ORDER if t in set(res_all.get("Tool", []))]
    tools_res += [t for t in sorted(set(res_all.get("Tool", [])) - set(tools_res))]

    resd = out_dir / "2_resources"
    tab = out_dir / "3_tables"
    for d in (resd, tab):
        d.mkdir(parents=True, exist_ok=True)
    RunLayout(out_dir).write_comparison_manifest(
        out_dir, sources=source_paths, dataset=_dataset_title(res_all, dataset_label),
        regimes=[REGIME_DISPLAY.get(str(r["configuration"]), str(r["configuration"]))
                 for _, r in res_all.drop_duplicates("run").iterrows()],
        input_shas=list(dict.fromkeys(input_shas)),
    )

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
    regime_counts = Counter(regime_of.values())
    run_display = dict(zip(labels, regimes_disp))
    for source in labels:
        if regime_counts[regime_of[source]] <= 1:
            continue
        # Shortest unique source suffix keeps repeated regimes identifiable
        # without changing bar styling, hatch order, or legend placement.
        parts = Path(source).parts
        for depth in range(1, len(parts) + 1):
            suffix = Path(*parts[-depth:])
            if sum(Path(*Path(other).parts[-depth:]) == suffix for other in labels) == 1:
                break
        run_display[source] += f" ({suffix})"

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
                  "compare_ram", fmt_ram_value)]
        for col, ylabel, title, stem, fmt in specs:
            if col not in res_all.columns:
                continue
            sub = res_all[pd.to_numeric(res_all[col], errors="coerce").notna()]
            if sub.empty:
                continue
            tools = [t for t in tools_res if t in set(sub["Tool"])]
            vals = [float(sub[sub["Tool"] == t][col].iloc[0]) for t in tools]
            fig, ax = _comparison_figure(len(tools), 1)
            single_bars(ax, tools, vals, ylabel, title, fmt,
                        annotation_rotation=45)
            save_fig(fig, resd / stem)
            print(f"  [SAVED] 2_resources/{stem}")
    else:
        specs = [("wall_seconds", "Execution Time (s)",
                  _plot_title("Execution time", ds_title, total_n),
                  "compare_time", fmt_time_short),
                 ("peak_ram_mb", "Peak Memory Usage (MB)",
                  _plot_title("Peak RAM", ds_title, total_n),
                  "compare_ram", fmt_ram_value),
                 ("speedup_vs_slowest", "Speedup vs slowest config (x)",
                  _plot_title("Speedup", ds_title, total_n),
                  "compare_speedup", lambda v: f"{v:.2f}x")]
        for col, ylabel, title, stem, fmt in specs:
            if col not in res_all.columns:
                continue
            sub = res_all[pd.to_numeric(res_all[col], errors="coerce").notna()]
            if sub.empty:
                continue
            vals = {(r["Tool"], r["run"]): float(r[col]) for _, r in sub.iterrows()}
            fig, ax = _comparison_figure(len(tools_res), len(labels))
            regime_bars(ax, tools_res, labels, vals, ylabel, title, fmt,
                        regime_of, annotation_rotation=45, run_display=run_display)
            save_fig(fig, resd / stem)
            print(f"  [SAVED] 2_resources/{stem}")

    # --- GPU VRAM: weights vs peak (GPU tools only) ---
    GPU_TOOLS = {"PromoterLCNN [CNN]", "ProkBERT-mini [gLM]", "iPro-MP [gLM]"}
    vram_reference_label = None
    if {"model_size_mb", "peak_vram_mb"} <= set(res_all.columns):
        gpu_reference, vram_reference_label = _vram_reference(res_all, regime_of)
        vsub = gpu_reference[(pd.to_numeric(gpu_reference["peak_vram_mb"],
                                      errors="coerce").fillna(0) > 0) & (gpu_reference["Tool"].isin(GPU_TOOLS))]
        if not vsub.empty:
            order_idx = {t: i for i, t in enumerate(PRIMARY_ORDER)}
            vtools = sorted([t for t in tools_res if t in set(vsub["Tool"])],
                            key=lambda t: order_idx.get(t, 999))
            weights = [float(vsub[vsub["Tool"] == t]["model_size_mb"].iloc[0])
                       for t in vtools]
            peaks = [float(vsub[vsub["Tool"] == t]["peak_vram_mb"].iloc[0])
                     for t in vtools]
            fig, ax = _comparison_figure(len(vtools), 2)
            x = np.arange(len(vtools))
            w = 0.35
            # Grouped (not stacked: peak already contains resident weights).
            # Peak bars share one pastel: the value (height) compares,
            # not the color. Both bars are VRAM info by construction.
            b1 = ax.bar(x - w / 2, weights, w, label="Model weights",
                        color="#455A64", edgecolor="black", linewidth=0.8,
                        zorder=3)
            b2 = ax.bar(x + w / 2, peaks, w, label="Max Inference VRAM",
                        color="#C5CAE9", edgecolor="black", linewidth=0.8,
                        zorder=3)
            reference_display = run_display[vram_reference_label]
            _style_bar_axes(ax, vtools, "Memory (MB)",
                            _plot_title("GPU Memory", ds_title, total_n))
            _outside_legend(ax, title=f"Reference: {reference_display}")
            entries = [(b, v, fmt_ram(v), "black") for b, v in zip(b1, weights)]
            # Single-line GB labels fit horizontally above the bars.
            entries += [(b, v, f"{v / 1024:.2f} GB" if v >= 1024 else fmt_ram(v),
                         "black") for b, v in zip(b2, peaks)]
            _annotate_bars(ax, [e for e in entries if np.isfinite(e[1])])
            save_fig(fig, resd / "compare_vram")
            print("  [SAVED] 2_resources/compare_vram")

    with open(out_dir / "CONTENTS.md", "w") as fh:
        fh.write("# Cross-run compute comparison\n\nRuns:\n")
        for lbl in labels:
            fh.write(f"- {display_labels[lbl]}: {lbl}\n")
        fh.write("\n- 2_resources/compare_{time,ram,speedup}: "
                 "single bars (1 run) or grouped by regime (>1 run).\n"
                 "- 2_resources/compare_vram: weights vs peak VRAM (GPU tools).\n"
                 "- 3_tables/resources_compare.tsv: source table "
                 "(incl. speedup_vs_slowest per tool).\n")
        if vram_reference_label:
            fh.write(f"\nVRAM reference: {vram_reference_label}. "
                     "Model weights and peak VRAM come from this run only.\n")
    print(f"Done -> {out_dir}")


def default_comparison_dir(run_dirs) -> Path:
    """Derive output dir from input runs: date + run tokens.

    e.g. [output/260917_runs/2_d39v_tigr4_high/16cpu-gpu, ...] ->
    output/comparisons/260917_comparison_2_d39v_tigr4_high-16cpu-gpu[_...].
    Default output stays under the project's primary comparisons directory.
    """
    import re
    dates, tokens = [], []
    for rd in run_dirs:
        parts = [p for p in Path(rd).parts if p not in ("output",)]
        for p in parts:
            m = re.match(r"^(\d{6})_", p)
            if m:
                dates.append(m.group(1))
                break
        # token: last two meaningful components, date prefix stripped
        # (e.g. 2_d39v_tigr4_high + 16cpu-gpu, or 1cpu-gpu alone)
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
    base = ROOT / COMPARISONS_ROOT / stem
    # bump _02, _03... if it already exists and is non-empty
    out, i = base, 2
    while out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        out = Path(f"{base}_{i:02d}")
        i += 1
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
