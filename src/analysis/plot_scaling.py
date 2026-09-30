#!/usr/bin/env python3
"""Scaling curve for one tool across scale_db sizes (self-contained).

Reads execution time / peak RAM / AUC from each scale run's
`2_resources/resource_metrics.tsv` + `3_tables/metrics_rows.tsv` and writes:
  2_resources/scaling_time.{png,pdf}   — execution time vs N (log-log)
  2_resources/scaling_throughput.{png,pdf} — seq/s vs N
  3_tables/scaling.tsv                 — numbers per size
  MANIFEST.json                        — sources

Usage (single tool):
    pixi run python src/analysis/plot_scaling.py \
      output/2609XX_runs/scale_10k/16cpu-gpu \
      output/2609XX_runs/scale_30k/16cpu-gpu \
      output/2609XX_runs/scale_60k/16cpu-gpu \
      output/2609XX_runs/scale_100k/16cpu-gpu \
      --tool "ProkBERT-mini (NeuralBioInfo)" \
      --dataset "ProkBERT scale" \
      -o output/comparisons/2609XX_prokbert_scale_10k-100k

Usage (multi-tool, all 9 in primary order):
    pixi run python src/analysis/plot_scaling.py \
      output/2609XX_runs/scale_10k_9tools/16cpu-gpu [...] \
      --all --dataset "D39V+TIGR4 high" \
      -o output/comparisons/2609XX_scale9_10k-100k
"""
import argparse
import json
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

try:
    from bench_labels import TOOL_COLORS as _TOOL_COLORS
except Exception:
    _TOOL_COLORS = {}

REGISTRY_TO_DISPLAY = {
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
REGISTRY_ORDER = [
    "MLDSPP XGBoost", "MLDSPP XGBoost (75% spn)", "PromoTech RF-HOT (PG Max)",
    "PromoterLCNN", "Prompt MLP (B. subtilis 168)", "ProkBERT-mini (NeuralBioInfo)",
    "iPro-MP (H. pylori)", "FIMO + Prokaryote DB", "MEME Suite (STREME+FIMO)",
]


def load_point(run_dir: Path, tool: str):
    res = pd.read_csv(sorted((run_dir / "2_resources").glob("resource_metrics*.tsv"))[0], sep="\t")
    row = res[res["tool"] == tool]
    if row.empty:
        raise SystemExit(f"ERROR: tool {tool!r} not in {run_dir}")
    row = row.iloc[0]
    met = pd.read_csv(run_dir / "3_tables" / "metrics_rows.tsv", sep="\t")
    label = REGISTRY_TO_DISPLAY.get(tool, tool)
    mrow = met[met["tool"] == label]
    auc = float(mrow["auc"].iloc[0]) if not mrow.empty else float("nan")
    npos, nneg = int(mrow["n_pos"].iloc[0]), int(mrow["n_neg"].iloc[0]) if not mrow.empty else (0, 0)
    return {"run": str(run_dir), "n": npos + nneg,
            "wall": float(row["wall_seconds"]), "ram": float(row["peak_ram_mb"]),
            "vram": float(row["peak_vram_mb"]), "auc": auc}


def save(fig, path: Path):
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Scaling curve across scale_db runs.")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tool", default=None,
                    help="single registry tool name (single-tool mode)")
    ap.add_argument("--tools", nargs="*", default=None,
                    help="registry tool names for multi-tool mode")
    ap.add_argument("--all", action="store_true",
                    help="all 9 tools in primary order (multi-tool mode)")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--project-to", type=int, default=0,
                    help="multi-tool mode: dashed log-log projection to N "
                         "(extrapolation, clearly marked; skipped for tools "
                         "with --context-tsv points)")
    ap.add_argument("--context-tsv", default=None,
                    help="multi-tool mode: TSV with tool,n,wall context points "
                         "from another era (# comment lines allowed); drawn "
                         "grey dotted, excluded from projection")
    ap.add_argument("--x-max", type=int, default=0,
                    help="multi-tool mode: cap x axis at N (points above are "
                         "dropped; projection beyond is skipped)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    if args.all or args.tools:
        return main_multi(args)
    if not args.tool:
        ap.error("pass --tool, --tools or --all")

    pts = [load_point(Path(r), args.tool) for r in args.runs]
    pts.sort(key=lambda p: p["n"])
    df = pd.DataFrame(pts)

    out = Path(args.output)
    resd, tab = out / "2_resources", out / "3_tables"
    resd.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / "scaling.tsv", sep="\t", index=False)

    label = REGISTRY_TO_DISPLAY.get(args.tool, args.tool)
    ds = args.dataset or label
    n = df["n"].values.astype(float)
    w = df["wall"].values.astype(float)
    slope = float(np.polyfit(np.log10(n), np.log10(w), 1)[0])

    # 1. execution time vs N log-log
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.loglog(n, w, "o-", color="#2D004D", lw=2.2, ms=8, label=label)
    for x, y in zip(n, w):
        ax.annotate(f"{int(x/1000)}k: {y:.0f}s", (x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Dataset size N (pos+neg)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Execution time (s, log scale)", fontsize=12, fontweight="bold")
    ax.set_title(f"{ds} — scaling", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save(fig, resd / "scaling_time")
    print("  [SAVED] 2_resources/scaling_time.png/.pdf")

    # 2. throughput vs N
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    thr = n / w
    ax.semilogx(n, thr, "s-", color="#0060E6", lw=2.2, ms=8, label=label)
    for x, y in zip(n, thr):
        ax.annotate(f"{y:.1f}/s", (x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Dataset size N (pos+neg)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Throughput (seq/s)", fontsize=12, fontweight="bold")
    ax.set_title(f"{ds} — throughput vs size", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    save(fig, resd / "scaling_throughput")
    print("  [SAVED] 2_resources/scaling_throughput.png/.pdf")

    (out / "MANIFEST.json").write_text(json.dumps(
        {"type": "scaling", "dataset": ds, "tool": label,
         "sources": [p["run"] for p in pts],
         "loglog_slope": round(slope, 3)}, indent=2))
    print(f"Done -> {out}\n{df.to_string(index=False)}")


def _run_date(run_path: str) -> str:
    """YYMMDD prefix of a run dir ('' when absent: sorts oldest)."""
    import re
    m = re.search(r"(\d{6})", Path(run_path).name)
    if m:
        return m.group(1)
    m = re.search(r"(\d{6})", str(Path(run_path).parent))
    return m.group(1) if m else ""


def _load_context(path: str):
    """Context points from another era: {display: [(n, wall)]}. Skips # lines."""
    import csv
    out = {}
    with open(path) as fh:
        reader = csv.DictReader((l for l in fh if not l.startswith("#")))
        for row in reader:
            disp = row["tool"].strip()
            if disp in REGISTRY_TO_DISPLAY.values():
                label = disp
            else:
                label = REGISTRY_TO_DISPLAY.get(disp, disp)
            out.setdefault(label, []).append((float(row["n"]), float(row["wall"])))
    for v in out.values():
        v.sort()
    return out


def main_multi(args):
    tools = list(REGISTRY_ORDER) if args.all else list(args.tools)
    runs = sorted(args.runs)
    rows = []
    series = {}
    for tool in tools:
        pts = []
        for r in runs:
            try:
                pts.append(load_point(Path(r), tool))
            except SystemExit as e:
                print(f"  [scaling] skip: {e}")
        # Supersede policy: same tool+n keeps the newest run only, so a
        # fresh ladder transparently replaces stale points (e.g. new
        # ProkBERT runner over serial-era rows).
        by_n = {}
        for p in pts:
            key = p["n"]
            if key not in by_n or _run_date(p["run"]) >= _run_date(by_n[key]["run"]):
                by_n[key] = p
        pts = sorted(by_n.values(), key=lambda p: p["n"])
        if args.x_max:
            pts = [p for p in pts if p["n"] <= args.x_max]
        if len(pts) < 2:
            print(f"  [scaling] skip {tool}: <2 points")
            continue
        series[tool] = pts
        for p in pts:
            rows.append({"size": Path(p["run"]).parent.name, "n": p["n"],
                         "tool": REGISTRY_TO_DISPLAY.get(tool, tool),
                         "wall": round(p["wall"], 3), "ram": round(p["ram"], 1),
                         "vram": round(p["vram"], 1)})
    out = Path(args.output)
    resd, tab = out / "2_resources", out / "3_tables"
    resd.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tab / "scaling_all.tsv", sep="\t", index=False)

    ds = args.dataset or ""
    title_ds = f" ({ds})" if ds else ""
    context = _load_context(args.context_tsv) if args.context_tsv else {}
    specs = [("wall", True, "Execution time (s, log scale)",
              f"Execution time scaling — {len(series)} tools{title_ds}",
              "scaling_time_all"),
             ("thr", False, "Throughput (seq/s)",
              f"Throughput scaling — {len(series)} tools{title_ds}",
              "scaling_throughput_all"),
             ("ram", True, "Peak RAM (MB, log scale)",
              f"Peak RAM scaling — {len(series)} tools{title_ds}",
              "scaling_ram_all")]
    for col, logy, ylabel, title, stem in specs:
        fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=300)
        drew_projection = False
        for tool in series:
            pts = series[tool]
            n = np.array([p["n"] for p in pts], dtype=float)
            w = np.array([p[col] if col != "thr" else p["n"] / p["wall"]
                          for p in pts], dtype=float)
            disp = REGISTRY_TO_DISPLAY.get(tool, tool)
            color = _TOOL_COLORS.get(disp, "#333333")
            if logy:
                ax.loglog(n, w, "o-", color=color, lw=2.2, ms=7, label=disp)
            else:
                ax.semilogx(n, w, "o-", color=color, lw=2.2, ms=7, label=disp)
            if (args.project_to and col == "wall" and disp not in context
                    and len(n) >= 2
                    and (not args.x_max or args.project_to <= args.x_max)):
                b, a = np.polyfit(np.log10(n), np.log10(w), 1)
                nw = np.array([n[-1], float(args.project_to)])
                ax.loglog(nw, 10 ** (a + b * np.log10(nw)), ":",
                          color=color, lw=1.6)
                drew_projection = True
        if col == "wall":
            for disp, cnw in context.items():
                kept = [(x, y) for x, y in cnw
                        if not args.x_max or x <= args.x_max]
                if len(kept) < 2:
                    continue
                cx = np.array([p[0] for p in kept])
                cw = np.array([p[1] for p in kept])
                ax.loglog(cx, cw, "x:", color="#9E9E9E", lw=1.6, ms=6,
                          label=f"{disp} (homologous era)")
        if drew_projection:
            ax.loglog([], [], ":", color="grey", lw=1.6,
                      label=f"projected to {args.project_to // 1000}k")
        if args.x_max:
            ax.set_xlim(right=float(args.x_max))
        ax.set_xlabel("Dataset size N (pos+neg)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Dataset size N (pos+neg)", fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        leg = ax.legend(fontsize=9, loc="upper left")
        leg.get_title().set_fontweight("bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        save(fig, resd / stem)
        print(f"  [SAVED] 2_resources/{stem}.png/.pdf")

    (out / "MANIFEST.json").write_text(json.dumps(
        {"type": "scaling_all", "dataset": ds,
         "tools": [REGISTRY_TO_DISPLAY.get(t, t) for t in series],
         "sources": runs,
         "project_to": args.project_to or None,
         "context": args.context_tsv or None}, indent=2))
    print(f"Done -> {out}")
    return



if __name__ == "__main__":
    main()
