#!/usr/bin/env python3
"""Scaling analysis — per-tool time(n)/RAM(n) fits and extrapolation.

Reads campaign metrics from /home/fierro/Desktop/scale_db/<iter>/predictions/
plus the local CPU smoke run, consolidates them into
output/tables/scaling_dataset.tsv, fits linear/power/quadratic models per
(tool, resource_class, machine), and writes:

    output/tables/scaling_dataset.tsv   consolidated observations
    output/tables/extrapolation.tsv     predicted time/RAM at target sizes
    output/plots/scaling/*.{png,svg}    time & RAM scaling plots

Usage:
    pixi run python src/analysis/scaling_analysis.py
"""
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

_parser = argparse.ArgumentParser(description="Scaling analysis")
_parser.add_argument("--scale-db", default="/home/fierro/Desktop/scale_db",
                     help="Root with <iter>/predictions/resource_metrics*.tsv folders")
_parser.add_argument("--metrics-name", default="resource_metrics.tsv",
                     help="Metrics filename to read per iteration")
_parser.add_argument("--local-suffix", default="mldspp75_local",
                     help="Suffix of local-only metrics files (e.g. _mldspp75_local)")
_args = _parser.parse_args()

SCALE_DB = Path(_args.scale_db)
METRICS_NAME = _args.metrics_name
LOCAL_SUFFIX = _args.local_suffix
ITERATIONS = sorted(d.name for d in SCALE_DB.iterdir()
                    if d.is_dir() and (d / "predictions").is_dir())
SMOKE_TSV = ROOT / "output" / "tables" / "resource_metrics.tsv"
OUT_TSV = ROOT / "output" / "tables" / "scaling_dataset.tsv"
EXTRAP_TSV = ROOT / "output" / "tables" / "extrapolation.tsv"
PLOT_DIR = ROOT / "output" / "plots" / "scaling"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GPU_TOOLS = {"PromoterLCNN", "iPro-MP (H. pylori)"}
TOOL_ORDER = ["PromoTech RF-HOT (PG Max)", "PromoterLCNN", "MLDSPP XGBoost",
              "MLDSPP XGBoost (75% spn)", "iPro-MP (H. pylori)",
              "FIMO + Prokaryote DB", "MEME Suite (STREME+FIMO)"]
TARGETS = [10_000, 25_000, 50_000, 100_000, 150_000, 200_000]
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
    for it in ITERATIONS:
        path = SCALE_DB / it / "predictions" / METRICS_NAME
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            df["machine"] = "ws"
            df["iteration"] = it
            frames.append(df)
        local = SCALE_DB / it / "predictions" / f"resource_metrics_{LOCAL_SUFFIX}.tsv"
        if local.exists():
            df = pd.read_csv(local, sep="\t")
            df["machine"] = "local"
            df["iteration"] = it
            frames.append(df)
    if SMOKE_TSV.exists():
        df = pd.read_csv(SMOKE_TSV, sep="\t")
        df["machine"] = "local"
        df["iteration"] = "smoke"
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["success"].fillna(False).astype(bool)]
    df["resource_class"] = np.where(df["tool"].isin(GPU_TOOLS) & (df["machine"] == "ws"),
                                    "gpu-3090", "cpu-1thread")
    cols = ["tool", "n_sequences", "resource_class", "machine", "iteration",
            "wall_seconds", "time_s", "peak_ram_mb", "peak_vram_mb",
            "mean_cpu_pct", "gpu_util_pct", "gpu_name", "success"]
    return df[cols].sort_values(["tool", "n_sequences"])


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


def fit_group(group):
    n = group["n_sequences"].values
    t = group["time_s"].values
    r = group["peak_ram_mb"].values
    name_t, fn_t, p0_t, r2_t, quad_r2_t = best_fit(n, t)
    popt_t, _ = curve_fit(fn_t, n, t, p0=p0_t, maxfev=20000)
    popt_r, _ = curve_fit(lambda x, a, b: a * x + b, n, r, p0=(1e-3, 100.0))
    res_t = t - fn_t(n, *popt_t)
    s_t = float(np.sqrt(np.sum(res_t ** 2) / max(len(n) - len(popt_t), 1)))
    return {
        "model": name_t, "popt_t": popt_t, "fn_t": fn_t, "r2_t": r2_t, "s_t": s_t,
        "quad_r2_t": quad_r2_t, "popt_r": popt_r,
    }


def main():
    df = load_all()
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Dataset: {len(df)} rows -> {OUT_TSV}")

    fits = {}
    for (tool, cls, machine), g in df.groupby(["tool", "resource_class", "machine"]):
        if len(g) < 2:
            continue
        fits[(tool, cls, machine)] = fit_group(g)

    rows = []
    for (tool, cls, machine), f in sorted(fits.items()):
        for target in TARGETS:
            t_pred = float(f["fn_t"](target, *f["popt_t"]))
            r_pred = float(f["popt_r"][0] * target + f["popt_r"][1])
            rows.append({"tool": tool, "resource_class": cls, "machine": machine,
                         "model": f["model"], "r2_adjusted": f["r2_t"],
                         "n_target": target, "time_s_pred": t_pred,
                         "ram_mb_pred": r_pred})
    extrap = pd.DataFrame(rows)
    extrap.to_csv(EXTRAP_TSV, sep="\t", index=False)

    budget_rows = []
    for cls in ["cpu-1thread", "gpu-3090"]:
        total = 0.0
        for tool in TOOL_ORDER:
            key = (tool, cls, "ws")
            if key not in fits and cls == "cpu-1thread":
                key = (tool, cls, "local")
            if key not in fits:
                continue
            f = fits[key]
            total += float(f["fn_t"](MAX_TARGET, *f["popt_t"]))
            budget_rows.append({"resource_class": cls, "tool": tool,
                                "n_target": MAX_TARGET,
                                "time_s_pred": float(f["fn_t"](MAX_TARGET, *f["popt_t"])),
                                "ram_mb_pred": float(f["popt_r"][0] * MAX_TARGET + f["popt_r"][1]),
                                "machine": key[2]})
        budget_rows.append({"resource_class": cls, "tool": "TOTAL",
                            "n_target": MAX_TARGET, "time_s_pred": total,
                            "ram_mb_pred": None, "machine": "ws"})

    print("\n=== FITS (mejor modelo por tool x clase; lineal/potencia) ===")
    for (tool, cls, machine), f in sorted(fits.items()):
        print(f"  {tool:<28} {cls:<12} {machine:<5} {f['model']:<10} "
              f"R2adj={f['r2_t']:.3f}  quadR2={f['quad_r2_t'] if f['quad_r2_t'] is not None else 0:.3f}  "
              f"t(n) params={np.round(f['popt_t'], 6)}")

    print(f"\n=== PROYECCIONES A {MAX_TARGET:,} SEQ DE 81bp (pos+neg) ===")
    for row in budget_rows:
        ram = f"  RAM {row['ram_mb_pred']:>12,.0f} MB" if row["ram_mb_pred"] else ""
        print(f"  {row['resource_class']:<12} {row['tool']:<32} {row['time_s_pred']:>14,.1f} s{ram}")
    print(f"\nExtrapolation -> {EXTRAP_TSV}")

    n_plot = 2
    fig, axes = plt.subplots(2, len(TOOL_ORDER), figsize=(3.2 * len(TOOL_ORDER), 7))
    fit_lookup = {}
    for (tool, cls, machine), f in fits.items():
        key = (tool, cls)
        if key not in fit_lookup or (fit_lookup[key][0] != "ws" and machine == "ws"):
            fit_lookup[key] = (machine, f)
    for j, tool in enumerate(TOOL_ORDER):
        for i, (col, ylabel) in enumerate([("time_s", "seconds"),
                                           ("peak_ram_mb", "peak RAM (MB)")]):
            ax = axes[i, j]
            sub = df[df["tool"] == tool]
            if sub.empty:
                ax.set_visible(False)
                continue
            x_obs = sub["n_sequences"].values
            y_obs = sub[col].values
            for _, row in sub.iterrows():
                ax.scatter(row["n_sequences"], row[col], s=28,
                           label=row["machine"], marker="o" if row["machine"] == "ws" else "x")
            cls = "gpu-3090" if tool in GPU_TOOLS else "cpu-1thread"
            entry = fit_lookup.get((tool, cls))
            if entry:
                machine, f = entry
                n_fit = np.linspace(x_obs.min(), x_obs.max(), 100)
                pred_fit = f["fn_t"](n_fit, *f["popt_t"]) if col == "time_s" \
                    else f["popt_r"][0] * n_fit + f["popt_r"][1]
                ax.plot(n_fit, pred_fit, color="#1f77b4", lw=1.5)
                t_target = f["fn_t"](TARGETS[-1], *f["popt_t"])
                r_target = f["popt_r"][0] * TARGETS[-1] + f["popt_r"][1]
                target = t_target if col == "time_s" else r_target
                ax.annotate(f"200K -> {target:,.0f}", xy=(0.97, 0.97),
                            xycoords="axes fraction", ha="right", va="top",
                            fontsize=7, color="#c62828")
                fit_tag = f"{f['model']} R2={f['r2_t']:.2f}" if f['r2_t'] is not None else "no fit"
            else:
                fit_tag = "no fit"
            ax.set_title(f"{tool}\n({cls}, {fit_tag})", fontsize=8)
            ax.set_xlabel("n seqs (pos+neg)", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_xlim(0, x_obs.max() * 1.12)
            ax.set_ylim(0, y_obs.max() * 1.25)
            ax.grid(alpha=0.3, ls="--", lw=0.4)
            ax.tick_params(labelsize=6)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle(f"Scaling: time(n) and RAM(n) — linear axes, red = projected at {MAX_TARGET:,} seqs (81bp)", fontsize=10)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(PLOT_DIR / f"scaling.{ext}", dpi=200, bbox_inches="tight")
    print(f"Plots -> {PLOT_DIR}/scaling.{{png,svg}}")


if __name__ == "__main__":
    main()