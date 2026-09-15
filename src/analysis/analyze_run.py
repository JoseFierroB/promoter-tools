#!/usr/bin/env python3
"""Analyze one CLI run: ROC + AUC rows for a single dataset.

Agnostic: works on any run dir produced by `src/cli.py run`. Reads the
canonical layout predictions/{name}_{fam}[.tsv] (new) with fallback to the
legacy predictions/{name}_{fam}_{fam}[.tsv] layout, and writes
plots/roc_auc_{name}.{png,svg,pdf} + metrics_rows.tsv into the run dir.
No dependency on canonical dataset names (no DS_DISPLAY lookup).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

try:
    from bench_labels import PALETTE, TOOL_ORDER  # canonical (single source)
except Exception:  # fallback when run as script without package context
    PALETTE = {
        "MLDSPP 0% [BDT]":       {"color": "#880E4F", "ls": "-",  "lw": 1.8, "method": "BDT"},
        "MLDSPP 75% [BDT]":      {"color": "#C2185B", "ls": "-",  "lw": 1.8, "method": "BDT"},
        "PromoTech RF-HOT [RF]": {"color": "#EF6C00", "ls": "-",  "lw": 1.9, "method": "RF"},
        "PromoterLCNN [CNN]":    {"color": "#2E7D32", "ls": "-",  "lw": 1.9, "method": "CNN"},
        "prompt [NN]":           {"color": "#0288D1", "ls": "-",  "lw": 1.8, "method": "NN"},
        "ProkBERT-mini [gLM]":   {"color": "#D81B60", "ls": "-",  "lw": 2.3, "method": "gLM"},
        "iPro-MP [gLM]":         {"color": "#7E57C2", "ls": "-",  "lw": 2.1, "method": "gLM"},
        "FIMO ProkDB [PWMs]":    {"color": "#00897B", "ls": "--", "lw": 1.7, "method": "PWMs"},
        "STREME+FIMO [motif]":   {"color": "#795548", "ls": ":",  "lw": 1.7, "method": "motif"},
    }
    TOOL_ORDER = [
        "MLDSPP 0% [BDT]", "MLDSPP 75% [BDT]", "PromoTech RF-HOT [RF]",
        "PromoterLCNN [CNN]", "prompt [NN]", "ProkBERT-mini [gLM]",
        "iPro-MP [gLM]", "FIMO ProkDB [PWMs]", "STREME+FIMO [motif]",
    ]

# registry key -> (display label, loader kind) — insertion order = TOOL_ORDER above
TOOLS = {
    "mldspp": ("MLDSPP 0% [BDT]", "posneg_dir"),
    "mldspp_75": ("MLDSPP 75% [BDT]", "posneg_dir"),
    "promotech_hot": ("PromoTech RF-HOT [RF]", "posneg_dir"),
    "lcnn": ("PromoterLCNN [CNN]", "posneg_dir"),
    "prompt": ("prompt [NN]", "single_tsv"),
    "prokbert": ("ProkBERT-mini [gLM]", "single_tsv"),
    "ipromp_sp12": ("iPro-MP [gLM]", "ipromp_dir"),
    "fimo_prok": ("FIMO ProkDB [PWMs]", "fimo_dir"),
    "meme": ("STREME+FIMO [motif]", "meme_dir"),
}
FAM = {"mldspp": "mldspp", "mldspp_75": "mldspp_75",
       "promotech_hot": "promotech", "lcnn": "lcnn", "prompt": "prompt",
       "prokbert": "prokbert", "ipromp_sp12": "ipromp",
       "fimo_prok": "fimo", "meme": "meme"}


def _posneg(p, n):
    pos = pd.read_csv(p, sep="\t")
    neg = pd.read_csv(n, sep="\t")
    return (np.hstack([np.ones(len(pos)), np.zeros(len(neg))]),
            np.hstack([pos["PRED"].values, neg["PRED"].values]))


def _bases(pred_root: Path, name: str, fam: str, legacy_suffix: str = ""):
    """Candidate base paths, new layout first, legacy layout as fallback."""
    cands = [pred_root / f"{name}_{fam}"]
    legacy = f"{name}_{fam}_{fam}{legacy_suffix}"
    if legacy != f"{name}_{fam}":
        cands.append(pred_root / legacy)
    return [c for c in cands if c.exists()]


def load_tool(pred_root: Path, name: str, key: str, n_pos_hint=None):
    """Return (y_true, y_score) or None.

    n_pos_hint fixes the iPro-MP split: that file has no LABEL column, so
    without a hint the split defaults to len//2, which mislabels when the
    total is odd (e.g. 1727+1738=3465 -> 1732/1733).
    """
    fam = FAM[key]
    kind = TOOLS[key][1]
    if kind == "single_tsv":
        for f in [pred_root / f"{name}_{fam}.tsv",
                  pred_root / f"{name}_{fam}_{fam}.tsv"]:
            if f.exists():
                df = pd.read_csv(f, sep="\t")
                return df["LABEL"].values, df["PRED"].values
        return None
    if kind == "ipromp_dir":
        for base in _bases(pred_root, name, fam):
            f = base / "ipromp" / "ipromp_12_predictions.csv"
            if f.exists():
                df = pd.read_csv(f, sep="\t")
                col = "PRED" if "PRED" in df.columns else "Probability"
                n_pos = n_pos_hint if n_pos_hint else len(df) // 2
                return (np.hstack([np.ones(n_pos), np.zeros(len(df) - n_pos)]),
                        df[col].values)
        return None
    if kind == "posneg_dir":
        for base in _bases(pred_root, name, fam):
            if key == "lcnn":
                p, n = base / "lcnn" / "lcnn_pos.csv", base / "lcnn" / "lcnn_neg.csv"
            elif key == "promotech_hot":
                p = base / "promotech" / "workdir" / "hot_pg_pos" / "sequences_predictions.csv"
                n = base / "promotech" / "workdir" / "hot_pg_neg" / "sequences_predictions.csv"
            elif key == "mldspp":
                p, n = base / "mldspp_pos.csv", base / "mldspp_neg.csv"
            elif key == "mldspp_75":
                p, n = base / "mldspp_75spn_pos.csv", base / "mldspp_75spn_neg.csv"
                if not (p.exists() and n.exists()):
                    p, n = base / "mldspp_pos.csv", base / "mldspp_neg.csv"
            else:
                continue
            if p.exists() and n.exists():
                return _posneg(p, n)
        return None
    if kind == "fimo_dir":
        for base in _bases(pred_root, name, fam, legacy_suffix="_dir"):
            p, n = base / "fimo_prok_pos.csv", base / "fimo_prok_neg.csv"
            if p.exists() and n.exists():
                return _posneg(p, n)
        # legacy layout with _fimo_fimo_dir suffix
        legacy = pred_root / f"{name}_{fam}_{fam}_dir"
        p, n = legacy / "fimo_prok_pos.csv", legacy / "fimo_prok_neg.csv"
        if p.exists() and n.exists():
            return _posneg(p, n)
        return None
    if kind == "meme_dir":
        for base in _bases(pred_root, name, fam):
            p, n = base / "meme_pos.csv", base / "meme_neg.csv"
            if p.exists() and n.exists():
                return _posneg(p, n)
        return None
    return None


def load_all(pred_root: Path, name: str, only=None):
    """Load all tools in TOOLS order. Returns {key: (label, y_true, y_score)}.

    Two passes: non-iPro-MP tools first to get the consensus n_pos, then
    iPro-MP with that hint (its file lacks a LABEL column).
    """
    keys = [k for k in TOOLS if only is None or k in only]
    scored = {}
    for key in keys:
        if key == "ipromp_sp12":
            continue
        loaded = load_tool(pred_root, name, key)
        if loaded is not None:
            scored[key] = (TOOLS[key][0],) + loaded
    n_pos_hint = None
    if scored:
        counts = [int(v[1].sum()) for v in scored.values()]
        n_pos_hint = max(set(counts), key=counts.count)
    if "ipromp_sp12" in keys:
        loaded = load_tool(pred_root, name, "ipromp_sp12", n_pos_hint)
        if loaded is not None:
            scored["ipromp_sp12"] = (TOOLS["ipromp_sp12"][0],) + loaded
    return {k: scored[k] for k in keys if k in scored}


def analyze_run(pred_root: Path, name: str, out_dir: Path) -> pd.DataFrame:
    """ROC plot + metrics rows for one run dir. Returns metrics DataFrame."""
    curves, rows = [], []
    for key, (label, y_true, y_score) in load_all(pred_root, name).items():
        n_pos = int(y_true.sum())
        if np.all(y_score == y_score[0]):
            fpr, tpr, a = np.array([0.0, 1.0]), np.array([0.0, 1.0]), 0.500
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            a = auc(fpr, tpr)
        curves.append((label, fpr, tpr, a))
        rows.append({"dataset": name, "tool": label,
                     "n_pos": n_pos, "n_neg": len(y_true) - n_pos,
                     "auc": round(float(a), 4)})
    if not curves:
        print("  [analyze] no predictions found, skipping plots")
        return pd.DataFrame(rows)

    # canonical: 1_inference for ROC, 3_tables for metrics
    try:
        from run_layout import RunLayout as _RL
    except ImportError:
        try:
            from src.analysis.run_layout import RunLayout as _RL
        except ImportError:
            _RL = None
    if _RL is not None:
        try:
            layout = _RL(out_dir)
            plots = layout.inference
            plots.mkdir(parents=True, exist_ok=True)
            tables = layout.tables
            tables.mkdir(parents=True, exist_ok=True)
        except Exception:
            plots = out_dir / "1_inference"
            plots.mkdir(parents=True, exist_ok=True)
            tables = out_dir / "3_tables"
            tables.mkdir(parents=True, exist_ok=True)
    else:
        plots = out_dir / "1_inference"
        plots.mkdir(parents=True, exist_ok=True)
        tables = out_dir / "3_tables"
        tables.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    order_idx = {t: i for i, t in enumerate(TOOL_ORDER)}
    for cname, fpr, tpr, a in sorted(curves, key=lambda x: order_idx.get(x[0], 999)):
        style = PALETTE.get(cname, {"color": "#333333", "ls": "-", "lw": 1.8})
        ax.plot(fpr, tpr, color=style["color"], linestyle=style["ls"],
                linewidth=style["lw"], label=f"{cname} (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    n_pos, n_neg = rows[0]["n_pos"], rows[0]["n_neg"]
    # dataset and configuration always declared
    try:
        from bench_labels import DS_DISPLAY as _DS
    except Exception:
        _DS = {}
    import re as _re
    ds_disp = _DS.get(name, name)
    cfg = Path(out_dir).name if _re.match(r"^\d+cpu(-gpu)?$", Path(out_dir).name) else ""
    cfg_disp = {"1cpu": "1CPU", "16cpu": "16CPU", "1cpu-gpu": "1CPU+GPU", "16cpu-gpu": "16CPU+GPU"}.get(cfg, cfg)
    title = f"Receiver Operating Characteristic (ROC)\n{ds_disp} (N={n_pos + n_neg:,})" + (f" — {cfg_disp}" if cfg_disp else "")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"roc_auc_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [analyze] ROC saved: 1_inference/roc_auc_{name}.png/.pdf")

    met = pd.DataFrame(rows)
    # canonical tables location
    try:
        met_path = tables / "metrics_rows.tsv"
    except NameError:
        met_path = out_dir / "3_tables" / "metrics_rows.tsv"
    met_path.parent.mkdir(parents=True, exist_ok=True)
    met.to_csv(met_path, sep="\t", index=False)
    print(f"  [analyze] metrics: {len(met)} tool rows -> 3_tables/metrics_rows.tsv")
    # legacy compat: also keep root metrics_rows for old readers (symlink if possible)
    try:
        legacy = out_dir / "metrics_rows.tsv"
        if not legacy.exists() and met_path.exists():
            try:
                legacy.symlink_to(Path("3_tables") / "metrics_rows.tsv")
            except Exception:
                import shutil as _sh
                _sh.copy2(met_path, legacy)
    except Exception:
        pass
    return met


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Analyze one CLI run (single dataset).")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--db", "--name", dest="name", required=True)
    p.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    analyze_run(Path(a.pred_dir), a.name, Path(a.output))
