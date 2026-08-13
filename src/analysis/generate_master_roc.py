#!/usr/bin/env python3
"""
Master ROC plot — S. pneumoniae D39V / TIGR4.
MLDSPP 0%/75%, LCNN, PromoTech HOT, iPro-MP, MEME, FIMO Prok.
"""
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "output" / "plots" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _count_positives():
    from src.config import config
    return config.n_positives


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions-dir", default=None,
                   help="Directory with prediction CSVs (default: output/predictions)")
    p.add_argument("-o", "--output", default=None,
                   help="Output filename prefix (default: master_benchmark_roc)")
    return p.parse_args()


def auc_from_csv(pos_file, neg_file):
    pos = pd.read_csv(pos_file, sep="\t")
    neg = pd.read_csv(neg_file, sep="\t")
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.hstack([pos["PRED"].values, neg["PRED"].values])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, auc(fpr, tpr)


def auc_from_ipromp(csv_file, n_pos=988, n_neg=1000):
    """iPro-MP format: Sequence,Prediction,Probability"""
    df = pd.read_csv(csv_file)
    if len(df) >= n_pos + n_neg:
        y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
        s = df["Probability"].values[:n_pos + n_neg]
        fpr, tpr, _ = roc_curve(y, s)
        return fpr, tpr, auc(fpr, tpr)
    return None


def main():
    args = parse_args()
    pr_dir = Path(args.predictions_dir) if args.predictions_dir else ROOT / "output" / "predictions"
    out_prefix = args.output or "master_benchmark_roc"

    curves = []

    # ── 1. iPro-MP sp12 ──
    ipromp_file = pr_dir / "ipromp" / "ipromp_12_predictions.csv"
    if not ipromp_file.exists():
        ipromp_file = pr_dir / "ipromp_sp12_predictions.csv"
    if ipromp_file.exists():
        content_sample = ipromp_file.read_text()[:200]
        sep = "\t" if "\t" in content_sample else ","
        df = pd.read_csv(ipromp_file, sep=sep)
        n_total = len(df)
        n_pos = _count_positives()
        n_neg = n_total - n_pos
        if "Probability" in df.columns:
            y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
            s = df["Probability"].values[:n_total]
        elif "PRED" in df.columns:
            y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
            s = df["PRED"].values[:n_total]
        else:
            y = None; s = None
        if y is not None:
            fpr, tpr, _ = roc_curve(y, s)
            curves.append(("iPro-MP (sp 12)", fpr, tpr, auc(fpr, tpr), "#3D185A", "-", 2.0))
            print(f"iPro-MP sp12: AUC={auc(fpr, tpr):.4f}")

    # ── 2. PromoTech RF-HOT ──
    hot_pos = pr_dir / "promotech/workdir/hot_pg_pos/sequences_predictions.csv"
    hot_neg = pr_dir / "promotech/workdir/hot_pg_neg/sequences_predictions.csv"
    if hot_pos.exists() and hot_neg.exists():
        fpr, tpr, a = auc_from_csv(hot_pos, hot_neg)
        curves.append(("PromoTech RF-HOT", fpr, tpr, a, "#E07614", "-", 2.0))
        print(f"PromoTech RF-HOT: AUC={a:.4f}")

    # ── 3. PromoterLCNN ──
    lcnn_pos = pr_dir / "lcnn/lcnn_pos.csv"
    lcnn_neg = pr_dir / "lcnn/lcnn_neg.csv"
    if lcnn_pos.exists() and lcnn_neg.exists():
        fpr, tpr, a = auc_from_csv(lcnn_pos, lcnn_neg)
        curves.append(("PromoterLCNN", fpr, tpr, a, "#228B22", "-", 2.0))
        print(f"PromoterLCNN: AUC={a:.4f}")

    # ── 4. FIMO + Prokaryote DB (zero-shot) ──
    fimo_prok_pos = pr_dir / "fimo_prok_pos.csv"
    fimo_prok_neg = pr_dir / "fimo_prok_neg.csv"
    if fimo_prok_pos.exists() and fimo_prok_neg.exists():
        fpr, tpr, a = auc_from_csv(fimo_prok_pos, fimo_prok_neg)
        curves.append(("FIMO (Prok DB, 838)", fpr, tpr, a, "#00BFC4", "-.", 1.5))
        print(f"FIMO_PROK: AUC={a:.4f}")

    # ── 5. MEME Suite (STREME+FIMO) ──
    meme_pos = pr_dir / "meme_pos.csv"
    meme_neg = pr_dir / "meme_neg.csv"
    if meme_pos.exists() and meme_neg.exists():
        fpr, tpr, a = auc_from_csv(meme_pos, meme_neg)
        curves.append(("MEME Suite (STREME+FIMO)", fpr, tpr, a, "#5A9BD5", "-", 1.8))
        print(f"MEME: AUC={a:.4f}")

    # ── 6. MLDSPP 0% strepto (cross-species) ──
    if (pr_dir / "mldspp_pos.csv").exists():
        fpr, tpr, a = auc_from_csv(pr_dir/"mldspp_pos.csv", pr_dir/"mldspp_neg.csv")
        curves.append(("MLDSPP XGBoost (0% strepto)", fpr, tpr, a, "#942C76", "-", 1.8))
        print(f"MLDSPP 0%: AUC={a:.4f}")

    # ── 7. MLDSPP 75% strepto* ──
    if (pr_dir / "mldspp_75spn_pos.csv").exists():
        fpr, tpr, a = auc_from_csv(pr_dir/"mldspp_75spn_pos.csv", pr_dir/"mldspp_75spn_neg.csv")
        curves.append(("MLDSPP XGBoost (75% strepto)*", fpr, tpr, a, "#B07AA1", "--", 2.5))
        print(f"MLDSPP 75%: AUC={a:.4f}")

    # ── PLOT ──
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=300)

    for name, fpr, tpr, auc_val, color, ls, lw in curves:
        ax.plot(fpr, tpr, lw=lw, ls=ls, color=color, alpha=0.9,
                label=f"{name}  (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)

    ds_name = pr_dir.name.replace("predictions_", "").replace("tigr4_", "TIGR4 ").replace("d39v", "D39V").replace("_", " ").replace("ext nogc", "2K").replace("high nogc", "high").title()
    pos_csv = next(pr_dir.glob("*pos*.csv"), None)
    neg_csv = next(pr_dir.glob("*neg*.csv"), None)
    n_pos = len(pd.read_csv(pos_csv, sep="\t")) if pos_csv and pos_csv.exists() else "?"
    n_neg = len(pd.read_csv(neg_csv, sep="\t")) if neg_csv and neg_csv.exists() else "?"
    title = f"Promoter Prediction — {ds_name} ({n_pos} pos + {n_neg} neg)"
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

    ax.text(0.98, 0.04, "* 75% S. pneumoniae in training",
            transform=ax.transAxes, fontsize=7, ha="right", color="#B07AA1", fontstyle="italic")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}.svg", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: {OUT_DIR}/{out_prefix}.{{svg,png}}")
    print(f"\nCurves plotted ({len(curves)}):")
    for name, _, _, a, _, _, _ in sorted(curves, key=lambda x: -x[3]):
        star = " *" if "*" in name else ""
        print(f"  {name:<35} AUC={a:.4f}{star}")


if __name__ == "__main__":
    main()
