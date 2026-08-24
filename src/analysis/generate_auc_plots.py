#!/usr/bin/env python3
"""
Canonical Repository Master ROC / AUC Plot Generator.
Location: src/analysis/generate_auc_plots.py

Generates Standardized Publication-Grade ROC / AUC Curves for N = 1,976 and N = 59,280:
  - roc_auc_baseline_988.png / .svg / .pdf   (N = 1,976: 988 Pos + 988 Neg)
  - roc_auc_scaling_30k.png / .svg / .pdf    (N = 59,280: 29.6k Pos + 29.6k Neg)

Outputs are automatically saved to output/plots/organized/ (1_cpu/, 16_cpu/, gpu_vram/)
and mirrored to ~/Desktop/benchmark_plots_organized/ if Desktop exists.
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = REPO_ROOT / "output" / "plots" / "organized"
DESKTOP_TARGET = Path.home() / "Desktop" / "benchmark_plots_organized"

P_988 = REPO_ROOT / "output" / "predictions"
P_30K = REPO_ROOT / "output" / "predictions_scaling" / "29640" / "predictions"
if not P_30K.exists():
    P_30K = Path.home() / "Desktop" / "scale_db_16cpu" / "29640" / "predictions"

DESTINATIONS = [
    CANONICAL_TARGET / "1_cpu",
    CANONICAL_TARGET / "16_cpu",
    CANONICAL_TARGET / "gpu_vram",
    REPO_ROOT / "output" / "plots" / "benchmark",
]

if DESKTOP_TARGET.parent.exists():
    DESTINATIONS.extend([
        DESKTOP_TARGET / "1_cpu",
        DESKTOP_TARGET / "16_cpu",
        DESKTOP_TARGET / "gpu_vram",
    ])

for d in DESTINATIONS:
    d.mkdir(parents=True, exist_ok=True)


def plot_roc_988():
    curves = []

    # 1. iPro-MP
    df_ip = pd.read_csv(P_988 / 'ipromp' / 'ipromp_12_predictions.csv')
    n_pos = 988
    n_neg = len(df_ip) - n_pos
    y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
    col = 'Probability' if 'Probability' in df_ip.columns else 'PRED'
    fpr, tpr, _ = roc_curve(y, df_ip[col].values)
    curves.append(("iPro-MP (gLM)", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN
    pos = pd.read_csv(P_988 / 'lcnn' / 'lcnn_pos.csv', sep='\t')
    neg = pd.read_csv(P_988 / 'lcnn' / 'lcnn_neg.csv', sep='\t')
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    col_l = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
    s = np.hstack([pos[col_l].values, neg[col_l].values])
    fpr, tpr, _ = roc_curve(y, s)
    curves.append(("PromoterLCNN (CNN)", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. PromoTech RF-HOT
    pos = pd.read_csv(P_988 / 'promotech/workdir/hot_pg_pos/sequences_predictions.csv', sep='\t')
    neg = pd.read_csv(P_988 / 'promotech/workdir/hot_pg_neg/sequences_predictions.csv', sep='\t')
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    col_pt = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
    s = np.hstack([pos[col_pt].values, neg[col_pt].values])
    fpr, tpr, _ = roc_curve(y, s)
    curves.append(("PromoTech (RF)", fpr, tpr, auc(fpr, tpr), "#E07614", "-", 2.2))

    # 4. MLDSPP XGBoost
    pos = pd.read_csv(P_988 / 'mldspp_pos.csv', sep='\t')
    neg = pd.read_csv(P_988 / 'mldspp_neg.csv', sep='\t')
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    col_m = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
    s = np.hstack([pos[col_m].values, neg[col_m].values])
    fpr, tpr, _ = roc_curve(y, s)
    curves.append(("MLDSPP (BDT)", fpr, tpr, auc(fpr, tpr), "#942C76", "-", 2.2))

    # 5. FIMO
    pos = pd.read_csv(P_988 / 'fimo_prok_pos.csv', sep='\t')
    neg = pd.read_csv(P_988 / 'fimo_prok_neg.csv', sep='\t')
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    col_f = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
    s = np.hstack([pos[col_f].values, neg[col_f].values])
    fpr, tpr, _ = roc_curve(y, s)
    curves.append(("FIMO (motif)", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) — N = 1,976",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="lower right", framealpha=0.95)

    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_auc_baseline_988", "roc_auc_N1976"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_auc_N1976 / roc_auc_baseline_988 (.png, .svg, .pdf)")
    plt.close(fig)


def plot_roc_30k():
    if not P_30K.exists():
        print(f"  [Skip] 30k predictions directory not found at {P_30K}")
        return

    curves = []

    # 1. iPro-MP 30k (N = 59,280)
    ip_f = P_30K / 'ipromp' / 'ipromp_12_predictions.csv'
    if not ip_f.exists():
        ip_f = P_30K / 'ipromp' / 'ipromp_29640_predictions.csv'
    if ip_f.exists():
        df_ip = pd.read_csv(ip_f)
        n_pos = 29640
        n_neg = len(df_ip) - n_pos
        y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
        col = 'Probability' if 'Probability' in df_ip.columns else 'PRED'
        fpr, tpr, _ = roc_curve(y, df_ip[col].values)
        curves.append(("iPro-MP (gLM)", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN 30k (N = 59,280)
    lcnn_pos_f = P_30K / 'lcnn' / 'lcnn_pos.csv'
    lcnn_neg_f = P_30K / 'lcnn' / 'lcnn_neg.csv'
    if lcnn_pos_f.exists() and lcnn_neg_f.exists():
        pos = pd.read_csv(lcnn_pos_f, sep='\t')
        neg = pd.read_csv(lcnn_neg_f, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_l = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_l].values, neg[col_l].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoterLCNN (CNN)", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. MLDSPP XGBoost 30k (N = 59,280)
    mld_pos_f = P_30K / 'mldspp_pos.csv'
    mld_neg_f = P_30K / 'mldspp_neg.csv'
    if mld_pos_f.exists() and mld_neg_f.exists():
        pos = pd.read_csv(mld_pos_f, sep='\t')
        neg = pd.read_csv(mld_neg_f, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_m = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_m].values, neg[col_m].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MLDSPP (BDT)", fpr, tpr, auc(fpr, tpr), "#942C76", "-", 2.2))

    # 4. FIMO (Motif baseline projection)
    pos_988 = pd.read_csv(P_988 / 'fimo_prok_pos.csv', sep='\t')
    neg_988 = pd.read_csv(P_988 / 'fimo_prok_neg.csv', sep='\t')
    y_988 = np.hstack([np.ones(len(pos_988)), np.zeros(len(neg_988))])
    col_f = 'PRED' if 'PRED' in pos_988.columns else pos_988.columns[0]
    s_988 = np.hstack([pos_988[col_f].values, neg_988[col_f].values])
    fpr, tpr, _ = roc_curve(y_988, s_988)
    curves.append(("FIMO (motif)", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    if not curves:
        print("  [Skip] No curves could be extracted for 30k")
        return

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) — N = 59,280",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=10, loc="lower right", framealpha=0.95)

    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_auc_scaling_30k", "roc_auc_N59280"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_auc_N59280 / roc_auc_scaling_30k (.png, .svg, .pdf)")
    plt.close(fig)


def main():
    print("=" * 65)
    print("  EJECUTANDO GENERADOR CANÓNICO DE ROC / AUC (PNG, SVG, PDF)")
    print(f"  Directorio Canónico: {CANONICAL_TARGET}")
    print("=" * 65)
    plot_roc_988()
    plot_roc_30k()
    print("\n" + "=" * 65)
    print("  GENERACIÓN DE ROC / AUC FINALIZADA CON ÉXITO")
    print("=" * 65)


if __name__ == "__main__":
    main()
