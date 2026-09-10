#!/usr/bin/env python3
"""
Canonical Repository Master ROC / AUC Plot Generator.
Location: src/analysis/generate_auc_plots.py

Generates Standardized Publication-Grade ROC / AUC Curves across all 6 datasets:
  1. roc_auc_baseline_989.png / .svg / .pdf          (D39V N = 1,978: 989 Pos + 989 Neg)
  2. roc_auc_scaling_30k.png / .svg / .pdf           (D39V Scaling N = 59,280: 29.6k Pos + 29.6k Neg)
  3. roc_tigr4_high.png / .svg / .pdf                (TIGR4 High-Confidence N = 1,476: 738 Pos + 738 Neg)
  4. roc_tigr4_all.png / .svg / .pdf                 (TIGR4 All TSS N = 4,000: 2,000 Pos + 2,000 Neg)
  5. roc_combined_d39v_tigr4_high.png / .svg / .pdf  (Combined D39V + TIGR4 High N = 3,454: 1,727 Pos + 1,727 Neg)
  6. roc_combined_d39v_tigr4_all.png / .svg / .pdf   (Combined D39V + TIGR4 All N = 5,978: 2,989 Pos + 2,989 Neg)

Outputs are automatically saved to output/plots/organized/ (1_cpu/, 16_cpu/, gpu_vram/)
and mirrored to ~/Desktop/benchmark_plots_organized/ and ~/Desktop/roc_curves_master/.
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
MASTER_ROC_TARGET = Path.home() / "Desktop" / "roc_curves_master"

P_988 = REPO_ROOT / "output" / "predictions"
P_T4_HIGH = REPO_ROOT / "output" / "tigr4" / "predictions"
P_T4_EXT = REPO_ROOT / "output" / "predictions" / "tigr4_extended"

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
        MASTER_ROC_TARGET,
    ])

for d in DESTINATIONS:
    d.mkdir(parents=True, exist_ok=True)


def plot_roc_989():
    curves = []

    # 1. iPro-MP (H. pylori) [gLM]
    ip_f = P_988 / 'ipromp' / 'ipromp_12_predictions.csv'
    if ip_f.exists():
        df_ip = pd.read_csv(ip_f)
        n_pos = 989
        n_neg = len(df_ip) - n_pos
        y = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
        col = 'Probability' if 'Probability' in df_ip.columns else 'PRED'
        fpr, tpr, _ = roc_curve(y, df_ip[col].values)
        curves.append(("iPro-MP (H. pylori) [gLM]", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN [CNN]
    lcnn_pos = P_988 / 'lcnn' / 'lcnn_pos.csv'
    lcnn_neg = P_988 / 'lcnn' / 'lcnn_neg.csv'
    if lcnn_pos.exists() and lcnn_neg.exists():
        pos = pd.read_csv(lcnn_pos, sep='\t')
        neg = pd.read_csv(lcnn_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_l = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_l].values, neg[col_l].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. PromoTech RF-HOT [RF]
    pt_pos = P_988 / 'promotech/workdir/hot_pg_pos/sequences_predictions.csv'
    pt_neg = P_988 / 'promotech/workdir/hot_pg_neg/sequences_predictions.csv'
    if pt_pos.exists() and pt_neg.exists():
        pos = pd.read_csv(pt_pos, sep='\t')
        neg = pd.read_csv(pt_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_pt = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_pt].values, neg[col_pt].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoTech RF-HOT [RF]", fpr, tpr, auc(fpr, tpr), "#E07614", "-", 2.2))

    # 4. MLDSPP 75%* (Fine-tuned) [BDT]
    mld75_pos = P_988 / 'mldspp_75spn_pos.csv'
    mld75_neg = P_988 / 'mldspp_75spn_neg.csv'
    if mld75_pos.exists() and mld75_neg.exists():
        pos = pd.read_csv(mld75_pos, sep='\t')
        neg = pd.read_csv(mld75_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_m75 = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_m75].values, neg[col_m75].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MLDSPP 75%* [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", "-", 2.3))

    # 5. MLDSPP 0% (Zero-shot) [BDT]
    mld0_pos = P_988 / 'mldspp_pos.csv'
    mld0_neg = P_988 / 'mldspp_neg.csv'
    if mld0_pos.exists() and mld0_neg.exists():
        pos = pd.read_csv(mld0_pos, sep='\t')
        neg = pd.read_csv(mld0_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_m = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_m].values, neg[col_m].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", ":", 2.0))

    # 6. MEME Suite (STREME+FIMO) [De novo]
    meme_pos = P_988 / 'meme_pos.csv'
    meme_neg = P_988 / 'meme_neg.csv'
    if meme_pos.exists() and meme_neg.exists():
        pos = pd.read_csv(meme_pos, sep='\t')
        neg = pd.read_csv(meme_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_me = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_me].values, neg[col_me].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MEME Suite [De novo]", fpr, tpr, auc(fpr, tpr), "#3949AB", "-.", 2.0))

    # 7. FIMO + Prokaryote DB [PWM]
    fimo_pos = P_988 / 'fimo_prok_pos.csv'
    fimo_neg = P_988 / 'fimo_prok_neg.csv'
    if fimo_pos.exists() and fimo_neg.exists():
        pos = pd.read_csv(fimo_pos, sep='\t')
        neg = pd.read_csv(fimo_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_f = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_f].values, neg[col_f].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("FIMO + Prokaryote DB [PWM]", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) — D39V Baseline (N = 1,978)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_auc_baseline_989", "roc_auc_N1978", "roc_auc_baseline_988", "roc_auc_N1976"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_auc_N1978 / roc_auc_baseline_989 (alias 988/N1976) (.png, .svg, .pdf)")
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
        curves.append(("iPro-MP (H. pylori) [gLM]", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

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
        curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

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
        curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", ":", 2.2))

    # 4. FIMO (Motif baseline projection)
    pos_988 = pd.read_csv(P_988 / 'fimo_prok_pos.csv', sep='\t')
    neg_988 = pd.read_csv(P_988 / 'fimo_prok_neg.csv', sep='\t')
    y_988 = np.hstack([np.ones(len(pos_988)), np.zeros(len(neg_988))])
    col_f = 'PRED' if 'PRED' in pos_988.columns else pos_988.columns[0]
    s_988 = np.hstack([pos_988[col_f].values, neg_988[col_f].values])
    fpr, tpr, _ = roc_curve(y_988, s_988)
    curves.append(("FIMO + Prokaryote DB [PWM]", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

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
    ax.set_title("Receiver Operating Characteristic (ROC) — D39V Scaling (N = 59,280)",
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


def plot_roc_tigr4_high():
    curves = []
    p_high = REPO_ROOT / 'output' / 'predictions' / 'tigr4_high'
    ip_f = REPO_ROOT / 'output' / 'predictions' / 'ipromp' / 'ipromp_tigr4_predictions.csv'

    # 1. iPro-MP (H. pylori) [gLM]
    if ip_f.exists():
        df_ip = pd.read_csv(ip_f)
        y_ip = np.hstack([np.ones(738), np.zeros(len(df_ip) - 738)])
        col = 'Probability' if 'Probability' in df_ip.columns else 'PRED'
        fpr, tpr, _ = roc_curve(y_ip, df_ip[col].values)
        curves.append(("iPro-MP (H. pylori) [gLM]", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN [CNN]
    lcnn_pos = p_high / 'lcnn' / 'lcnn_pos.csv'
    lcnn_neg = p_high / 'lcnn' / 'lcnn_neg.csv'
    if lcnn_pos.exists() and lcnn_neg.exists():
        pos = pd.read_csv(lcnn_pos, sep='\t')
        neg = pd.read_csv(lcnn_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_l = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_l].values, neg[col_l].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. PromoTech RF-HOT [RF]
    pt_pos = p_high / 'promotech_hot/promotech/workdir/hot_pg_pos/sequences_predictions.csv'
    pt_neg = p_high / 'promotech_hot/promotech/workdir/hot_pg_neg/sequences_predictions.csv'
    if pt_pos.exists() and pt_neg.exists():
        pos = pd.read_csv(pt_pos, sep='\t')
        neg = pd.read_csv(pt_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_pt = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_pt].values, neg[col_pt].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoTech RF-HOT [RF]", fpr, tpr, auc(fpr, tpr), "#E07614", "-", 2.2))

    # 4. PromoTech RF-TETRA [RF]
    pt_t_pos = p_high / 'promotech_tetra/promotech/workdir/tetra_pg_pos/sequences_predictions.csv'
    pt_t_neg = p_high / 'promotech_tetra/promotech/workdir/tetra_pg_neg/sequences_predictions.csv'
    if pt_t_pos.exists() and pt_t_neg.exists():
        pos = pd.read_csv(pt_t_pos, sep='\t')
        neg = pd.read_csv(pt_t_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_ptt = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_ptt].values, neg[col_ptt].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoTech RF-TETRA [RF]", fpr, tpr, auc(fpr, tpr), "#FB8C00", "--", 1.8))

    # 5. MLDSPP 75%* [BDT]
    mld75_pos = P_T4_HIGH / 'mldspp_75spn_pos.csv'
    mld75_neg = P_T4_HIGH / 'mldspp_75spn_neg.csv'
    if mld75_pos.exists() and mld75_neg.exists():
        pos = pd.read_csv(mld75_pos, sep='\t')
        neg = pd.read_csv(mld75_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_m75 = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_m75].values, neg[col_m75].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MLDSPP 75%* [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", "-", 2.3))

    # 6. MLDSPP 0% [BDT]
    mld_pos = p_high / 'mldspp_pos.csv'
    mld_neg = p_high / 'mldspp_neg.csv'
    if mld_pos.exists() and mld_neg.exists():
        pos = pd.read_csv(mld_pos, sep='\t')
        neg = pd.read_csv(mld_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_m = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_m].values, neg[col_m].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", ":", 2.0))

    # 7. MEME Suite [De novo]
    meme_pos = P_T4_HIGH / 'meme_pos.csv'
    meme_neg = P_T4_HIGH / 'meme_neg.csv'
    if meme_pos.exists() and meme_neg.exists():
        pos = pd.read_csv(meme_pos, sep='\t')
        neg = pd.read_csv(meme_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_me = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_me].values, neg[col_me].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MEME Suite [De novo]", fpr, tpr, auc(fpr, tpr), "#3949AB", "-.", 2.0))

    # 8. FIMO + Prokaryote DB [PWM]
    fimo_pos = P_T4_HIGH / 'fimo_prok_pos.csv'
    fimo_neg = P_T4_HIGH / 'fimo_prok_neg.csv'
    if fimo_pos.exists() and fimo_neg.exists():
        pos = pd.read_csv(fimo_pos, sep='\t')
        neg = pd.read_csv(fimo_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_f = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_f].values, neg[col_f].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("FIMO + Prokaryote DB [PWM]", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) — TIGR4 High-Confidence (N = 1,476)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_tigr4_high", "roc_auc_tigr4_high"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_tigr4_high / roc_auc_tigr4_high (.png, .svg, .pdf)")
    plt.close(fig)


def plot_roc_tigr4_all():
    curves = []
    p_ext = P_T4_EXT

    # 1. iPro-MP (H. pylori) [gLM]
    ip_ext = p_ext / 'ipromp_tigr4_extended.csv'
    if ip_ext.exists():
        df_ip = pd.read_csv(ip_ext)
        y_ip = np.hstack([np.ones(2000), np.zeros(len(df_ip) - 2000)])
        col = 'Probability' if 'Probability' in df_ip.columns else 'PRED'
        fpr, tpr, _ = roc_curve(y_ip, df_ip[col].values)
        curves.append(("iPro-MP (H. pylori) [gLM]", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN [CNN]
    lcnn_pos = p_ext / 'lcnn' / 'lcnn_pos.csv'
    lcnn_neg = p_ext / 'lcnn' / 'lcnn_neg.csv'
    if lcnn_pos.exists() and lcnn_neg.exists():
        pos = pd.read_csv(lcnn_pos, sep='\t')
        neg = pd.read_csv(lcnn_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_l = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_l].values, neg[col_l].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. PromoTech RF-HOT [RF]
    pt_pos = p_ext / 'promotech_hot/promotech/workdir/hot_pg_pos/sequences_predictions.csv'
    pt_neg = p_ext / 'promotech_hot/promotech/workdir/hot_pg_neg/sequences_predictions.csv'
    if pt_pos.exists() and pt_neg.exists():
        pos = pd.read_csv(pt_pos, sep='\t')
        neg = pd.read_csv(pt_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_pt = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_pt].values, neg[col_pt].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("PromoTech RF-HOT [RF]", fpr, tpr, auc(fpr, tpr), "#E07614", "-", 2.2))

    # 4. MLDSPP 0% [BDT]
    mld_pos = p_ext / 'mldspp_pos.csv'
    mld_neg = p_ext / 'mldspp_neg.csv'
    if mld_pos.exists() and mld_neg.exists():
        pos = pd.read_csv(mld_pos, sep='\t')
        neg = pd.read_csv(mld_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_m = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_m].values, neg[col_m].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", ":", 2.0))

    # 5. FIMO + Prokaryote DB [PWM]
    fimo_pos = p_ext / 'fimo_prok_pos.csv'
    fimo_neg = p_ext / 'fimo_prok_neg.csv'
    if fimo_pos.exists() and fimo_neg.exists():
        pos = pd.read_csv(fimo_pos, sep='\t')
        neg = pd.read_csv(fimo_neg, sep='\t')
        y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        col_f = 'PRED' if 'PRED' in pos.columns else pos.columns[0]
        s = np.hstack([pos[col_f].values, neg[col_f].values])
        fpr, tpr, _ = roc_curve(y, s)
        curves.append(("FIMO + Prokaryote DB [PWM]", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) — TIGR4 All TSS / Extended (N = 4,000)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_tigr4_all", "roc_auc_tigr4_all"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_tigr4_all / roc_auc_tigr4_all (.png, .svg, .pdf)")
    plt.close(fig)


def plot_roc_combined_d39v_tigr4_high():
    curves = []

    # 1. iPro-MP
    df_ip_d = pd.read_csv(P_988 / 'ipromp/ipromp_12_predictions.csv')
    y_ip_d = np.hstack([np.ones(989), np.zeros(len(df_ip_d) - 989)])
    s_ip_d = df_ip_d['Probability' if 'Probability' in df_ip_d.columns else 'PRED'].values

    df_ip_t = pd.read_csv(REPO_ROOT / 'output/predictions/ipromp/ipromp_tigr4_predictions.csv')
    y_ip_t = np.hstack([np.ones(738), np.zeros(len(df_ip_t) - 738)])
    s_ip_t = df_ip_t['PRED'].values

    y_ip = np.hstack([y_ip_d, y_ip_t])
    s_ip = np.hstack([s_ip_d, s_ip_t])
    fpr, tpr, _ = roc_curve(y_ip, s_ip)
    curves.append(("iPro-MP (H. pylori) [gLM]", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN
    pos_lc_d = pd.read_csv(P_988 / 'lcnn/lcnn_pos.csv', sep='\t')
    neg_lc_d = pd.read_csv(P_988 / 'lcnn/lcnn_neg.csv', sep='\t')
    y_lc_d = np.hstack([np.ones(len(pos_lc_d)), np.zeros(len(neg_lc_d))])
    s_lc_d = np.hstack([pos_lc_d.iloc[:,0].values, neg_lc_d.iloc[:,0].values])

    pos_lc_t = pd.read_csv(REPO_ROOT / 'output/predictions/tigr4_high/lcnn/lcnn_pos.csv', sep='\t')
    neg_lc_t = pd.read_csv(REPO_ROOT / 'output/predictions/tigr4_high/lcnn/lcnn_neg.csv', sep='\t')
    y_lc_t = np.hstack([np.ones(len(pos_lc_t)), np.zeros(len(neg_lc_t))])
    s_lc_t = np.hstack([pos_lc_t.iloc[:,0].values, neg_lc_t.iloc[:,0].values])

    y_lc = np.hstack([y_lc_d, y_lc_t])
    s_lc = np.hstack([s_lc_d, s_lc_t])
    fpr, tpr, _ = roc_curve(y_lc, s_lc)
    curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. PromoTech RF-HOT
    pos_pt_d = pd.read_csv(P_988 / 'promotech/workdir/hot_pg_pos/sequences_predictions.csv', sep='\t')
    neg_pt_d = pd.read_csv(P_988 / 'promotech/workdir/hot_pg_neg/sequences_predictions.csv', sep='\t')
    y_pt_d = np.hstack([np.ones(len(pos_pt_d)), np.zeros(len(neg_pt_d))])
    s_pt_d = np.hstack([pos_pt_d.iloc[:,0].values, neg_pt_d.iloc[:,0].values])

    pos_pt_t = pd.read_csv(REPO_ROOT / 'output/predictions/tigr4_high/promotech_hot/promotech/workdir/hot_pg_pos/sequences_predictions.csv', sep='\t')
    neg_pt_t = pd.read_csv(REPO_ROOT / 'output/predictions/tigr4_high/promotech_hot/promotech/workdir/hot_pg_neg/sequences_predictions.csv', sep='\t')
    y_pt_t = np.hstack([np.ones(len(pos_pt_t)), np.zeros(len(neg_pt_t))])
    s_pt_t = np.hstack([pos_pt_t.iloc[:,0].values, neg_pt_t.iloc[:,0].values])

    y_pt = np.hstack([y_pt_d, y_pt_t])
    s_pt = np.hstack([s_pt_d, s_pt_t])
    fpr, tpr, _ = roc_curve(y_pt, s_pt)
    curves.append(("PromoTech RF-HOT [RF]", fpr, tpr, auc(fpr, tpr), "#E07614", "-", 2.2))

    # 4. MLDSPP 0%
    pos_m0_d = pd.read_csv(P_988 / 'mldspp_pos.csv', sep='\t')
    neg_m0_d = pd.read_csv(P_988 / 'mldspp_neg.csv', sep='\t')
    y_m0_d = np.hstack([np.ones(len(pos_m0_d)), np.zeros(len(neg_m0_d))])
    s_m0_d = np.hstack([pos_m0_d.iloc[:,0].values, neg_m0_d.iloc[:,0].values])

    pos_m0_t = pd.read_csv(REPO_ROOT / 'output/predictions/tigr4_high/mldspp_pos.csv', sep='\t')
    neg_m0_t = pd.read_csv(REPO_ROOT / 'output/predictions/tigr4_high/mldspp_neg.csv', sep='\t')
    y_m0_t = np.hstack([np.ones(len(pos_m0_t)), np.zeros(len(neg_m0_t))])
    s_m0_t = np.hstack([pos_m0_t.iloc[:,0].values, neg_m0_t.iloc[:,0].values])

    y_m0 = np.hstack([y_m0_d, y_m0_t])
    s_m0 = np.hstack([s_m0_d, s_m0_t])
    fpr, tpr, _ = roc_curve(y_m0, s_m0)
    curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", ":", 2.0))

    # 5. MEME Suite [De novo]
    meme_pos_d = pd.read_csv(P_988 / 'meme_pos.csv', sep='\t')
    meme_neg_d = pd.read_csv(P_988 / 'meme_neg.csv', sep='\t')
    meme_pos_t = pd.read_csv(P_T4_HIGH / 'meme_pos.csv', sep='\t')
    meme_neg_t = pd.read_csv(P_T4_HIGH / 'meme_neg.csv', sep='\t')
    y_meme = np.hstack([np.ones(len(meme_pos_d) + len(meme_pos_t)), np.zeros(len(meme_neg_d) + len(meme_neg_t))])
    s_meme = np.hstack([meme_pos_d.iloc[:,0].values, meme_pos_t.iloc[:,0].values,
                        meme_neg_d.iloc[:,0].values, meme_neg_t.iloc[:,0].values])
    fpr, tpr, _ = roc_curve(y_meme, s_meme)
    curves.append(("MEME Suite [De novo]", fpr, tpr, auc(fpr, tpr), "#3949AB", "-.", 2.0))

    # 6. FIMO + Prokaryote DB [PWM]
    fimo_pos_d = pd.read_csv(P_988 / 'fimo_prok_pos.csv', sep='\t')
    fimo_neg_d = pd.read_csv(P_988 / 'fimo_prok_neg.csv', sep='\t')
    fimo_pos_t = pd.read_csv(P_T4_HIGH / 'fimo_prok_pos.csv', sep='\t')
    fimo_neg_t = pd.read_csv(P_T4_HIGH / 'fimo_prok_neg.csv', sep='\t')
    y_fimo = np.hstack([np.ones(len(fimo_pos_d) + len(fimo_pos_t)), np.zeros(len(fimo_neg_d) + len(fimo_neg_t))])
    s_fimo = np.hstack([fimo_pos_d.iloc[:,0].values, fimo_pos_t.iloc[:,0].values,
                        fimo_neg_d.iloc[:,0].values, fimo_neg_t.iloc[:,0].values])
    fpr, tpr, _ = roc_curve(y_fimo, s_fimo)
    curves.append(("FIMO + Prokaryote DB [PWM]", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Combined ROC — D39V + TIGR4 High-Confidence (N = 3,454: 1:1 Balanced)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_combined_d39v_tigr4_high", "roc_auc_combined_high"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_combined_d39v_tigr4_high (.png, .svg, .pdf)")
    plt.close(fig)


def plot_roc_combined_d39v_tigr4_all():
    curves = []
    p_ext = P_T4_EXT

    # 1. iPro-MP
    df_ip_d = pd.read_csv(P_988 / 'ipromp/ipromp_12_predictions.csv')
    y_ip_d = np.hstack([np.ones(989), np.zeros(len(df_ip_d) - 989)])
    s_ip_d = df_ip_d['Probability' if 'Probability' in df_ip_d.columns else 'PRED'].values

    df_ip_a = pd.read_csv(p_ext / 'ipromp_tigr4_extended.csv')
    y_ip_a = np.hstack([np.ones(2000), np.zeros(len(df_ip_a) - 2000)])
    s_ip_a = df_ip_a['PRED'].values

    y_ip = np.hstack([y_ip_d, y_ip_a])
    s_ip = np.hstack([s_ip_d, s_ip_a])
    fpr, tpr, _ = roc_curve(y_ip, s_ip)
    curves.append(("iPro-MP (H. pylori) [gLM]", fpr, tpr, auc(fpr, tpr), "#7E57C2", "-", 2.4))

    # 2. PromoterLCNN
    pos_lc_d = pd.read_csv(P_988 / 'lcnn/lcnn_pos.csv', sep='\t')
    neg_lc_d = pd.read_csv(P_988 / 'lcnn/lcnn_neg.csv', sep='\t')
    y_lc_d = np.hstack([np.ones(len(pos_lc_d)), np.zeros(len(neg_lc_d))])
    s_lc_d = np.hstack([pos_lc_d.iloc[:,0].values, neg_lc_d.iloc[:,0].values])

    pos_lc_a = pd.read_csv(p_ext / 'lcnn/lcnn_pos.csv', sep='\t')
    neg_lc_a = pd.read_csv(p_ext / 'lcnn/lcnn_neg.csv', sep='\t')
    y_lc_a = np.hstack([np.ones(len(pos_lc_a)), np.zeros(len(neg_lc_a))])
    s_lc_a = np.hstack([pos_lc_a.iloc[:,0].values, neg_lc_a.iloc[:,0].values])

    y_lc = np.hstack([y_lc_d, y_lc_a])
    s_lc = np.hstack([s_lc_d, s_lc_a])
    fpr, tpr, _ = roc_curve(y_lc, s_lc)
    curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr), "#228B22", "-", 2.4))

    # 3. PromoTech RF-HOT
    pos_pt_d = pd.read_csv(P_988 / 'promotech/workdir/hot_pg_pos/sequences_predictions.csv', sep='\t')
    neg_pt_d = pd.read_csv(P_988 / 'promotech/workdir/hot_pg_neg/sequences_predictions.csv', sep='\t')
    y_pt_d = np.hstack([np.ones(len(pos_pt_d)), np.zeros(len(neg_pt_d))])
    s_pt_d = np.hstack([pos_pt_d.iloc[:,0].values, neg_pt_d.iloc[:,0].values])

    pos_pt_a = pd.read_csv(p_ext / 'promotech_hot/promotech/workdir/hot_pg_pos/sequences_predictions.csv', sep='\t')
    neg_pt_a = pd.read_csv(p_ext / 'promotech_hot/promotech/workdir/hot_pg_neg/sequences_predictions.csv', sep='\t')
    y_pt_a = np.hstack([np.ones(len(pos_pt_a)), np.zeros(len(neg_pt_a))])
    s_pt_a = np.hstack([pos_pt_a.iloc[:,0].values, neg_pt_a.iloc[:,0].values])

    y_pt = np.hstack([y_pt_d, y_pt_a])
    s_pt = np.hstack([s_pt_d, s_pt_a])
    fpr, tpr, _ = roc_curve(y_pt, s_pt)
    curves.append(("PromoTech RF-HOT [RF]", fpr, tpr, auc(fpr, tpr), "#E07614", "-", 2.2))

    # 4. MLDSPP 0%
    pos_m0_d = pd.read_csv(P_988 / 'mldspp_pos.csv', sep='\t')
    neg_m0_d = pd.read_csv(P_988 / 'mldspp_neg.csv', sep='\t')
    y_m0_d = np.hstack([np.ones(len(pos_m0_d)), np.zeros(len(neg_m0_d))])
    s_m0_d = np.hstack([pos_m0_d.iloc[:,0].values, neg_m0_d.iloc[:,0].values])

    pos_m0_a = pd.read_csv(p_ext / 'mldspp_pos.csv', sep='\t')
    neg_m0_a = pd.read_csv(p_ext / 'mldspp_neg.csv', sep='\t')
    y_m0_a = np.hstack([np.ones(len(pos_m0_a)), np.zeros(len(neg_m0_a))])
    s_m0_a = np.hstack([pos_m0_a.iloc[:,0].values, neg_m0_a.iloc[:,0].values])

    y_m0 = np.hstack([y_m0_d, y_m0_a])
    s_m0 = np.hstack([s_m0_d, s_m0_a])
    fpr, tpr, _ = roc_curve(y_m0, s_m0)
    curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr), "#942C76", ":", 2.0))

    # 5. FIMO + Prokaryote DB [PWM]
    fimo_pos_d = pd.read_csv(P_988 / 'fimo_prok_pos.csv', sep='\t')
    fimo_neg_d = pd.read_csv(P_988 / 'fimo_prok_neg.csv', sep='\t')
    fimo_pos_a = pd.read_csv(p_ext / 'fimo_prok_pos.csv', sep='\t')
    fimo_neg_a = pd.read_csv(p_ext / 'fimo_prok_neg.csv', sep='\t')
    y_fimo = np.hstack([np.ones(len(fimo_pos_d) + len(fimo_pos_a)), np.zeros(len(fimo_neg_d) + len(fimo_neg_a))])
    s_fimo = np.hstack([fimo_pos_d.iloc[:,0].values, fimo_pos_a.iloc[:,0].values,
                        fimo_neg_d.iloc[:,0].values, fimo_neg_a.iloc[:,0].values])
    fpr, tpr, _ = roc_curve(y_fimo, s_fimo)
    curves.append(("FIMO + Prokaryote DB [PWM]", fpr, tpr, auc(fpr, tpr), "#00ACC1", "-.", 2.0))

    fig, ax = plt.subplots(figsize=(9.5, 7.8), dpi=300)
    curves.sort(key=lambda x: x[3], reverse=True)

    for name, fpr, tpr, a, color, ls, lw in curves:
        ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw,
                label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", linewidth=1.2, label="Random Chance (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("Combined ROC — D39V + TIGR4 All TSS / Extended (N = 5,976: 1:1 Balanced)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", framealpha=0.95)
    plt.tight_layout()

    for d in DESTINATIONS:
        for fname in ["roc_combined_d39v_tigr4_all", "roc_auc_combined_all"]:
            fig.savefig(d / f"{fname}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.svg", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{fname}.pdf", dpi=300, bbox_inches="tight")
            if (d / "graph_png").exists():
                fig.savefig(d / "graph_png" / f"{fname}.png", dpi=300, bbox_inches="tight")
    print("  [Saved ROC] roc_combined_d39v_tigr4_all (.png, .svg, .pdf)")
    plt.close(fig)


def main():
    print("=" * 65)
    print("  RUNNING CANONICAL ROC / AUC GENERATOR (PNG, SVG, PDF)")
    print(f"  Canonical directory: {CANONICAL_TARGET}")
    print("=" * 65)
    plot_roc_989()
    plot_roc_30k()
    plot_roc_tigr4_high()
    plot_roc_tigr4_all()
    plot_roc_combined_d39v_tigr4_high()
    plot_roc_combined_d39v_tigr4_all()
    print("\n" + "=" * 65)
    print("  ROC / AUC GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
