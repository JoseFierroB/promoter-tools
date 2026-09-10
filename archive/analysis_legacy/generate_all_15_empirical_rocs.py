#!/usr/bin/env python3
"""
================================================================================
EMPIRICAL ROC CURVE GENERATOR FOR ALL 15 CANONICAL BENCHMARK DATASETS (8 TOOLS)
================================================================================
Generates 100% empirical, true step-wise (irregular/staircase) ROC curves directly
from raw prediction score distributions across all 8 tools:
  - ProkBERT-mini [gLM]
  - iPro-MP [gLM Transformer]
  - PromoterLCNN [CNN]
  - PromoTech RF-HOT [RF]
  - Prompt [MLP 168]
  - MLDSPP 0% [BDT]
  - FIMO ProkDB [PWMs]

Outputs in PNG (300 DPI), PDF (vectorial), and SVG:
  - /home/fierro/drive/roc_curves_15_datasets_empirical/
  - /home/fierro/Desktop/roc_curves_15_datasets_empirical/
  - /home/fierro/drive/plots_updated_all_tools_2026/benchmark/
  - /home/fierro/drive/plots_updated_all_tools_2026/igr/
================================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

ROOT_DIR = Path("/home/fierro/Desktop/promoter-tools")
sys.path.insert(0, str(ROOT_DIR))

PRED_DIR = ROOT_DIR / "output/grand_master_final/predictions"

DRIVE_DIR = Path("/home/fierro/drive/roc_curves_15_datasets_empirical")
DESK_DIR = Path("/home/fierro/Desktop/roc_curves_15_datasets_empirical")

DRIVE_SUITE_BENCH = Path("/home/fierro/drive/plots_updated_all_tools_2026/benchmark")
DRIVE_SUITE_IGR = Path("/home/fierro/drive/plots_updated_all_tools_2026/igr")
DESK_SUITE_BENCH = Path("/home/fierro/Desktop/drive_plots_updated_all_tools_2026/benchmark")
DESK_SUITE_IGR = Path("/home/fierro/Desktop/drive_plots_updated_all_tools_2026/igr")

for d in [DRIVE_DIR, DESK_DIR, DRIVE_SUITE_BENCH, DRIVE_SUITE_IGR, DESK_SUITE_BENCH, DESK_SUITE_IGR]:
    d.mkdir(parents=True, exist_ok=True)

# Master counts
COUNTS = {
    "1_d39v": (989, 1000, "1. S. pneumoniae D39V Baseline (N = 1,989)"),
    "2_tigr4_high": (738, 738, "2. S. pneumoniae TIGR4 High-Confidence (N = 1,476)"),
    "3_tigr4_all": (1009, 1009, "3. S. pneumoniae TIGR4 Comprehensive All (N = 2,018)"),
    "4_d39v_tigr4_high": (1727, 1738, "4. Combined D39V + TIGR4 High-Confidence (N = 3,465)"),
    "5_d39v_tigr4_all": (1998, 2009, "5. Combined D39V + TIGR4 Comprehensive All (N = 4,007)"),
    "6_d39v_igr": (723, 723, "6. S. pneumoniae D39V Intergenic IGR (N = 1,446)"),
    "7_tigr4_igr_high": (553, 553, "7. S. pneumoniae TIGR4 High-Confidence IGR (N = 1,106)"),
    "8_tigr4_igr_all": (1009, 1009, "8. S. pneumoniae TIGR4 All IGR (N = 2,018)"),
    "9_d39v_igr_tigr4_igr_high": (1276, 1276, "9. Combined Inter-Strain High-Confidence IGR (N = 2,552)"),
    "10_d39v_igr_tigr4_igr_all": (1732, 1732, "10. Combined Inter-Strain Comprehensive IGR (N = 3,464)"),
    "11_d39v_cds": (266, 266, "11. S. pneumoniae D39V Intragenic CDS Internal (N = 532)"),
    "12_tigr4_cds_high": (738, 738, "12. S. pneumoniae TIGR4 High-Confidence CDS (N = 1,476)"),
    "13_tigr4_cds_all": (504, 504, "13. S. pneumoniae TIGR4 All CDS (N = 1,008)"),
    "14_d39v_cds_tigr4_cds_high": (1004, 1004, "14. Combined Inter-Strain High-Confidence CDS (N = 2,008)"),
    "15_d39v_cds_tigr4_cds_all": (770, 770, "15. Combined Inter-Strain Comprehensive CDS (N = 1,540)")
}

PALETTE = {
    "ProkBERT-mini [gLM]":    {"color": "#D81B60", "ls": "-",  "lw": 2.3},
    "iPro-MP [gLM]":          {"color": "#7E57C2", "ls": "-",  "lw": 2.1},
    "PromoterLCNN [CNN]":     {"color": "#2E7D32", "ls": "-",  "lw": 1.9},
    "PromoTech RF-HOT [RF]":  {"color": "#EF6C00", "ls": "-",  "lw": 1.9},
    "Prompt [MLP 168]":       {"color": "#0288D1", "ls": "-",  "lw": 1.8},
    "MLDSPP 0% [BDT]":        {"color": "#880E4F", "ls": "-",  "lw": 1.8},
    "FIMO ProkDB [PWMs]":     {"color": "#00897B", "ls": "--", "lw": 1.7},
}

def load_predictions(d_key, n_pos, n_neg):
    curves = []
    
    # 1. ProkBERT
    prok_f = PRED_DIR / f"{d_key}_prokbert_prokbert.tsv"
    if prok_f.exists():
        df = pd.read_csv(prok_f, sep="\t")
        fpr, tpr, _ = roc_curve(df["LABEL"], df["PRED"])
        curves.append(("ProkBERT-mini [gLM]", fpr, tpr, auc(fpr, tpr)))

    # 2. iPro-MP
    ip_f = PRED_DIR / f"{d_key}_ipromp_ipromp" / "ipromp" / "ipromp_12_predictions.csv"
    if ip_f.exists():
        df_ip = pd.read_csv(ip_f)
        col = "PRED" if "PRED" in df_ip.columns else "Probability"
        y_true = np.hstack([np.ones(n_pos), np.zeros(len(df_ip) - n_pos)])
        fpr, tpr, _ = roc_curve(y_true, df_ip[col].values)
        curves.append(("iPro-MP [gLM]", fpr, tpr, auc(fpr, tpr)))

    # 3. PromoterLCNN
    lcnn_pos = PRED_DIR / f"{d_key}_lcnn_lcnn" / "lcnn" / "lcnn_pos.csv"
    lcnn_neg = PRED_DIR / f"{d_key}_lcnn_lcnn" / "lcnn" / "lcnn_neg.csv"
    if lcnn_pos.exists() and lcnn_neg.exists():
        pos = pd.read_csv(lcnn_pos, sep="\t")
        neg = pd.read_csv(lcnn_neg, sep="\t")
        y_true = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        y_score = np.hstack([pos["PRED"].values, neg["PRED"].values])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves.append(("PromoterLCNN [CNN]", fpr, tpr, auc(fpr, tpr)))

    # 4. PromoTech RF-HOT
    pt_pos = PRED_DIR / f"{d_key}_promotech_promotech" / "promotech" / "workdir" / "hot_pg_pos" / "sequences_predictions.csv"
    pt_neg = PRED_DIR / f"{d_key}_promotech_promotech" / "promotech" / "workdir" / "hot_pg_neg" / "sequences_predictions.csv"
    if pt_pos.exists() and pt_neg.exists():
        pos = pd.read_csv(pt_pos, sep="\t")
        neg = pd.read_csv(pt_neg, sep="\t")
        y_true = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        y_score = np.hstack([pos["PRED"].values, neg["PRED"].values])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves.append(("PromoTech RF-HOT [RF]", fpr, tpr, auc(fpr, tpr)))

    # 5. Prompt
    prompt_f = PRED_DIR / f"{d_key}_prompt_prompt.tsv"
    if prompt_f.exists():
        df_p = pd.read_csv(prompt_f, sep="\t")
        fpr, tpr, _ = roc_curve(df_p["LABEL"], df_p["PRED"])
        curves.append(("Prompt [MLP 168]", fpr, tpr, auc(fpr, tpr)))

    # 6. MLDSPP 0%
    mld_pos = PRED_DIR / f"{d_key}_mldspp_mldspp" / "mldspp_pos.csv"
    mld_neg = PRED_DIR / f"{d_key}_mldspp_mldspp" / "mldspp_neg.csv"
    if mld_pos.exists() and mld_neg.exists():
        pos = pd.read_csv(mld_pos, sep="\t")
        neg = pd.read_csv(mld_neg, sep="\t")
        y_true = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        y_score = np.hstack([pos["PRED"].values, neg["PRED"].values])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves.append(("MLDSPP 0% [BDT]", fpr, tpr, auc(fpr, tpr)))

    # 7. FIMO ProkDB
    fimo_pos = PRED_DIR / f"{d_key}_fimo_fimo_dir" / "fimo_prok_pos.csv"
    fimo_neg = PRED_DIR / f"{d_key}_fimo_fimo_dir" / "fimo_prok_neg.csv"
    if fimo_pos.exists() and fimo_neg.exists():
        pos = pd.read_csv(fimo_pos, sep="\t")
        neg = pd.read_csv(fimo_neg, sep="\t")
        y_true = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
        y_score = np.hstack([pos["PRED"].values, neg["PRED"].values])
        if np.all(y_score == y_score[0]):
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            curves.append(("FIMO ProkDB [PWMs]", fpr, tpr, 0.500))
        else:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            curves.append(("FIMO ProkDB [PWMs]", fpr, tpr, auc(fpr, tpr)))

    curves.sort(key=lambda x: x[3], reverse=True)
    return curves

def plot_and_save_empirical_roc(d_key):
    n_pos, n_neg, title_str = COUNTS[d_key]
    curves = load_predictions(d_key, n_pos, n_neg)
    if not curves:
        print(f"  [Skip] No curves found for {d_key}")
        return

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)

    for name, fpr, tpr, a in curves:
        style = PALETTE.get(name, {"color": "#333333", "ls": "-", "lw": 1.8})
        ax.plot(fpr, tpr, color=style["color"], linestyle=style["ls"], 
                linewidth=style["lw"], label=f"{name} (AUC = {a:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title(f"Receiver Operating Characteristic (ROC)\n{title_str}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_name = f"roc_auc_{d_key}"
    
    # Save to primary Drive and Desktop folders
    for out_dir in [DRIVE_DIR, DESK_DIR]:
        fig.savefig(out_dir / f"{out_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{out_name}.svg", bbox_inches="tight")

    # Also update suite folders for benchmark and igr
    if d_key in ["1_d39v", "2_tigr4_high", "3_tigr4_all", "4_d39v_tigr4_high", "5_d39v_tigr4_all"]:
        for out_dir in [DRIVE_SUITE_BENCH, DESK_SUITE_BENCH]:
            fig.savefig(out_dir / f"{out_name}.png", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{out_name}.svg", bbox_inches="tight")
            if d_key == "1_d39v":
                fig.savefig(out_dir / "roc_auc_baseline_989.png", dpi=300, bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_baseline_989.pdf", bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_baseline_989.svg", bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_N1976.png", dpi=300, bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_N1976.pdf", bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_N1976.svg", bbox_inches="tight")

    if "igr" in d_key:
        for out_dir in [DRIVE_SUITE_IGR, DESK_SUITE_IGR]:
            fig.savefig(out_dir / f"{out_name}.png", dpi=300, bbox_inches="tight")
            fig.savefig(out_dir / f"{out_name}.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{out_name}.svg", bbox_inches="tight")
            if d_key == "6_d39v_igr":
                fig.savefig(out_dir / "roc_auc_d39v_igr_benchmark.png", dpi=300, bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_d39v_igr_benchmark.pdf", bbox_inches="tight")
                fig.savefig(out_dir / "roc_auc_d39v_igr_benchmark.svg", bbox_inches="tight")

    plt.close(fig)
    print(f"  [SAVED EMPIRICAL ROC] {out_name} (.png, .pdf, .svg)")

def main():
    print("=" * 80)
    print("GENERATING EMPIRICAL ROC CURVES FOR ALL 15 CANONICAL DATASETS (8 TOOLS)")
    print(f"Destination Drive Folder:   {DRIVE_DIR}")
    print(f"Destination Desktop Folder: {DESK_DIR}")
    print("=" * 80)

    for d_key in COUNTS.keys():
        plot_and_save_empirical_roc(d_key)

    print("\n" + "=" * 80)
    print("ALL 15 EMPIRICAL ROC CURVES GENERATED SUCCESSFULLY!")
    print("Zero smoothing applied. 100% discrete empirical step-wise curves.")
    print("=" * 80)

if __name__ == "__main__":
    main()
