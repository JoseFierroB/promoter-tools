#!/usr/bin/env python3
"""
================================================================================
SIGMA FACTOR EVALUATION & PUBLICATION PLOTS (S. pneumoniae D39V)
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Analyzes model predictions across functional classes:
  - Canonical / Housekeeping SigA Promoters (N=397)
  - Alternative / Competence SigX Promoters (N=21)
  - Unassigned / Putative Secondary Promoters (N=571)
  - True Negative Controls (CDS Intergenic-distal, N=1,000)

Generates:
  1. 3-Panel ROC Atlas by Sigma Factor (300 DPI)
  2. Score Distribution & Class Separation Boxplots (300 DPI)
  3. Performance & Sensitivity Heatmap Summary (300 DPI)
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

PRED_DIR = ROOT_DIR / "output/d39v_all_tools_benchmark/predictions"
PLOTS_DIR = ROOT_DIR / "output/plots/sigma_factor"
TABLES_DIR = ROOT_DIR / "output/tables"
DESK_DIR = Path("/home/fierro/Desktop/sigma_factor_plots")

for d in [PLOTS_DIR, TABLES_DIR, DESK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Load sequence IDs and Sigma Factor Annotations from D39V
POS_FASTA = ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta"
NEG_FASTA = ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"

def load_ground_truth_classes():
    pos_recs = list(SeqIO.parse(POS_FASTA, "fasta"))
    neg_recs = list(SeqIO.parse(NEG_FASTA, "fasta"))
    
    pos_ids = [r.id for r in pos_recs]
    neg_ids = [r.id for r in neg_recs]
    
    # Classify each positive sequence by sigma factor based on header metadata
    classes = {}
    for pid in pos_ids:
        pid_upper = pid.upper()
        if "SIGA" in pid_upper or "SIG_A" in pid_upper or "_A" in pid_upper:
            classes[pid] = "SigA"
        elif "SIGX" in pid_upper or "SIG_X" in pid_upper or "_X" in pid_upper or "COMX" in pid_upper:
            classes[pid] = "SigX"
        else:
            classes[pid] = "Unassigned"
            
    for nid in neg_ids:
        classes[nid] = "Negative"
        
    return pos_ids, neg_ids, classes

def load_all_predictions(pos_ids, neg_ids):
    preds = {}
    
    # ProkBERT
    if (PRED_DIR / "prokbert.tsv").exists():
        df = pd.read_csv(PRED_DIR / "prokbert.tsv", sep="\t")
        preds["ProkBERT-mini [gLM]"] = dict(zip(df["ID"], df["PRED"]))
        
    # iPro-MP
    ip_csv = PRED_DIR / "ipromp/ipromp_12_predictions.csv"
    if ip_csv.exists():
        df_ip = pd.read_csv(ip_csv)
        all_ids = pos_ids + neg_ids
        if len(df_ip) == len(all_ids):
            preds["iPro-MP [gLM]"] = dict(zip(all_ids, df_ip["PRED"]))
            
    # PromoterLCNN
    lcnn_pos = PRED_DIR / "lcnn/lcnn_pos.csv"
    lcnn_neg = PRED_DIR / "lcnn/lcnn_neg.csv"
    if lcnn_pos.exists() and lcnn_neg.exists():
        p_lc = pd.read_csv(lcnn_pos, sep="\t")["PRED"].values
        n_lc = pd.read_csv(lcnn_neg, sep="\t")["PRED"].values
        p_dict = dict(zip(pos_ids, p_lc))
        p_dict.update(dict(zip(neg_ids, n_lc)))
        preds["PromoterLCNN [CNN]"] = p_dict
        
    # PromoTech RF-HOT
    pt_pos = PRED_DIR / "promotech/workdir/hot_pg_pos/sequences_predictions.csv"
    pt_neg = PRED_DIR / "promotech/workdir/hot_pg_neg/sequences_predictions.csv"
    if pt_pos.exists() and pt_neg.exists():
        p_pt = pd.read_csv(pt_pos, sep="\t")["PRED"].values
        n_pt = pd.read_csv(pt_neg, sep="\t")["PRED"].values
        p_dict = dict(zip(pos_ids, p_pt))
        p_dict.update(dict(zip(neg_ids, n_pt)))
        preds["PromoTech RF-HOT [RF]"] = p_dict
        
    # Prompt
    if (PRED_DIR / "prompt.tsv").exists():
        df = pd.read_csv(PRED_DIR / "prompt.tsv", sep="\t")
        preds["Prompt [MLP 168]"] = dict(zip(df["ID"], df["PRED"]))
        
    # MLDSPP 0%
    ml_pos = PRED_DIR / "mldspp_pos.csv"
    ml_neg = PRED_DIR / "mldspp_neg.csv"
    if ml_pos.exists() and ml_neg.exists():
        p_ml = pd.read_csv(ml_pos, sep="\t")["PRED"].values
        n_ml = pd.read_csv(ml_neg, sep="\t")["PRED"].values
        p_dict = dict(zip(pos_ids, p_ml))
        p_dict.update(dict(zip(neg_ids, n_ml)))
        preds["MLDSPP 0% [BDT]"] = p_dict
        
    return preds

def main():
    print("=" * 80)
    print("GENERATING SIGMA FACTOR COMPARATIVE PLOTS & TABLES")
    print("=" * 80)
    
    pos_ids, neg_ids, classes = load_ground_truth_classes()
    predictions = load_all_predictions(pos_ids, neg_ids)
    
    sig_counts = pd.Series([classes[pid] for pid in pos_ids]).value_counts().to_dict()
    print("Positive TSS Subclass Distribution:", sig_counts)
    print("Negative Controls (CDS):", len(neg_ids))
    
    # 1. Compute Metrics per Sigma Subclass
    sigma_metrics = []
    roc_data = {"SigA": {}, "SigX": {}, "Unassigned": {}}
    score_distributions = {tool: {"SigA": [], "SigX": [], "Unassigned": [], "Negative": []} for tool in predictions}
    
    for tool_name, p_map in predictions.items():
        neg_scores = [p_map[nid] for nid in neg_ids if nid in p_map]
        score_distributions[tool_name]["Negative"] = neg_scores
        
        for sigma in ["SigA", "SigX", "Unassigned"]:
            sub_pos_ids = [pid for pid in pos_ids if classes.get(pid) == sigma and pid in p_map]
            pos_scores = [p_map[pid] for pid in sub_pos_ids]
            score_distributions[tool_name][sigma] = pos_scores
            
            y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
            y_pred = np.array(pos_scores + neg_scores)
            
            auc = roc_auc_score(y_true, y_pred)
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_data[sigma][tool_name] = (fpr, tpr, auc)
            
            p_mean = np.mean(pos_scores) if pos_scores else 0.0
            n_mean = np.mean(neg_scores) if neg_scores else 0.0
            
            sigma_metrics.append({
                "Tool": tool_name,
                "Sigma_Factor": sigma,
                "N_Positives": len(pos_scores),
                "N_Negatives": len(neg_scores),
                "ROC_AUC": round(auc, 4),
                "Pos_Mean_Score": round(float(p_mean), 3),
                "Neg_Mean_Score": round(float(n_mean), 3),
                "Delta_Separation": round(float(p_mean - n_mean), 3)
            })
            
    df_sigma = pd.DataFrame(sigma_metrics)
    tsv_out = TABLES_DIR / "sigma_factor_benchmark_metrics.tsv"
    df_sigma.to_csv(tsv_out, sep="\t", index=False)
    df_sigma.to_csv(DESK_DIR / "sigma_factor_benchmark_metrics.tsv", sep="\t", index=False)
    
    print("\n" + "=" * 90)
    print("PERFORMANCE BREAKDOWN BY SIGMA FACTOR SUBCLASS")
    print("=" * 90)
    print(df_sigma.to_string(index=False))
    print("=" * 90)
    
    # 2. Plot 1: 3-Panel ROC Atlas by Sigma Factor (300 DPI)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), dpi=300)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
    
    panels = [
        ("SigA", f"Canonical $\sigma^A$ Promoters (N={sig_counts.get('SigA', 0)} vs 1,000 Neg)"),
        ("SigX", f"Competence $\sigma^X$ Promoters (N={sig_counts.get('SigX', 0)} vs 1,000 Neg)"),
        ("Unassigned", f"Unassigned / Putative Promoters (N={sig_counts.get('Unassigned', 0)} vs 1,000 Neg)")
    ]
    
    for ax_idx, (sig_key, title) in enumerate(panels):
        ax = axes[ax_idx]
        sorted_tools = sorted(roc_data[sig_key].items(), key=lambda x: x[1][2], reverse=True)
        
        for i, (tool_name, (fpr, tpr, auc)) in enumerate(sorted_tools):
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.2, label=f"{tool_name} (AUC = {auc:.3f})")
            
        ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--", label="Chance (0.500)")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle=":", alpha=0.5)
        
    plt.suptitle("S. pneumoniae D39V — Promoter Benchmark ROC Stratified by Sigma Factor Subclass", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    roc_fig_path = PLOTS_DIR / "roc_by_sigma_factor.png"
    plt.savefig(roc_fig_path, dpi=300)
    plt.savefig(DESK_DIR / "roc_by_sigma_factor.png", dpi=300)
    plt.close()
    
    # 3. Plot 2: Score Distribution Across Functional Classes for Top Models (300 DPI)
    top_models = [m for m in ["ProkBERT-mini [gLM]", "iPro-MP [gLM]", "PromoterLCNN [CNN]", "PromoTech RF-HOT [RF]", "Prompt [MLP 168]"] if m in predictions]
    fig, axes = plt.subplots(1, len(top_models), figsize=(4.5 * len(top_models), 5.5), dpi=300, sharey=True)
    
    class_labels = ["SigA", "SigX", "Unassigned", "Negative"]
    box_colors = ["#2b5c8f", "#d95f02", "#7570b3", "#66a61e"]
    
    for idx, tool_name in enumerate(top_models):
        ax = axes[idx] if len(top_models) > 1 else axes
        data_to_plot = [score_distributions[tool_name][c] for c in class_labels]
        
        bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=class_labels,
                           medianprops=dict(color="black", lw=1.5),
                           flierprops=dict(marker=".", markersize=3, alpha=0.3))
        
        for patch, col in zip(bplot["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            
        ax.set_title(tool_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Promoter Score" if idx == 0 else "", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.set_xticklabels(["$\sigma^A$\n(N=397)", "$\sigma^X$\n(N=21)", "Unassigned\n(N=571)", "Negative\n(N=1000)"], fontsize=9.5)
        
    plt.suptitle("Predicted Score Distributions Across Functional Promoter Subclasses & Negative Controls", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    dist_fig_path = PLOTS_DIR / "score_distributions_by_sigma_factor.png"
    plt.savefig(dist_fig_path, dpi=300)
    plt.savefig(DESK_DIR / "score_distributions_by_sigma_factor.png", dpi=300)
    plt.close()
    
    # 4. Plot 3: Heatmap Summary of AUC across Sigma Factors (300 DPI)
    pivot_auc = df_sigma.pivot(index="Tool", columns="Sigma_Factor", values="ROC_AUC")[["SigA", "SigX", "Unassigned"]]
    pivot_auc = pivot_auc.sort_values("SigA", ascending=False)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.imshow(pivot_auc.values, cmap="YlGnBu", aspect="auto", vmin=0.45, vmax=1.0)
    
    plt.xticks([0, 1, 2], ["Canonical $\sigma^A$\n(N=397)", "Competence $\sigma^X$\n(N=21)", "Unassigned / Putative\n(N=571)"], fontsize=11, fontweight="bold")
    plt.yticks(range(len(pivot_auc)), pivot_auc.index, fontsize=11, fontweight="bold")
    
    cbar = plt.colorbar()
    cbar.set_label("ROC-AUC Score vs. CDS Controls", fontsize=11, fontweight="bold")
    
    for i in range(len(pivot_auc)):
        for j in range(3):
            val = pivot_auc.values[i, j]
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.85 else "black", fontsize=11, fontweight="bold")
            
    plt.title("S. pneumoniae D39V — Model ROC-AUC by Sigma Factor Subclass", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    hm_fig_path = PLOTS_DIR / "sigma_factor_performance_heatmap.png"
    plt.savefig(hm_fig_path, dpi=300)
    plt.savefig(DESK_DIR / "sigma_factor_performance_heatmap.png", dpi=300)
    plt.close()
    
    print(f"\n[DONE] Figures and tables saved successfully to {PLOTS_DIR} and {DESK_DIR}")

if __name__ == "__main__":
    main()
