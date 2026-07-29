#!/usr/bin/env python3
"""
Generate Master Evaluation Plots for S. pneumoniae Promoter Prediction.
Features:
  1. Loads all 23 species of iPro-MP (loads pre-computed codon folder or fallbacks).
  2. Loads both PromoTech models (RF-HOT and RF-TETRA).
  3. Loads MLDSPP (XGBoost, SVM, RF) and PromoterLCNN.
  4. Dynamically calculates ROC/PR AUC and highlights only the Top 5 models.
  5. Plots all other species/models in thin light gray background.
  6. Plots Youden's Optimal Threshold Point (solid dot) on each of the Top 5 curves.
  7. Generates Overall, SigA, SigX, and individual model panels.

Author: Antigravity Code Assistant
Date: July 15, 2026
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def get_temp_dir():
    """Pick the best temp directory: nobackup on compute nodes, research as fallback."""
    env_tmp = os.environ.get("TMPDIR", "")
    if env_tmp and os.path.isdir(os.path.dirname(env_tmp)):
        return Path(env_tmp)
    candidates = [
        "/hps/nobackup/jlees/fierro/tmp",
        "/nfs/research/jlees/fierro/tmp",
    ]
    for p in candidates:
        parent = os.path.dirname(p)
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            return Path(p)
    return Path("/nfs/research/jlees/fierro/tmp")


def main():
    parser = argparse.ArgumentParser(
        description="Generate master evaluation plots from benchmark results."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="Directory containing prediction results (CSV files) and output for plots."
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory containing PromoTech temp files (default: auto-detect)."
    )
    args = parser.parse_args()

    root_dir = Path(os.getcwd()).resolve()
    resultados_dir = Path(args.output_dir).resolve()
    if args.temp_dir:
        temp_dir = Path(args.temp_dir).resolve()
    else:
        temp_dir = get_temp_dir() / "temp_promotech_pipelines"
    metadata_path = root_dir / "data/benchmark/positives_81bp_metadata.tsv"
    ipromp_src_dir = resultados_dir / "ipromp_out"
    
    # 1. Load Metadata to separate SigA and SigX
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    metadata = pd.read_csv(metadata_path, sep='\t').set_index("Sequence_ID")
    
    siga_ids = metadata[metadata["Sigma_Factor"] == "SigA"].index
    sigx_ids = metadata[metadata["Sigma_Factor"] == "SigX"].index
    
    # 2. Define prediction paths for standard models
    pred_files = {
        "MLDSPP (XGBoost)": (resultados_dir / "mldspp_pos.csv", resultados_dir / "mldspp_neg.csv"),
        "MLDSPP (SVM)": (resultados_dir / "mldspp_svm_pos.csv", resultados_dir / "mldspp_svm_neg.csv"),
        "MLDSPP (Random Forest)": (resultados_dir / "mldspp_rf_pos.csv", resultados_dir / "mldspp_rf_neg.csv"),
        "PromoterLCNN": (resultados_dir / "lcnn_pos.csv", resultados_dir / "lcnn_neg.csv"),
    }
    
    models_data = {}
    for name, (pos_p, neg_p) in pred_files.items():
        if pos_p.exists() and neg_p.exists():
            pos_df = pd.read_csv(pos_p, sep='\t').set_index("CHROM")
            neg_df = pd.read_csv(neg_p, sep='\t').set_index("CHROM")
            models_data[name] = {
                "pos": pos_df["PRED"].to_dict(),
                "neg": neg_df["PRED"].values
            }
            
    # 3. Load PromoTech RF-HOT and RF-TETRA
    pt_models = {
        "PromoTech RF-HOT": (
            temp_dir / "hot_s_pos/sequences_predictions.csv",
            temp_dir / "hot_s_neg/sequences_predictions.csv",
            resultados_dir / "benchmark_results_rfhot/promotech_out/sequences_predictions.csv"
        ),
        "PromoTech RF-TETRA": (
            temp_dir / "tetra_s_pos/sequences_predictions.csv",
            temp_dir / "tetra_s_neg/sequences_predictions.csv",
            resultados_dir / "benchmark_results_rftetra/promotech_out/sequences_predictions.csv"
        )
    }
    
    for name, (pos_p, neg_p, fallback_p) in pt_models.items():
        if pos_p.exists() and neg_p.exists():
            pos_df = pd.read_csv(pos_p, sep='\t').set_index("CHROM")
            neg_df = pd.read_csv(neg_p, sep='\t').set_index("CHROM")
            models_data[name] = {
                "pos": pos_df["PRED"].to_dict(),
                "neg": neg_df["PRED"].values
            }
        elif fallback_p.exists():
            df = pd.read_csv(fallback_p, sep='\t')
            df['is_pos'] = df['CHROM'].str.startswith('TSS_')
            pos_df = df[df['is_pos']].set_index("CHROM")
            neg_df = df[~df['is_pos']]
            models_data[name] = {
                "pos": pos_df["PRED"].to_dict(),
                "neg": neg_df["PRED"].values
            }

    # 4. Load all 23 species of iPro-MP
    if ipromp_src_dir.exists():
        print("Loading pre-computed iPro-MP (23 species) predictions...")
        for sp_id in range(1, 24):
            csv_path = ipromp_src_dir / f"ipromp_{sp_id}_predictions.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                pos_probs = dict(zip(metadata.index, df["Probability"].values[:len(metadata)]))
                neg_probs = df["Probability"].values[len(metadata):]
                models_data[f"iPro-MP (sp {sp_id})"] = {
                    "pos": pos_probs,
                    "neg": neg_probs
                }
    else:
        # Fallback: load species 18 and 23 which are already computed in promoter-tools resultados
        print("Codon iPro-MP directory not found. Loading local sp 18 and 23 fallback...")
        for sp_id in [18, 23]:
            pos_p = resultados_dir / f"ipromp/ipromp_{sp_id}_pos.csv"
            neg_p = resultados_dir / f"ipromp/ipromp_{sp_id}_neg.csv"
            if pos_p.exists() and neg_p.exists():
                pos_df = pd.read_csv(pos_p)
                pos_probs = dict(zip(metadata.index, pos_df["Probability"].values))
                neg_df = pd.read_csv(neg_p)
                models_data[f"iPro-MP (sp {sp_id})"] = {
                    "pos": pos_probs,
                    "neg": neg_df["Probability"].values
                }

    print(f"Loaded predictions for {len(models_data)} models total.")
    
    # iPro-MP species name lookup (from iPro-MP/iPro-MP_predict.py)
    ipromp_species_names = {
        1: "Acinetobacter baumannii ATCC 17978",
        2: "Bradyrhizobium japonicum USDA 110",
        3: "Burkholderia cenocepacia J2315",
        4: "Campylobacter jejuni RM1221",
        5: "Campylobacter jejuni subsp. jejuni 81116",
        6: "Campylobacter jejuni subsp. jejuni 81-176",
        7: "Campylobacter jejuni subsp. jejuni NCTC 11168",
        8: "Corynebacterium diphtheriae NCTC 13129",
        9: "Corynebacterium glutamicum ATCC 13032",
        10: "Escherichia coli str K-12 substr. MG1655",
        11: "Haloferax volcanii DS2",
        12: "Helicobacter pylori strain 26695",
        13: "Nostoc sp. PCC7120",
        14: "Paenibacillus riograndensis SBR5",
        15: "Pseudomonas putida KT2440",
        16: "Shigella flexneri 5a str. M90T",
        17: "Sinorhizobium meliloti 1021",
        18: "Staphylococcus aureus subsp. aureus MW2",
        19: "Staphylococcus epidermidis ATCC 12228",
        20: "Synechococcus elongatus PCC 7942",
        21: "Thermococcus kodakarensis KOD1",
        22: "Xanthomonas campestris pv. campestrie B100",
        23: "Bacillus subtilis subsp. subtilis str. 168",
    }
    
    def ipromp_label(model_name):
        import re
        m = re.match(r"iPro-MP \(sp (\d+)\)", model_name)
        if m:
            sp_id = int(m.group(1))
            sp_name = ipromp_species_names.get(sp_id, f"sp {sp_id}")
            return f"iPro-MP ({sp_name})"
        return model_name
    
    # Configure global plotting parameters (Nature / Science style)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 8
    
    # Helper function to plot ROC and PR subplots with Top 5 highlighting and Youden optimal points
    def make_double_plot(subset_name, pos_ids_list, filename):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5), dpi=300)
        
        # Calculate ROC AUC dynamically for all loaded models
        model_aucs = {}
        model_curves = {}
        
        for name, data in models_data.items():
            pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
            neg_scores = data["neg"]
            
            y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
            y_scores = np.concatenate([pos_scores, neg_scores])
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(rec, prec)
            
            # Find optimal Youden threshold
            youden = tpr - fpr
            opt_idx = np.argmax(youden)
            opt_thresh = thresholds[opt_idx]
            
            # Calculate precision/recall exactly at the optimal threshold
            y_pred = (y_scores >= opt_thresh).astype(int)
            opt_sens = np.mean(y_pred[y_true == 1]) # Sensitivity / Recall
            opt_prec = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
            
            model_aucs[name] = roc_auc
            model_curves[name] = {
                "fpr": fpr, "tpr": tpr, "auc": roc_auc,
                "rec": rec, "prec": prec, "pr_auc": pr_auc,
                "opt_fpr": fpr[opt_idx], "opt_tpr": tpr[opt_idx],
                "opt_rec": opt_sens, "opt_prec": opt_prec
            }
            
        # Determine Top 6 models based on ROC AUC
        sorted_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)
        top_n = 6
        top_n_names = [m[0] for m in sorted_models[:top_n]]
        
        # Define high-contrast colors for Top N
        top_colors = ["#002D62", "#D95319", "#7E2F8E", "#00A087", "#3C5488", "#1f77b4"]
        color_mapping = {name: top_colors[i] for i, name in enumerate(top_n_names)}
        
        # Plot curves: Top N in color/highlighted, others in thin semi-transparent gray
        for name in model_curves:
            curve = model_curves[name]
            if name in top_n_names:
                color = color_mapping[name]
                # ROC
                ax1.plot(curve["fpr"], curve["tpr"], label=f"{name} (AUC={curve['auc']:.3f})", color=color, lw=1.3)
                # Plot Youden optimal point (circle marker)
                ax1.plot(curve["opt_fpr"], curve["opt_tpr"], marker='o', markersize=4.5,
                         markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.6)
                
                # PR
                ax2.plot(curve["rec"], curve["prec"], label=f"{name} (AUC={curve['pr_auc']:.3f})", color=color, lw=1.3)
                # Plot Youden optimal point (circle marker)
                ax2.plot(curve["opt_rec"], curve["opt_prec"], marker='o', markersize=4.5,
                         markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.6)
            else:
                # Thin gray line for background models
                ax1.plot(curve["fpr"], curve["tpr"], color="#DDDDDD", lw=0.4, alpha=0.4)
                ax2.plot(curve["rec"], curve["prec"], color="#DDDDDD", lw=0.4, alpha=0.4)
                
        # Format ROC plot
        ax1.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.set_xlabel("False Positive Rate (FPR)")
        ax1.set_ylabel("True Positive Rate (TPR)")
        ax1.set_title(f"ROC Curves - {subset_name}", fontsize=9, fontweight="bold")
        ax1.legend(frameon=False, loc="lower right", fontsize=6)
        
        # Format PR plot
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        ax2.set_xlabel("Recall (Sensitivity)")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"PR Curves - {subset_name}", fontsize=9, fontweight="bold")
        ax2.legend(frameon=False, loc="lower left", fontsize=6)
        
        # Note on Youden points in the figure
        fig.suptitle("Solid markers (o) represent the Youden optimal operating threshold.", y=0.02, fontsize=7.5, fontstyle='italic', alpha=0.7)
        
        plt.tight_layout()
        plot_p = resultados_dir / filename
        plt.savefig(plot_p, bbox_inches="tight")
        print(f"Saved {subset_name} plot to: {plot_p}")
        plt.close()

    # 1. Overall ROC & PR Plot
    make_double_plot("Overall Dataset", list(metadata.index), "master_plot_overall.svg")
    
    # 2. SigA-specific ROC & PR Plot
    make_double_plot("SigA Housekeeping", list(siga_ids), "master_plot_siga.svg")
    
    # 3. SigX-specific ROC & PR Plot
    make_double_plot("SigX Competence", list(sigx_ids), "master_plot_sigx.svg")

    # Helper function to plot 2x2 individual panels for a specific positive subset
    def make_individual_panels_plot(pos_ids_list, subset_name, filename):
        print(f"Generating individual subplots panel for {subset_name}...")
        fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.5), dpi=300)
        
        # 4.1. MLDSPP Models Subplot
        ax_mldspp = axes[0, 0]
        ax_mldspp.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
        mldspp_names = ["MLDSPP (XGBoost)", "MLDSPP (SVM)", "MLDSPP (Random Forest)"]
        mldspp_colors = {"MLDSPP (XGBoost)": "#002D62", "MLDSPP (SVM)": "#4DBBD5", "MLDSPP (Random Forest)": "#3C5488"}
        
        for name in mldspp_names:
            if name in models_data:
                data = models_data[name]
                pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
                neg_scores = data["neg"]
                y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
                y_scores = np.concatenate([pos_scores, neg_scores])
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                color = mldspp_colors[name]
                ax_mldspp.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color, lw=1.2)
                youden = tpr - fpr
                opt_idx = np.argmax(youden)
                ax_mldspp.plot(fpr[opt_idx], tpr[opt_idx], marker='o', markersize=4.0,
                               markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.5)
                               
        ax_mldspp.spines["top"].set_visible(False)
        ax_mldspp.spines["right"].set_visible(False)
        ax_mldspp.set_xlim([-0.02, 1.02])
        ax_mldspp.set_ylim([-0.02, 1.02])
        ax_mldspp.set_xlabel("FPR")
        ax_mldspp.set_ylabel("TPR")
        ax_mldspp.set_title(f"MLDSPP Models ({subset_name})", fontsize=9, fontweight="bold")
        ax_mldspp.legend(frameon=False, loc="lower right", fontsize=6)

        # 4.2. iPro-MP Models Subplot (All 23 species, highlighting Top 6 on this subset)
        ax_ipromp = axes[0, 1]
        ax_ipromp.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
        ipromp_names = [name for name in models_data if name.startswith("iPro-MP (sp ")]
        
        # Calculate AUCs dynamically for this subset to sort species
        ipromp_aucs = {}
        for name in ipromp_names:
            data = models_data[name]
            pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
            neg_scores = data["neg"]
            y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
            y_scores = np.concatenate([pos_scores, neg_scores])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ipromp_aucs[name] = auc(fpr, tpr)
            
        sorted_ipromp = sorted(ipromp_aucs.items(), key=lambda x: x[1], reverse=True)
        top_n = 6
        top_ipromp_names = [m[0] for m in sorted_ipromp[:top_n]]
        
        # Distinct colors for the top 6 species of iPro-MP
        top_ipromp_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        ipromp_color_map = {name: top_ipromp_colors[i] for i, name in enumerate(top_ipromp_names)}
        
        for name in ipromp_names:
            data = models_data[name]
            pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
            neg_scores = data["neg"]
            y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
            y_scores = np.concatenate([pos_scores, neg_scores])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            if name in top_ipromp_names:
                color = ipromp_color_map[name]
                display_name = ipromp_label(name)
                ax_ipromp.plot(fpr, tpr, label=f"{display_name} (AUC={roc_auc:.3f})", color=color, lw=1.2)
                youden = tpr - fpr
                opt_idx = np.argmax(youden)
                ax_ipromp.plot(fpr[opt_idx], tpr[opt_idx], marker='o', markersize=4.0,
                               markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.5)
            else:
                # Other 17 species in background light gray
                ax_ipromp.plot(fpr, tpr, color="#DDDDDD", lw=0.4, alpha=0.4)
                
        ax_ipromp.spines["top"].set_visible(False)
        ax_ipromp.spines["right"].set_visible(False)
        ax_ipromp.set_xlim([-0.02, 1.02])
        ax_ipromp.set_ylim([-0.02, 1.02])
        ax_ipromp.set_xlabel("FPR")
        ax_ipromp.set_ylabel("TPR")
        ax_ipromp.set_title(f"iPro-MP Models ({subset_name})", fontsize=9, fontweight="bold")
        if top_5_ipromp:
            ax_ipromp.legend(frameon=False, loc="lower right", fontsize=5.5)

        # 4.3. PromoterLCNN Model Subplot
        ax_lcnn = axes[1, 0]
        ax_lcnn.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
        if "PromoterLCNN" in models_data:
            data = models_data["PromoterLCNN"]
            pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
            neg_scores = data["neg"]
            y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
            y_scores = np.concatenate([pos_scores, neg_scores])
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax_lcnn.plot(fpr, tpr, label=f"PromoterLCNN (AUC={roc_auc:.3f})", color="#7E2F8E", lw=1.2)
            youden = tpr - fpr
            opt_idx = np.argmax(youden)
            ax_lcnn.plot(fpr[opt_idx], tpr[opt_idx], marker='o', markersize=4.0,
                         markerfacecolor="#7E2F8E", markeredgecolor='black', markeredgewidth=0.5)
                         
        ax_lcnn.spines["top"].set_visible(False)
        ax_lcnn.spines["right"].set_visible(False)
        ax_lcnn.set_xlim([-0.02, 1.02])
        ax_lcnn.set_ylim([-0.02, 1.02])
        ax_lcnn.set_xlabel("FPR")
        ax_lcnn.set_ylabel("TPR")
        ax_lcnn.set_title(f"PromoterLCNN Model ({subset_name})", fontsize=9, fontweight="bold")
        ax_lcnn.legend(frameon=False, loc="lower right", fontsize=6)

        # 4.4. PromoTech Models Subplot
        ax_pt = axes[1, 1]
        ax_pt.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
        pt_names = ["PromoTech RF-HOT", "PromoTech RF-TETRA"]
        pt_colors = {"PromoTech RF-HOT": "#00A087", "PromoTech RF-TETRA": "#3C5488"}
        
        for name in pt_names:
            if name in models_data:
                data = models_data[name]
                pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
                neg_scores = data["neg"]
                y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
                y_scores = np.concatenate([pos_scores, neg_scores])
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                color = pt_colors[name]
                ax_pt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=color, lw=1.2)
                youden = tpr - fpr
                opt_idx = np.argmax(youden)
                ax_pt.plot(fpr[opt_idx], tpr[opt_idx], marker='o', markersize=4.0,
                           markerfacecolor=color, markeredgecolor='black', markeredgewidth=0.5)
                           
        ax_pt.spines["top"].set_visible(False)
        ax_pt.spines["right"].set_visible(False)
        ax_pt.set_xlim([-0.02, 1.02])
        ax_pt.set_ylim([-0.02, 1.02])
        ax_pt.set_xlabel("FPR")
        ax_pt.set_ylabel("TPR")
        ax_pt.set_title(f"PromoTech Models ({subset_name})", fontsize=9, fontweight="bold")
        ax_pt.legend(frameon=False, loc="lower right", fontsize=6)
        
        # Note on Youden points for 2x2 panel
        fig.suptitle(f"Solid markers (o) represent the Youden optimal operating threshold - {subset_name}.", y=0.015, fontsize=7.5, fontstyle='italic', alpha=0.7)
        
        plt.tight_layout()
        plot_p = resultados_dir / filename
        plt.savefig(plot_p, bbox_inches="tight")
        print(f"Saved {subset_name} individual panels plot to: {plot_p}")
        plt.close()

    # Generate iPro-MP Gram classification plot
    def make_ipromp_gram_plot(subset_name, pos_ids_list, filename):
        print(f"Generating iPro-MP Gram classification plot for {subset_name}...")
        
        # Gram classification for each iPro-MP species
        # Gram-negative (Proteobacteria)
        gram_neg = {1, 2, 3, 4, 5, 6, 7, 10, 12, 15, 16, 17, 22}
        # Gram-positive (Firmicutes + Actinobacteria)
        gram_pos = {8, 9, 14, 18, 19, 23}
        # Cyanobacteria
        cyano = {13, 20}
        # Archaea
        archaea = {11, 21}
        
        gram_to_group = {}
        for sp in gram_neg: gram_to_group[sp] = "Gram-negative"
        for sp in gram_pos: gram_to_group[sp] = "Gram-positive"
        for sp in cyano: gram_to_group[sp] = "Cyanobacteria"
        for sp in archaea: gram_to_group[sp] = "Archaea"
        
        group_colors = {
            "Gram-negative": "#2166AC",
            "Gram-positive": "#B2182B",
            "Cyanobacteria": "#4393C3",
            "Archaea": "#762A83"
        }
        group_labels = {
            "Gram-negative": "Gram-negative (Proteobacteria)",
            "Gram-positive": "Gram-positive (Firmicutes/Actinobacteria)",
            "Cyanobacteria": "Cyanobacteria",
            "Archaea": "Archaea"
        }
        
        # Collect all iPro-MP models
        ipromp_curves = []
        for name, data in models_data.items():
            if not name.startswith("iPro-MP (sp "):
                continue
            import re
            m = re.match(r"iPro-MP \(sp (\d+)\)", name)
            if not m:
                continue
            sp_id = int(m.group(1))
            
            pos_scores = [data["pos"][seq_id] for seq_id in pos_ids_list if seq_id in data["pos"]]
            if len(pos_scores) == 0:
                continue
            neg_scores = data["neg"]
            y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
            y_scores = np.concatenate([pos_scores, neg_scores])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(rec, prec)
            
            sp_name = ipromp_species_names.get(sp_id, f"sp {sp_id}")
            group = gram_to_group.get(sp_id, "Other")
            
            ipromp_curves.append({
                "sp_id": sp_id,
                "sp_name": sp_name,
                "group": group,
                "fpr": fpr, "tpr": tpr,
                "rec": rec, "prec": prec,
                "roc_auc": roc_auc, "pr_auc": pr_auc
            })
        
        # Sort within each group by AUC descending
        group_order = ["Gram-negative", "Gram-positive", "Cyanobacteria", "Archaea"]
        for g in group_order:
            group_curves = [c for c in ipromp_curves if c["group"] == g]
            group_curves.sort(key=lambda x: x["roc_auc"], reverse=True)
            for i, c in enumerate(group_curves):
                c["rank_in_group"] = i
                c["n_in_group"] = len(group_curves)
        
        # Find best per group
        best_per_group = {}
        for g in group_order:
            gc = [c for c in ipromp_curves if c["group"] == g]
            if gc:
                gc.sort(key=lambda x: x["roc_auc"], reverse=True)
                best_per_group[g] = gc[0]
        
        matplotlib.use('cairo')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        
        # ROC plot
        ax1.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
        for c in ipromp_curves:
            is_best = (best_per_group.get(c["group"]) is c)
            alpha = 0.35 if not is_best else 1.0
            lw = 0.8 if not is_best else 1.8
            color = group_colors[c["group"]]
            ax1.plot(c["fpr"], c["tpr"], color=color, lw=lw, alpha=alpha)
        
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.set_xlabel("FPR")
        ax1.set_ylabel("TPR")
        ax1.set_title(f"ROC - iPro-MP by Gram Class ({subset_name})", fontsize=9, fontweight="bold")
        
        # PR plot
        for c in ipromp_curves:
            is_best = (best_per_group.get(c["group"]) is c)
            alpha = 0.35 if not is_best else 1.0
            lw = 0.8 if not is_best else 1.8
            color = group_colors[c["group"]]
            ax2.plot(c["rec"], c["prec"], color=color, lw=lw, alpha=alpha)
        
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title(f"PR - iPro-MP by Gram Class ({subset_name})", fontsize=9, fontweight="bold")
        
        # No legend on the plot itself — save a separate legend page as a table
        
        plt.tight_layout()
        pdf_path = str(resultados_dir / filename.replace(".svg", ".pdf"))
        plt.savefig(pdf_path, bbox_inches="tight", format="pdf")
        print(f"Saved iPro-MP Gram plot PDF to: {pdf_path}")
        plt.close()
    
    make_ipromp_gram_plot(list(metadata.index), "Overall Dataset", "master_plot_ipromp_gram.pdf")
    make_ipromp_gram_plot(list(siga_ids), "SigA Housekeeping", "master_plot_ipromp_gram_siga.pdf")
    make_ipromp_gram_plot(list(sigx_ids), "SigX Competence", "master_plot_ipromp_gram_sigx.pdf")
    
    # Generate 2x2 individual panel plots for each subset
    make_individual_panels_plot(list(metadata.index), "Overall Dataset", "master_plot_individual_panels.svg")
    make_individual_panels_plot(list(siga_ids), "SigA Housekeeping", "master_plot_individual_panels_siga.svg")
    make_individual_panels_plot(list(sigx_ids), "SigX Competence", "master_plot_individual_panels_sigx.svg")

    # Clean up temporary PromoTech pipelines directory after plotting is finished
    if temp_dir.exists():
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary PromoTech directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up {temp_dir}: {e}")

if __name__ == "__main__":
    main()
