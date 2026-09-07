#!/usr/bin/env python3
"""
================================================================================
COMPREHENSIVE EVALUATION & MECHANISTIC INTERPRETABILITY: Prompt (Du et al., 2024)
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Evaluates:
  1. Checkpoint Training Dynamics: B. subtilis 168 vs E. coli across epochs (50 to 500).
  2. Cross-Strain & Subclass Generalization: D39V, TIGR4, SigA, SigX, 1:1 Orthologs.
  3. Neural Network Positional Weight Attribution across the 81 bp window (TSS +1).
  4. Publication Figures (300 DPI, English).
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

PLOTS_DIR = ROOT_DIR / "output/plots/prompt_evaluation"
TABLES_DIR = ROOT_DIR / "output/tables"
DESK_DIR = Path("/home/fierro/Desktop/prompt_evaluation_results")

for d in [PLOTS_DIR, TABLES_DIR, DESK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

class MLP_(nn.Module):                    
    def __init__(self, num_classes=2):
        super(MLP_, self).__init__()      
        self.layer1 = nn.Linear(81, 16)
        self.Tanh = nn.Tanh()
        self.layer3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.Tanh(x)
        x = self.layer3(x)
        return x

DATA_DICT = {'A':0, 'T':1, 'C':2, 'G':3, 'N':4, 'M':5, 'Y':6, 'W':7}

def predict_fasta_with_model(model, fasta_path):
    if not fasta_path.exists():
        return np.array([])
    recs = list(SeqIO.parse(fasta_path, "fasta"))
    seqs = [str(r.seq).upper()[:81] for r in recs]
    padded = []
    for s in seqs:
        if len(s) < 81:
            s = s + 'A' * (81 - len(s))
        padded.append([DATA_DICT.get(ch, 0) for ch in s])
    mat = np.array(padded, dtype=np.float32)
    with torch.no_grad():
        out = model(torch.tensor(mat, dtype=torch.float32))
        probs = torch.softmax(out, dim=1)[:, 1].numpy()
    return probs

def evaluate_checkpoints():
    pos_fasta = ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta"
    neg_fasta = ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"
    
    ckpt_dir = ROOT_DIR / "tools/Prompt/mlp/ckpt"
    ckpts = sorted(list(ckpt_dir.glob("*/*.pth")))
    
    records = []
    for cp in ckpts:
        species = "B. subtilis (168)" if cp.parent.name == "168" else "E. coli (coli)"
        epoch_str = cp.stem.replace("model_", "")
        epoch = int(epoch_str)
        
        model = torch.load(cp, map_location="cpu", weights_only=False)
        model.eval()
        
        p_preds = predict_fasta_with_model(model, pos_fasta)
        n_preds = predict_fasta_with_model(model, neg_fasta)
        
        y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
        y_pred = np.concatenate([p_preds, n_preds])
        
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        mcc = matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
        
        records.append({
            "Species": species,
            "Epoch": epoch,
            "ROC_AUC": round(auc, 4),
            "Accuracy": round(acc, 4),
            "MCC": round(mcc, 4),
            "Pos_Mean": round(float(p_preds.mean()), 4),
            "Neg_Mean": round(float(n_preds.mean()), 4),
            "Delta": round(float(p_preds.mean() - n_preds.mean()), 4)
        })
        
    return pd.DataFrame(records).sort_values(["Species", "Epoch"])

def evaluate_cross_datasets(best_model):
    datasets = [
        ("D39V Baseline (All TSS)", ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta", ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"),
        ("D39V Canonical SigA", ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr_SigA.fasta", ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"),
        ("D39V Competence SigX", ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr_SigX.fasta", ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"),
        ("TIGR4 High Conf IGR", ROOT_DIR / "data/tigr4/positives_high_81bp.fasta", ROOT_DIR / "data/tigr4/negatives_high_81bp.fasta"),
        ("TIGR4 All IGR Subsets", ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/positives_81bp.fasta", ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/negatives_81bp.fasta"),
        ("Inter-strain 1:1 Orthologs", ROOT_DIR / "data/benchmark/specialized_niches/orthologs_1to1/positives_81bp.fasta", ROOT_DIR / "data/benchmark/specialized_niches/orthologs_1to1/negatives_81bp.fasta")
    ]
    
    rows = []
    for d_name, p_f, n_f in datasets:
        if not p_f.exists() or not n_f.exists():
            continue
        p_preds = predict_fasta_with_model(best_model, p_f)
        n_preds = predict_fasta_with_model(best_model, n_f)
        
        y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
        y_pred = np.concatenate([p_preds, n_preds])
        
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
        mcc = matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
        
        rows.append({
            "Dataset_Evaluation": d_name,
            "N_Positives": len(p_preds),
            "N_Negatives": len(n_preds),
            "ROC_AUC": round(auc, 4),
            "Accuracy_0.5": round(acc, 4),
            "MCC": round(mcc, 4),
            "Pos_Mean": round(float(p_preds.mean()), 4),
            "Neg_Mean": round(float(n_preds.mean()), 4),
            "Delta": round(float(p_preds.mean() - n_preds.mean()), 4)
        })
        
    return pd.DataFrame(rows)

def extract_positional_attribution(model):
    W1 = model.layer1.weight.detach().numpy() # (16, 81)
    W3 = model.layer3.weight.detach().numpy() # (2, 16)
    
    # Effective linear combination of weights projecting into the promoter logit:
    # Score_diff = sum_h (W3[1, h] - W3[0, h]) * W1[h, i]
    promoter_direction = W3[1, :] - W3[0, :]
    effective_weights = np.dot(promoter_direction, W1) # (81,)
    return effective_weights

def main():
    print("=" * 85)
    print("RUNNING COMPREHENSIVE EVALUATION & ATTRIBUTION OF PROMPT")
    print("=" * 85)
    
    # 1. Checkpoint evaluation across training epochs
    df_ckpts = evaluate_checkpoints()
    df_ckpts.to_csv(TABLES_DIR / "prompt_checkpoints_epoch_dynamics.tsv", sep="\t", index=False)
    df_ckpts.to_csv(DESK_DIR / "prompt_checkpoints_epoch_dynamics.tsv", sep="\t", index=False)
    print("\n--- Checkpoint Training Dynamics ---")
    print(df_ckpts.to_string(index=False))
    
    # 2. Cross-Dataset Evaluation using the top model (B. subtilis 168 model_450 / model_500)
    best_model_path = ROOT_DIR / "tools/Prompt/mlp/ckpt/168/model_500.pth"
    best_model = torch.load(best_model_path, map_location="cpu", weights_only=False)
    best_model.eval()
    
    df_cross = evaluate_cross_datasets(best_model)
    df_cross.to_csv(TABLES_DIR / "prompt_cross_dataset_performance.tsv", sep="\t", index=False)
    df_cross.to_csv(DESK_DIR / "prompt_cross_dataset_performance.tsv", sep="\t", index=False)
    print("\n--- Cross-Dataset Generalization ---")
    print(df_cross.to_string(index=False))
    
    # 3. Mechanistic Positional Attribution
    pos_weights = extract_positional_attribution(best_model)
    
    # 4. Generate Master Publication Figure (3-Panel, 300 DPI, English)
    fig = plt.figure(figsize=(18, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.3, wspace=0.25)
    
    # Panel A: Checkpoint Dynamics (ROC-AUC vs Epoch)
    ax_a = fig.add_subplot(gs[0, 0])
    sub168 = df_ckpts[df_ckpts["Species"] == "B. subtilis (168)"]
    sub_coli = df_ckpts[df_ckpts["Species"] == "E. coli (coli)"]
    
    ax_a.plot(sub168["Epoch"], sub168["ROC_AUC"], marker="o", lw=2.5, color="#1f77b4", label="B. subtilis 168 (Firmicute)")
    ax_a.plot(sub_coli["Epoch"], sub_coli["ROC_AUC"], marker="s", lw=2.5, color="#d62728", linestyle="--", label="E. coli (Gammaproteobacteria)")
    ax_a.set_xlabel("Training Epochs", fontsize=11, fontweight="bold")
    ax_a.set_ylabel("ROC-AUC on S. pneumoniae", fontsize=11, fontweight="bold")
    ax_a.set_title("A. Cross-Phylum Transfer Dynamics (Firmicute vs. Proteobacteria)", fontsize=12, fontweight="bold", pad=10)
    ax_a.set_ylim([0.70, 0.90])
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="lower right", fontsize=10, frameon=True)
    
    # Panel B: Cross-Dataset Performance Bar Chart
    ax_b = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(len(df_cross))
    bars = ax_b.barh(y_pos, df_cross["ROC_AUC"], color="#2ca02c", alpha=0.85, edgecolor="#1b611b", height=0.6)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(df_cross["Dataset_Evaluation"], fontsize=10, fontweight="bold")
    ax_b.set_xlabel("ROC-AUC Score", fontsize=11, fontweight="bold")
    ax_b.set_title("B. Generalization Across Niches & Ortholog Datasets", fontsize=12, fontweight="bold", pad=10)
    ax_b.set_xlim([0.5, 1.0])
    ax_b.grid(True, axis="x", linestyle=":", alpha=0.6)
    
    for bar, val in zip(bars, df_cross["ROC_AUC"]):
        ax_b.text(val + 0.01, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9.5, fontweight="bold")
        
    # Panel C: Mechanistic Positional Attribution Curve aligned to TSS +1 (Spans both bottom columns)
    ax_c = fig.add_subplot(gs[1, :])
    rel_positions = np.arange(-60, 21) # 81 bp, index 60 is TSS +1 -> 0
    
    # Plot bar chart of weights
    colors_pos = ["#d62728" if w < 0 else "#1f77b4" for w in pos_weights]
    ax_c.bar(rel_positions, pos_weights, color=colors_pos, width=0.8, alpha=0.8, edgecolor="black", lw=0.4)
    
    # Highlight Biological Functional Motifs
    # 1. -35 Box (-37 to -30)
    ax_c.axvspan(-37, -30, color="#ffe6cc", alpha=0.7, label="-35 Box Region (TTGACA)")
    # 2. Extended -10 (TG motif around -17 to -13)
    ax_c.axvspan(-17, -13, color="#d5e8d4", alpha=0.7, label="Extended -10 (TRTGNT)")
    # 3. -10 Pribnow Box (-12 to -6)
    ax_c.axvspan(-12, -6, color="#dae8fc", alpha=0.7, label="-10 Pribnow Box (TATAAT)")
    # 4. TSS +1 (position 0)
    ax_c.axvline(0, color="black", linestyle="--", lw=2, label="Transcription Start Site (TSS +1)")
    
    ax_c.set_xlabel("Nucleotide Position Relative to TSS +1 (bp)", fontsize=12, fontweight="bold")
    ax_c.set_ylabel("Neural Network Weight Attribution", fontsize=12, fontweight="bold")
    ax_c.set_title("C. Mechanistic Neural Attribution Map — Prompt Aligns with Canonical Bacterial Promoter Geometry", fontsize=13, fontweight="bold", pad=12)
    ax_c.set_xlim([-61, 21])
    ax_c.grid(True, linestyle=":", alpha=0.5)
    ax_c.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.95)
    
    plt.suptitle("Comprehensive Evaluation & Mechanistic Interpretability Atlas: Prompt (MLP Architecture)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    out_fig = PLOTS_DIR / "prompt_comprehensive_evaluation_atlas.png"
    plt.savefig(out_fig, dpi=300)
    plt.savefig(DESK_DIR / "prompt_comprehensive_evaluation_atlas.png", dpi=300)
    plt.close()
    
    print(f"\n[DONE] Figures and tables saved successfully to {PLOTS_DIR} and {DESK_DIR}")

if __name__ == "__main__":
    main()
