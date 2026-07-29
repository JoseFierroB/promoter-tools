#!/usr/bin/env python3
"""
PromoTech Experiment - AUC Sliding Window Analysis for RF-HOT and RF-TETRA.
Slides a 40-nt window in steps of 5 bp across 81-bp positive and negative sequences,
runs predictions using both models, calculates classification AUC, and generates comparative plots.
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.metrics import roc_curve, auc

# Set paths for PromoTech imports
PROMOTECH_DIR = "/home/fierro/Desktop/Promotech"
sys.path.append(PROMOTECH_DIR)
sys.path.append(os.path.join(PROMOTECH_DIR, "core"))

from utils import dataConverter, predict_proba

def predict_in_batches(model, seqs, model_type, tokenizer_path, batch_size=100):
    """Predict in batches to avoid sandbox memory allocation crash."""
    all_scores = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i+batch_size]
        X = dataConverter(seqs=np.array(chunk), data_type=model_type, tokenizer_path=tokenizer_path, print_fn=lambda *a,**k: None)
        probs = predict_proba(model, X)
        scores = probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
        all_scores.extend(scores)
    return np.array(all_scores)

def main():
    pos_path = "data/benchmark/positives_81bp.fasta"
    neg_path = "data/benchmark/negatives_81bp.fasta"
    out_tsv = "output/plots/promotech/promotech_auc_sliding.tsv"
    out_plot = "output/plots/promotech/promotech_auc_sliding.png"
    
    # Load sequences
    print(f"Loading positives from: {pos_path}")
    pos_records = list(SeqIO.parse(pos_path, "fasta"))
    print(f"Loading negatives from: {neg_path}")
    neg_records = list(SeqIO.parse(neg_path, "fasta"))
    
    pos_seqs = [str(r.seq).upper() for r in pos_records]
    neg_seqs = [str(r.seq).upper() for r in neg_records]
    print(f"Loaded {len(pos_seqs)} positives and {len(neg_seqs)} negatives.")
    
    # Change cwd to Promotech folder to load files correctly
    os.chdir(PROMOTECH_DIR)
    tokenizer_path = "models/tokenizer.data"
    
    model_configs = [
        {"name": "RF-HOT", "type": "RF-HOT", "model_path": "models/RF-HOT.model", "color": "crimson"},
        {"name": "RF-TETRA", "type": "RF-TETRA", "model_path": "models/RF-TETRA.model", "color": "royalblue"}
    ]
    
    start_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    all_results = []
    
    plt.figure(figsize=(10, 6))
    
    for config in model_configs:
        model_name = config["name"]
        model_type = config["type"]
        model_file = config["model_path"]
        plot_color = config["color"]
        
        print(f"\n==================================================")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"==================================================")
        print(f"Loading PromoTech model from: {model_file}...")
        model = joblib.load(model_file)
        
        auc_values = []
        rel_starts = []
        
        print("Starting sliding window AUC evaluation...")
        for start_idx in start_indices:
            rel_start = start_idx - 40
            rel_end = rel_start + 40
            window_label = f"[{rel_start}, {rel_end-1}]"
            
            # Slice sequences
            pos_slices = [s[start_idx:start_idx+40] for s in pos_seqs]
            neg_slices = [s[start_idx:start_idx+40] for s in neg_seqs]
            
            # Predict
            pos_scores = predict_in_batches(model, pos_slices, model_type, tokenizer_path)
            neg_scores = predict_in_batches(model, neg_slices, model_type, tokenizer_path)
            
            # Compute ROC and AUC
            y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
            y_scores = np.concatenate([pos_scores, neg_scores])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_val = auc(fpr, tpr)
            
            print(f"  Window starting at index {start_idx} (TSS relative: {window_label}) -> AUC: {auc_val:.4f} | Mean Pos: {np.mean(pos_scores):.4f} | Mean Neg: {np.mean(neg_scores):.4f}")
            
            all_results.append({
                "Model": model_name,
                "Start_Index": start_idx,
                "TSS_Relative_Start": rel_start,
                "TSS_Relative_End": rel_end - 1,
                "Window": window_label,
                "AUC": auc_val,
                "Mean_Pos_Score": np.mean(pos_scores),
                "Mean_Neg_Score": np.mean(neg_scores)
            })
            
            auc_values.append(auc_val)
            rel_starts.append(rel_start)
            
        # Plot curve for this model
        plt.plot(rel_starts, auc_values, marker="o", color=plot_color, lw=2, markersize=8, label=f"{model_name} AUC")
        
        # Clean memory before loading next model
        del model
        gc.collect()
        
    df_res = pd.DataFrame(all_results)
    
    # Save TSV Report
    df_res.to_csv(out_tsv, sep="\t", index=False)
    print(f"\n[SUCCESS] TSV report saved to: {out_tsv}")
    
    # Format plot
    plt.xlabel("Window Start Position (relative to TSS at 0)")
    plt.ylabel("Classification AUC (Pos vs Neg)")
    plt.title("PromoTech Performance vs. 40-nt Window Shift (on 81-bp sequences)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim([0.3, 1.05])
    plt.xticks(start_indices - np.array([40]*len(start_indices)))
    
    # Add vertical line for the centered optimal window
    plt.axvline(x=-20, color="green", linestyle="--", alpha=0.7, label="Centered Window [-20, +19]")
    
    plt.legend(loc="lower left")
    plt.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Comparative plot saved to: {out_plot}")
    
    print("\nSUMMARY TABLE:")
    print("="*90)
    print(df_res.to_string(index=False))
    print("="*90)

if __name__ == "__main__":
    main()
