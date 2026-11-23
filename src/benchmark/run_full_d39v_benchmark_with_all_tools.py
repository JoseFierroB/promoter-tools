#!/usr/bin/env python3
"""
================================================================================
MASTER BENCHMARK: S. pneumoniae D39V Baseline (N=1,989) — All 7 Tools
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Evaluates:
  1. ProkBERT-mini (NeuralBioInfo / gLM)
  2. iPro-MP (Transformer / gLM)
  3. PromoterLCNN (CNN)
  4. PromoTech RF-HOT (Random Forest)
  5. Prompt MLP 168 (Firmicute MLP)
  6. MLDSPP 0% (Zero-shot BDT)
  7. EvoSNR-Prom (FastText Lexicon)
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

POS_FASTA = ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta"
NEG_FASTA = ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta"

OUT_DIR = ROOT_DIR / "output/d39v_all_tools_benchmark"
PRED_DIR = OUT_DIR / "predictions"
TABLES_DIR = ROOT_DIR / "output/tables"
PLOTS_DIR = ROOT_DIR / "output/plots/d39v_all_tools"
DESK_DIR = Path("/home/fierro/Desktop/d39v_all_tools_results")

for d in [PRED_DIR, TABLES_DIR, PLOTS_DIR, DESK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def run_command(cmd, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return elapsed, res.returncode, res.stdout, res.stderr

def main():
    parser = argparse.ArgumentParser(description="Run full D39V benchmark across all 7 tools.")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads to use (default: 16).")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU index (default: 0).")
    args = parser.parse_args()

    thread_env = {
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
        "TF_NUM_INTRAOP_THREADS": str(args.threads),
        "TF_NUM_INTEROP_THREADS": str(args.threads),
        "CUDA_VISIBLE_DEVICES": args.gpu_id
    }

    n_pos = len(list(SeqIO.parse(POS_FASTA, "fasta")))
    n_neg = len(list(SeqIO.parse(NEG_FASTA, "fasta")))
    n_total = n_pos + n_neg

    print("=" * 90)
    print(f"STARTING FULL D39V BENCHMARK (N={n_total}: {n_pos} Pos / {n_neg} Neg)")
    print(f"Hardware Configuration: {args.threads} Threads | GPU: {args.gpu_id}")
    print("=" * 90)

    # 1. Tool execution definitions
    tool_cmds = [
        ("ProkBERT-mini [gLM]", f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prokbert_mini.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR}/prokbert.tsv"),
        ("Prompt [MLP 168]", f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prompt.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR}/prompt.tsv"),
        ("EvoSNR-Prom [Lexicon]", f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/evosnr.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR}/evosnr.tsv"),
        ("PromoterLCNN [CNN]", f"tools/Promoters/.pixi/envs/default/bin/python src/runners/lcnn.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR}"),
        ("iPro-MP [gLM]", f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/ipromp_sp12.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR} -m tools/iPro-MP/07-final -d tools/iPro-MP/DNABERT-6"),
        ("PromoTech RF-HOT [RF]", f"tools/Promotech/.pixi/envs/default/bin/python src/runners/promotech_hot.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR}"),
        ("MLDSPP 0% [BDT]", f"pixi run python src/runners/mldspp.py --pos {POS_FASTA} --neg {NEG_FASTA} -o {PRED_DIR}")
    ]

    resource_metrics = []

    for name, cmd in tool_cmds:
        print(f"\n--> Running {name}...", flush=True)
        elapsed, retcode, stdout, stderr = run_command(cmd, thread_env)
        status = "OK" if retcode == 0 else "FAIL"
        print(f"    [{status}] Elapsed: {elapsed:.2f}s | Throughput: {n_total/max(elapsed, 0.001):.1f} seq/s")
        if retcode != 0:
            print(f"    Error: {stderr[:300]}")
        resource_metrics.append({
            "Tool": name,
            "N_Sequences": n_total,
            "Wall_Seconds": round(elapsed, 3),
            "Throughput_seq_s": round(n_total / max(elapsed, 0.001), 1),
            "Status": status
        })

    df_metrics = pd.DataFrame(resource_metrics)
    metrics_tsv = TABLES_DIR / "resource_metrics_d39v_all_tools.tsv"
    df_metrics.to_csv(metrics_tsv, sep="\t", index=False)
    df_metrics.to_csv(DESK_DIR / "resource_metrics_d39v_all_tools.tsv", sep="\t", index=False)

    # 2. Extract predictions and compute performance metrics
    pred_data = {}
    
    # ProkBERT
    if (PRED_DIR / "prokbert.tsv").exists():
        df_p = pd.read_csv(PRED_DIR / "prokbert.tsv", sep="\t")
        pred_data["ProkBERT-mini [gLM]"] = (df_p["LABEL"].values, df_p["PRED"].values)
        
    # iPro-MP
    ip_csv = PRED_DIR / "ipromp/ipromp_12_predictions.csv"
    if ip_csv.exists():
        ip_preds = pd.read_csv(ip_csv)["PRED"].values
        y_true_ip = np.array([1]*n_pos + [0]*n_neg)
        pred_data["iPro-MP [gLM]"] = (y_true_ip, ip_preds)

    # LCNN
    lcnn_pos = PRED_DIR / "lcnn/lcnn_pos.csv"
    lcnn_neg = PRED_DIR / "lcnn/lcnn_neg.csv"
    if lcnn_pos.exists() and lcnn_neg.exists():
        p_lc = pd.read_csv(lcnn_pos, sep="\t")["PRED"].values
        n_lc = pd.read_csv(lcnn_neg, sep="\t")["PRED"].values
        pred_data["PromoterLCNN [CNN]"] = (np.array([1]*len(p_lc) + [0]*len(n_lc)), np.concatenate([p_lc, n_lc]))

    # PromoTech
    pt_pos = PRED_DIR / "promotech/workdir/hot_pg_pos/sequences_predictions.csv"
    pt_neg = PRED_DIR / "promotech/workdir/hot_pg_neg/sequences_predictions.csv"
    if pt_pos.exists() and pt_neg.exists():
        p_pt = pd.read_csv(pt_pos, sep="\t")["PRED"].values
        n_pt = pd.read_csv(pt_neg, sep="\t")["PRED"].values
        pred_data["PromoTech RF-HOT [RF]"] = (np.array([1]*len(p_pt) + [0]*len(n_pt)), np.concatenate([p_pt, n_pt]))

    # Prompt
    if (PRED_DIR / "prompt.tsv").exists():
        df_pr = pd.read_csv(PRED_DIR / "prompt.tsv", sep="\t")
        pred_data["Prompt [MLP 168]"] = (df_pr["LABEL"].values, df_pr["PRED"].values)

    # MLDSPP 0%
    ml_pos = PRED_DIR / "mldspp_pos.csv"
    ml_neg = PRED_DIR / "mldspp_neg.csv"
    if ml_pos.exists() and ml_neg.exists():
        p_ml = pd.read_csv(ml_pos, sep="\t")["PRED"].values
        n_ml = pd.read_csv(ml_neg, sep="\t")["PRED"].values
        pred_data["MLDSPP 0% [BDT]"] = (np.array([1]*len(p_ml) + [0]*len(n_ml)), np.concatenate([p_ml, n_ml]))

    # EvoSNR
    if (PRED_DIR / "evosnr.tsv").exists():
        df_ev = pd.read_csv(PRED_DIR / "evosnr.tsv", sep="\t")
        pred_data["EvoSNR-Prom [Lexicon]"] = (df_ev["LABEL"].values, df_ev["PRED"].values)

    # 3. Compute Confusion Matrices and Statistical Metrics
    conf_rows = []
    roc_curves = {}

    for tool_name, (y_true, y_pred) in pred_data.items():
        auc = roc_auc_score(y_true, y_pred)
        fpr, tpr, th = roc_curve(y_true, y_pred)
        roc_curves[tool_name] = (fpr, tpr, auc)
        
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_th = th[best_idx]
        
        y_bin = (y_pred >= best_th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = f1_score(y_true, y_bin)
        mcc = matthews_corrcoef(y_true, y_bin)
        
        conf_rows.append({
            "Tool": tool_name,
            "N_Total": len(y_true),
            "ROC_AUC": round(auc, 4),
            "Optimal_Threshold": round(float(best_th), 4),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Sensitivity_TPR": round(sens, 4),
            "Specificity_TNR": round(spec, 4),
            "Precision_PPV": round(ppv, 4),
            "NPV": round(npv, 4),
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1, 4),
            "MCC": round(mcc, 4)
        })

    df_conf = pd.DataFrame(conf_rows).sort_values("ROC_AUC", ascending=False)
    conf_tsv = TABLES_DIR / "confusion_matrices_d39v_all_tools.tsv"
    df_conf.to_csv(conf_tsv, sep="\t", index=False)
    df_conf.to_csv(DESK_DIR / "confusion_matrices_d39v_all_tools.tsv", sep="\t", index=False)

    print("\n" + "=" * 110)
    print("D39V BASELINE BENCHMARK — PERFORMANCE & CONFUSION MATRIX SUMMARY (N=1,989)")
    print("=" * 110)
    print(df_conf.to_string(index=False))
    print("=" * 110)

    # 4. Generate Publication Plots (300 DPI, Pure English)
    # 4.1 Master ROC Curve Plot
    plt.figure(figsize=(9, 8), dpi=300)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]
    
    for i, (tool_name, (fpr, tpr, auc)) in enumerate(sorted(roc_curves.items(), key=lambda x: x[1][2], reverse=True)):
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5, label=f"{tool_name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=13, fontweight="bold")
    plt.title("S. pneumoniae D39V Baseline Promoter Benchmark\nReceiver Operating Characteristic (ROC) Atlas (N=1,989)", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10.5, frameon=True, framealpha=0.95, edgecolor="#cccccc")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    roc_plot_path = PLOTS_DIR / "roc_curves_d39v_all_tools.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.savefig(DESK_DIR / "roc_curves_d39v_all_tools.png", dpi=300)
    plt.close()

    # 4.2 Master Confusion Matrix Heatmap Panel
    n_tools = len(df_conf)
    cols = 4
    rows = int(np.ceil(n_tools / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), dpi=300)
    axes = axes.flatten()

    for idx, (_, r) in enumerate(df_conf.iterrows()):
        ax = axes[idx]
        cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
        
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{r['Tool']}\nAUC = {r['ROC_AUC']:.3f} | Acc = {r['Accuracy']*100:.1f}%", fontsize=11, fontweight="bold")
        
        classes = ["Actual Pos", "Actual Neg"]
        preds = ["Pred Pos", "Pred Neg"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(preds, fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)
        
        thresh = cm.max() / 2.
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                total_class = n_pos if i == 0 else n_neg
                pct = val / total_class * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        color="white" if val > thresh else "black", fontsize=11, fontweight="bold")
                
        ax.set_ylabel("True Class", fontsize=10)
        ax.set_xlabel("Predicted Class", fontsize=10)

    for j in range(n_tools, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("S. pneumoniae D39V Baseline — Tool-by-Tool Confusion Matrix Atlas (N=1,989)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    
    cm_plot_path = PLOTS_DIR / "confusion_matrices_d39v_all_tools.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.savefig(DESK_DIR / "confusion_matrices_d39v_all_tools.png", dpi=300)
    plt.close()

    print(f"\n[DONE] Figures and tables saved successfully to {PLOTS_DIR} and {DESK_DIR}")

if __name__ == "__main__":
    main()
