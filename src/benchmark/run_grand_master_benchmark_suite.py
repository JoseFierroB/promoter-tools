#!/usr/bin/env python3
"""
================================================================================
GRAND MASTER BENCHMARK SUITE: S. pneumoniae D39V, TIGR4 & Specialized Niches
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Hardware Configuration: 16 CPU Threads + NVIDIA RTX 3090 GPU (PCI_BUS_ID 0)

14 Datasets:
  1. D39V Baseline All (N=1,989)
  2. D39V Canonical SigA (N=1,310)
  3. D39V Competence SigX (N=1,012)
  4. D39V Unassigned Secondary (N=1,570)
  5. TIGR4 High-Confidence Primary Sub1 (N=1,106)
  6. TIGR4 High-Confidence All Sub2 (N=1,156)
  7. TIGR4 Primary All Sub3 (N=1,942)
  8. TIGR4 All Comprehensive Sub4 (N=2,018)
  9. TIGR4 Curated Data (N=1,476)
  10. Combined 1:1 Orthologs D39V <-> TIGR4 (N=1,232)
  11. Combined High-Confidence IGR (N=1,576)
  12. Combined All Comprehensive IGR (N=2,488)
  13. CDS Internal High-Confidence (N=964)
  14. CDS Internal All (N=2,796)

7 Tool Architectures (Best Optimized Configurations):
  1. ProkBERT-mini (gLM) -> GPU RTX 3090
  2. iPro-MP (gLM Transformer) -> GPU RTX 3090
  3. PromoterLCNN (CNN) -> GPU RTX 3090
  4. PromoTech RF-HOT (Random Forest) -> 16 CPUs
  5. Prompt MLP 168 (Firmicute Epoch 500) -> 16 CPUs
  6. EvoSNR-Prom Dual PWM (Slager + ProkDB) -> 16 CPUs
  7. MLDSPP 0% (Zero-shot BDT) -> 16 CPUs
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

OUT_DIR = ROOT_DIR / "output/grand_master_suite"
PRED_DIR = OUT_DIR / "predictions"
TABLES_DIR = ROOT_DIR / "output/tables"
PLOTS_DIR = ROOT_DIR / "output/plots/grand_master_suite"
DESK_DIR = Path("/home/fierro/Desktop/grand_master_results")

for d in [PRED_DIR, TABLES_DIR, PLOTS_DIR, DESK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "1_D39V_Baseline": {
        "pos": ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta",
        "category": "D39V Pure",
        "label": "D39V Baseline (N=1,989)"
    },
    "2_D39V_SigA": {
        "pos": ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr_SigA.fasta",
        "neg": ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta",
        "category": "D39V Pure",
        "label": "D39V SigA (N=1,310)"
    },
    "3_D39V_SigX": {
        "pos": ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr_SigX.fasta",
        "neg": ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta",
        "category": "D39V Pure",
        "label": "D39V SigX (N=1,012)"
    },
    "4_TIGR4_High_Prim_Sub1": {
        "pos": ROOT_DIR / "data/benchmark_igr/tigr4/subset_1_high_conf_primary/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark_igr/tigr4/subset_1_high_conf_primary/negatives_81bp.fasta",
        "category": "TIGR4 IGR",
        "label": "TIGR4 High Prim (N=1,106)"
    },
    "5_TIGR4_High_All_Sub2": {
        "pos": ROOT_DIR / "data/benchmark_igr/tigr4/subset_2_high_conf_all/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark_igr/tigr4/subset_2_high_conf_all/negatives_81bp.fasta",
        "category": "TIGR4 IGR",
        "label": "TIGR4 High All (N=1,156)"
    },
    "6_TIGR4_Prim_All_Sub3": {
        "pos": ROOT_DIR / "data/benchmark_igr/tigr4/subset_3_all_primary/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark_igr/tigr4/subset_3_all_primary/negatives_81bp.fasta",
        "category": "TIGR4 IGR",
        "label": "TIGR4 Prim All (N=1,942)"
    },
    "7_TIGR4_Comprehensive_Sub4": {
        "pos": ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/negatives_81bp.fasta",
        "category": "TIGR4 IGR",
        "label": "TIGR4 All Exhaustive (N=2,018)"
    },
    "8_TIGR4_Curated": {
        "pos": ROOT_DIR / "data/tigr4/positives_high_81bp.fasta",
        "neg": ROOT_DIR / "data/tigr4/negatives_high_81bp.fasta",
        "category": "TIGR4 IGR",
        "label": "TIGR4 Curated (N=1,476)"
    },
    "9_Combined_Orthologs": {
        "pos": ROOT_DIR / "data/benchmark/specialized_niches/orthologs_1to1/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark/specialized_niches/orthologs_1to1/negatives_81bp.fasta",
        "category": "Combined Inter-strain",
        "label": "Combined Orthologs 1:1 (N=1,232)"
    },
    "10_Combined_High_IGR": {
        "pos": ROOT_DIR / "data/benchmark/specialized_niches/combined_interstrain_high_confidence/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark/specialized_niches/combined_interstrain_high_confidence/negatives_81bp.fasta",
        "category": "Combined Inter-strain",
        "label": "Combined High-Conf IGR (N=1,576)"
    },
    "11_Combined_All_IGR": {
        "pos": ROOT_DIR / "data/benchmark/specialized_niches/combined_interstrain_all/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark/specialized_niches/combined_interstrain_all/negatives_81bp.fasta",
        "category": "Combined Inter-strain",
        "label": "Combined All IGR (N=2,488)"
    },
    "12_CDS_Internal_High": {
        "pos": ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_high_confidence/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_high_confidence/negatives_81bp.fasta",
        "category": "CDS Internals",
        "label": "CDS Internals High-Conf (N=964)"
    },
    "13_CDS_Internal_All": {
        "pos": ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_all/positives_81bp.fasta",
        "neg": ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_all/negatives_81bp.fasta",
        "category": "CDS Internals",
        "label": "CDS Internals All (N=2,796)"
    }
}

def run_command(cmd, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return elapsed, res.returncode, res.stdout, res.stderr

def run_model_on_dataset(model_key, pos_path, neg_path, out_prefix, env_vars):
    if model_key == "prokbert":
        out_f = f"{out_prefix}_prokbert.tsv"
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prokbert_mini.py --pos {pos_path} --neg {neg_path} -o {out_f}"
        el, rc, out, err = run_command(cmd, env_vars)
        if rc == 0 and Path(out_f).exists():
            df = pd.read_csv(out_f, sep="\t")
            return df["LABEL"].values, df["PRED"].values, el
    elif model_key == "prompt":
        out_f = f"{out_prefix}_prompt.tsv"
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prompt.py --pos {pos_path} --neg {neg_path} -o {out_f}"
        el, rc, out, err = run_command(cmd, env_vars)
        if rc == 0 and Path(out_f).exists():
            df = pd.read_csv(out_f, sep="\t")
            return df["LABEL"].values, df["PRED"].values, el
    elif model_key == "evosnr":
        out_f = f"{out_prefix}_evosnr.tsv"
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/evosnr.py --pos {pos_path} --neg {neg_path} -o {out_f}"
        el, rc, out, err = run_command(cmd, env_vars)
        if rc == 0 and Path(out_f).exists():
            df = pd.read_csv(out_f, sep="\t")
            return df["LABEL"].values, df["PRED"].values, el
    elif model_key == "ipromp":
        out_d = Path(f"{out_prefix}_ipromp")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/ipromp_sp12.py --pos {pos_path} --neg {neg_path} -o {out_d} -m tools/iPro-MP/07-final -d tools/iPro-MP/DNABERT-6"
        el, rc, out, err = run_command(cmd, env_vars)
        pred_f = out_d / "ipromp/ipromp_12_predictions.csv"
        if pred_f.exists():
            preds = pd.read_csv(pred_f)["PRED"].values
            n_pos = len(list(SeqIO.parse(pos_path, "fasta")))
            n_neg = len(list(SeqIO.parse(neg_path, "fasta")))
            y_true = np.array([1]*n_pos + [0]*n_neg)
            return y_true, preds, el
    elif model_key == "lcnn":
        out_d = Path(f"{out_prefix}_lcnn")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"tools/Promoters/.pixi/envs/default/bin/python src/runners/lcnn.py --pos {pos_path} --neg {neg_path} -o {out_d}"
        el, rc, out, err = run_command(cmd, env_vars)
        pos_f = out_d / "lcnn/lcnn_pos.csv"
        neg_f = out_d / "lcnn/lcnn_neg.csv"
        if pos_f.exists() and neg_f.exists():
            p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
            n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
            y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
            return y_true, np.concatenate([p_preds, n_preds]), el
    elif model_key == "promotech":
        out_d = Path(f"{out_prefix}_promotech")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"tools/Promotech/.pixi/envs/default/bin/python src/runners/promotech_hot.py --pos {pos_path} --neg {neg_path} -o {out_d}"
        el, rc, out, err = run_command(cmd, env_vars)
        pos_f = out_d / "promotech/workdir/hot_pg_pos/sequences_predictions.csv"
        neg_f = out_d / "promotech/workdir/hot_pg_neg/sequences_predictions.csv"
        if pos_f.exists() and neg_f.exists():
            p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
            n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
            y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
            return y_true, np.concatenate([p_preds, n_preds]), el
    elif model_key == "mldspp":
        out_d = Path(f"{out_prefix}_mldspp")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"pixi run python src/runners/mldspp.py --pos {pos_path} --neg {neg_path} -o {out_d}"
        el, rc, out, err = run_command(cmd, env_vars)
        pos_f = out_d / "mldspp_pos.csv"
        neg_f = out_d / "mldspp_neg.csv"
        if pos_f.exists() and neg_f.exists():
            p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
            n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
            y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
            return y_true, np.concatenate([p_preds, n_preds]), el
    return None, None, 0.0

def main():
    parser = argparse.ArgumentParser(description="Run Grand Master Benchmark Suite.")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads (default: 16).")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU index (default: 0).")
    args = parser.parse_args()

    env_vars = {
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
        "TF_NUM_INTRAOP_THREADS": str(args.threads),
        "TF_NUM_INTEROP_THREADS": str(args.threads),
        "CUDA_VISIBLE_DEVICES": args.gpu_id
    }

    print("=" * 105)
    print("STARTING GRAND MASTER BENCHMARK SUITE (13 DATASETS x 7 TOOL ARCHITECTURES)")
    print(f"Hardware: {args.threads} Threads | GPU: {args.gpu_id}")
    print("=" * 105)

    models = [
        ("ProkBERT-mini [gLM]", "prokbert"),
        ("iPro-MP [gLM]", "ipromp"),
        ("PromoterLCNN [CNN]", "lcnn"),
        ("PromoTech RF-HOT [RF]", "promotech"),
        ("Prompt [MLP 168]", "prompt"),
        ("EvoSNR-Prom [Lexicon]", "evosnr"),
        ("MLDSPP 0% [BDT]", "mldspp")
    ]

    master_results = []
    roc_atlas_data = {d_key: {} for d_key in DATASETS}

    total_evals = len(DATASETS) * len(models)
    curr_eval = 0

    for d_key, d_info in DATASETS.items():
        pos_p = d_info["pos"]
        neg_p = d_info["neg"]
        d_cat = d_info["category"]
        d_label = d_info["label"]

        if not pos_p.exists() or not neg_p.exists():
            print(f"Skipping {d_key} (files missing: {pos_p} or {neg_p})")
            continue

        print(f"\n================================================================================")
        print(f"DATASET: {d_label} [{d_cat}]")
        print(f"================================================================================")

        for m_name, m_key in models:
            curr_eval += 1
            out_prefix = str(PRED_DIR / f"{d_key}_{m_key}")
            print(f"[{curr_eval}/{total_evals}] Running {m_name} on {d_key}...", end=" ", flush=True)

            y_true, y_pred, elapsed = run_model_on_dataset(m_key, pos_p, neg_p, out_prefix, env_vars)

            if y_true is not None and len(y_true) > 0:
                auc = roc_auc_score(y_true, y_pred)
                fpr, tpr, th = roc_curve(y_true, y_pred)
                roc_atlas_data[d_key][m_name] = (fpr, tpr, auc)

                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_th = th[best_idx]

                y_bin = (y_pred >= best_th).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

                sens = tp / (tp + fn) if (tp + fn) else 0.0
                spec = tn / (tn + fp) if (tn + fp) else 0.0
                ppv = tp / (tp + fp) if (tp + fp) else 0.0
                acc = (tp + tn) / (tp + tn + fp + fn)
                f1 = f1_score(y_true, y_bin)
                mcc = matthews_corrcoef(y_true, y_bin)

                print(f"[OK] AUC={auc:.4f} | Acc={acc*100:.1f}% ({elapsed:.2f}s)")

                master_results.append({
                    "Category": d_cat,
                    "Dataset_Key": d_key,
                    "Dataset_Label": d_label,
                    "Model": m_name,
                    "N_Positives": int(np.sum(y_true == 1)),
                    "N_Negatives": int(np.sum(y_true == 0)),
                    "N_Total": len(y_true),
                    "ROC_AUC": round(auc, 4),
                    "Optimal_Threshold": round(float(best_th), 4),
                    "Sensitivity_TPR": round(sens, 4),
                    "Specificity_TNR": round(spec, 4),
                    "Precision_PPV": round(ppv, 4),
                    "Accuracy": round(acc, 4),
                    "F1_Score": round(f1, 4),
                    "MCC": round(mcc, 4),
                    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
                    "Elapsed_Seconds": round(elapsed, 2)
                })
            else:
                print(f"[FAIL]")

    df_master = pd.DataFrame(master_results)
    tsv_out = TABLES_DIR / "grand_master_suite_benchmark_metrics.tsv"
    df_master.to_csv(tsv_out, sep="\t", index=False)
    df_master.to_csv(DESK_DIR / "grand_master_suite_benchmark_metrics.tsv", sep="\t", index=False)

    # 4. Generate Master Heatmap of AUC across All Datasets x 7 Models (300 DPI)
    pivot_auc = df_master.pivot(index="Dataset_Label", columns="Model", values="ROC_AUC")
    cols_order = [m[0] for m in models if m[0] in pivot_auc.columns]
    pivot_auc = pivot_auc[cols_order]

    plt.figure(figsize=(15, 11), dpi=300)
    plt.imshow(pivot_auc.values, cmap="YlGnBu", aspect="auto", vmin=0.45, vmax=1.0)
    plt.xticks(range(len(pivot_auc.columns)), pivot_auc.columns, fontsize=11, fontweight="bold", rotation=25, ha="right")
    plt.yticks(range(len(pivot_auc.index)), pivot_auc.index, fontsize=10.5, fontweight="bold")
    cbar = plt.colorbar()
    cbar.set_label("ROC-AUC Score", fontsize=12, fontweight="bold")

    for i in range(len(pivot_auc)):
        for j in range(len(pivot_auc.columns)):
            val = pivot_auc.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.85 else "black", fontsize=10.5, fontweight="bold")

    plt.title("S. pneumoniae Grand Master Benchmark Suite\nROC-AUC Performance Matrix: 13 Datasets x 7 Tool Architectures", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    hm_path = PLOTS_DIR / "grand_master_performance_heatmap.png"
    plt.savefig(hm_path, dpi=300)
    plt.savefig(DESK_DIR / "grand_master_performance_heatmap.png", dpi=300)
    plt.close()

    # 5. Generate 6-Panel Grand Master ROC Atlas
    fig, axes = plt.subplots(3, 2, figsize=(18, 22), dpi=300)
    axes = axes.flatten()
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"]

    selected_panels = [
        ("1_D39V_Baseline", "A. D39V Canonical Baseline (N=1,989)"),
        ("2_D39V_SigA", "B. D39V Canonical SigA Promoters (N=1,310)"),
        ("4_TIGR4_High_Prim_Sub1", "C. TIGR4 High-Confidence IGRs (N=1,106)"),
        ("7_TIGR4_Comprehensive_Sub4", "D. TIGR4 Comprehensive All IGRs (N=2,018)"),
        ("9_Combined_Orthologs", "E. Cross-Strain 1:1 Orthologs (N=1,232)"),
        ("12_CDS_Internal_High", "F. CDS Internal High-Confidence Promoters (N=964)")
    ]

    for idx, (d_key, title) in enumerate(selected_panels):
        ax = axes[idx]
        if d_key in roc_atlas_data and roc_atlas_data[d_key]:
            sorted_m = sorted(roc_atlas_data[d_key].items(), key=lambda x: x[1][2], reverse=True)
            for i, (m_name, (fpr, tpr, auc)) in enumerate(sorted_m):
                ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.4, label=f"{m_name} (AUC = {auc:.3f})")
            ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle="--", label="Chance (0.500)")
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.02])
            ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
            ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.95)
            ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("S. pneumoniae Grand Master Benchmark — 6-Panel Cross-Niche ROC Atlas (300 DPI)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    atlas_path = PLOTS_DIR / "grand_master_6panel_roc_atlas.png"
    plt.savefig(atlas_path, dpi=300)
    plt.savefig(DESK_DIR / "grand_master_6panel_roc_atlas.png", dpi=300)
    plt.close()

    print("\n" + "=" * 110)
    print("GRAND MASTER BENCHMARK SUITE COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {TABLES_DIR} and {DESK_DIR}")
    print("=" * 110)

if __name__ == "__main__":
    main()
