#!/usr/bin/env python3
"""
================================================================================
RUN SPECIALIZED NICHES BENCHMARK (CDS & 1:1 ORTHOLOGS)
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools
================================================================================
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUT_BASE = ROOT_DIR / "output/benchmark_specialized_niches"
OUT_PREDS = OUT_BASE / "predictions"
OUT_PLOTS = OUT_BASE / "plots"
OUT_TABLES = OUT_BASE / "tables"

PY_LCNN = str(ROOT_DIR / "tools/Promoters/.pixi/envs/default/bin/python")
PY_IPRO = str(ROOT_DIR / "tools/iPro-MP/.pixi/envs/ipro-mp/bin/python")

def run_cmd(cmd, desc):
    t0 = time.perf_counter()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    el = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"  [ERROR in {desc}]: {res.stderr.strip()[:200]}")
    else:
        print(f"  [{desc} DONE in {el:.2f}s]")
    return res

def evaluate_subset(sub_key, sub_name, pos_fa, neg_fa):
    print(f"\n--- EVALUATING: {sub_name} ({sub_key}) ---")
    sub_pred_dir = OUT_PREDS / sub_key
    sub_pred_dir.mkdir(parents=True, exist_ok=True)
    
    pos_recs = list(SeqIO.parse(pos_fa, "fasta"))
    neg_recs = list(SeqIO.parse(neg_fa, "fasta"))
    n_pos = len(pos_recs)
    n_neg = len(neg_recs)
    y_true = np.array([1] * n_pos + [0] * n_neg)
    model_scores = {}
    
    # 1. PromoterLCNN
    lcnn_py = PY_LCNN if Path(PY_LCNN).exists() else "python"
    run_cmd(f"{lcnn_py} src/runners/lcnn.py --pos {pos_fa} --neg {neg_fa} -o {sub_pred_dir}", "PromoterLCNN")
    lcnn_pos = sub_pred_dir / "lcnn" / "lcnn_pos.csv"
    lcnn_neg = sub_pred_dir / "lcnn" / "lcnn_neg.csv"
    if lcnn_pos.exists() and lcnn_neg.exists():
        p_sc = pd.read_csv(lcnn_pos, sep="\t")["PRED"].values
        n_sc = pd.read_csv(lcnn_neg, sep="\t")["PRED"].values
        model_scores["PromoterLCNN [CNN]"] = np.concatenate([p_sc, n_sc])

    # 2. iPro-MP
    ipro_py = PY_IPRO if Path(PY_IPRO).exists() else "python"
    run_cmd(f"{ipro_py} src/runners/ipromp_sp12.py --pos {pos_fa} --neg {neg_fa} -o {sub_pred_dir}", "iPro-MP")
    ip_pos = sub_pred_dir / "ipromp" / "ipromp_12_predictions.csv"
    if ip_pos.exists():
        y_sc = pd.read_csv(ip_pos)["PRED"].values
        if len(y_sc) == len(y_true):
            model_scores["iPro-MP [gLM]"] = y_sc

    # 3. PromoTech RF-HOT
    run_cmd(f"pixi run python src/runners/promotech_hot.py --pos {pos_fa} --neg {neg_fa} -o {sub_pred_dir}", "PromoTech RF-HOT")
    pt_pos = sub_pred_dir / "promotech" / "workdir" / "hot_pg_pos" / "sequences_predictions.csv"
    pt_neg = sub_pred_dir / "promotech" / "workdir" / "hot_pg_neg" / "sequences_predictions.csv"
    if pt_pos.exists() and pt_neg.exists():
        p_sc = pd.read_csv(pt_pos, sep="\t")["PRED"].values
        n_sc = pd.read_csv(pt_neg, sep="\t")["PRED"].values
        model_scores["PromoTech RF-HOT [RF]"] = np.concatenate([p_sc, n_sc])

    # 4. MLDSPP 0%
    run_cmd(f"pixi run python src/runners/mldspp.py --pos {pos_fa} --neg {neg_fa} -o {sub_pred_dir}", "MLDSPP 0%")
    ml_pos = sub_pred_dir / "mldspp_pos.csv"
    ml_neg = sub_pred_dir / "mldspp_neg.csv"
    if ml_pos.exists() and ml_neg.exists():
        p_sc = pd.read_csv(ml_pos, sep="\t")["PRED"].values
        n_sc = pd.read_csv(ml_neg, sep="\t")["PRED"].values
        model_scores["MLDSPP 0% [BDT]"] = np.concatenate([p_sc, n_sc])

    metrics = []
    for m_name, y_score in model_scores.items():
        if len(y_score) != len(y_true):
            continue
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, th = roc_curve(y_true, y_score)
        best_idx = np.argmax(tpr - fpr)
        best_th = th[best_idx]
        y_pred = (y_score >= best_th).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        sens = ((y_pred == 1) & (y_true == 1)).sum() / n_pos
        spec = ((y_pred == 0) & (y_true == 0)).sum() / n_neg
        mcc = matthews_corrcoef(y_true, y_pred)
        
        print(f"  {m_name:<25} | AUC={auc:.4f} | Acc={acc*100:.2f}% | Sens={sens*100:.2f}% | Spec={spec*100:.2f}% | MCC={mcc:.3f}")
        metrics.append({
            "Subset_Key": sub_key,
            "Subset_Name": sub_name,
            "Total_N": len(y_true),
            "Positives_N": n_pos,
            "Negatives_N": n_neg,
            "Model": m_name,
            "ROC_AUC": round(auc, 4),
            "Accuracy": round(acc * 100, 2),
            "Sensitivity": round(sens * 100, 2),
            "Specificity": round(spec * 100, 2),
            "MCC": round(mcc, 3)
        })
    return pd.DataFrame(metrics)

def main():
    print("=" * 80)
    print("EVALUATING SPECIALIZED NICHES BENCHMARK (CDS & 1:1 ORTHOLOGS)")
    print("=" * 80)
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    
    subsets = [
        ("d39v_cds_internal", "D39V CDS Internal Promoters (Pos vs CDS Neg)", ROOT_DIR / "data/benchmark_cds/d39v_cds_internal"),
        ("tigr4_cds_high_conf", "TIGR4 CDS High Confidence Promoters (Pos vs CDS Neg)", ROOT_DIR / "data/benchmark_cds/tigr4_cds_high_conf"),
        ("tigr4_cds_all", "TIGR4 CDS All Promoters (Pos vs CDS Neg)", ROOT_DIR / "data/benchmark_cds/tigr4_cds_all"),
        ("d39v_ortho_1to1_siga", "D39V 1:1 Ortholog IGR SigA Promoters (Pos vs Ortho IGR Neg)", ROOT_DIR / "data/benchmark_ortho_1to1/d39v_ortho_1to1_siga"),
        ("tigr4_ortho_1to1_high_conf", "TIGR4 1:1 Ortholog IGR High Conf Promoters (Pos vs Ortho IGR Neg)", ROOT_DIR / "data/benchmark_ortho_1to1/tigr4_ortho_1to1_high_conf"),
    ]
    
    all_dfs = []
    for sub_key, sub_name, sub_path in subsets:
        pos_fa = sub_path / "positives_81bp.fasta"
        neg_fa = sub_path / "negatives_81bp.fasta"
        if pos_fa.exists() and neg_fa.exists():
            df_res = evaluate_subset(sub_key, sub_name, pos_fa, neg_fa)
            all_dfs.append(df_res)
            
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        tsv_path = OUT_TABLES / "specialized_niches_benchmark_metrics.tsv"
        master_df.to_csv(tsv_path, sep="\t", index=False)
        print(f"\n[SAVED] Consolidated Specialized Metrics -> {tsv_path}")
        
        # Plot multi-panel ROC
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
        axes = axes.flatten()
        palette = {'iPro-MP [gLM]': '#8A2BE2', 'PromoTech RF-HOT [RF]': '#2A9D8F', 'PromoterLCNN [CNN]': '#E63946', 'MLDSPP 0% [BDT]': '#457B9D'}
        
        for idx, (sub_key, sub_name, sub_path) in enumerate(subsets):
            ax = axes[idx]
            ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5)
            s_dir = OUT_PREDS / sub_key
            pos_recs = list(SeqIO.parse(sub_path / "positives_81bp.fasta", "fasta"))
            neg_recs = list(SeqIO.parse(sub_path / "negatives_81bp.fasta", "fasta"))
            y_t = np.array([1]*len(pos_recs) + [0]*len(neg_recs))
            
            # LCNN
            if (s_dir / "lcnn" / "lcnn_pos.csv").exists():
                p = pd.read_csv(s_dir / "lcnn" / "lcnn_pos.csv", sep="\t")["PRED"].values
                n = pd.read_csv(s_dir / "lcnn" / "lcnn_neg.csv", sep="\t")["PRED"].values
                ys = np.concatenate([p, n])
                auc = roc_auc_score(y_t, ys)
                fpr, tpr, _ = roc_curve(y_t, ys)
                ax.plot(fpr, tpr, color=palette['PromoterLCNN [CNN]'], lw=2, label=f"LCNN (AUC={auc:.3f})")
                
            # iPro-MP
            if (s_dir / "ipromp" / "ipromp_12_predictions.csv").exists():
                ys = pd.read_csv(s_dir / "ipromp" / "ipromp_12_predictions.csv")["PRED"].values
                auc = roc_auc_score(y_t, ys)
                fpr, tpr, _ = roc_curve(y_t, ys)
                ax.plot(fpr, tpr, color=palette['iPro-MP [gLM]'], lw=2, label=f"iPro-MP (AUC={auc:.3f})")

            # PromoTech
            pt_p = s_dir / "promotech" / "workdir" / "hot_pg_pos" / "sequences_predictions.csv"
            pt_n = s_dir / "promotech" / "workdir" / "hot_pg_neg" / "sequences_predictions.csv"
            if pt_p.exists() and pt_n.exists():
                p = pd.read_csv(pt_p, sep="\t")["PRED"].values
                n = pd.read_csv(pt_n, sep="\t")["PRED"].values
                ys = np.concatenate([p, n])
                auc = roc_auc_score(y_t, ys)
                fpr, tpr, _ = roc_curve(y_t, ys)
                ax.plot(fpr, tpr, color=palette['PromoTech RF-HOT [RF]'], lw=2, label=f"PromoTech (AUC={auc:.3f})")

            # MLDSPP
            if (s_dir / "mldspp_pos.csv").exists():
                p = pd.read_csv(s_dir / "mldspp_pos.csv", sep="\t")["PRED"].values
                n = pd.read_csv(s_dir / "mldspp_neg.csv", sep="\t")["PRED"].values
                ys = np.concatenate([p, n])
                auc = roc_auc_score(y_t, ys)
                fpr, tpr, _ = roc_curve(y_t, ys)
                ax.plot(fpr, tpr, color=palette['MLDSPP 0% [BDT]'], lw=2, label=f"MLDSPP (AUC={auc:.3f})")

            ax.set_title(f"{sub_name}\n(N={len(y_t)}: {len(pos_recs)} Pos / {len(neg_recs)} Neg)", fontsize=10, fontweight="bold")
            ax.set_xlabel("FPR", fontsize=9)
            ax.set_ylabel("TPR", fontsize=9)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)

        axes[5].axis("off")
        plt.suptitle("Specialized Niches Benchmark: Internal Promoters in CDS & Conserved 1:1 Orthologous IGRs", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        plot_f = OUT_PLOTS / "roc_specialized_niches_master.png"
        plt.savefig(plot_f, dpi=300)
        plt.close()
        print(f"[SAVED] Master ROC Plot -> {plot_f}")

if __name__ == "__main__":
    main()
