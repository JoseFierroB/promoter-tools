#!/usr/bin/env python3
"""
Process and Plot Master Benchmark Results for IGR Dataset vs CDS Dataset.

Audits:
  - Model AUCs on IGR Benchmark (Promoters in IGR vs Intergenic Non-Promoter Controls)
  - Delta AUC (IGR Background vs CDS Background)
  - Publication-ready ROC curves in PNG, SVG, PDF
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from Bio import SeqIO

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runners._shared import extract_aligned


def load_igr_predictions(pr_dir, n_pos=723, n_neg=723):
    y_true = np.hstack([np.ones(n_pos), np.zeros(n_neg)])
    models = {}

    # 1. iPro-MP
    ipromp_file = pr_dir / "ipromp" / "ipromp_12_predictions.csv"
    if ipromp_file.exists():
        df = pd.read_csv(ipromp_file, sep="\t" if "\t" in ipromp_file.read_text()[:200] else ",")
        col = "Probability" if "Probability" in df.columns else "PRED"
        models["iPro-MP [gLM]"] = df[col].values[:len(y_true)]

    # 2. PromoterLCNN
    lcnn_pos = pr_dir / "lcnn" / "lcnn_pos.csv"
    lcnn_neg = pr_dir / "lcnn" / "lcnn_neg.csv"
    if lcnn_pos.exists() and lcnn_neg.exists():
        p = pd.read_csv(lcnn_pos, sep="\t")["PRED"].values
        n = pd.read_csv(lcnn_neg, sep="\t")["PRED"].values
        models["PromoterLCNN [CNN]"] = np.hstack([p, n])

    # 3. PromoTech RF-HOT
    hot_pos = pr_dir / "promotech" / "workdir" / "hot_pg_pos" / "sequences_predictions.csv"
    hot_neg = pr_dir / "promotech" / "workdir" / "hot_pg_neg" / "sequences_predictions.csv"
    if hot_pos.exists() and hot_neg.exists():
        p = pd.read_csv(hot_pos, sep="\t")["PRED"].values
        n = pd.read_csv(hot_neg, sep="\t")["PRED"].values
        models["PromoTech RF-HOT [RF]"] = np.hstack([p, n])

    # 4. PromoTech RF-TETRA
    tetra_pos = pr_dir / "promotech" / "workdir" / "tetra_pg_pos" / "sequences_predictions.csv"
    tetra_neg = pr_dir / "promotech" / "workdir" / "tetra_pg_neg" / "sequences_predictions.csv"
    if tetra_pos.exists() and tetra_neg.exists():
        p = pd.read_csv(tetra_pos, sep="\t")["PRED"].values
        n = pd.read_csv(tetra_neg, sep="\t")["PRED"].values
        models["PromoTech RF-TETRA [RF]"] = np.hstack([p, n])

    # 5. MLDSPP 0% Zero-Shot
    mld_pos = pr_dir / "mldspp_pos.csv"
    mld_neg = pr_dir / "mldspp_neg.csv"
    if mld_pos.exists() and mld_neg.exists():
        p = pd.read_csv(mld_pos, sep="\t")["PRED"].values
        n = pd.read_csv(mld_neg, sep="\t")["PRED"].values
        models["MLDSPP 0% [BDT]"] = np.hstack([p, n])

    # 6. MLDSPP 75%* (5-Fold Stratified CV on IGR)
    pos_fasta = ROOT / "data/benchmark_igr/d39v/positives_81bp_igr.fasta"
    neg_fasta = ROOT / "data/benchmark_igr/d39v/negatives_81bp_igr.fasta"
    pos_recs = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_recs = list(SeqIO.parse(neg_fasta, "fasta"))
    all_recs = pos_recs + neg_recs

    X = np.array([extract_aligned(str(r.seq)[:80]) for r in all_recs])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mld_75_preds = np.zeros(len(y_true))
    for train_idx, test_idx in skf.split(X, y_true):
        clf = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric="logloss")
        clf.fit(X[train_idx], y_true[train_idx])
        mld_75_preds[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    models["MLDSPP 75%* [BDT]"] = mld_75_preds

    # 7. FIMO + Prokaryote DB
    fimo_pos = pr_dir / "fimo_prok_pos.csv"
    fimo_neg = pr_dir / "fimo_prok_neg.csv"
    if fimo_pos.exists() and fimo_neg.exists():
        p = pd.read_csv(fimo_pos, sep="\t")["PRED"].values
        n = pd.read_csv(fimo_neg, sep="\t")["PRED"].values
        models["FIMO + Prokaryote DB [PWM]"] = np.hstack([p, n])

    # 8. MEME Suite
    meme_pos = pr_dir / "meme_pos.csv"
    meme_neg = pr_dir / "meme_neg.csv"
    if meme_pos.exists() and meme_neg.exists():
        p = pd.read_csv(meme_pos, sep="\t")["PRED"].values
        n = pd.read_csv(meme_neg, sep="\t")["PRED"].values
        models["MEME Suite (STREME) [De novo]"] = np.hstack([p, n])

    return y_true, models


def main():
    pr_dir = ROOT / "output/predictions_igr/d39v"
    plots_dir = ROOT / "output/plots/igr_benchmark"
    tables_dir = ROOT / "output/tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    y_true, models = load_igr_predictions(pr_dir, n_pos=723, n_neg=723)

    # Baseline D39V CDS AUCs for comparison
    cds_aucs = {
        "iPro-MP [gLM]": 0.955,
        "MLDSPP 75%* [BDT]": 0.957,
        "PromoterLCNN [CNN]": 0.949,
        "PromoTech RF-HOT [RF]": 0.943,
        "PromoTech RF-TETRA [RF]": 0.917,
        "MLDSPP 0% [BDT]": 0.865,
        "MEME Suite (STREME) [De novo]": 0.862,
        "FIMO + Prokaryote DB [PWM]": 0.759,
    }

    metrics = []
    print("=== D39V IGR Benchmark Performance (N = 1,446: 723 Pos / 723 Neg) ===")

    for name, s in models.items():
        roc_auc = roc_auc_score(y_true, s)
        fpr, tpr, thresholds = roc_curve(y_true, s)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (s >= best_thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acc = (tp + tn) / len(y_true)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        cds_auc = cds_aucs.get(name, np.nan)
        delta_auc = roc_auc - cds_auc if not np.isnan(cds_auc) else np.nan

        metrics.append({
            "Model": name,
            "AUC_IGR": round(float(roc_auc), 3),
            "AUC_CDS": round(float(cds_auc), 3),
            "Delta_AUC": round(float(delta_auc), 3),
            "Accuracy": round(float(acc), 3),
            "Sensitivity": round(float(sens), 3),
            "Specificity": round(float(spec), 3),
            "MCC": round(float(mcc), 3),
            "Optimal_Threshold": round(float(best_thresh), 3)
        })
        print(f"{name:<33} | IGR AUC: {roc_auc:.3f} | CDS AUC: {cds_auc:.3f} (Δ = {delta_auc:+.3f}) | MCC: {mcc:.3f}")

    df_metrics = pd.DataFrame(metrics).sort_values("AUC_IGR", ascending=False)
    metrics_tsv = tables_dir / "igr_vs_cds_benchmark_metrics.tsv"
    df_metrics.to_csv(metrics_tsv, sep="\t", index=False)
    print(f"\nSaved metrics comparison table to: {metrics_tsv}")

    # Colors and styles
    colors = {
        "iPro-MP [gLM]": "#2B5C8F",
        "PromoterLCNN [CNN]": "#E05A47",
        "PromoTech RF-HOT [RF]": "#2BA84A",
        "PromoTech RF-TETRA [RF]": "#3A86C8",
        "MLDSPP 0% [BDT]": "#942C76",
        "MLDSPP 75%* [BDT]": "#942C76",
        "FIMO + Prokaryote DB [PWM]": "#D95F02",
        "MEME Suite (STREME) [De novo]": "#E6AB02"
    }

    # Plot 1: Master ROC Curves for IGR Benchmark
    plt.figure(figsize=(8.5, 7.5), dpi=300)
    for name, s in models.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        roc_auc = roc_auc_score(y_true, s)
        color = colors.get(name, "#333333")
        linestyle = ":" if "0%" in name else "--" if "TETRA" in name else "-." if "PWM" in name else "-"
        label = f"{name} (AUC = {roc_auc:.3f})"
        plt.plot(fpr, tpr, label=label, color=color, linestyle=linestyle, linewidth=2.0)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.2, label="Random Guess (AUC = 0.500)")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves - S. pneumoniae D39V IGR Benchmark\n(Promoters in IGR vs. Non-Promoter IGR Background, N = 1,446)", fontsize=12, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=9.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(plots_dir / "roc_auc_d39v_igr_benchmark.png", dpi=300)
    plt.savefig(plots_dir / "roc_auc_d39v_igr_benchmark.svg")
    plt.savefig(plots_dir / "roc_auc_d39v_igr_benchmark.pdf")
    plt.close()

    # Plot 2: Side-by-Side Bar Chart comparing CDS Benchmark vs IGR Benchmark
    plt.figure(figsize=(10, 6), dpi=300)
    df_sorted = df_metrics.sort_values("AUC_IGR", ascending=True)
    y_pos = np.arange(len(df_sorted))
    bar_width = 0.38

    plt.barh(y_pos + bar_width/2, df_sorted["AUC_CDS"], height=bar_width, label="CDS Controls (N = 1,976)", color="#4A90E2", alpha=0.85, edgecolor="black", linewidth=0.8)
    plt.barh(y_pos - bar_width/2, df_sorted["AUC_IGR"], height=bar_width, label="IGR Controls (N = 1,446)", color="#50E3C2", alpha=0.85, edgecolor="black", linewidth=0.8)

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        plt.text(row["AUC_CDS"] + 0.01, i + bar_width/2, f"{row['AUC_CDS']:.3f}", va="center", fontsize=8.5, fontweight="bold", color="#1B365D")
        plt.text(row["AUC_IGR"] + 0.01, i - bar_width/2, f"{row['AUC_IGR']:.3f}", va="center", fontsize=8.5, fontweight="bold", color="#0E5A44")

    plt.yticks(y_pos, df_sorted["Model"], fontsize=10, fontweight="medium")
    plt.xlabel("ROC-AUC Score", fontsize=12, fontweight="bold")
    plt.xlim([0.45, 1.05])
    plt.title("Promoter Classification Generalizability:\nCDS Background vs. Intergenic (IGR) Background", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(plots_dir / "auc_comparison_cds_vs_igr_benchmark.png", dpi=300)
    plt.savefig(plots_dir / "auc_comparison_cds_vs_igr_benchmark.svg")
    plt.savefig(plots_dir / "auc_comparison_cds_vs_igr_benchmark.pdf")
    plt.close()

    print(f"\nSaved comparison plots to: {plots_dir}")


if __name__ == "__main__":
    main()
