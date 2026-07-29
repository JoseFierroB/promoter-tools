#!/usr/bin/env python3
"""
Analyze PromoTech predictions: load CSVs from working directory,
compute per-sigma and overall metrics, generate summary TSV and ROC plots.

Usage:
  pixi run python analyze_promotech.py -i output/workdir -o output/eval
"""

import os, sys, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             matthews_corrcoef)
from Bio import SeqIO


def parse_args():
    p = argparse.ArgumentParser(description="Analyze PromoTech predictions.")
    p.add_argument('-i', '--input-dir', default="output/predictions/promotech/workdir",
                   help="Working directory with predictions")
    p.add_argument('-o', '--output-dir', default="output/plots/promotech",
                   help="Output directory for TSV and plots")
    p.add_argument('-p', '--positives', default="data/benchmark/positives_81bp.fasta",
                   help="Positive FASTA (needed for sigma info)")
    p.add_argument('-n', '--negatives', default="data/benchmark/negatives_81bp.fasta",
                   help="Negative FASTA")
    p.add_argument('--best-offset-hot', type=int, default=21,
                   help="Best single offset for HOT aggregation")
    p.add_argument('--best-offset-tetra', type=int, default=23,
                   help="Best single offset for TETRA aggregation")
    return p.parse_args()


def extract_sigma(seq_id):
    m = re.search(r'Sig(\w+)', seq_id)
    return f"Sig{m.group(1)}" if m else "SigUnknown"


def load_pt_matrix(csv_path, ids, window_max=42):
    id2i = {s: i for i, s in enumerate(ids)}
    matrix = np.zeros((len(ids), window_max))
    if not csv_path.exists():
        return matrix
    df = pd.read_csv(csv_path, sep='\t')
    for _, row in df.iterrows():
        chrom = str(row['chrom'])
        strand = str(row['strand'])
        if strand != '+':
            continue
        base_id = chrom.rsplit('_offset_', 1)[0]
        try:
            offset = int(chrom.rsplit('_offset_', 1)[1])
        except (IndexError, ValueError):
            continue
        if base_id in id2i and offset < window_max:
            matrix[id2i[base_id], offset] = float(row['score'])
    return matrix





def compute_metrics(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    opt_thresh = thresholds[opt_idx]
    y_pred = (y_scores >= opt_thresh).astype(int)
    return {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "Opt_Threshold": round(float(opt_thresh), 4),
        "Sensitivity": round(float(tpr[opt_idx]), 4),
        "Specificity": round(float(1 - fpr[opt_idx]), 4),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }


def main():
    args = parse_args()
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_records = list(SeqIO.parse(args.positives, "fasta"))
    neg_records = list(SeqIO.parse(args.negatives, "fasta"))
    pos_ids = [r.id for r in pos_records]
    neg_ids = [r.id for r in neg_records]
    all_ids = pos_ids + neg_ids
    y_true = np.array([1] * len(pos_ids) + [0] * len(neg_ids))

    # Sigma factor labels
    pos_sigma = [extract_sigma(i) for i in pos_ids]
    all_sigma = pos_sigma + ["Neg"] * len(neg_ids)

    # ── Define prediction sets ──
    pred_sets = {}

    # Direct mode
    for tag, model in [("hot", "RF-HOT"), ("tetra", "RF-TETRA")]:
        for label in ["pos", "neg"]:
            csv = in_dir / f"{tag}_s_{label}" / "sequences_predictions.csv"
            ids = pos_ids if label == "pos" else neg_ids
            if csv.exists():
                df = pd.read_csv(csv, sep='\t').set_index("CHROM")
                pred_sets[f"{tag}_s_{label}"] = df.loc[ids, "PRED"].values

    if "hot_s_pos" in pred_sets:
        pred_sets["Direct 40bp RF-HOT"] = np.concatenate(
            [pred_sets["hot_s_pos"], pred_sets["hot_s_neg"]])
        pred_sets["Direct 40bp RF-TETRA"] = np.concatenate(
            [pred_sets["tetra_s_pos"], pred_sets["tetra_s_neg"]])

    # PG + aggregation (no Gaussian — best offset + raw max/mean)
    for tag, model, best_off in [("hot", "RF-HOT", args.best_offset_hot),
                                  ("tetra", "RF-TETRA", args.best_offset_tetra)]:
        pos_mat = load_pt_matrix(in_dir / f"{tag}_pg_pos" / "genome_predictions.csv", pos_ids)
        neg_mat = load_pt_matrix(in_dir / f"{tag}_pg_neg" / "genome_predictions.csv", neg_ids)
        if pos_mat.sum() > 0 or neg_mat.sum() > 0:
            best_pos = np.concatenate([pos_mat[:, best_off], neg_mat[:, best_off]])
            raw_max = np.concatenate([np.max(pos_mat, axis=1),
                                      np.max(neg_mat, axis=1)])
            raw_mean = np.concatenate([np.mean(pos_mat, axis=1),
                                       np.mean(neg_mat, axis=1)])
            pred_sets[f"SW BestOffset{best_off} {tag.upper()}"] = best_pos
            pred_sets[f"SW RawMax {tag.upper()}"] = raw_max
            pred_sets[f"SW RawMean {tag.upper()}"] = raw_mean



    # ══════════════════════════════════════════════════════════════
    # COMPUTE METRICS
    # ══════════════════════════════════════════════════════════════
    rows = []
    for name, y_scores in sorted(pred_sets.items()):
        if len(y_scores) != len(y_true):
            continue
        m = compute_metrics(y_true, y_scores)
        m["Model"] = name
        rows.append(m)

    df_all = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    out_tsv = out_dir / "promotech_summary.tsv"
    df_all.to_csv(out_tsv, sep='\t', index=False)
    print(f"\nSummary: {out_tsv}")
    print(df_all.to_string(index=False))

    # Per-sigma metrics (each sigma vs all negatives)
    sigma_groups = sorted(set(s for s in all_sigma if s != "Neg"))
    neg_mask = np.array([s == "Neg" for s in all_sigma])
    rows_sigma = []
    for name, y_scores in sorted(pred_sets.items()):
        if len(y_scores) != len(all_sigma):
            continue
        for sg in sigma_groups:
            pos_mask = np.array([s == sg for s in all_sigma])
            if pos_mask.sum() < 5:
                continue
            mask = pos_mask | neg_mask
            m = compute_metrics(y_true[mask], y_scores[mask])
            m["Model"] = name
            m["Sigma"] = sg
            m["Pos_N"] = int(pos_mask.sum())
            rows_sigma.append(m)

    if rows_sigma:
        df_sigma = pd.DataFrame(rows_sigma).sort_values(["Sigma", "ROC_AUC"], ascending=[True, False])
        out_sigma = out_dir / "promotech_summary_per_sigma.tsv"
        df_sigma.to_csv(out_sigma, sep='\t', index=False)
        print(f"\nPer-sigma: {out_sigma}")
        print(df_sigma.to_string(index=False))

    # ══════════════════════════════════════════════════════════════
    # ROC PLOT
    # ══════════════════════════════════════════════════════════════
    plot_keys = [k for k in pred_sets if any(x in k for x in
                  ["Direct 40bp RF", "SW BestOffset", "SW RawMax"])]
    colors = {"Direct 40bp RF-HOT": "#002D62",
              "Direct 40bp RF-TETRA": "#7E2F8E",
              "SW BestOffset21 HOT": "#D95319",
              "SW BestOffset23 TETRA": "#00A087",
              "SW RawMax HOT": "#D95319",
              "SW RawMax TETRA": "#00A087"}
    styles = {"Direct 40bp RF-HOT": "-",
              "Direct 40bp RF-TETRA": "-",
              "SW BestOffset21 HOT": "--",
              "SW BestOffset23 TETRA": "--",
              "SW RawMax HOT": ":",
              "SW RawMax TETRA": ":"}

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    ax.plot([0, 1], [0, 1], '--', color='#AAAAAA', lw=0.8)
    for key in plot_keys:
        if key not in pred_sets:
            continue
        fpr, tpr, _ = roc_curve(y_true, pred_sets[key])
        roc_auc = auc(fpr, tpr)
        c = colors.get(key, "#888888")
        ls = styles.get(key, "-")
        label = key.replace("SW BestOffset21 ", "BestOff21 ").replace("SW BestOffset23 ", "BestOff23 ").replace("SW RawMax", "RawMax")
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})",
                color=c, ls=ls, lw=1.3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("PromoTech Benchmark")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    plt.tight_layout()
    plot_p = out_dir / "promotech_roc.png"
    fig.savefig(plot_p, bbox_inches="tight")
    plt.close()
    print(f"ROC plot: {plot_p}")


if __name__ == "__main__":
    main()
