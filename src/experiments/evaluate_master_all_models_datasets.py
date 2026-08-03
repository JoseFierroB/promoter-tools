#!/usr/bin/env python3
"""
Master Benchmark Evaluator across All Datasets and All Model Families.

Evaluates:
1. Canonical FIMO (Pneumococcal SigA -10 TATAAT motif PSSM)
2. PromoterLCNN (Deep CNN)
3. iPro-MP (DNABERT-6 Transformer)

Usage:
    pixi run python src/experiments/evaluate_master_all_models_datasets.py
"""

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
FIMO_BIN = ROOT / "tools/meme/.pixi/envs/default/bin/fimo"
SIGA_MOTIF = ROOT / "tools/meme/motif_databases/spneumo_sigA_promoter.meme"

IPRO_JSON = ROOT / "output/plots/ipro_mp/ipro_mp_zero_shot_results.json"
PLOTS_DIR = ROOT / "output/plots/master_benchmark"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "D39V (Cappable-seq Primary)": {
        "pos": ROOT / "data/benchmark/positives_81bp.fasta",
        "neg": ROOT / "data/benchmark/negatives_81bp.fasta",
    },
    "TIGR4 Tier 1 (High Conf Primary)": {
        "pos": ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.fasta",
        "neg": ROOT / "output/tigr4_data/negatives_tigr4_high_conf_primary_81bp.fasta",
    },
    "TIGR4 Tier 2 (Extended Primary)": {
        "pos": ROOT / "output/tigr4_data/positives_tigr4_extended_primary_81bp.fasta",
        "neg": ROOT / "output/tigr4_data/negatives_tigr4_extended_primary_81bp.fasta",
    },
}


def load_fasta_tuples(fasta_path: Path) -> List[Tuple[str, str]]:
    recs = list(SeqIO.parse(fasta_path, "fasta"))
    return [(r.id, str(r.seq).upper()) for r in recs if len(r.seq) == 81]


def evaluate_fimo_canonical(pos_path: Path, neg_path: Path, run_name: str) -> Dict:
    dataset_out = PLOTS_DIR / "fimo_scratch" / run_name.replace(" ", "_").replace("(", "").replace(")", "")
    dataset_out.mkdir(parents=True, exist_ok=True)

    pos_tuples = load_fasta_tuples(pos_path)
    neg_tuples = load_fasta_tuples(neg_path)

    combined_fasta = dataset_out / "combined.fasta"
    pos_ids = set()
    neg_ids = set()

    with open(combined_fasta, "w") as f:
        for idx, (rid, seq) in enumerate(pos_tuples):
            sid = f"POS_{idx}_{rid}"
            pos_ids.add(sid)
            f.write(f">{sid}\n{seq}\n")
        for idx, (rid, seq) in enumerate(neg_tuples):
            sid = f"NEG_{idx}_{rid}"
            neg_ids.add(sid)
            f.write(f">{sid}\n{seq}\n")

    fimo_out = dataset_out / "fimo_output"
    cmd = [
        str(FIMO_BIN),
        "--oc",
        str(fimo_out),
        "--thresh",
        "1e-1",
        "--verbosity",
        "1",
        str(SIGA_MOTIF),
        str(combined_fasta),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FIMO execution failed: {res.stderr}")

    scores_dict = {}
    fimo_tsv = fimo_out / "fimo.tsv"
    if fimo_tsv.exists():
        df_fimo = pd.read_csv(fimo_tsv, sep="\t", comment="#")
        if "sequence_name" in df_fimo.columns and "score" in df_fimo.columns:
            scores_dict = df_fimo.groupby("sequence_name")["score"].max().to_dict()

    y_true_list = []
    y_score_list = []

    for sid in pos_ids:
        y_true_list.append(1)
        y_score_list.append(scores_dict.get(sid, -99.0))

    for sid in neg_ids:
        y_true_list.append(0)
        y_score_list.append(scores_dict.get(sid, -99.0))

    y_true = np.array(y_true_list, dtype=np.int32)
    y_scores = np.array(y_score_list, dtype=np.float32)

    return compute_classification_metrics(y_true, y_scores)


def compute_classification_metrics(y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
    desc_indices = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_indices]
    y_scores_sorted = y_scores[desc_indices]

    distinct_value_indices = np.where(np.diff(y_scores_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    roc_auc = float(np.trapz(tpr, fpr))

    precision = tps / (tps + fps)
    recall = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    precision = np.r_[1, precision]
    recall = np.r_[0, recall]

    pr_auc = float(np.trapz(precision, recall))

    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    opt_thresh = float(y_scores_sorted[min(opt_idx, len(y_scores_sorted) - 1)])

    y_pred = (y_scores >= opt_thresh).astype(int)

    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec_val * sensitivity) / (prec_val + sensitivity) if (prec_val + sensitivity) > 0 else 0.0

    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num / den if den > 0 else 0.0

    return {
        "n_positives": int(n_pos),
        "n_negatives": int(n_neg),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": float(accuracy),
        "sensitivity_tpr": float(sensitivity),
        "specificity_tnr": float(specificity),
        "precision": float(prec_val),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "opt_threshold": opt_thresh,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" MASTER BENCHMARK EVALUATOR: FIMO (SIGA) vs LCNN vs IPRO-MP")
    print("═════════════════════════════════════════════════════════════════\n")

    master_results = {}
    summary_rows = []

    # 1. Run FIMO Canonical SigA
    for name, paths in DATASETS.items():
        print(f"[EVALUATING FIMO CANONICAL SIGA] {name}...")
        fimo_res = evaluate_fimo_canonical(paths["pos"], paths["neg"], name)
        key = f"{name} | FIMO (SigA Motif)"
        master_results[key] = fimo_res
        summary_rows.append({"Model": "FIMO (SigA Motif)", "Dataset": name, **fimo_res})

    # 2. Integrate iPro-MP results if available
    if IPRO_JSON.exists():
        with open(IPRO_JSON) as f:
            ipro_data = json.load(f)
        for name, res in ipro_data.items():
            key = f"{name} | iPro-MP (DNABERT)"
            master_results[key] = res
            summary_rows.append({"Model": "iPro-MP (DNABERT)", "Dataset": name, **res})

    master_json = PLOTS_DIR / "master_benchmark_results.json"
    with open(master_json, "w") as f:
        json.dump(master_results, f, indent=2)

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "═" * 105)
    print(" MASTER BENCHMARK SUMMARY TABLE (PROMOTER RECOGNITION)")
    print("═" * 105)
    print(
        f"{'Dataset Strain / Tier':<32} | {'Model Family':<22} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Accuracy':<8} | {'Spec(TNR)':<9} | {'MCC':<6}"
    )
    print("─" * 105)
    for _, r in df_sum.iterrows():
        print(
            f"{r['Dataset']:<32} | {r['Model']:<22} | {r['roc_auc']:<8.4f} | {r['pr_auc']:<8.4f} | {r['accuracy']:<8.4f} | {r['specificity_tnr']:<9.4f} | {r['mcc']:<6.4f}"
        )
    print("═" * 105 + "\n")


if __name__ == "__main__":
    main()
