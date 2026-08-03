#!/usr/bin/env python3
"""
Zero-Shot Evaluation & Cross-Strain Reproducibility of PromoterLCNN.

Evaluates the pre-trained PromoterLCNN model (IsPromoter_fold_5) on both:
1. D39V Benchmark Dataset (988 Positives vs 1,000 Negatives).
2. TIGR4 High Confidence Primary Dataset (738 Positives vs 738 Negatives).
3. TIGR4 Extended Primary Dataset (2,000 Positives vs 2,000 Negatives).

Computes ROC-AUC, PR-AUC, Accuracy, Sensitivity (TPR), Specificity (1-FPR),
Precision, F1-Score, and Matthews Correlation Coefficient (MCC).

Usage:
    pixi run --manifest-path tools/Promoters/pixi.toml python src/experiments/evaluate_promoter_lcnn_zero_shot.py
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Force TensorFlow CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5"
PLOTS_DIR = ROOT / "output/plots/promoter_lcnn"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "D39V (Cappable-seq Primary)": {
        "pos": ROOT / "data/benchmark/positives_81bp.fasta",
        "neg": ROOT / "data/benchmark/negatives_81bp.fasta",
        "color": "#1f77b4",
    },
    "TIGR4 (High Conf Primary)": {
        "pos": ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.fasta",
        "neg": ROOT / "output/tigr4_data/negatives_tigr4_high_conf_primary_81bp.fasta",
        "color": "#2ca02c",
    },
    "TIGR4 (Extended Primary)": {
        "pos": ROOT / "output/tigr4_data/positives_tigr4_extended_primary_81bp.fasta",
        "neg": ROOT / "output/tigr4_data/negatives_tigr4_extended_primary_81bp.fasta",
        "color": "#ff7f0e",
    },
}


def one_hot_encode_sequences(sequences: List[str]) -> np.ndarray:
    encoding_map = {
        "A": [1, 0, 0, 0],
        "T": [0, 1, 0, 0],
        "C": [0, 0, 1, 0],
        "G": [0, 0, 0, 1],
    }
    encoded_list = []
    for seq in sequences:
        seq_clean = seq.upper()
        if len(seq_clean) != 81:
            raise ValueError(f"Sequence length must be 81 bp, got {len(seq_clean)} bp")
        matrix = [encoding_map.get(base, [0.25, 0.25, 0.25, 0.25]) for base in seq_clean]
        encoded_list.append(matrix)

    return np.array(encoded_list, dtype=np.float32)


def load_fasta_sequences(fasta_path: Path) -> List[str]:
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found at {fasta_path}")
    seqs = []
    curr = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr:
                    s = "".join(curr).upper()
                    if len(s) == 81:
                        seqs.append(s)
                    curr = []
            else:
                curr.append(line)
        if curr:
            s = "".join(curr).upper()
            if len(s) == 81:
                seqs.append(s)
    return seqs


def compute_roc_curve_and_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes exact ROC curve points (FPR, TPR) and trapezoidal AUC using pure NumPy."""
    desc_indices = np.argsort(-y_probs)
    y_true_sorted = y_true[desc_indices]
    y_probs_sorted = y_probs[desc_indices]

    distinct_value_indices = np.where(np.diff(y_probs_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    tpr = tps / n_pos if n_pos > 0 else np.zeros_like(tps)
    fpr = fps / n_neg if n_neg > 0 else np.zeros_like(fps)

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]

    roc_auc = np.trapz(tpr, fpr)
    return fpr, tpr, float(roc_auc)


def compute_pr_curve_and_auc(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes exact Precision-Recall curve and PR-AUC (Average Precision) using pure NumPy."""
    desc_indices = np.argsort(-y_probs)
    y_true_sorted = y_true[desc_indices]
    y_probs_sorted = y_probs[desc_indices]

    distinct_value_indices = np.where(np.diff(y_probs_sorted))[0]
    threshold_indices = np.r_[distinct_value_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    n_pos = y_true.sum()

    precision = tps / (tps + fps)
    recall = tps / n_pos if n_pos > 0 else np.zeros_like(tps)

    precision = np.r_[1, precision]
    recall = np.r_[0, recall]

    pr_auc = np.trapz(precision, recall)
    return precision, recall, float(pr_auc)


def evaluate_dataset(
    model: tf.keras.Model, pos_path: Path, neg_path: Path
) -> Dict:
    pos_seqs = load_fasta_sequences(pos_path)
    neg_seqs = load_fasta_sequences(neg_path)

    n_pos = len(pos_seqs)
    n_neg = len(neg_seqs)

    all_seqs = pos_seqs + neg_seqs
    y_true = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)

    X_encoded = one_hot_encode_sequences(all_seqs)

    raw_preds = model.predict(X_encoded, batch_size=64, verbose=0)
    y_probs = raw_preds[:, 1]  # Probability of Promoter class

    fpr, tpr, roc_auc = compute_roc_curve_and_auc(y_true, y_probs)
    precision_curve, recall_curve, pr_auc = compute_pr_curve_and_auc(y_true, y_probs)

    # Youden's J statistic for optimal threshold selection (J = TPR - FPR)
    j_scores = tpr - fpr
    opt_idx = np.argmax(j_scores)
    # Estimate threshold
    desc_indices = np.argsort(-y_probs)
    y_probs_sorted = y_probs[desc_indices]
    opt_thresh = float(y_probs_sorted[min(opt_idx, len(y_probs_sorted) - 1)])

    y_pred_opt = (y_probs >= opt_thresh).astype(int)

    tp = np.sum((y_true == 1) & (y_pred_opt == 1))
    tn = np.sum((y_true == 0) & (y_pred_opt == 0))
    fp = np.sum((y_true == 0) & (y_pred_opt == 1))
    fn = np.sum((y_true == 1) & (y_pred_opt == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Matthews Correlation Coefficient (MCC)
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num / den if den > 0 else 0.0

    return {
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "total_eval": n_pos + n_neg,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "opt_threshold": opt_thresh,
        "accuracy_opt": float(accuracy),
        "sensitivity_tpr": float(sensitivity),
        "specificity_tnr": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "mcc": float(mcc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision_curve": precision_curve.tolist(),
        "recall_curve": recall_curve.tolist(),
    }


def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" PROMOTER-LCNN ZERO-SHOT EVALUATION & CROSS-STRAIN BENCHMARK")
    print("═════════════════════════════════════════════════════════════════\n")

    if not MODEL_PATH.exists():
        print(f"[ERROR] PromoterLCNN model weights not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Loading pre-trained PromoterLCNN model weights from {MODEL_PATH.name}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    eval_results = {}
    summary_table_rows = []

    for name, paths in DATASETS.items():
        pos_path = paths["pos"]
        neg_path = paths["neg"]

        if not pos_path.exists() or not neg_path.exists():
            print(f"[SKIP] Datasets missing for {name}: {pos_path}")
            continue

        print(f"[EVALUATING] {name}...")
        res = evaluate_dataset(model, pos_path, neg_path)
        eval_results[name] = res

        summary_table_rows.append({
            "Dataset": name,
            "N_Pos": res["n_positives"],
            "N_Neg": res["n_negatives"],
            "ROC_AUC": res["roc_auc"],
            "PR_AUC": res["pr_auc"],
            "Opt_Thresh": res["opt_threshold"],
            "Accuracy": res["accuracy_opt"],
            "Sensitivity": res["sensitivity_tpr"],
            "Specificity": res["specificity_tnr"],
            "Precision": res["precision"],
            "F1_Score": res["f1_score"],
            "MCC": res["mcc"],
        })

    json_path = PLOTS_DIR / "promoter_lcnn_zero_shot_results.json"
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[SUCCESS] Detailed evaluation results saved ➔ {json_path}")

    # Executive Console Summary Table
    print("\n" + "═" * 95)
    print(" PROMOTER-LCNN ZERO-SHOT COMPARATIVE PERFORMANCE TABLE")
    print("═" * 95)
    print(
        f"{'Dataset Tier / Strain':<30} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Accuracy':<8} | {'Sens(TPR)':<9} | {'Spec(TNR)':<9} | {'F1-Score':<8} | {'MCC':<6}"
    )
    print("─" * 95)
    for r in summary_table_rows:
        print(
            f"{r['Dataset']:<30} | {r['ROC_AUC']:<8.4f} | {r['PR_AUC']:<8.4f} | {r['Accuracy']:<8.4f} | {r['Sensitivity']:<9.4f} | {r['Specificity']:<9.4f} | {r['F1_Score']:<8.4f} | {r['MCC']:<6.4f}"
        )
    print("═" * 95)

    if "D39V (Cappable-seq Primary)" in eval_results and "TIGR4 (High Conf Primary)" in eval_results:
        auc_d39v = eval_results["D39V (Cappable-seq Primary)"]["roc_auc"]
        auc_tigr4 = eval_results["TIGR4 (High Conf Primary)"]["roc_auc"]
        delta_auc = abs(auc_d39v - auc_tigr4)

        print("\nCROSS-STRAIN REPRODUCIBILITY ANALYSIS:")
        print(f" • D39V  Primary High Conf ROC-AUC: {auc_d39v:.4f}")
        print(f" • TIGR4 Primary High Conf ROC-AUC: {auc_tigr4:.4f}")
        print(f" • Inter-Strain Performance Delta:  ΔAUC = {delta_auc:.4f} ({delta_auc * 100:.2f}%)")
        if delta_auc < 0.05:
            print(" • Conclusion: High cross-strain model stability (ΔAUC < 0.05). Reproducibility CONFIRMED.")
        else:
            print(" • Conclusion: Moderate performance divergence across strains.")
    print("═" * 95 + "\n")


if __name__ == "__main__":
    main()
