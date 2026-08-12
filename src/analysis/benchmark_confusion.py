#!/usr/bin/env python3
"""
Confusion matrix analysis for all benchmark tools.
Computes optimal threshold (Youden's J), TP/FP/TN/FN, and derived metrics.

Output: output/tables/benchmark_confusion.tsv
"""

import csv
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "output" / "tables" / "benchmark_confusion.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_fasta_ids(fasta_path):
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def load_simple_scores(csv_path):
    scores = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                scores.append(float(row[0]))
    return scores


def load_ipromp_scores(csv_path, n_pos=988, n_neg=1000):
    pos_scores = []
    neg_scores = []
    with open(csv_path) as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) >= 3:
                score = float(parts[2])
            elif len(parts) >= 2:
                score = float(parts[1])
            else:
                score = float(parts[0])
            if i < n_pos:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
    return pos_scores[:n_pos], neg_scores[:n_neg]


def confusion_at_best(pos_scores, neg_scores):
    """Compute confusion matrix at optimal threshold (Youden's J)."""
    y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    y_score = np.array(pos_scores + list(neg_scores))

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    y_pred = (y_score >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "AUC": auc, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Sensitivity": sens, "Specificity": spec,
        "Precision": prec, "F1": f1, "Accuracy": acc,
        "Threshold": best_thresh,
    }


def process_strain(strain, pos_fasta, neg_fasta, pred_dir, tool_list):
    """Process all tools for one strain."""
    pos_ids = load_fasta_ids(pos_fasta)
    neg_ids = load_fasta_ids(neg_fasta)
    n_pos, n_neg = len(pos_ids), len(neg_ids)

    results = []
    for name, pos_p, neg_p, ttype in tool_list:
        if not Path(pos_p).exists():
            print(f"  SKIP {name}: {pos_p} not found")
            continue
        if ttype != "ipromp" and not Path(neg_p).exists():
            print(f"  SKIP {name}: {neg_p} not found")
            continue

        print(f"  {name}...")
        try:
            if ttype == "ipromp":
                pos_s, neg_s = load_ipromp_scores(pos_p, n_pos, n_neg)
            else:
                pos_s = load_simple_scores(pos_p)
                neg_s = load_simple_scores(neg_p)

            c = confusion_at_best(pos_s, neg_s)
            c["tool"] = name
            c["strain"] = strain
            c["n_positives"] = len(pos_s)
            c["n_negatives"] = len(neg_s)
            results.append(c)
        except Exception as e:
            print(f"    ERROR: {e}")

    return results


def main():
    # ---- D39V tools ----
    d39v_tools = [
        ("meme",          "output/predictions/meme_pos.csv",        "output/predictions/meme_neg.csv",        "simple"),
        ("fimo_db",       "output/predictions/fimo_db_pos.csv",     "output/predictions/fimo_db_neg.csv",     "simple"),
        ("fimo_prok",     "output/predictions/fimo_prok_pos.csv",   "output/predictions/fimo_prok_neg.csv",   "simple"),
        ("mldspp",        "output/predictions/mldspp_pos.csv",      "output/predictions/mldspp_neg.csv",      "simple"),
        ("mldspp_75",     "output/predictions/mldspp_75spn_pos.csv","output/predictions/mldspp_75spn_neg.csv","simple"),
        ("lcnn",          "output/predictions/lcnn/lcnn_pos.csv",   "output/predictions/lcnn/lcnn_neg.csv",   "simple"),
        ("promotech_hot",  "output/predictions/promotech/workdir/hot_pg_pos/sequences_predictions.csv",
                           "output/predictions/promotech/workdir/hot_pg_neg/sequences_predictions.csv", "simple"),
        ("promotech_tetra","output/predictions/promotech/workdir/tetra_pg_pos/sequences_predictions.csv",
                           "output/predictions/promotech/workdir/tetra_pg_neg/sequences_predictions.csv", "simple"),
        ("ipromp_sp12",   "output/predictions/ipromp/ipromp_12_predictions.csv", "", "ipromp"),
    ]

    # ---- TIGR4 tools ----
    tigr4_tools = [
        ("meme",          "output/tigr4/predictions/meme_pos.csv",        "output/tigr4/predictions/meme_neg.csv",        "simple"),
        ("fimo_db",       "output/tigr4/predictions/fimo_db_pos.csv",     "output/tigr4/predictions/fimo_db_neg.csv",     "simple"),
        ("fimo_prok",     "output/tigr4/predictions/fimo_prok_pos.csv",   "output/tigr4/predictions/fimo_prok_neg.csv",   "simple"),
        ("mldspp",        "output/tigr4/predictions/mldspp_pos.csv",      "output/tigr4/predictions/mldspp_neg.csv",      "simple"),
        ("mldspp_75",     "output/tigr4/predictions/mldspp_75spn_pos.csv","output/tigr4/predictions/mldspp_75spn_neg.csv","simple"),
        ("lcnn",          "output/tigr4/predictions/lcnn/lcnn_pos.csv",   "output/tigr4/predictions/lcnn/lcnn_neg.csv",   "simple"),
        ("promotech_hot",  "output/tigr4/predictions/promotech/workdir/hot_pg_pos/sequences_predictions.csv",
                           "output/tigr4/predictions/promotech/workdir/hot_pg_neg/sequences_predictions.csv", "simple"),
        ("promotech_tetra","output/tigr4/predictions/promotech/workdir/tetra_pg_pos/sequences_predictions.csv",
                           "output/tigr4/predictions/promotech/workdir/tetra_pg_neg/sequences_predictions.csv", "simple"),
        ("ipromp_sp12",   "output/tigr4/predictions/ipromp_sp12_predictions.csv", "", "ipromp"),
    ]

    all_results = []

    print("=== D39V ===")
    d39v_r = process_strain("D39V",
                             ROOT / "data/benchmark/d39v/positives_81bp.fasta",
                              ROOT / "data/benchmark/d39v/negatives_81bp.fasta",
                             ROOT / "output/predictions",
                             d39v_tools)
    all_results.extend(d39v_r)

    print("\n=== TIGR4 ===")
    tigr4_r = process_strain("TIGR4",
                              ROOT / "data/tigr4/positives_high_81bp.fasta",
                              ROOT / "data/tigr4/negatives_high_81bp.fasta",
                              ROOT / "output/tigr4/predictions",
                              tigr4_tools)
    all_results.extend(tigr4_r)

    # Write
    if all_results:
        fields = ["tool", "strain", "n_positives", "n_negatives",
                  "AUC", "TP", "FN", "FP", "TN",
                  "Sensitivity", "Specificity", "Precision", "F1", "Accuracy", "Threshold"]
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(all_results)

        # Print
        print(f"\n{'='*110}")
        print("  CONFUSION MATRIX — ALL TOOLS AT OPTIMAL THRESHOLD (Youden's J)")
        print(f"{'='*110}")
        print(f"{'Tool':<20} {'Strain':>6} {'AUC':>7} {'TP':>5} {'FN':>5} {'FP':>5} {'TN':>5} {'Sens':>7} {'Spec':>7} {'Prec':>7} {'F1':>7} {'Acc':>7}")
        print("-" * 100)
        for r in all_results:
            print(f"{r['tool']:<20} {r['strain']:>6} {r['AUC']:7.4f} {r['TP']:5d} {r['FN']:5d} "
                  f"{r['FP']:5d} {r['TN']:5d} "
                  f"{r['Sensitivity']:6.3f} {r['Specificity']:6.3f} "
                  f"{r['Precision']:6.3f} {r['F1']:6.3f} {r['Accuracy']:6.3f}")

        # Summary: best FP/FN balance
        print(f"\n  === FP (falsos positivos) ===")
        print(f"  {'Tool':<20} {'D39V FP':>8} {'D39V %':>8} {'TIGR4 FP':>10} {'TIGR4 %':>8}")
        d39v_map = {r["tool"]: r for r in d39v_r}
        tigr4_map = {r["tool"]: r for r in tigr4_r}
        for tool_name in sorted(set(list(d39v_map.keys()) + list(tigr4_map.keys()))):
            d = d39v_map.get(tool_name, {})
            t = tigr4_map.get(tool_name, {})
            d_fp = d.get("FP", 0)
            d_pct = d_fp / (d.get("FP", 0) + d.get("TN", 1)) * 100 if d else 0
            t_fp = t.get("FP", 0)
            t_pct = t_fp / (t.get("FP", 0) + t.get("TN", 1)) * 100 if t else 0
            if d or t:
                print(f"  {tool_name:<20} {d_fp:>8} {d_pct:>7.1f}% {t_fp:>10} {t_pct:>7.1f}%")

        print(f"\n  Output: {OUT}")


if __name__ == "__main__":
    main()
