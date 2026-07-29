#!/usr/bin/env python3
"""
MLDSPP Transfer Learning Experiment.
Trains on original MLDSPP data (other organisms), tests on S. pneumoniae (D39V).

Key difference from run_mldspp_cv_predictions.py:
  - Training data: MLDSPP's 10-species promoter dataset (~4,800 seqs)
  - Test data: our S. pneumoniae dataset (never seen during training)
  - No data leakage — honest generalization test.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef

STABILITY_MAP = {
    'AA': -1.00, 'TT': -1.00,
    'AT': -0.88, 'TA': -0.58,
    'AG': -1.30, 'GA': -1.30,
    'AC': -1.45, 'CA': -1.45,
    'TG': -1.44, 'GT': -1.44,
    'TC': -1.28, 'CT': -1.28,
    'CC': -1.84, 'GG': -1.84,
    'CG': -2.24, 'GC': -2.27,
}

MLDSPP_DATA = Path(__file__).resolve().parent.parent.parent / "tools" / "MLDSPP-Promoter-prediction" / "Sample Dataset" / "Promoter Sequences"


def extract_features(seq, target_len=81):
    s = seq[:target_len].upper()
    return np.array([STABILITY_MAP.get(s[i:i+2], -1.35) for i in range(len(s) - 1)])

def extract_aligned(seq):
    """Extract 80bp (-60 to +19) with TSS at position 60.
    For 100bp external (-80/+19, TSS at 80): slice [20:100].
    For 81bp S. pneumoniae (-60/+20, TSS at 60): slice [0:80]."""
    seq = seq.upper()
    seq_len = len(seq)
    if seq_len == 100:
        s = seq[20:100]  # external: -60 to +19
    else:
        s = seq[:80]     # spn 81bp: -60 to +19 (drop +20)
    if len(s) < 80:
        return np.zeros(79)
    return np.array([STABILITY_MAP.get(s[i:i+2], -1.35) for i in range(79)])


def load_training_data(data_dir, target_len=81):
    pos_features, neg_features = [], []
    files = sorted(data_dir.glob("Sequences_80-20_B*.txt"))
    print(f"[TRAIN] Loading {len(files)} promoter files from: {data_dir}")
    rng = np.random.RandomState(42)
    for f in files:
        with open(f) as fh:
            for line in fh:
                seq = line.strip()
                if len(seq) < target_len:
                    continue
                feats = extract_aligned(seq)
                pos_features.append(feats)
                neg_features.append(rng.permutation(feats))
    X = np.vstack(pos_features + neg_features)
    y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
    print(f"[TRAIN] {len(pos_features)} positives + {len(neg_features)} shuffled negatives = {len(X)} total")
    return X, y, len(pos_features)


def load_test_data(pos_fasta, neg_fasta):
    pos_records = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_records = list(SeqIO.parse(neg_fasta, "fasta"))
    pos_ids = [r.id for r in pos_records]
    neg_ids = [r.id for r in neg_records]
    pos_seqs = [str(r.seq) for r in pos_records]
    neg_seqs = [str(r.seq) for r in neg_records]
    X_pos = np.array([extract_aligned(s) for s in pos_seqs])
    X_neg = np.array([extract_aligned(s) for s in neg_seqs])
    X_test = np.vstack([X_pos, X_neg])
    y_test = np.hstack([np.ones(len(pos_ids)), np.zeros(len(neg_ids))])
    chroms = pos_ids + neg_ids
    seqs = pos_seqs + neg_seqs
    print(f"[TEST] {len(pos_ids)} positives + {len(neg_ids)} negatives = {len(X_test)} total")
    return X_test, y_test, chroms, seqs


def compute_metrics(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    youden = tpr - fpr
    opt_idx = np.argmax(youden)
    y_pred = (y_scores >= tpr[int(opt_idx)] if opt_idx < len(tpr) else 0.5).astype(int)
    return {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
    }


def save_predictions(chroms, seqs, scores, n_pos, path):
    df = pd.DataFrame({"CHROM": chroms, "SEQ": seqs, "PRED": scores})
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path, index=False, sep='\t')
    pos_out = path
    neg_out = path.replace("_pos.csv", "_neg.csv").replace("mldspp_transfer_", "mldspp_transfer_")
    if "_pos" in str(path):
        pos_mask = np.array([True] * n_pos + [False] * (len(scores) - n_pos))
        df[pos_mask].to_csv(path, index=False, sep='\t')
        df[~pos_mask].to_csv(neg_out, index=False, sep='\t')
    print(f"  Saved: {path} ({len(df)} rows)")


def main():
    p = argparse.ArgumentParser(description="MLDSPP Transfer Learning Experiment")
    p.add_argument("-p", "--positives", default="data/benchmark/positives_81bp.fasta")
    p.add_argument("-n", "--negatives", default="data/benchmark/negatives_81bp.fasta")
    p.add_argument("-o", "--output-dir", default="output/tables/mldspp_transfer")
    p.add_argument("--train-dir", default=str(MLDSPP_DATA))
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("  MLDSPP Transfer Experiment")
    print("  Train: other organisms  |  Test: S. pneumoniae D39V")
    print("="*60)

    X_train, y_train, n_train_pos = load_training_data(Path(args.train_dir))
    X_test, y_test, chroms, seqs = load_test_data(args.positives, args.negatives)

    results = []

    # ── XGBoost ──
    print("\n[1/3] XGBoost...")
    xgb = XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                        eval_metric='logloss', verbosity=0)
    xgb.fit(X_train, y_train)
    scores = xgb.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, scores)
    m["Model"] = "MLDSPP Transfer (XGBoost)"
    results.append(m)
    save_predictions(chroms, seqs, scores, len(chroms)-len(y_test),
                     str(out / "mldspp_transfer_pos.csv"))
    print(f"  ROC AUC: {m['ROC_AUC']}  PR AUC: {m['PR_AUC']}")

    # ── SVM ──
    print("\n[2/3] SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    scores = svm.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, scores)
    m["Model"] = "MLDSPP Transfer (SVM)"
    results.append(m)
    save_predictions(chroms, seqs, scores, len(chroms)-len(y_test),
                     str(out / "mldspp_transfer_svm_pos.csv"))
    print(f"  ROC AUC: {m['ROC_AUC']}  PR AUC: {m['PR_AUC']}")

    # ── Random Forest ──
    print("\n[3/3] Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    scores = rf.predict_proba(X_test)[:, 1]
    m = compute_metrics(y_test, scores)
    m["Model"] = "MLDSPP Transfer (Random Forest)"
    results.append(m)
    save_predictions(chroms, seqs, scores, len(chroms)-len(y_test),
                     str(out / "mldspp_transfer_rf_pos.csv"))
    print(f"  ROC AUC: {m['ROC_AUC']}  PR AUC: {m['PR_AUC']}")

    # ── Summary ──
    df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    print("\n" + "="*60)
    print("  RESULTS")
    print("="*60)
    print(df.to_string(index=False))

    summary_path = out / "mldspp_transfer_summary.tsv"
    df.to_csv(summary_path, sep='\t', index=False)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
