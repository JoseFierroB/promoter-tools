#!/usr/bin/env python3
"""
MLDSPP cross-species predictions with XGBoost, SVM, and Random Forest.
Trains on 12 external species (MLDSPP paper data), tests on S. pneumoniae.
TSS-aligned 80bp (-60/+19) window for both train and test.

Generates 6 CSVs:
  - mldspp_pos.csv / mldspp_neg.csv (XGBoost)
  - mldspp_svm_pos.csv / mldspp_svm_neg.csv (SVM)
  - mldspp_rf_pos.csv / mldspp_rf_neg.csv (Random Forest)
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

STABILITY_MAP = {
    'AA': -1.00, 'TT': -1.00, 'AT': -0.88, 'TA': -0.58,
    'AG': -1.30, 'GA': -1.30, 'AC': -1.45, 'CA': -1.45,
    'TG': -1.44, 'GT': -1.44, 'TC': -1.28, 'CT': -1.28,
    'CC': -1.84, 'GG': -1.84, 'CG': -2.24, 'GC': -2.27,
}

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = ROOT / "tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences"


def extract_aligned(seq: str) -> np.ndarray:
    """80bp (-60/+19) with TSS at position 60. 79 dinucleotides."""
    seq = seq.upper()
    if len(seq) >= 100:
        s = seq[20:100]
    else:
        s = seq[:80]
    return np.array([STABILITY_MAP.get(s[i:i+2], -1.35) for i in range(79)])


def load_training_data() -> tuple:
    """Load external species data. Returns (X_train, y_train)."""
    rng = np.random.RandomState(42)
    pos_feats = []
    for f in sorted(TRAIN_DIR.glob("Sequences_80-20_B*.txt")):
        with open(f) as fh:
            for line in fh:
                seq = line.strip()
                if len(seq) >= 100:
                    pos_feats.append(extract_aligned(seq))
    X_pos = np.array(pos_feats)
    X_neg = np.array([rng.permutation(row) for row in X_pos])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    return X, y


def load_test_data(pos_fasta: str, neg_fasta: str) -> tuple:
    """Load S. pneumoniae test data."""
    pos_rec = [r for r in SeqIO.parse(pos_fasta, "fasta")]
    neg_rec = [r for r in SeqIO.parse(neg_fasta, "fasta")]
    pos_ids = [r.id for r in pos_rec]
    neg_ids = [r.id for r in neg_rec]
    pos_seqs = [str(r.seq) for r in pos_rec]
    neg_seqs = [str(r.seq) for r in neg_rec]
    X_pos = np.array([extract_aligned(s) for s in pos_seqs])
    X_neg = np.array([extract_aligned(s) for s in neg_seqs])
    X_test = np.vstack([X_pos, X_neg])
    y_test = np.hstack([np.ones(len(pos_ids)), np.zeros(len(neg_ids))])
    return X_test, y_test, pos_ids + neg_ids, pos_seqs + neg_seqs


def save_csv(chroms, seqs, scores, path):
    df = pd.DataFrame({"CHROM": chroms, "SEQ": seqs, "PRED": scores})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, sep='\t')
    print(f"  Saved: {path} ({len(df)} predictions)")


def main():
    p = argparse.ArgumentParser(description="MLDSPP cross-species predictions.")
    p.add_argument("-p", "--positives", required=True)
    p.add_argument("-n", "--negatives", required=True)
    p.add_argument("-o", "--output-dir", required=True)
    args = p.parse_args()

    print(f"[INFO] Loading external training data from {TRAIN_DIR}...")
    X_train, y_train = load_training_data()
    print(f"  Training: {len(X_train)} sequences ({X_train.shape[1]} features)")

    print(f"[INFO] Loading S. pneumoniae test data...")
    X_test, y_test, ids, seqs = load_test_data(args.positives, args.negatives)
    n_pos = int(sum(y_test))
    n_neg = len(y_test) - n_pos
    print(f"  Test: {n_pos} pos + {n_neg} neg = {len(y_test)} total")

    for name, model in [
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                                   eval_metric='logloss', verbosity=0)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10,
                                                  random_state=42, n_jobs=4)),
        ("SVM", SVC(kernel='rbf', probability=True, random_state=42)),
    ]:
        print(f"\n[INFO] Training {name}...")
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]

        pos_scores = scores[:n_pos]
        neg_scores = scores[n_pos:]
        pos_ids = ids[:n_pos]
        neg_ids = ids[n_pos:]
        pos_seqs = seqs[:n_pos]
        neg_seqs = seqs[n_pos:]

        suffix = "svm" if name == "SVM" else "rf" if name == "Random Forest" else ""
        pos_file = os.path.join(args.output_dir, f"mldspp{'_' + suffix if suffix else ''}_pos.csv")
        neg_file = os.path.join(args.output_dir, f"mldspp{'_' + suffix if suffix else ''}_neg.csv")
        save_csv(pos_ids, pos_seqs, pos_scores, pos_file)
        save_csv(neg_ids, neg_seqs, neg_scores, neg_file)

    print("\n[DONE] All MLDSPP predictions saved.")


if __name__ == "__main__":
    main()
