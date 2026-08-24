#!/usr/bin/env python3
"""MLDSPP XGBoost — dinucleotide stability + cross-species training."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.runners._shared import extract_aligned

TRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences"


def load_training():
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


def main():
    p = argparse.ArgumentParser(description="MLDSPP XGBoost cross-species")
    p.add_argument("--pos", required=True, help="Positive test FASTA")
    p.add_argument("--neg", required=True, help="Negative test FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    args = p.parse_args()

    t0 = time.perf_counter()
    X_train, y_train = load_training()
    pos = list(SeqIO.parse(args.pos, "fasta"))
    neg = list(SeqIO.parse(args.neg, "fasta"))
    X_pos = np.array([extract_aligned(str(r.seq)) for r in pos])
    X_neg = np.array([extract_aligned(str(r.seq)) for r in neg])
    X_test = np.vstack([X_pos, X_neg])

    model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                          eval_metric="logloss", verbosity=0,
                          n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1") or 1))
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PRED": probs[:len(pos)]}).to_csv(
        out_dir / "mldspp_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": probs[len(pos):]}).to_csv(
        out_dir / "mldspp_neg.csv", sep="\t", index=False)

    print(f"MLDSPP: {len(pos) + len(neg)} seqs in {elapsed:.4f}s (train {train_s:.3f}s)")


if __name__ == "__main__":
    main()
