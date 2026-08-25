#!/usr/bin/env python3
"""MLDSPP XGBoost (75% S. pneumoniae) — dinucleotide stability + mixed training.
Uses pre-built 75/25 splits from data/benchmark/mldspp_75_split_*.npz (seed=42).
"""
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

from src.runners._shared import extract_aligned, MLDSPP_XGB_PARAMS

TRAIN_DIR = ROOT / "tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences"
SPLIT_DIR = ROOT / "data/benchmark"
RNG = np.random.RandomState(42)


def load_external_training():
    pos_feats = []
    for f in sorted(TRAIN_DIR.glob("Sequences_80-20_B*.txt")):
        with open(f) as fh:
            for line in fh:
                seq = line.strip()
                if len(seq) >= 100:
                    pos_feats.append(extract_aligned(seq))
    return np.array(pos_feats)


def main():
    p = argparse.ArgumentParser(description="MLDSPP XGBoost (75% S. pneumoniae)")
    p.add_argument("--pos", required=True, help="Positive test FASTA")
    p.add_argument("--neg", required=True, help="Negative test FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    p.add_argument("--split", default=None,
                   help="Pre-built split .npz from data/benchmark/ (seed=42, ratio=0.75)")
    args = p.parse_args()

    t0 = time.perf_counter()
    X_ext = load_external_training()

    pos = list(SeqIO.parse(args.pos, "fasta"))
    neg = list(SeqIO.parse(args.neg, "fasta"))
    X_pos_all = np.array([extract_aligned(str(r.seq)) for r in pos])
    X_neg_all = np.array([extract_aligned(str(r.seq)) for r in neg])

    # Load pre-built split or generate random one
    if args.split:
        split_data = np.load(SPLIT_DIR / args.split)
        train_idx = split_data["train_idx"]
        if int(train_idx.max()) >= len(pos):
            sys.exit(f"ERROR: split '{args.split}' indexes up to {train_idx.max()} "
                     f"but positive FASTA has only {len(pos)} sequences. "
                     f"Use the split matching this dataset (see data/benchmark/mldspp_75_split_*.npz).")
    else:
        n_spn = int(len(pos) * 0.75)
        idx = RNG.permutation(len(pos))
        train_idx = idx[:n_spn]

    X_spn_pos = X_pos_all[train_idx]
    X_spn_neg = np.array([RNG.permutation(row) for row in X_spn_pos])
    X_ext_neg = np.array([RNG.permutation(row) for row in X_ext])

    X_train = np.vstack([X_ext, X_ext_neg, X_spn_pos, X_spn_neg])
    y_train = np.hstack([
        np.ones(len(X_ext)), np.zeros(len(X_ext)),
        np.ones(len(X_spn_pos)), np.zeros(len(X_spn_neg)),
    ])

    X_test = np.vstack([X_pos_all, X_neg_all])
    model = XGBClassifier(**MLDSPP_XGB_PARAMS,
                          n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1") or 1))
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - t0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PRED": probs[:len(pos)]}).to_csv(
        out_dir / "mldspp_75spn_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": probs[len(pos):]}).to_csv(
        out_dir / "mldspp_75spn_neg.csv", sep="\t", index=False)

    print(f"MLDSPP_75: {len(pos) + len(neg)} seqs in {elapsed:.4f}s (train {train_s:.3f}s)")


if __name__ == "__main__":
    main()
