#!/usr/bin/env python3
"""PromoterLCNN runner — one-hot encode + TF predict."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf

MODEL_DIR = "tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5"


def main():
    p = argparse.ArgumentParser(description="PromoterLCNN prediction")
    p.add_argument("--pos", required=True, help="Positive test FASTA")
    p.add_argument("--neg", required=True, help="Negative test FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    p.add_argument("-m", "--model", default=MODEL_DIR, help="Model directory")
    args = p.parse_args()

    pos = list(SeqIO.parse(args.pos, "fasta"))
    neg = list(SeqIO.parse(args.neg, "fasta"))
    seqs = [str(r.seq).upper() for r in pos + neg]

    m = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}
    X = np.array([[m[c] for c in s] for s in seqs], dtype=np.float32)

    model = tf.keras.models.load_model(args.model, compile=False)
    t0 = time.perf_counter()
    probs = model.predict(X, verbose=0, batch_size=128)
    elapsed = time.perf_counter() - t0
    probs = probs[:, 1] if probs.ndim == 2 and probs.shape[1] >= 2 else probs.ravel()

    out_dir = Path(args.output) / "lcnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PRED": probs[:len(pos)]}).to_csv(
        out_dir / "lcnn_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": probs[len(pos):]}).to_csv(
        out_dir / "lcnn_neg.csv", sep="\t", index=False)

    print(f"LCNN: {len(seqs)} seqs in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
