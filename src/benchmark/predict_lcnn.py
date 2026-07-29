#!/usr/bin/env python3
"""
PromoterLCNN Prediction Script.
Loads the IsPromoter_fold_5 TensorFlow model, one-hot encodes 81bp sequences,
and saves prediction scores.

Outputs:
  - lcnn_pos.csv (CHROM, SEQ, PRED)
  - lcnn_neg.csv (CHROM, SEQ, PRED)

Author: Antigravity Code Assistant
Date: July 15, 2026
"""

import argparse
import os
import sys
import re
import numpy as np
import pandas as pd

# Suppress TF verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


# One-hot encoding: A/T/C/G → 4D binary vectors
ENCODING = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0],
            "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}


def one_hot_encode(seq: str) -> np.ndarray:
    """Convert 81bp DNA string → (81, 4) one-hot matrix."""
    seq = seq.upper()
    return np.array([ENCODING.get(base, [0, 0, 0, 0]) for base in seq])


def load_fasta(fasta_path: str) -> tuple:
    """Parse FASTA manually (no BioPython dep). Returns (ids, sequences, encoded_matrix)."""
    ids, seqs, encoded = [], [], []
    current_id, current_seq = None, []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    s = "".join(current_seq).upper()
                    s = re.sub(r'[^ATCG]', 'N', s)
                    if len(s) == 81:
                        ids.append(current_id)
                        seqs.append(s)
                        encoded.append(one_hot_encode(s))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            s = "".join(current_seq).upper()
            s = re.sub(r'[^ATCG]', 'N', s)
            if len(s) == 81:
                ids.append(current_id)
                seqs.append(s)
                encoded.append(one_hot_encode(s))
    return ids, seqs, np.array(encoded) if encoded else np.empty((0, 81, 4))


def parse_args():
    p = argparse.ArgumentParser(description="PromoterLCNN prediction script.")
    p.add_argument("-p", "--positives", required=True, help="Positive 81bp FASTA")
    p.add_argument("-n", "--negatives", required=True, help="Negative 81bp FASTA")
    p.add_argument("-o", "--output-dir", required=True, help="Output directory for CSVs")
    p.add_argument("-m", "--model-dir", required=True, help="Path to PromoterLCNN model folder")
    return p.parse_args()


def save_csv(chroms, seqs, scores, path):
    df = pd.DataFrame({"CHROM": chroms, "SEQ": seqs, "PRED": scores})
    df.to_csv(path, index=False, sep='\t')
    print(f"  Saved: {path} ({len(df)} predictions)")


def main():
    args = parse_args()

    print(f"[INFO] Loading PromoterLCNN model from: {args.model_dir}")
    model = tf.keras.models.load_model(args.model_dir, compile=False)
    print(f"[INFO] Model loaded. Input: {model.input_shape}, Output: {model.output_shape}")

    # Load and encode sequences
    print("[INFO] Loading positive sequences...")
    pos_ids, pos_seqs, pos_encoded = load_fasta(args.positives)
    print(f"  Positives: {len(pos_ids)} valid sequences (shape: {pos_encoded.shape})")

    print("[INFO] Loading negative sequences...")
    neg_ids, neg_seqs, neg_encoded = load_fasta(args.negatives)
    print(f"  Negatives: {len(neg_ids)} valid sequences (shape: {neg_encoded.shape})")

    # Predict
    print("[INFO] Running predictions on positives...")
    pos_probs = model.predict(pos_encoded, verbose=1)
    # Softmax output: take probability of class 1 (promoter)
    if pos_probs.ndim == 2 and pos_probs.shape[1] >= 2:
        pos_scores = pos_probs[:, 1]
    else:
        pos_scores = pos_probs.ravel()

    print("[INFO] Running predictions on negatives...")
    neg_probs = model.predict(neg_encoded, verbose=1)
    if neg_probs.ndim == 2 and neg_probs.shape[1] >= 2:
        neg_scores = neg_probs[:, 1]
    else:
        neg_scores = neg_probs.ravel()

    # Save
    import os as _os
    _os.makedirs(os.path.join(args.output_dir, "lcnn"), exist_ok=True)
    save_csv(pos_ids, pos_seqs, pos_scores, os.path.join(args.output_dir, "lcnn", "lcnn_pos.csv"))
    save_csv(neg_ids, neg_seqs, neg_scores, os.path.join(args.output_dir, "lcnn", "lcnn_neg.csv"))

    print("\n[SUCCESS] PromoterLCNN predictions complete.")


if __name__ == "__main__":
    main()
