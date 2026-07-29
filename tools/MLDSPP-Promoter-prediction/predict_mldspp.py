#!/usr/bin/env python3
import argparse
import sys
import os
import glob
import numpy as np
import pandas as pd
from Bio import SeqIO
from xgboost import XGBClassifier

# SantaLucia Stability parameters map
STABILITY_MAP = {
    'AA': -1.00, 'TT': -1.00,
    'AT': -0.88,
    'TA': -0.58,
    'AG': -1.30, 'GA': -1.30,
    'AC': -1.45, 'CA': -1.45,
    'TG': -1.44, 'GT': -1.44,
    'TC': -1.28, 'CT': -1.28,
    'CC': -1.84, 'GG': -1.84,
    'CG': -2.24,
    'GC': -2.27
}

def extract_stability_features(sequence):
    """Converts a 100 bp sequence into 99 dinucleotide stability values."""
    sequence = sequence.upper()
    features = []
    for pos in range(len(sequence) - 1):
        pair = sequence[pos:pos+2]
        val = STABILITY_MAP.get(pair, -1.35)  # fallback default if there is any N base
        features.append(val)
    return features

import random

def load_mldspp_training_data(dataset_dir):
    """Loads all promoter sequences, extracts stability features, and shuffles them for negatives."""
    promoter_dir = os.path.join(dataset_dir, "Promoter Sequences")
    
    X = []
    y = []
    
    # Initialize random seed for reproducibility
    random.seed(42)
    
    print("[INFO] Loading promoter training sequences and generating shuffled negatives...")
    for file_path in glob.glob(os.path.join(promoter_dir, "*.txt")):
        with open(file_path) as f:
            for line in f:
                seq = line.strip()
                if len(seq) >= 100:
                    seq_100 = seq[:100]
                    # Positive sample (Promoter)
                    X.append(extract_stability_features(seq_100))
                    y.append(1)
                    
                    # Negative sample (Shuffled Promoter)
                    seq_list = list(seq_100)
                    random.shuffle(seq_list)
                    shuffled_seq = "".join(seq_list)
                    X.append(extract_stability_features(shuffled_seq))
                    y.append(0)
                    
    return np.array(X), np.array(y)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and run MLDSPP (SantaLucia stability + XGBoost) predictions.")
    parser.add_argument('-i', '--input_file', type=str, required=True, help="Input FASTA file to predict (101 bp sequences).")
    parser.add_argument('-o', '--output_file', type=str, required=True, help="Output CSV predictions path.")
    parser.add_argument('-d', '--dataset_dir', type=str, default="/home/fierro/Desktop/MLDSPP-Promoter-prediction/Sample Dataset", help="Path to MLDSPP training Sample Dataset.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load and train XGBoost classifier
    X_train, y_train = load_mldspp_training_data(args.dataset_dir)
    print(f"[INFO] Loaded training data shape: {X_train.shape}. Positives: {np.sum(y_train == 1)}, Negatives: {np.sum(y_train == 0)}")
    
    print("[INFO] Training XGBoost classifier...")
    model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    
    # 2. Load FASTA file to predict
    if not os.path.exists(args.input_file):
        sys.exit(f"[ERROR] Input file {args.input_file} does not exist.")
        
    print(f"[INFO] Loading sequences to predict from {args.input_file}...")
    records = list(SeqIO.parse(args.input_file, "fasta"))
    chroms = [r.id for r in records]
    seqs = [str(r.seq).upper() for r in records]
    
    X_pred = []
    valid_indices = []
    
    for i, seq in enumerate(seqs):
        # We need exactly 100 bp (slice if longer, warn if shorter)
        if len(seq) < 100:
            print(f"[WARNING] Skipping sequence {chroms[i]} because it is shorter than 100 bp ({len(seq)} bp).")
            continue
        X_pred.append(extract_stability_features(seq[:100]))
        valid_indices.append(i)
        
    X_pred = np.array(X_pred)
    print(f"[INFO] Extracted features for {len(X_pred)} test sequences.")
    
    # 3. Predict
    print("[INFO] Running predictions...")
    y_probs = model.predict_proba(X_pred)[:, 1]
    
    # 4. Save predictions
    output_chroms = [chroms[i] for i in valid_indices]
    output_seqs = [seqs[i] for i in valid_indices]
    
    df = pd.DataFrame({
        "CHROM": output_chroms,
        "SEQ": output_seqs,
        "PRED": y_probs
    })
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_csv(args.output_file, index=None, sep='\t')
    print(f"[SUCCESS] MLDSPP predictions saved to {args.output_file}")

if __name__ == "__main__":
    main()
