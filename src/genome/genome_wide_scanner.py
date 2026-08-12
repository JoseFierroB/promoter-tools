#!/usr/bin/env python3
"""
Genome-Wide Promoter Scanner using MLDSPP (Biophysical Stability + XGBoost)
Scans a whole-genome FASTA in a fast, vectorized sliding window (using NumPy sliding window views)
and outputs predicted TSS positions in GFF3 and TSV formats.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from Bio import SeqIO
from xgboost import XGBClassifier

from src.runners._shared import STABILITY_MAP

def extract_stability_features(sequence):
    sequence = sequence.upper()
    features = []
    for pos in range(len(sequence) - 1):
        pair = sequence[pos:pos+2]
        val = STABILITY_MAP.get(pair, -1.35)
        features.append(val)
    return features

def train_champion_model(pos_path, neg_path):
    print(f"[INFO] Training champion MLDSPP model using:\n  - Pos: {pos_path}\n  - Neg: {neg_path} ...", file=sys.stderr)
    pos_records = list(SeqIO.parse(pos_path, "fasta"))
    neg_records = list(SeqIO.parse(neg_path, "fasta"))
    
    X_pos = np.array([extract_stability_features(str(r.seq)) for r in pos_records])
    X_neg = np.array([extract_stability_features(str(r.seq)) for r in neg_records])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    
    model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    print("[INFO] Model trained successfully.", file=sys.stderr)
    return model

def scan_sequence_vectorized(seq_str, model, threshold, window_size=81):
    L = len(seq_str)
    if L < window_size:
        return []
        
    # 1. Map sequence to stability values
    # Pre-allocate array
    stability_profile = np.zeros(L - 1, dtype=np.float32)
    # Convert sequence to upper case string
    seq_str = seq_str.upper()
    
    # Fill stability profile
    for i in range(L - 1):
        pair = seq_str[i:i+2]
        stability_profile[i] = STABILITY_MAP.get(pair, -1.35)
        
    # 2. Use numpy stride tricks to construct sliding window matrix
    # sliding_window_view constructs a view of shape (L - window_size + 1, window_size - 1)
    # Shape of stability_profile is L - 1. We want windows of size 80 (window_size - 1).
    from numpy.lib.stride_tricks import sliding_window_view
    X = sliding_window_view(stability_profile, window_size - 1)
    
    # 3. Batch predict
    probs = model.predict_proba(X)[:, 1]
    
    # 4. Filter indices above threshold
    passed_indices = np.where(probs >= threshold)[0]
    
    # Return list of tuples: (relative_TSS_index_in_seq, probability)
    # The TSS is located at index 60 of the 81bp window.
    # So if the window starts at index 'start', the TSS coordinate is 'start + 60'.
    results = []
    for idx in passed_indices:
        results.append((idx + 60, float(probs[idx])))
    return results

def non_maximum_suppression(hits, cluster_dist):
    """
    Cluster adjacent predictions (NMS) within a distance window.
    Only keeps the prediction with the highest probability.
    """
    if not hits:
        return []
    # Sort hits by TSS position
    hits = sorted(hits, key=lambda x: x[0])
    
    clustered = []
    current_cluster = [hits[0]]
    
    for h in hits[1:]:
        # If current hit is close to the cluster
        if h[0] - current_cluster[-1][0] <= cluster_dist:
            current_cluster.append(h)
        else:
            # End of cluster: pick the best hit
            best_hit = max(current_cluster, key=lambda x: x[1])
            clustered.append(best_hit)
            current_cluster = [h]
            
    # Append the last cluster
    if current_cluster:
        best_hit = max(current_cluster, key=lambda x: x[1])
        clustered.append(best_hit)
        
    return clustered

def main():
    parser = argparse.ArgumentParser(description="Genome-wide promoter scanner using MLDSPP biophysical model.")
    parser.add_argument("-g", "--genome", required=True, help="Path to genome FASTA file.")
    parser.add_argument("-p", "--positives", default="data/benchmark/d39v/positives_81bp.fasta", help="Path to positive training FASTA.")
    parser.add_argument("-n", "--negatives", default="data/benchmark/d39v/negatives_81bp.fasta", help="Path to negative training FASTA.")
    parser.add_argument("-t", "--threshold", type=float, default=0.6822, help="Probability threshold (default: 0.6822).")
    parser.add_argument("-c", "--cluster-distance", type=int, default=50, help="Clustering distance for overlapping peaks (default: 50 bp).")
    parser.add_argument("-o", "--output", default="mldspp_genome_predictions", help="Output prefix for GFF3/TSV files.")
    args = parser.parse_args()
    
    # 1. Train model
    model = train_champion_model(args.positives, args.negatives)
    
    # 2. Load genome
    print(f"[INFO] Loading genome from {args.genome}...", file=sys.stderr)
    genome_records = list(SeqIO.parse(args.genome, "fasta"))
    print(f"[INFO] Loaded {len(genome_records)} contig(s)/chromosome(s).", file=sys.stderr)
    
    all_predictions = []
    
    # 3. Scan contig by contig
    for rec in genome_records:
        chrom_id = rec.id
        seq_fwd = str(rec.seq)
        seq_rev = str(rec.seq.reverse_complement())
        
        print(f"[INFO] Scanning contig '{chrom_id}' (Length: {len(seq_fwd)} bp)...", file=sys.stderr)
        
        # Scan Forward strand
        print("  - Scanning forward strand (+)...", file=sys.stderr)
        hits_fwd = scan_sequence_vectorized(seq_fwd, model, args.threshold)
        hits_fwd_nms = non_maximum_suppression(hits_fwd, args.cluster_distance)
        for tss_pos, score in hits_fwd_nms:
            # 1-based coordinate for GFF3
            all_predictions.append((chrom_id, tss_pos + 1, "+", score))
            
        # Scan Reverse strand
        print("  - Scanning reverse strand (-)...", file=sys.stderr)
        hits_rev = scan_sequence_vectorized(seq_rev, model, args.threshold)
        hits_rev_nms = non_maximum_suppression(hits_rev, args.cluster_distance)
        L = len(seq_fwd)
        for tss_pos_rev_strand, score in hits_rev_nms:
            # Map TSS coordinate back to forward strand:
            # The TSS is at index `tss_pos_rev_strand` on the reverse complement.
            # Forward index is `L - 1 - tss_pos_rev_strand`
            tss_pos_fwd_strand = L - 1 - tss_pos_rev_strand
            # 1-based coordinate for GFF3
            all_predictions.append((chrom_id, tss_pos_fwd_strand + 1, "-", score))
            
    print(f"[INFO] Found {len(all_predictions)} predicted promoters across all contigs.", file=sys.stderr)
    
    # 4. Save results to TSV
    tsv_out = f"{args.output}.tsv"
    df = pd.DataFrame(all_predictions, columns=["Chromosome", "TSS_Position_1based", "Strand", "Probability"])
    df.sort_values(by=["Chromosome", "TSS_Position_1based"], inplace=True)
    df.to_csv(tsv_out, index=False, sep='\t')
    print(f"[SUCCESS] Exported TSV report to: {tsv_out}", file=sys.stderr)
    
    # 5. Save results to GFF3
    gff_out = f"{args.output}.gff3"
    with open(gff_out, "w") as f:
        f.write("##gff-version 3\n")
        for chrom, pos, strand, score in all_predictions:
            f.write(f"{chrom}\tMLDSPP_Scanner\ttranscription_start_site\t{pos}\t{pos}\t{score:.4f}\t{strand}\t.\tID=TSS_pred_{pos};Name=Predicted_TSS_{pos}\n")
    print(f"[SUCCESS] Exported GFF3 annotations to: {gff_out}", file=sys.stderr)

if __name__ == "__main__":
    main()
