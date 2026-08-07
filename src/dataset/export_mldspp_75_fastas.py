#!/usr/bin/env python3
"""
Exporter for MLDSPP 75/25 Train/Test FASTA Splits.
Reads .npz index files in data/benchmark/ and exports exact 75% train and 25% test FASTAs.

Outputs written to: data/benchmark/splits/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmark"
TIGR4_DIR = ROOT / "data" / "tigr4"
SPLITS_DIR = BENCHMARK_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of .npz split file to source FASTA file
SPLIT_MAPPINGS = [
    {
        "npz": BENCHMARK_DIR / "mldspp_75_split_d39v.npz",
        "fasta": BENCHMARK_DIR / "positives_81bp.fasta",
        "prefix": "d39v"
    },
    {
        "npz": BENCHMARK_DIR / "mldspp_75_split_tigr4_high.npz",
        "fasta": TIGR4_DIR / "positives_high_81bp.fasta",
        "prefix": "tigr4_high"
    },
    {
        "npz": BENCHMARK_DIR / "mldspp_75_split_tigr4_extended_high.npz",
        "fasta": TIGR4_DIR / "positives_extended_81bp.fasta",
        "prefix": "tigr4_extended_high"
    },
]

def export_splits():
    print("Exporting MLDSPP 75/25 Train/Test FASTA Splits...")

    for item in SPLIT_MAPPINGS:
        npz_path = item["npz"]
        fasta_path = item["fasta"]
        prefix = item["prefix"]

        if not npz_path.exists():
            print(f"Skipping {prefix}: NPZ not found at {npz_path}")
            continue
        if not fasta_path.exists():
            print(f"Skipping {prefix}: FASTA not found at {fasta_path}")
            continue

        # Load indices and records
        data = np.load(npz_path)
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]

        records = list(SeqIO.parse(fasta_path, "fasta"))
        n_total = len(records)

        train_records = [records[i] for i in train_idx if i < n_total]
        test_records = [records[i] for i in test_idx if i < n_total]

        out_train_fasta = SPLITS_DIR / f"{prefix}_train_75.fasta"
        out_test_fasta = SPLITS_DIR / f"{prefix}_test_25.fasta"

        SeqIO.write(train_records, out_train_fasta, "fasta")
        SeqIO.write(test_records, out_test_fasta, "fasta")

        print(f"\n{prefix.upper()}:")
        print(f"  • Source FASTA:  {fasta_path.name} ({n_total} records)")
        print(f"  • Train (75%):   {out_train_fasta} ({len(train_records)} records)")
        print(f"  • Test (25%):    {out_test_fasta} ({len(test_records)} records)")

if __name__ == "__main__":
    export_splits()
