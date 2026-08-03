#!/usr/bin/env python3
"""
Generates a Consolidated CSV Summary Table of Dataset Extraction Metrics.

Extracts summary statistics (GC content, Cohen's d, Z-score, Steric conflicts,
+1 Purine %, -10 Box match %, 5'-UTR %) across all generated D39V and TIGR4 datasets.

Usage:
    pixi run python src/dataset/generate_extraction_summary_table.py
"""

import math
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_TO_SUMMARIZE = [
    {
        "Name": "TIGR4 Tier 1 (High Conf Primary Positives)",
        "Type": "Positive",
        "Species": "TIGR4",
        "Fasta": ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.fasta",
        "Genome_Fasta": ROOT / "data/reference/NC_003028.fasta",
        "Tss_Evaluated": 742,
        "Steric_Conflicts": 4,
    },
    {
        "Name": "TIGR4 Tier 1 (High Conf Primary Negatives)",
        "Type": "Negative",
        "Species": "TIGR4",
        "Fasta": ROOT / "output/tigr4_data/negatives_tigr4_high_conf_primary_81bp.fasta",
        "Genome_Fasta": ROOT / "data/reference/NC_003028.fasta",
        "Tss_Evaluated": 138544,
        "Steric_Conflicts": 0,
    },
    {
        "Name": "TIGR4 Tier 2 (Extended Primary Positives)",
        "Type": "Positive",
        "Species": "TIGR4",
        "Fasta": ROOT / "output/tigr4_data/positives_tigr4_extended_primary_81bp.fasta",
        "Genome_Fasta": ROOT / "data/reference/NC_003028.fasta",
        "Tss_Evaluated": 2028,
        "Steric_Conflicts": 28,
    },
    {
        "Name": "TIGR4 Tier 2 (Extended Primary Negatives)",
        "Type": "Negative",
        "Species": "TIGR4",
        "Fasta": ROOT / "output/tigr4_data/negatives_tigr4_extended_primary_81bp.fasta",
        "Genome_Fasta": ROOT / "data/reference/NC_003028.fasta",
        "Tss_Evaluated": 138544,
        "Steric_Conflicts": 0,
    },
    {
        "Name": "D39V Primary Positives",
        "Type": "Positive",
        "Species": "D39V",
        "Fasta": ROOT / "data/benchmark/positives_81bp.fasta",
        "Genome_Fasta": ROOT / "data/reference/D39V.fna",
        "Tss_Evaluated": 1002,
        "Steric_Conflicts": 14,
    },
    {
        "Name": "D39V Primary Negatives",
        "Type": "Negative",
        "Species": "D39V",
        "Fasta": ROOT / "data/benchmark/negatives_81bp.fasta",
        "Genome_Fasta": ROOT / "data/reference/D39V.fna",
        "Tss_Evaluated": 130000,
        "Steric_Conflicts": 0,
    },
]


def compute_fasta_metrics(fasta_path: Path, genome_fasta: Path, upstream: int = 60) -> Dict:
    if not fasta_path.exists():
        return {}

    recs = list(SeqIO.parse(fasta_path, "fasta"))
    if not recs:
        return {}

    seqs = [str(r.seq).upper() for r in recs]
    window_size = len(seqs[0])

    gc_list = [((s.count("G") + s.count("C")) / window_size * 100.0) for s in seqs]
    mean_gc = float(np.mean(gc_list))
    sd_gc = float(np.std(gc_list, ddof=1)) if len(gc_list) > 1 else 0.0

    genome_rec = list(SeqIO.parse(genome_fasta, "fasta"))[0]
    gen_seq = str(genome_rec.seq).upper()
    gen_gc = ((gen_seq.count("G") + gen_seq.count("C")) / len(gen_seq)) * 100.0

    n = len(gc_list)
    z_score = (mean_gc - gen_gc) / (sd_gc / math.sqrt(n)) if n > 0 and sd_gc > 0 else 0.0
    cohen_d = (mean_gc - gen_gc) / sd_gc if sd_gc > 0 else 0.0

    plus1_bases = [s[upstream] for s in seqs if len(s) > upstream]
    n_p1 = len(plus1_bases)
    purines = ((plus1_bases.count("A") + plus1_bases.count("G")) / n_p1 * 100.0) if n_p1 > 0 else 0.0

    pribnow_matches = 0
    for s in seqs:
        if len(s) >= upstream + 21:
            win = s[40:57]
            if "TATAAT" in win or re.search(r"TA[ATGC]{2,3}[AT]", win):
                pribnow_matches += 1

    pribnow_pct = (pribnow_matches / n * 100.0) if n > 0 else 0.0

    return {
        "N_Sequences": n,
        "Window_Size_bp": window_size,
        "Total_Sampled_nt": n * window_size,
        "Sampled_GC_Mean_Pct": round(mean_gc, 2),
        "Sampled_GC_SD_Pct": round(sd_gc, 2),
        "Genome_Background_GC_Pct": round(gen_gc, 2),
        "Z_Score": round(z_score, 2),
        "Cohens_D": round(cohen_d, 2),
        "Plus1_Purines_Pct": round(purines, 1),
        "Pribnow_Minus10_Match_Pct": round(pribnow_pct, 1),
    }


def main():
    rows = []
    for item in DATASETS_TO_SUMMARIZE:
        m = compute_fasta_metrics(item["Fasta"], item["Genome_Fasta"])
        if not m:
            continue

        eval_count = item["Tss_Evaluated"]
        n_seqs = m["N_Sequences"]
        eff_pct = round((n_seqs / eval_count * 100.0), 1) if eval_count > 0 else 100.0

        row = {
            "Dataset_Name": item["Name"],
            "Dataset_Type": item["Type"],
            "Species": item["Species"],
            "Total_Evaluated": eval_count,
            "Extracted_Sequences": n_seqs,
            "Extraction_Efficiency_Pct": eff_pct,
            "Window_Size_bp": m["Window_Size_bp"],
            "Total_Sampled_nt": m["Total_Sampled_nt"],
            "Steric_Conflicts_Discarded": item["Steric_Conflicts"],
            "Sampled_GC_Mean_Pct": m["Sampled_GC_Mean_Pct"],
            "Sampled_GC_SD_Pct": m["Sampled_GC_SD_Pct"],
            "Genome_Background_GC_Pct": m["Genome_Background_GC_Pct"],
            "Z_Score": m["Z_Score"],
            "Cohens_D": m["Cohens_D"],
            "Plus1_Purines_Pct": m["Plus1_Purines_Pct"],
            "Pribnow_Minus10_Match_Pct": m["Pribnow_Minus10_Match_Pct"],
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "summary_dataset_extraction_metrics.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Consolidated extraction summary table saved ➔ {out_csv}\n")

    print("═" * 110)
    print(" CONSOLIDATED DATASET EXTRACTION SUMMARY TABLE")
    print("═" * 110)
    print(df_summary.to_string(index=False))
    print("═" * 110 + "\n")


if __name__ == "__main__":
    main()
