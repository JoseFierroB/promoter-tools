#!/usr/bin/env python3
"""
TIGR4 Negative TSS Sequence Extractor (Master Pool Approach).

Parses CDS gene boundaries (Locus_start to Locus_end) and experimental TSS positions
from S1_TSS.xlsx (Aprianto et al., 2018) against the TIGR4 genome (NC_003028.fasta).

Features:
- Builds a complete Master Pool of clean non-promoter k-mers located strictly inside CDS regions.
- Enforces safety margins from CDS boundaries (--margin) and TSS proximity (--tss-margin).
- Generates balanced 1:1 negative datasets for TIGR4 promoter prediction benchmarks.

Usage:
    pixi run python src/dataset/negatives_tss_tigr4.py \
      --xlsx data/tigr4/S1_TSS.xlsx \
      --fasta data/reference/NC_003028.fasta \
      --tier high_conf_primary \
      --limit 738 \
      -o data/tigr4/negatives_high_81bp
"""

import argparse
import math
import os
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

_RC_TRANS = str.maketrans("ACGT", "TGCA")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract negative non-promoter windows from TIGR4 CDS regions."
    )

    parser.add_argument(
        "--xlsx",
        default="data/tigr4/S1_TSS.xlsx",
        help="Path to TIGR4 S1_TSS.xlsx file.",
    )
    parser.add_argument(
        "--fasta",
        default="data/reference/NC_003028.fasta",
        help="Path to TIGR4 genome FASTA file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="data/tigr4/negatives_high_81bp",
        help="Output prefix (generates .fasta and .tsv).",
    )
    parser.add_argument(
        "--tier",
        choices=["high_conf_primary", "extended_primary", "all_tss"],
        default="high_conf_primary",
        help="Target tier to match TSS exclusions and GC target (default: 'high_conf_primary').",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=81,
        help="Window size / k-mer length (default: 81).",
    )
    parser.add_argument(
        "-s",
        "--step",
        type=int,
        default=10,
        help="Step size for scanning CDS k-mers (default: 10).",
    )
    parser.add_argument(
        "-m",
        "--margin",
        type=int,
        default=20,
        help="Safety margin from CDS start/end boundaries (default: 20).",
    )
    parser.add_argument(
        "--tss-margin",
        type=int,
        default=200,
        help="Minimum distance allowed to any TSS on same strand (default: 200).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of negative sequences to sample (0 = match positive tier count).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility (default: 42).",
    )
    parser.add_argument(
        "--target-gc",
        type=float,
        default=None,
        help="Target GC content percentage to match the positive promoter collection.",
    )
    parser.add_argument(
        "--gc-tolerance",
        type=float,
        default=5.0,
        help="GC content tolerance percentage when --target-gc is specified (default: 5.0).",
    )
    parser.add_argument(
        "--dedup-rc",
        action="store_true",
        help="Deduplicate reverse complements of extracted windows as well.",
    )

    return parser.parse_args()


def load_genome(fasta_path: Path) -> Tuple[str, Seq, int]:
    if not fasta_path.exists():
        alt_fasta = fasta_path.parent / "TIGR4.fasta"
        if alt_fasta.exists():
            fasta_path = alt_fasta
        else:
            print(f"[ERROR] FASTA file not found at {fasta_path}", file=sys.stderr)
            sys.exit(1)

    genome_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    if not genome_dict:
        print(f"[ERROR] Could not parse FASTA from {fasta_path}", file=sys.stderr)
        sys.exit(1)

    chrom_id = list(genome_dict.keys())[0]
    seq = genome_dict[chrom_id].seq
    seq_len = len(seq)

    print(f"[INFO] Loaded genome '{chrom_id}' (length: {seq_len:,} bp) from {fasta_path.name}")
    return chrom_id, seq, seq_len


def load_cds_and_tss_tables(
    xlsx_path: Path, tier: str
) -> Tuple[pd.DataFrame, List[int], List[int]]:
    if not xlsx_path.exists():
        print(f"[ERROR] Excel file not found at {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    xl = pd.ExcelFile(xlsx_path)

    # Always load all sheet entries to extract CDS gene boundaries
    all_dfs = []
    for sheet in xl.sheet_names:
        df_s = pd.read_excel(xlsx_path, sheet_name=sheet)
        all_dfs.append(df_s)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Extract all TSS positions for exclusion masking
    pos_tss = []
    neg_tss = []

    for _, row in combined_df.iterrows():
        strand = str(row.get("Strand", "+")).strip()
        tss_col = (
            "TSS_position"
            if "TSS_position" in row
            else ("Primary_TSS" if "Primary_TSS" in row else "TSS")
        )
        if tss_col in row and pd.notna(row[tss_col]):
            pos = int(row[tss_col])
            if strand == "+":
                pos_tss.append(pos)
            else:
                neg_tss.append(pos)

    pos_tss.sort()
    neg_tss.sort()

    print(
        f"[INFO] Loaded {len(combined_df):,} CDS annotations and {len(pos_tss) + len(neg_tss):,} TSS positions for exclusion"
    )
    return combined_df, pos_tss, neg_tss


def build_master_pool(
    df_cds: pd.DataFrame,
    genome_seq: Seq,
    pos_tss: List[int],
    neg_tss: List[int],
    window_size: int,
    step_size: int,
    edge_margin: int,
    tss_margin: int,
    target_gc: float = None,
    gc_tolerance: float = 5.0,
    dedup_rc: bool = False,
) -> Tuple[List[Dict], Dict[str, int]]:
    seq_str = str(genome_seq).upper()
    seq_len = len(seq_str)

    master_pool = []
    exclusion_stats = {
        "total_evaluated": 0,
        "skipped_short_cds": 0,
        "skipped_tss_proximity": 0,
        "skipped_invalid_n": 0,
        "skipped_gc_bias": 0,
    }

    seen_seqs = set()

    for idx, row in df_cds.iterrows():
        if "Locus_start" not in row or "Locus_end" not in row:
            continue

        try:
            start = int(row["Locus_start"])
            end = int(row["Locus_end"])
            strand = str(row.get("Strand", "+")).strip()
            locus = str(row.get("Locus_tag", f"TIGR4_CDS_{idx}")).strip()
        except (ValueError, TypeError):
            continue

        if start > end:
            start, end = end, start

        # Apply edge safety margins inside CDS
        sub_start = start + edge_margin
        sub_end = end - edge_margin

        if (sub_end - sub_start + 1) < window_size:
            exclusion_stats["skipped_short_cds"] += 1
            continue

        tss_list = pos_tss if strand == "+" else neg_tss

        # Scan k-mers with specified step size
        for pos in range(sub_start, sub_end - window_size + 2, step_size):
            exclusion_stats["total_evaluated"] += 1
            kmer_start_idx = pos - 1
            kmer_end_idx = kmer_start_idx + window_size

            if kmer_start_idx < 0 or kmer_end_idx > seq_len:
                continue

            center_pos = pos + (window_size // 2)

            # Check distance to closest TSS on same strand
            if tss_list:
                # Binary search for closest TSS
                idx_c = np.searchsorted(tss_list, center_pos)
                min_dist = float("inf")
                if idx_c < len(tss_list):
                    min_dist = min(min_dist, abs(tss_list[idx_c] - center_pos))
                if idx_c > 0:
                    min_dist = min(min_dist, abs(tss_list[idx_c - 1] - center_pos))

                if min_dist < tss_margin:
                    exclusion_stats["skipped_tss_proximity"] += 1
                    continue

            # Extract sequence
            raw_seq = seq_str[kmer_start_idx:kmer_end_idx]
            if strand == "-":
                raw_seq = raw_seq.translate(_RC_TRANS)[::-1]

            if len(raw_seq) != window_size or "N" in raw_seq:
                exclusion_stats["skipped_invalid_n"] += 1
                continue

            if raw_seq in seen_seqs:
                continue
            if dedup_rc:
                raw_rc = raw_seq.translate(_RC_TRANS)[::-1]
                if raw_rc in seen_seqs:
                    continue

            gc_count = raw_seq.count("G") + raw_seq.count("C")
            gc_content = (gc_count / window_size) * 100.0

            if target_gc is not None:
                if not (target_gc - gc_tolerance <= gc_content <= target_gc + gc_tolerance):
                    exclusion_stats["skipped_gc_bias"] += 1
                    continue

            clean_locus = str(locus).replace(" ", "_")
            s_tag = "f" if strand == "+" else "r"
            neg_id = f"TIGR4_NEG_{clean_locus}-{pos}-{s_tag}"
            seen_seqs.add(raw_seq)
            master_pool.append({
                "Sequence_ID": neg_id,
                "Locus_Tag": clean_locus,
                "Position": pos,
                "Strand": strand,
                "Window_Size": window_size,
                "Sequence": raw_seq,
                "GC_Content": round(gc_content, 2),
            })

    return master_pool, exclusion_stats


def sample_negative_dataset(
    master_pool: List[Dict], limit: int, seed: int = 42
) -> List[Dict]:
    if not master_pool:
        return []

    if limit <= 0 or limit >= len(master_pool):
        return master_pool

    random.seed(seed)
    sampled = random.sample(master_pool, limit)
    # Sort by position for clean genomic ordering
    sampled.sort(key=lambda x: (x["Locus_Tag"], x["Position"]))
    return sampled


def compute_gc_statistics(records: List[Dict], genome_seq: Seq) -> Dict:
    if not records:
        return {}

    sampled_gc = [r["GC_Content"] for r in records]
    mean_gc = float(np.mean(sampled_gc))
    stdev_gc = float(np.std(sampled_gc, ddof=1)) if len(sampled_gc) > 1 else 0.0

    gen_seq_str = str(genome_seq).upper()
    gc_gen_count = gen_seq_str.count("G") + gen_seq_str.count("C")
    gen_gc_mean = (gc_gen_count / len(gen_seq_str)) * 100.0

    n = len(sampled_gc)
    std_error = stdev_gc / math.sqrt(n) if n > 0 and stdev_gc > 0 else 1.0
    z_score = (mean_gc - gen_gc_mean) / std_error if std_error > 0 else 0.0
    cohen_d = (mean_gc - gen_gc_mean) / stdev_gc if stdev_gc > 0 else 0.0

    return {
        "n_samples": n,
        "mean_gc": mean_gc,
        "stdev_gc": stdev_gc,
        "genome_gc_mean": gen_gc_mean,
        "z_score": z_score,
        "cohen_d": cohen_d,
    }


def write_dataset_files(records: List[Dict], out_prefix: Path) -> Tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fasta_out = out_prefix.with_suffix(".fasta")
    tsv_out = out_prefix.with_suffix(".tsv")

    with open(fasta_out, "w") as f:
        for r in records:
            f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")

    df_meta = pd.DataFrame(records)
    if "Sequence" in df_meta.columns:
        df_meta_clean = df_meta.drop(columns=["Sequence"])
    else:
        df_meta_clean = df_meta

    df_meta_clean.to_csv(tsv_out, sep="\t", index=False)
    return fasta_out, tsv_out


def report_summary(
    sampled_records: List[Dict],
    master_pool_len: int,
    exclusion_stats: Dict,
    gc_stats: Dict,
    tier: str,
    fasta_out: Path,
    tsv_out: Path,
    window_size: int = 81,
):
    n_pos = sum(1 for r in sampled_records if r["Strand"] == "+")
    n_neg = sum(1 for r in sampled_records if r["Strand"] == "-")
    total_eval = exclusion_stats.get("total_evaluated", 0)
    n_sampled = len(sampled_records)
    total_nt = n_sampled * window_size

    print("\n" + "═" * 65)
    print(" TIGR4 NEGATIVE TSS EXTRACTION SUMMARY")
    print("═" * 65)
    print(f"Target Tier Context:        {tier}")
    print(f"Total CDS K-mers Evaluated: {total_eval:,}")
    print(f"Total Clean Master Pool:    {master_pool_len:,} negative 81-mers")
    print(f"Total Sequences Extracted:  {n_sampled:,} (Balanced 1:1 Sampled)")
    print(f"K-mer Window Size (bp):     {window_size} bp")
    print(f"Total Nucleotides Sampled:   {total_nt:,} bp")
    print("─" * 65)
    print("EXTRACTION EXCLUSIONS & FILTERS:")
    print(f" • TSS Proximity Exclusions (<200 bp): {exclusion_stats.get('skipped_tss_proximity', 0):,}")
    print(f" • Short CDS Edge Exclusions (<20 bp):  {exclusion_stats.get('skipped_short_cds', 0):,}")
    print(f" • Skipped Invalid 'N' Bases:            {exclusion_stats.get('skipped_invalid_n', 0):,}")
    gc_skipped = exclusion_stats.get("skipped_gc_bias", 0)
    if gc_skipped:
        print(f" • GC Composition Exclusions:          {gc_skipped:,}")
    print("─" * 65)
    print("DATASET COMPOSITION & BIAS:")
    print(f" • Strand Distribution:     + strand: {n_pos:,} | - strand: {n_neg:,}")
    print(f" • Sampled GC Content:       {gc_stats.get('mean_gc', 0):.2f}% ± {gc_stats.get('stdev_gc', 0):.2f}%")
    print(f" • Genome Background GC:     {gc_stats.get('genome_gc_mean', 0):.2f}%")
    print(f" • GC Bias Significance:     Z-score = {gc_stats.get('z_score', 0):+.2f} | Cohen's d = {gc_stats.get('cohen_d', 0):+.2f}")
    print("═" * 65)
    print(f"[SUCCESS] FASTA dataset ➔ {fasta_out}")
    print(f"[SUCCESS] Metadata TSV ➔ {tsv_out}\n")


def main():
    args = parse_arguments()

    tier_limits = {
        "high_conf_primary": 738,
        "extended_primary": 2000,
        "all_tss": 2000,
    }

    sample_limit = args.limit if args.limit > 0 else tier_limits.get(args.tier, 738)

    chrom_id, genome_seq, seq_len = load_genome(Path(args.fasta))
    df_cds, pos_tss, neg_tss = load_cds_and_tss_tables(Path(args.xlsx), args.tier)

    master_pool, exclusion_stats = build_master_pool(
        df_cds,
        genome_seq,
        pos_tss,
        neg_tss,
        window_size=args.window,
        step_size=args.step,
        edge_margin=args.margin,
        tss_margin=args.tss_margin,
        target_gc=args.target_gc,
        gc_tolerance=args.gc_tolerance,
        dedup_rc=args.dedup_rc,
    )

    sampled_records = sample_negative_dataset(master_pool, sample_limit, args.seed)
    gc_stats = compute_gc_statistics(sampled_records, genome_seq)

    out_prefix = Path(args.output)
    fasta_out, tsv_out = write_dataset_files(sampled_records, out_prefix)

    report_summary(
        sampled_records,
        len(master_pool),
        exclusion_stats,
        gc_stats,
        args.tier,
        fasta_out,
        tsv_out,
        args.window,
    )


if __name__ == "__main__":
    main()
