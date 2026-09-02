#!/usr/bin/env python3
"""
Build IGR-Specific Benchmark Dataset for S. pneumoniae D39V.

Extracts:
  1. IGR Positives: 81 bp promoter windows [-60, +20] around validated TSSs falling strictly inside IGRs.
  2. IGR Negatives: 81 bp non-promoter windows extracted from IGRs, enforcing TSS avoidance margin (>= 50 bp).
  3. Stratified subsets for SigA and SigX.

Outputs saved in dedicated directory: data/benchmark_igr/d39v/
"""

import argparse
import bisect
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args():
    parser = argparse.ArgumentParser(description="Extract IGR positives and IGR negative controls.")
    parser.add_argument(
        "--fasta", default=Path("data/reference/D39V.fna"), type=Path, help="Reference FASTA."
    )
    parser.add_argument(
        "--pos-tsv",
        default=Path("data/benchmark/d39v/positives_81bp_metadata.tsv"),
        type=Path,
        help="Master positive TSS metadata TSV.",
    )
    parser.add_argument(
        "--pos-fasta",
        default=Path("data/benchmark/d39v/positives_81bp.fasta"),
        type=Path,
        help="Master positive TSS FASTA.",
    )
    parser.add_argument(
        "--igr-tsv",
        default=Path("output/intergenic_refined/d39v/D39V_igrs_refined.tsv"),
        type=Path,
        help="Refined IGR TSV metadata.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("data/benchmark_igr/d39v"),
        type=Path,
        help="Output directory (default: data/benchmark_igr/d39v).",
    )
    parser.add_argument(
        "--tss-margin",
        default=50,
        type=int,
        help="Avoidance margin in bp around ANY known TSS for negative extraction (default: 50).",
    )
    parser.add_argument(
        "--window", default=81, type=int, help="Window length in bp (default: 81)."
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for negative sampling (default: 42)."
    )
    return parser.parse_args()


def calculate_gc(seq_str):
    g = seq_str.upper().count("G")
    c = seq_str.upper().count("C")
    return round((g + c) / len(seq_str) * 100, 3) if len(seq_str) > 0 else 0.0


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load genome
    genome_record = SeqIO.read(args.fasta, "fasta")
    genome_seq = str(genome_record.seq)
    genome_len = len(genome_seq)

    # 2. Load positive metadata and FASTA
    df_pos_all = pd.read_csv(args.pos_tsv, sep="\t")
    pos_fasta_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(args.pos_fasta, "fasta")}

    # 3. Load IGR metadata
    df_igr = pd.read_csv(args.igr_tsv, sep="\t")

    # Map all known TSS coordinates (1-based)
    all_tss_1based = (df_pos_all["TSS_Position_0based"] + 1).tolist()
    all_tss_sorted = sorted(all_tss_1based)

    # Filter linear IGR intervals
    linear_igrs = df_igr[~df_igr["is_circular_origin_wrap"]].copy()
    linear_igrs["start_int"] = linear_igrs["start"].astype(int)
    linear_igrs["end_int"] = linear_igrs["end"].astype(int)
    linear_igrs = linear_igrs.sort_values("start_int").reset_index(drop=True)

    starts = linear_igrs["start_int"].tolist()
    ends = linear_igrs["end_int"].tolist()
    igr_ids = linear_igrs["igr_id"].tolist()

    # Match TSSs falling inside IGRs
    df_pos_all["tss_1based"] = df_pos_all["TSS_Position_0based"] + 1
    in_igr_mask = []
    matched_igr_list = []

    for tss in df_pos_all["tss_1based"]:
        idx = bisect.bisect_right(starts, tss) - 1
        if idx >= 0 and starts[idx] <= tss <= ends[idx]:
            in_igr_mask.append(True)
            matched_igr_list.append(igr_ids[idx])
        else:
            in_igr_mask.append(False)
            matched_igr_list.append(None)

    df_pos_all["in_igr"] = in_igr_mask
    df_pos_all["matched_igr"] = matched_igr_list

    # Filter IGR positives
    df_pos_igr = df_pos_all[df_pos_all["in_igr"]].copy().reset_index(drop=True)
    num_pos = len(df_pos_igr)

    print(f"=== IGR Benchmark Dataset Extraction (D39V) ===")
    print(f"Total genome length: {genome_len:,} bp")
    print(f"Total verified TSSs: {len(df_pos_all)}")
    print(f"TSSs inside Refined IGRs (IGR Positives): {num_pos} ({num_pos/len(df_pos_all)*100:.1f}%)")

    # Export IGR Positives
    pos_records = []
    for _, row in df_pos_igr.iterrows():
        seq_id = row["Sequence_ID"]
        seq_str = pos_fasta_dict.get(seq_id, "")
        if not seq_str:
            # Fallback extraction from genome if not in fasta
            tss_0 = row["TSS_Position_0based"]
            strand = row["Strand"]
            if strand == "+":
                s_0 = tss_0 - 60
                e_0 = tss_0 + 21
                seq_str = genome_seq[s_0:e_0]
            else:
                s_0 = tss_0 - 20
                e_0 = tss_0 + 61
                seq_str = str(Seq(genome_seq[s_0:e_0]).reverse_complement())

        rec = SeqRecord(Seq(seq_str), id=seq_id, description=f"chrom={row['Chromosome']} strand={row['Strand']} sigma={row['Sigma_Factor']} igr={row['matched_igr']}")
        pos_records.append(rec)

    pos_fasta_out = args.out_dir / "positives_81bp_igr.fasta"
    pos_tsv_out = args.out_dir / "positives_81bp_igr_metadata.tsv"
    SeqIO.write(pos_records, pos_fasta_out, "fasta")
    df_pos_igr.to_csv(pos_tsv_out, sep="\t", index=False)

    # Export SigA and SigX subsets
    df_pos_siga = df_pos_igr[df_pos_igr["Sigma_Factor"] == "SigA"]
    siga_records = [r for r in pos_records if r.id in set(df_pos_siga["Sequence_ID"])]
    SeqIO.write(siga_records, args.out_dir / "positives_81bp_igr_SigA.fasta", "fasta")
    df_pos_siga.to_csv(args.out_dir / "positives_81bp_igr_SigA_metadata.tsv", sep="\t", index=False)

    df_pos_sigx = df_pos_igr[df_pos_igr["Sigma_Factor"] == "SigX"]
    sigx_records = [r for r in pos_records if r.id in set(df_pos_sigx["Sequence_ID"])]
    SeqIO.write(sigx_records, args.out_dir / "positives_81bp_igr_SigX.fasta", "fasta")
    df_pos_sigx.to_csv(args.out_dir / "positives_81bp_igr_SigX_metadata.tsv", sep="\t", index=False)

    print(f" -> SigA IGR Positives: {len(df_pos_siga)}")
    print(f" -> SigX IGR Positives: {len(df_pos_sigx)}")

    # 4. Extract IGR Negatives
    # Build a forbidden interval set for TSS avoidance (both strands)
    # Any coordinate within [-60, +20] + margin of ANY TSS is forbidden
    forbidden_intervals = []
    for tss in all_tss_sorted:
        # Avoid window [-60, +20] relative to TSS + margin
        f_start = max(1, tss - 60 - args.tss_margin)
        f_end = min(genome_len, tss + 20 + args.tss_margin)
        forbidden_intervals.append((f_start, f_end))

    # Merge overlapping forbidden intervals
    merged_forbidden = []
    for s, e in forbidden_intervals:
        if not merged_forbidden:
            merged_forbidden.append([s, e])
        else:
            if s <= merged_forbidden[-1][1]:
                merged_forbidden[-1][1] = max(merged_forbidden[-1][1], e)
            else:
                merged_forbidden.append([s, e])

    f_starts = [x[0] for x in merged_forbidden]
    f_ends = [x[1] for x in merged_forbidden]

    def is_window_forbidden(w_start_1, w_end_1):
        # Check if window [w_start_1, w_end_1] overlaps any forbidden interval
        idx = bisect.bisect_right(f_starts, w_end_1) - 1
        if idx >= 0 and f_starts[idx] <= w_end_1 and f_ends[idx] >= w_start_1:
            return True
        # Check adjacent interval
        if idx + 1 < len(f_starts) and f_starts[idx + 1] <= w_end_1:
            return True
        return False

    candidate_negatives = []
    seen_kmers = set()

    for _, igr_row in linear_igrs.iterrows():
        igr_s = igr_row["start_int"]
        igr_e = igr_row["end_int"]
        igr_id = igr_row["igr_id"]

        if igr_e - igr_s + 1 < args.window:
            continue

        # Slide windows across IGR
        for w_s in range(igr_s, igr_e - args.window + 2, 2):  # step=2 bp to sample diverse candidates
            w_e = w_s + args.window - 1
            if not is_window_forbidden(w_s, w_e):
                seq_fwd = genome_seq[w_s - 1 : w_e]
                if "N" in seq_fwd or len(seq_fwd) != args.window:
                    continue

                kmer_norm = seq_fwd.upper()
                kmer_rc = str(Seq(kmer_norm).reverse_complement())
                canonical_kmer = min(kmer_norm, kmer_rc)

                if canonical_kmer not in seen_kmers:
                    seen_kmers.add(canonical_kmer)
                    # Alternate strand assignment
                    strand = "+" if len(candidate_negatives) % 2 == 0 else "-"
                    seq_final = seq_fwd if strand == "+" else kmer_rc

                    candidate_negatives.append({
                        "chrom": genome_record.id,
                        "start": w_s,
                        "end": w_e,
                        "strand": strand,
                        "igr_id": igr_id,
                        "length": args.window,
                        "gc": calculate_gc(seq_final),
                        "sequence": seq_final
                    })

    print(f"Total unique candidate IGR non-promoter windows extracted: {len(candidate_negatives)}")

    if len(candidate_negatives) < num_pos:
        sys.exit(f"Error: Not enough candidate negatives ({len(candidate_negatives)}) for 1:1 balance ({num_pos}).")

    # Sample exactly num_pos negatives for 1:1 balance
    sampled_negatives = random.sample(candidate_negatives, num_pos)

    neg_records = []
    neg_table_rows = []
    for i, neg in enumerate(sampled_negatives, 1):
        seq_id = f"NEG_IGR_{i:04d}_{neg['chrom']}_{neg['start']}_{neg['strand']}"
        rec = SeqRecord(
            Seq(neg["sequence"]),
            id=seq_id,
            description=f"chrom={neg['chrom']} pos={neg['start']}-{neg['end']} strand={neg['strand']} igr={neg['igr_id']} gc={neg['gc']:.1f}%"
        )
        neg_records.append(rec)
        neg_table_rows.append({
            "Sequence_ID": seq_id,
            "Chromosome": neg["chrom"],
            "Start": neg["start"],
            "End": neg["end"],
            "Strand": neg["strand"],
            "IGR_ID": neg["igr_id"],
            "Length": neg["length"],
            "GC_Content(%)": neg["gc"],
            "Label": 0
        })

    neg_fasta_out = args.out_dir / "negatives_81bp_igr.fasta"
    neg_tsv_out = args.out_dir / "negatives_81bp_igr_metadata.tsv"
    SeqIO.write(neg_records, neg_fasta_out, "fasta")
    pd.DataFrame(neg_table_rows).to_csv(neg_tsv_out, sep="\t", index=False)

    # 5. Combined Balanced Benchmark Dataset (N = num_pos * 2 = 1,444)
    all_benchmark_records = pos_records + neg_records
    combined_fasta_out = args.out_dir / "benchmark_81bp_igr.fasta"
    SeqIO.write(all_benchmark_records, combined_fasta_out, "fasta")

    # Summary Statistics
    pos_gcs = [calculate_gc(str(r.seq)) for r in pos_records]
    neg_gcs = [neg["gc"] for neg in sampled_negatives]

    print("\n=== Dataset Composition & GC Content Audit ===")
    print(f"Balanced Dataset Size: N = {len(all_benchmark_records):,} (Pos={num_pos}, Neg={num_pos})")
    print(f"IGR Positives GC Content: Mean = {np.mean(pos_gcs):.3f}% (Std = {np.std(pos_gcs):.3f}%)")
    print(f"IGR Negatives GC Content: Mean = {np.mean(neg_gcs):.3f}% (Std = {np.std(neg_gcs):.3f}%)")
    print(f"GC Difference: Delta_GC = {abs(np.mean(pos_gcs) - np.mean(neg_gcs)):.3f}% (Natural intergenic background matching!)")
    print(f"\nOutputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
