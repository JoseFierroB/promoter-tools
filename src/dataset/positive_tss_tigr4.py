#!/usr/bin/env python3
"""
TIGR4 Positive TSS Sequence Extractor.

Parses experimental TSS annotations from S1_TSS.xlsx (Aprianto et al., 2018)
against the TIGR4 genome (NC_003028.fasta).

Features:
- Multi-tier dataset generation ('high_conf_primary', 'extended_primary', 'all_tss').
- Extraction efficiency metrics (Theoretical maximum TSS candidates vs extracted 81-mers).
- Same-strand steric hindrance conflict resolution.
- Advanced GC bias statistics (Z-score, Cohen's d).
- Biological validation metrics (+1 Purine preference, -10 box match, UTR spacing).

Usage:
    pixi run python src/dataset/positive_tss_tigr4.py \
      --xlsx data/tigr4/S1_TSS.xlsx \
      --fasta data/reference/NC_003028.fasta \
      --tier high_conf_primary \
      -u 60 -d 20 \
      -o data/tigr4/positives_high_81bp
"""

import argparse
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


# ════════════════════════════════════════════════════════════════
# 1. CLI Arguments & Configuration
# ════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract positive promoter windows centered on TIGR4 TSS."
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
        default="data/tigr4/positives_high_81bp",
        help="Output prefix (generates .fasta and .tsv).",
    )
    parser.add_argument(
        "--tier",
        choices=["high_conf_primary", "extended_high", "extended_primary", "all_tss"],
        default="high_conf_primary",
        help="Dataset tier: 'high_conf_primary' (742), 'extended_high' (742+48 sec), 'extended_primary' (2028), 'all_tss' (2150).",
    )
    parser.add_argument(
        "-u",
        "--upstream",
        type=int,
        default=60,
        help="Upstream bp from TSS (default: 60).",
    )
    parser.add_argument(
        "-d",
        "--downstream",
        type=int,
        default=20,
        help="Downstream bp from TSS (default: 20).",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=int,
        default=25,
        help="Distance threshold (bp) to resolve same-strand steric hindrance conflicts (default: 25).",
    )

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
# 2. Genome Loading & Validation
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 3. Excel Sheet Filtering & Loading
# ════════════════════════════════════════════════════════════════

def load_tigr4_tss_table(xlsx_path: Path, tier: str) -> Tuple[pd.DataFrame, List[str]]:
    if not xlsx_path.exists():
        print(f"[ERROR] Excel file not found at {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    tier_sheet_map = {
        "high_conf_primary": ["High Confidence (TSS_100.4)"],
        "extended_high": ["High Confidence (TSS_100.4)", "Secondary TSS, High confidence"],
        "extended_primary": ["High Confidence (TSS_100.4)", "Low Confidence (TSS_2.1)"],
        "all_tss": [
            "High Confidence (TSS_100.4)",
            "Secondary TSS, High confidence",
            "Low Confidence (TSS_2.1)",
            "Secondary TSS, Low confidence",
        ],
    }

    target_sheets = tier_sheet_map.get(tier, tier_sheet_map["high_conf_primary"])
    xl = pd.ExcelFile(xlsx_path)

    dfs = []
    for sheet in target_sheets:
        if sheet in xl.sheet_names:
            df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet)
            df_sheet["_Source_Sheet"] = sheet
            dfs.append(df_sheet)

    if not dfs:
        print(f"[ERROR] No matching sheets found for tier '{tier}' in {xlsx_path}", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Tier '{tier}': Loaded {len(combined_df):,} TSS entries across {len(dfs)} sheet(s)")
    return combined_df, target_sheets


# ════════════════════════════════════════════════════════════════
# 4. Biological Window Extraction
# ════════════════════════════════════════════════════════════════

def extract_promoter_windows(
    df_tss: pd.DataFrame,
    genome_seq: Seq,
    chrom_id: str,
    upstream: int,
    downstream: int,
) -> Tuple[List[Dict], Dict[str, int]]:
    extracted_records = []
    seq_len = len(genome_seq)
    window_size = upstream + 1 + downstream

    exclusion_stats = {
        "total_evaluated": len(df_tss),
        "skipped_boundary": 0,
        "skipped_invalid_n": 0,
        "skipped_missing_tss": 0,
    }

    for idx, row in df_tss.iterrows():
        locus = str(row.get("Locus_tag", row.get("Locus", f"TIGR4_TSS_{idx}"))).strip()
        strand = str(row.get("Strand", "+")).strip()

        tss_col = (
            "TSS_position"
            if "TSS_position" in row and pd.notna(row["TSS_position"])
            else ("Secondary_TSS" if "Secondary_TSS" in row and pd.notna(row["Secondary_TSS"])
            else ("Primary_TSS" if "Primary_TSS" in row and pd.notna(row["Primary_TSS"])
            else "TSS"))
        )
        if tss_col not in row or pd.isna(row[tss_col]):
            exclusion_stats["skipped_missing_tss"] += 1
            continue

        tss_pos = int(row[tss_col])
        pos_0 = tss_pos - 1

        if strand == "+":
            start_idx = pos_0 - upstream
            end_idx = pos_0 + downstream + 1
            if start_idx < 0 or end_idx > seq_len:
                exclusion_stats["skipped_boundary"] += 1
                continue
            kmer = str(genome_seq[start_idx:end_idx]).upper()
        elif strand == "-":
            start_idx = pos_0 - downstream
            end_idx = pos_0 + upstream + 1
            if start_idx < 0 or end_idx > seq_len:
                exclusion_stats["skipped_boundary"] += 1
                continue
            kmer = str(genome_seq[start_idx:end_idx].reverse_complement()).upper()
        else:
            exclusion_stats["skipped_missing_tss"] += 1
            continue

        if len(kmer) != window_size or "N" in kmer:
            exclusion_stats["skipped_invalid_n"] += 1
            continue

        gc_count = kmer.count("G") + kmer.count("C")
        gc_content = (gc_count / window_size) * 100.0

        utr_len = row.get("5'-UTR_length", row.get("Primary_5'-UTR_length", None))
        loc_type = row.get("Within_coding_vs_intergenic", "intergenic")
        sheet_source = row.get("_Source_Sheet", "Unknown")

        seq_id = f"TIGR4_{locus}_TSS_{tss_pos}_{strand}"

        extracted_records.append({
            "Sequence_ID": seq_id,
            "Locus_Tag": locus,
            "Chromosome": chrom_id,
            "TSS_Position": tss_pos,
            "Strand": strand,
            "Upstream_bp": upstream,
            "Downstream_bp": downstream,
            "Window_Size": window_size,
            "Sequence": kmer,
            "GC_Content": round(gc_content, 2),
            "UTR5_Length": utr_len if pd.notna(utr_len) else "NA",
            "Location_Type": loc_type,
            "Confidence_Sheet": sheet_source,
            "Processed_Coverage": row.get("Processed_coverage", row.get("Primary_Processed_cov", "NA")),
            "Processed_Ratio": row.get("Processed/unprocessed ratio", row.get("Primary_Ratio", "NA")),
        })

    return extracted_records, exclusion_stats


# ════════════════════════════════════════════════════════════════
# 5. Steric Hindrance Conflict Resolution
# ════════════════════════════════════════════════════════════════

def resolve_steric_conflicts(records: List[Dict], threshold: int) -> Tuple[List[Dict], int]:
    if threshold <= 0 or not records:
        return records, 0

    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in records:
        key = (r["Chromosome"], r["Strand"])
        groups.setdefault(key, []).append(r)

    resolved = []
    discarded_count = 0

    for key, item_list in groups.items():
        sorted_items = sorted(item_list, key=lambda x: x["TSS_Position"])
        current_cluster = []

        for item in sorted_items:
            if not current_cluster:
                current_cluster.append(item)
            else:
                prev_pos = current_cluster[-1]["TSS_Position"]
                if item["TSS_Position"] - prev_pos < threshold:
                    current_cluster.append(item)
                else:
                    best = _pick_best_tss_in_cluster(current_cluster)
                    resolved.append(best)
                    discarded_count += len(current_cluster) - 1
                    current_cluster = [item]

        if current_cluster:
            best = _pick_best_tss_in_cluster(current_cluster)
            resolved.append(best)
            discarded_count += len(current_cluster) - 1

    return resolved, discarded_count


def _pick_best_tss_in_cluster(cluster: List[Dict]) -> Dict:
    def rank_key(item):
        sheet = item.get("Confidence_Sheet", "")
        sheet_score = 0
        if "High Confidence" in sheet:
            sheet_score = 3
        elif "Secondary TSS, High" in sheet:
            sheet_score = 2
        elif "Low Confidence" in sheet:
            sheet_score = 1

        try:
            cov = float(item.get("Processed_Coverage", 0))
        except (ValueError, TypeError):
            cov = 0.0

        return (sheet_score, cov)

    cluster.sort(key=rank_key, reverse=True)
    return cluster[0]


# ════════════════════════════════════════════════════════════════
# 6. Advanced GC Bias & Biological Validation Metrics
# ════════════════════════════════════════════════════════════════

def compute_gc_statistics(records: List[Dict], genome_seq: Seq, upstream: int = 60) -> Dict:
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

    plus1_bases = [r["Sequence"][upstream] for r in records if len(r["Sequence"]) > upstream]
    n_plus1 = len(plus1_bases)
    n_a = plus1_bases.count("A")
    n_g = plus1_bases.count("G")
    n_c = plus1_bases.count("C")
    n_t = plus1_bases.count("T")
    purines_pct = ((n_a + n_g) / n_plus1 * 100.0) if n_plus1 > 0 else 0.0

    pribnow_matches = 0
    pribnow_exact = 0
    for r in records:
        seq = r["Sequence"]
        if len(seq) == upstream + 1 + 20:
            window = seq[40:57]
            if "TATAAT" in window:
                pribnow_exact += 1
                pribnow_matches += 1
            elif re.search(r"TA[ATGC]{3}T", window) or re.search(r"TA[ATGC]{2}AT", window):
                pribnow_matches += 1

    pribnow_pct = (pribnow_matches / n * 100.0) if n > 0 else 0.0
    pribnow_exact_pct = (pribnow_exact / n * 100.0) if n > 0 else 0.0

    valid_utrs = []
    for r in records:
        u_len = r.get("UTR5_Length")
        try:
            u_num = float(u_len)
            if not math.isnan(u_num):
                valid_utrs.append(u_num)
        except (ValueError, TypeError):
            continue

    n_utrs = len(valid_utrs)
    canonical_utrs = [u for u in valid_utrs if 15 <= u <= 45]
    canonical_utr_pct = (len(canonical_utrs) / n_utrs * 100.0) if n_utrs > 0 else 0.0

    return {
        "n_samples": n,
        "mean_gc": mean_gc,
        "stdev_gc": stdev_gc,
        "genome_gc_mean": gen_gc_mean,
        "z_score": z_score,
        "cohen_d": cohen_d,
        "plus1_purines_pct": purines_pct,
        "plus1_a_pct": (n_a / n_plus1 * 100.0) if n_plus1 > 0 else 0.0,
        "plus1_g_pct": (n_g / n_plus1 * 100.0) if n_plus1 > 0 else 0.0,
        "pribnow_pct": pribnow_pct,
        "pribnow_exact_pct": pribnow_exact_pct,
        "canonical_utr_pct": canonical_utr_pct,
        "n_utrs": n_utrs,
    }


# ════════════════════════════════════════════════════════════════
# 7. Dataset Exporters (.fasta & .tsv)
# ════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════
# 8. Executive Console Summary
# ════════════════════════════════════════════════════════════════

def report_summary(
    records: List[Dict],
    stats: Dict,
    exclusion_stats: Dict,
    conflict_count: int,
    tier: str,
    fasta_out: Path,
    tsv_out: Path,
    upstream: int = 60,
    downstream: int = 20,
):
    n_pos = sum(1 for r in records if r["Strand"] == "+")
    n_neg = sum(1 for r in records if r["Strand"] == "-")
    total_eval = exclusion_stats.get("total_evaluated", len(records))
    n_extracted = stats.get("n_samples", len(records))
    yield_pct = (n_extracted / total_eval * 100.0) if total_eval > 0 else 0.0

    kmer_len = upstream + 1 + downstream
    total_nt = n_extracted * kmer_len

    print("\n" + "═" * 65)
    print(" TIGR4 POSITIVE TSS EXTRACTION SUMMARY")
    print("═" * 65)
    print(f"Dataset Tier:               {tier}")
    print(f"Total TSS Candidates Evaluated: {total_eval:,}")
    print(f"Total 81-mer Sequences Extracted: {n_extracted:,} (Efficiency: {yield_pct:.1f}%)")
    print(f"K-mer Window Size (bp):     {kmer_len} bp (-{upstream} to +{downstream} relative to TSS)")
    print(f"Total Nucleotides Sampled:   {total_nt:,} bp")
    print("─" * 65)
    print("EXTRACTION EXCLUSIONS & RESOLUTIONS:")
    print(f" • Steric Conflicts Discarded (<25 bp): {conflict_count:,}")
    print(f" • Skipped Boundary Coordinates:         {exclusion_stats.get('skipped_boundary', 0):,}")
    print(f" • Skipped Invalid 'N' Bases:            {exclusion_stats.get('skipped_invalid_n', 0):,}")
    print("─" * 65)
    print("DATASET COMPOSITION & BIAS:")
    print(f" • Strand Distribution:     + strand: {n_pos:,} | - strand: {n_neg:,}")
    print(f" • Sampled GC Content:       {stats.get('mean_gc', 0):.2f}% ± {stats.get('stdev_gc', 0):.2f}%")
    print(f" • Genome Background GC:     {stats.get('genome_gc_mean', 0):.2f}%")
    print(f" • GC Bias Significance:     Z-score = {stats.get('z_score', 0):+.2f} | Cohen's d = {stats.get('cohen_d', 0):+.2f}")
    print("─" * 65)
    print("BIOLOGICAL VALIDATION METRICS:")
    print(f" • +1 Initiator Purines (A+G): {stats.get('plus1_purines_pct', 0):.1f}% (A: {stats.get('plus1_a_pct', 0):.1f}%, G: {stats.get('plus1_g_pct', 0):.1f}%)")
    print(f" • -10 Box Variant Match:       {stats.get('pribnow_pct', 0):.1f}% (Exact TATAAT: {stats.get('pribnow_exact_pct', 0):.1f}%)")
    print(f" • Canonical 5'-UTR (15-45bp): {stats.get('canonical_utr_pct', 0):.1f}% of evaluated UTRs")
    print("═" * 65)
    print(f"[SUCCESS] FASTA dataset ➔ {fasta_out}")
    print(f"[SUCCESS] Metadata TSV ➔ {tsv_out}\n")


# ════════════════════════════════════════════════════════════════
# 9. Main Pipeline Orchestrator
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()

    chrom_id, genome_seq, seq_len = load_genome(Path(args.fasta))
    df_tss, target_sheets = load_tigr4_tss_table(Path(args.xlsx), args.tier)

    raw_records, exclusion_stats = extract_promoter_windows(
        df_tss, genome_seq, chrom_id, args.upstream, args.downstream
    )

    resolved_records, conflict_count = resolve_steric_conflicts(
        raw_records, args.conflict_threshold
    )

    gc_stats = compute_gc_statistics(resolved_records, genome_seq, args.upstream)

    out_prefix = Path(args.output)
    fasta_out, tsv_out = write_dataset_files(resolved_records, out_prefix)

    report_summary(
        resolved_records, gc_stats, exclusion_stats, conflict_count, args.tier, fasta_out, tsv_out, args.upstream, args.downstream
    )


if __name__ == "__main__":
    main()
