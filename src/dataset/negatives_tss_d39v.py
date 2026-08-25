#!/usr/bin/env python3
"""
Negative Sequence Extractor (Master Pool Version - Victor's Approach)
Pre-filters all possible CDS k-mers across the entire genome first, 
generating a complete "master pool" of clean negatives, and then 
randomly samples exactly N sequences from it.
"""

import argparse
import sys
import random
import bisect
import csv
import re
import statistics
from typing import Dict, List
from Bio import SeqIO
from BCBio import GFF
from collections import Counter

# Hardcoded chromosome mappings (e.g. NCBI RefSeq accession to friendly/custom name)

# Precompiled translation table for fast reverse complement computation
_RC_TRANS = str.maketrans('ACGT', 'TGCA')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract negative sequence windows from CDS using a pre-filtered Master Pool approach.")
    parser.add_argument("--gff-cds", required=True, help="Path to the structural GFF3 annotation (NCBI/CDS).")
    parser.add_argument("--fasta", required=True, help="Path to the genome FASTA file.")
    parser.add_argument("--gff-tss", required=False, default=None, help="(Optional) Path to the TSS GFF3 to exclude.")
    parser.add_argument("--dedup-rc", action="store_true", help="Erase or deduplicate reverse complement of extracted windows.")
    parser.add_argument("-o", "--output", default="negative_dataset_master", help="Output prefix (generates .fasta and .tsv).")
    parser.add_argument("-w", "--window", type=int, default=81, help="Window size/k-mer length (default: 81).")
    parser.add_argument("-s", "--step", type=int, default=10, help="Step size for extraction (default: 10).")
    parser.add_argument("-m", "--margin", type=int, default=20, help="Safety margin from CDS edges (default: 20).")
    parser.add_argument("--tss-margin", type=int, default=200, help="Minimum distance allowed to any TSS (default: 200).")
    parser.add_argument("--conflict-threshold", type=int, default=25, help="Distance threshold (bp) to flag same-strand TSS conflicts (default: 25).")
    parser.add_argument("--target-gc", type=float, default=None, help="Target GC content percentage to match the positive promoter collection.")
    parser.add_argument("--gc-tolerance", type=float, default=5.0, help="GC content tolerance percentage when --target-gc is specified (default: 5.0).")
    parser.add_argument("--limit", type=int, default=0, help="Max sequences to extract (0 = no limit / return all clean negatives).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling reproducibility (default: 42).")
    return parser.parse_args()

def get_all_tss(features):
    for f in features:
        is_tss = False
        if f.type == 'transcription_start_site':
            is_tss = True
        elif f.type in ('sequence_feature', 'misc_feature', 'regulatory'):
            for key, val_list in f.qualifiers.items():
                val_str = " ".join(val_list).lower()
                if "transcription start site" in val_str or "tss" in val_str:
                    is_tss = True
                    break
        if is_tss:
            yield f
        sub_feats = getattr(f, 'sub_features', None) or getattr(f, 'features', [])
        if sub_feats:
            yield from get_all_tss(sub_feats)

def load_tss_positions(tss_gff: str, use_fallback_if_empty: bool = True) -> Dict[str, List[dict]]:
    tss_data = {}
    if not tss_gff:
        return tss_data
    
    print(f"[INFO] Loading TSS coordinates from {tss_gff}...", file=sys.stderr)
    try:
        parsed_records = []
        with open(tss_gff) as f:
            for rec in GFF.parse(f):
                parsed_records.append(rec)
                
        tss_count = 0
        for rec in parsed_records:
            for feat in get_all_tss(rec.features):
                tss_count += 1
                
        use_fallback = (tss_count == 0) and use_fallback_if_empty
        for rec in parsed_records:
            chrom = rec.id
            if chrom not in tss_data:
                tss_data[chrom] = []
            features_to_load = rec.features if use_fallback else get_all_tss(rec.features)
            for feat in features_to_load:
                strand = '+' if feat.location.strand == 1 else '-'
                pos = int(feat.location.end) - 1 if strand == '-' else int(feat.location.start)
                score_raw = feat.qualifiers.get('score', ['0'])[0]
                score = int(score_raw) if score_raw.isdigit() else 0
                tss_data[chrom].append({
                    "pos": pos,
                    "strand": strand,
                    "score": score,
                    "id": feat.id
                })
        for chrom in tss_data:
            tss_data[chrom].sort(key=lambda x: x["pos"])
    except Exception as e:
        print(f"[WARNING] Could not parse TSS GFF file: {e}", file=sys.stderr)
    return tss_data

def get_nearest_tss_distance(chrom: str, pos: int, tss_data: Dict[str, List[dict]], chrom_map: Dict[str, str] = None) -> int:
    resolved_chrom = chrom
    if resolved_chrom not in tss_data or not tss_data[resolved_chrom]:
        return float('inf')
    positions = [t["pos"] for t in tss_data[resolved_chrom]]
    idx = bisect.bisect_left(positions, pos)
    distances = []
    if idx < len(positions):
        distances.append(abs(positions[idx] - pos))
    if idx > 0:
        distances.append(abs(pos - positions[idx - 1]))
    return min(distances) if distances else float('inf')

def get_all_cds(features):
    for f in features:
        if f.type == 'CDS':
            yield f
        sub_feats = getattr(f, 'sub_features', None) or getattr(f, 'features', [])
        if sub_feats:
            yield from get_all_cds(sub_feats)

def calculate_gc(seq_str: str) -> float:
    g = seq_str.count('G')
    c = seq_str.count('C')
    return ((g + c) / len(seq_str) * 100) if len(seq_str) > 0 else 0.0

def filter_tss_conflicts(tss_data: Dict[str, List[dict]], cds_data: Dict[str, List[tuple]], chrom_map: Dict, conflict_threshold=25) -> Dict[str, List[dict]]:
    filtered_tss_data = {}
    for chrom, tss_list in tss_data.items():
        if not tss_list:
            filtered_tss_data[chrom] = []
            continue
        by_strand = {'+': [], '-': []}
        for t in tss_list:
            by_strand[t['strand']].append(t)
        valid_tss = []
        resolved_cds_chrom = chrom
        cds_list = cds_data.get(resolved_cds_chrom, [])
        
        def get_dist_to_closest_cds(pos, strand):
            valid_cds_dists = [
                abs(pos - c_start) if strand == '+' else abs(pos - c_end)
                for c_start, c_end, c_strand, _ in cds_list
                if c_strand == strand
            ]
            return min(valid_cds_dists) if valid_cds_dists else float('inf')

        for strand, tss_sublist in by_strand.items():
            if not tss_sublist:
                continue
            tss_sublist.sort(key=lambda x: x['pos'])
            clusters = []
            current_cluster = []
            for t in tss_sublist:
                if not current_cluster:
                    current_cluster.append(t)
                else:
                    if t['pos'] - current_cluster[-1]['pos'] < conflict_threshold:
                        current_cluster.append(t)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [t]
            if current_cluster:
                clusters.append(current_cluster)
            for cluster in clusters:
                if len(cluster) == 1:
                    valid_tss.append(cluster[0])
                else:
                    best_t = min(cluster, key=lambda x: (
                        get_dist_to_closest_cds(x['pos'], x['strand']),
                        -x['score'],
                        x['pos']
                    ))
                    valid_tss.append(best_t)
        valid_tss.sort(key=lambda x: x['pos'])
        filtered_tss_data[chrom] = valid_tss
    return filtered_tss_data

def has_tss_inside(chrom: str, start: int, end: int, strand: str, tss_data: Dict, chrom_map: Dict = None) -> bool:
    resolved_chrom = chrom
    if resolved_chrom not in tss_data or not tss_data[resolved_chrom]:
        return False
    positions = [t["pos"] for t in tss_data[resolved_chrom]]
    idx_start = bisect.bisect_left(positions, start)
    idx_end = bisect.bisect_left(positions, end)
    if idx_end > idx_start:
        return True
    return False

def extract_negatives(args: argparse.Namespace = None) -> None:
    if args is None:
        args = parse_arguments()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"[INFO] Loading genome from {args.fasta}...", file=sys.stderr)
    genome = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))
    if not genome:
        sys.exit("[ERROR] FASTA file is empty or invalid.")


    cds_data = {}
    print(f"[INFO] Parsing CDS GFF3 from {args.gff_cds}...", file=sys.stderr)
    with open(args.gff_cds) as f:
        for rec in GFF.parse(f):
            chrom = rec.id
            if chrom not in cds_data:
                cds_data[chrom] = []
            for feature in get_all_cds(rec.features):
                start = int(feature.location.start)
                end = int(feature.location.end)
                strand_char = '+' if feature.location.strand == 1 else '-'
                locus = feature.qualifiers.get('locus_tag', feature.qualifiers.get('Name', ['unknown_cds']))[0]
                cds_data[chrom].append((start, end, strand_char, locus))

    raw_tss_data = load_tss_positions(args.gff_tss, use_fallback_if_empty=True)
    chrom_map = {}  # not needed: all data uses the same chromosome ID
    tss_data = filter_tss_conflicts(raw_tss_data, cds_data, chrom_map, args.conflict_threshold)

    valid_bases = {'A', 'C', 'G', 'T'}
    stats = Counter()
    master_pool: Dict[str, dict] = {}
    total_evaluated = 0

    print("[INFO] Phase 1: Scanning entire genome to build Master Pool of clean negatives...", file=sys.stderr)

    for chrom, cds_list in cds_data.items():
        resolved_genome_chrom = chrom
        if resolved_genome_chrom not in genome:
            continue
            
        for start, end, strand_char, locus in cds_list:
            cds_seq = genome[resolved_genome_chrom].seq[start:end]
            cds_str = str(cds_seq).upper()
            stats["total_cds_bases_A"] += cds_str.count('A')
            stats["total_cds_bases_C"] += cds_str.count('C')
            stats["total_cds_bases_G"] += cds_str.count('G')
            stats["total_cds_bases_T"] += cds_str.count('T')

            safe_start = start + args.margin
            safe_end = end - args.margin

            if safe_end - safe_start < args.window:
                stats["skipped_too_short_cds"] += 1
                continue

            cds_len = end - start
            n_kmers = max(0, (cds_len - 2 * args.margin - args.window) // args.step + 1)
            stats["theoretical_max_kmers"] += n_kmers

            for pos in range(safe_start, safe_end - args.window + 1, args.step):
                total_evaluated += 1
                
                # Filter 1: TSS Proximity
                window_center = pos + (args.window // 2)
                dist_to_tss = get_nearest_tss_distance(chrom, window_center, tss_data, chrom_map)
                if dist_to_tss <= args.tss_margin:
                    stats["skipped_tss"] += 1
                    continue

                # Filter 1.5: TSS inside window
                if has_tss_inside(chrom, pos, pos + args.window, strand_char, tss_data, chrom_map):
                    stats["skipped_tss_inside"] += 1
                    continue

                # Extract sequence
                subseq = genome[resolved_genome_chrom].seq[pos:pos + args.window]
                if strand_char == '-':
                    subseq = subseq.reverse_complement()
                seq_str = str(subseq).upper()

                # Filter 2: Ambiguous bases
                if not set(seq_str).issubset(valid_bases):
                    stats["skipped_invalid"] += 1
                    continue

                # Filter 2.5: GC composition
                gc_content = calculate_gc(seq_str)
                if args.target_gc is not None:
                    if not (args.target_gc - args.gc_tolerance <= gc_content <= args.target_gc + args.gc_tolerance):
                        stats["skipped_gc_bias"] += 1
                        continue

                # Filter 3: Redundancy in the master pool
                if seq_str in master_pool:
                    stats["skipped_repeats"] += 1
                    continue
                if args.dedup_rc:
                    seq_rc = seq_str.translate(_RC_TRANS)[::-1]
                    if seq_rc in master_pool:
                        stats["skipped_repeats"] += 1
                        continue

                # Add to Master Pool
                header = f"NEG_{locus}_{chrom}_{pos}_{strand_char}"
                master_pool[seq_str] = {
                    "id": header,
                    "chrom": chrom,
                    "pos": pos,
                    "strand": strand_char,
                    "locus": locus,
                    "gc_content": round(gc_content, 2),
                    "dist_to_tss": "No_TSS_Provided" if not tss_data else (round(dist_to_tss) if dist_to_tss != float('inf') else "Infinity")
                }

    total_clean_negatives = len(master_pool)
    print(f"[INFO] Phase 1 Finished. Total clean negatives in Master Pool: {total_clean_negatives}", file=sys.stderr)

    # Phase 2: Sampling from Master Pool
    sampled_pool = {}
    if args.limit > 0 and total_clean_negatives > args.limit:
        print(f"[INFO] Phase 2: Randomly sampling {args.limit} sequences from Master Pool...", file=sys.stderr)
        sampled_keys = random.sample(list(master_pool.keys()), args.limit)
        for key in sampled_keys:
            sampled_pool[key] = master_pool[key]
    else:
        print(f"[INFO] Phase 2: Keeping all {total_clean_negatives} clean negatives (no sampling/limit set).", file=sys.stderr)
        sampled_pool = master_pool

    # Output definitions
    fasta_out = f"{args.output}.fasta"
    tsv_out = f"{args.output}_metadata.tsv"

    # Write FASTA dataset
    with open(fasta_out, 'w') as f:
        for seq, meta in sampled_pool.items():
            f.write(f">{meta['id']}\n{seq}\n")

    # Write Metadata TSV
    with open(tsv_out, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(["Sequence_ID", "Chromosome", "Position", "Strand", "Associated_CDS", "GC_Content(%)", "Distance_to_Nearest_TSS(bp)"])
        for meta in sampled_pool.values():
            writer.writerow([meta["id"], meta["chrom"], meta["pos"], meta["strand"], meta["locus"], meta["gc_content"], meta["dist_to_tss"]])

    # Collect statistics for output pool
    saved_count = len(sampled_pool)
    base_A = base_C = base_G = base_T = 0
    strand_pos = strand_neg = 0
    gc_values = []
    
    for seq, meta in sampled_pool.items():
        base_A += seq.count('A')
        base_C += seq.count('C')
        base_G += seq.count('G')
        base_T += seq.count('T')
        if meta["strand"] == '+':
            strand_pos += 1
        else:
            strand_neg += 1
        gc_values.append(meta["gc_content"])

    total_bases = base_A + base_C + base_G + base_T
    global_gc = ((base_G + base_C) / total_bases * 100) if total_bases > 0 else 0
    mean_gc = statistics.mean(gc_values) if gc_values else 0.0
    stdev_gc = statistics.stdev(gc_values) if len(gc_values) > 1 else 0.0

    total_cds_bases = stats["total_cds_bases_A"] + stats["total_cds_bases_C"] + stats["total_cds_bases_G"] + stats["total_cds_bases_T"]
    global_cds_gc = ((stats["total_cds_bases_G"] + stats["total_cds_bases_C"]) / total_cds_bases * 100) if total_cds_bases > 0 else 0.0

    print("\n" + "="*50, file=sys.stderr)
    print("MASTER POOL EXPLORATION RESULTS:", file=sys.stderr)
    print(f"Theoretical maximum k-mers:           {stats['theoretical_max_kmers']}", file=sys.stderr)
    print(f"Total possible coordinates evaluated: {total_evaluated}", file=sys.stderr)
    print(f"Total clean negatives in master pool: {total_clean_negatives}", file=sys.stderr)
    print(f"Unique sequences saved (Sampled):     {saved_count}", file=sys.stderr)


    print("\nLOST K-MERS (Extraction Exclusions):", file=sys.stderr)
    print(f"Due to CDS length constraints:        {stats['skipped_too_short_cds']}", file=sys.stderr)
    print(f"Due to invalid 'N' bases:             {stats['skipped_invalid']}", file=sys.stderr)
    print(f"Due to identical/RC repeats:          {stats['skipped_repeats']}", file=sys.stderr)
    print(f"Due to TSS proximity (Conflict):      {stats['skipped_tss']} (Margin: {args.tss_margin}bp)", file=sys.stderr)
    print(f"Due to TSS inside k-mer (any strand): {stats['skipped_tss_inside']}", file=sys.stderr)
    if args.target_gc is not None:
        print(f"Due to GC composition selection:      {stats['skipped_gc_bias']} (Target: {args.target_gc}%)", file=sys.stderr)
    print("\nBASE & STRAND DISTRIBUTION (Saved Dataset):", file=sys.stderr)
    print(f"Global GC Content (Saved Dataset):    {global_gc:.2f}% (Mean: {mean_gc:.2f}% ± {stdev_gc:.2f}%)", file=sys.stderr)
    print(f"Positive strands (+):                 {strand_pos}", file=sys.stderr)
    print(f"Negative strands (-):                 {strand_neg}", file=sys.stderr)
    print("="*50, file=sys.stderr)
    print(f"[SUCCESS] Dataset generated -> {fasta_out}", file=sys.stderr)
    print(f"[SUCCESS] Metadata generated -> {tsv_out}", file=sys.stderr)

if __name__ == '__main__':
    args = parse_arguments()
    extract_negatives(args)
