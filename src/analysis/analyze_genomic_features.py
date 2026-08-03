#!/usr/bin/env python3
"""
Unified Genomic Feature & CDS Overlap Analyzer (Master Analysis Engine).

Consolidates all exploratory analysis features across D39V and TIGR4:
1. 5'-UTR Distribution (Leaderless <= 5 bp, Canonical 15-45 bp, Ultra-long >= 200 bp).
2. TSS Density & Spatial Clustering (< 100 bp).
3. Replicative Strand Bias (oriC to terC lagging vs leading strand transcription).
4. Intergenic vs. Intragenic CDS 40 nt Window Overlaps (CDS 5'-start, internal body, 3'-end).

Usage:
    pixi run python src/analysis/analyze_genomic_features.py
"""

import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent

# Reference files
TIGR4_GFF = ROOT / "data/reference/NC_003028.gff3"
TIGR4_FASTA = ROOT / "data/reference/NC_003028.fasta"
TIGR4_EXCEL = ROOT / "output/tigr4_data/S1_TSS.xlsx"

D39V_GFF_CDS = ROOT / "data/reference/sequence.gff3"
D39V_GFF_TSS = ROOT / "data/reference/D39V_annotation_TSS_Victor.gff"
D39V_FASTA = ROOT / "data/reference/D39V.fna"

OUTPUT_DIR = ROOT / "output/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════
# 1. CDS Boundary Reader
# ════════════════════════════════════════════════════════════════

def load_cds_features(gff_path: Path) -> List[Tuple[int, int, str, str]]:
    cds_list = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and parts[2] == "CDS":
                start, end, strand = int(parts[3]), int(parts[4]), parts[6]
                attr = dict(item.split("=") for item in parts[8].split(";") if "=" in item)
                locus = attr.get("locus_tag", attr.get("ID", attr.get("Name", "unknown_cds")))
                cds_list.append((start, end, strand, locus))
    return cds_list


# ════════════════════════════════════════════════════════════════
# 2. 5'-UTR & Spatial Clustering Analysis
# ════════════════════════════════════════════════════════════════

def analyze_5utr_and_clustering(tss_records: List[Dict], cluster_dist: int = 100) -> Dict:
    utrs = []
    positions = []

    for r in tss_records:
        pos = r["TSS_Position"]
        positions.append((pos, r["Strand"]))
        u_val = r.get("UTR5_Length")
        try:
            u_num = float(u_val)
            if not math.isnan(u_num):
                utrs.append(u_num)
        except (ValueError, TypeError):
            continue

    n_utrs = len(utrs)
    median_utr = float(pd.Series(utrs).median()) if utrs else 0.0
    leaderless = sum(1 for u in utrs if u <= 5)
    canonical = sum(1 for u in utrs if 15 <= u <= 45)
    long_utrs = sum(1 for u in utrs if u >= 200)

    positions.sort(key=lambda x: x[0])
    close_pairs = 0
    clusters = 0
    in_cluster = False

    for i in range(len(positions) - 1):
        d = positions[i + 1][0] - positions[i][0]
        if d < cluster_dist:
            close_pairs += 1
            if not in_cluster:
                clusters += 1
                in_cluster = True
        else:
            in_cluster = False

    return {
        "n_utrs": n_utrs,
        "median_utr": median_utr,
        "leaderless_count": leaderless,
        "leaderless_pct": (leaderless / n_utrs * 100.0) if n_utrs > 0 else 0.0,
        "canonical_count": canonical,
        "canonical_pct": (canonical / n_utrs * 100.0) if n_utrs > 0 else 0.0,
        "long_utrs_count": long_utrs,
        "long_utrs_pct": (long_utrs / n_utrs * 100.0) if n_utrs > 0 else 0.0,
        "close_pairs": close_pairs,
        "clusters": clusters,
    }


# ════════════════════════════════════════════════════════════════
# 3. Replicative Strand Bias (oriC to terC)
# ════════════════════════════════════════════════════════════════

def analyze_replicative_strand_bias(tss_records: List[Dict], ter_pos: int) -> Dict:
    leading = 0
    lagging = 0

    for r in tss_records:
        pos = r["TSS_Position"]
        strand = r["Strand"]

        if pos <= ter_pos:
            if strand == "+":
                leading += 1
            else:
                lagging += 1
        else:
            if strand == "-":
                leading += 1
            else:
                lagging += 1

    total = leading + lagging
    return {
        "leading": leading,
        "leading_pct": (leading / total * 100.0) if total > 0 else 0.0,
        "lagging": lagging,
        "lagging_pct": (lagging / total * 100.0) if total > 0 else 0.0,
    }


# ════════════════════════════════════════════════════════════════
# 4. CDS Overlap Analysis (40 nt Window)
# ════════════════════════════════════════════════════════════════

def analyze_cds_overlaps(
    tss_records: List[Dict], cds_list: List[Tuple[int, int, str, str]], window_size: int = 40
) -> Dict:
    n_total = len(tss_records)
    n_intragenic = 0
    n_intergenic = 0

    cds_position_types = {"5_prime_start": 0, "internal_body": 0, "3_prime_end": 0}

    for r in tss_records:
        tss_pos = r["TSS_Position"]
        is_inside = False
        pos_type = "intergenic"

        for c_start, c_end, c_strand, c_locus in cds_list:
            if c_start <= tss_pos <= c_end:
                is_inside = True
                dist_start = abs(tss_pos - c_start) if c_strand == "+" else abs(c_end - tss_pos)
                dist_end = abs(c_end - tss_pos) if c_strand == "+" else abs(tss_pos - c_start)

                if dist_start <= window_size:
                    pos_type = "5_prime_start"
                elif dist_end <= window_size:
                    pos_type = "3_prime_end"
                else:
                    pos_type = "internal_body"
                break

        if is_inside:
            n_intragenic += 1
            cds_position_types[pos_type] = cds_position_types.get(pos_type, 0) + 1
        else:
            n_intergenic += 1

    return {
        "n_total": n_total,
        "n_intergenic": n_intergenic,
        "freq_intergenic": (n_intergenic / n_total * 100.0) if n_total > 0 else 0.0,
        "n_intragenic": n_intragenic,
        "freq_intragenic": (n_intragenic / n_total * 100.0) if n_total > 0 else 0.0,
        "cds_position_types": cds_position_types,
    }


# ════════════════════════════════════════════════════════════════
# 5. Main Pipeline Orchestrator
# ════════════════════════════════════════════════════════════════

def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" UNIFIED GENOMIC FEATURE & CDS OVERLAP ANALYZER (MASTER ANALYSIS)")
    print("═════════════════════════════════════════════════════════════════\n")

    print("[INFO] Analyzing TIGR4 genome features...")
    tigr4_cds = load_cds_features(TIGR4_GFF)
    df_t4_hconf = pd.read_excel(TIGR4_EXCEL, sheet_name="High Confidence (TSS_100.4)")
    t4_records = [
        {
            "Locus_Tag": str(r["Locus_tag"]),
            "TSS_Position": int(r["TSS_position"]),
            "Strand": str(r["Strand"]),
            "UTR5_Length": r.get("5'-UTR_length", "NA"),
        }
        for _, r in df_t4_hconf.iterrows()
    ]

    t4_utr_res = analyze_5utr_and_clustering(t4_records)
    t4_strand_res = analyze_replicative_strand_bias(t4_records, ter_pos=1080000)
    t4_overlap_res = analyze_cds_overlaps(t4_records, tigr4_cds, window_size=40)

    print("[INFO] Analyzing D39V genome features...")
    d39v_cds = load_cds_features(D39V_GFF_CDS)
    d39v_records = []
    with open(D39V_GFF_TSS) as f:
        for idx, line in enumerate(f):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and any(k in parts[2].lower() for k in ["transcription_start_site", "tss"]):
                d39v_records.append({
                    "Locus_Tag": f"D39V_TSS_{idx}",
                    "TSS_Position": int(parts[3]),
                    "Strand": parts[6],
                    "UTR5_Length": "NA",
                })

    d39v_utr_res = analyze_5utr_and_clustering(d39v_records)
    d39v_strand_res = analyze_replicative_strand_bias(d39v_records, ter_pos=1020000)
    d39v_overlap_res = analyze_cds_overlaps(d39v_records, d39v_cds, window_size=40)

    print("\n" + "═" * 95)
    print(" UNIFIED GENOMIC & TRANSCRIPTOMIC FEATURES COMPARISON")
    print("═" * 95)
    print(f"{'Feature Category / Metric':<40} | {'D39V (Cappable-seq)':<24} | {'TIGR4 High Conf (Core)':<24}")
    print("─" * 95)
    print(f"{'Total Evaluated TSS Promoters':<40} | {d39v_overlap_res['n_total']:<24,} | {t4_overlap_res['n_total']:<24,}")
    print(f"{'5-UTR Median Length (bp)':<40} | {d39v_utr_res['median_utr']:<24.1f} | {t4_utr_res['median_utr']:<24.1f}")
    print(f"{'Leaderless mRNAs (<= 5 bp)':<40} | {d39v_utr_res['leaderless_count']} ({d39v_utr_res['leaderless_pct']:.1f}%)                 | {t4_utr_res['leaderless_count']} ({t4_utr_res['leaderless_pct']:.1f}%)")
    print(f"{'Canonical 5-UTR (15-45 bp)':<40} | {d39v_utr_res['canonical_count']} ({d39v_utr_res['canonical_pct']:.1f}%)                 | {t4_utr_res['canonical_count']} ({t4_utr_res['canonical_pct']:.1f}%)")
    print(f"{'Ultra-long 5-UTR (>= 200 bp)':<40} | {d39v_utr_res['long_utrs_count']} ({d39v_utr_res['long_utrs_pct']:.1f}%)                 | {t4_utr_res['long_utrs_count']} ({t4_utr_res['long_utrs_pct']:.1f}%)")
    print(f"{'TSS Density Clusters (< 100 bp)':<40} | {d39v_utr_res['clusters']:<24,} | {t4_utr_res['clusters']:<24,}")
    print(f"{'Replicative Lagging Strand (%)':<40} | {d39v_strand_res['lagging_pct']:<24.1f}% | {t4_strand_res['lagging_pct']:<24.1f}%")
    print(f"{'Intergenic TSS Promoters (%)':<40} | {d39v_overlap_res['freq_intergenic']:<24.1f}% | {t4_overlap_res['freq_intergenic']:<24.1f}%")
    print(f"{'Intragenic TSS Promoters (%)':<40} | {d39v_overlap_res['freq_intragenic']:<24.1f}% | {t4_overlap_res['freq_intragenic']:<24.1f}%")
    print("═" * 95 + "\n")


if __name__ == "__main__":
    main()
