#!/usr/bin/env python3
"""
Exploratory Dataset Feature Analysis for Pneumococcal TSS Datasets (D39V & TIGR4).

Performs 3 specific genomic analyses directly on dataset metadata & GFF annotations:
1. 5'-UTR Length Distribution & Extremes (Leaderless vs Ultra-Long UTRs).
2. Genomic TSS Density & Clustering (Hotspot Promoters < 100 bp).
3. Replicative Strand Bias (Leading vs Lagging Strand relative to oriC).

Usage:
    pixi run python src/analysis/analyze_dataset_features.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS = {
    "D39V (Cappable-seq Primary)": {
        "tsv": ROOT / "data/benchmark/positives_81bp_metadata.tsv",
        "gff_tss": ROOT / "data/reference/D39V_annotation_TSS_Victor.gff",
        "gff_cds": ROOT / "data/reference/sequence.gff3",
        "strain": "D39V"
    },
    "TIGR4 (High Conf Primary)": {
        "tsv": ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.tsv",
        "strain": "TIGR4"
    },
    "TIGR4 (Extended Primary)": {
        "tsv": ROOT / "output/tigr4_data/positives_tigr4_extended_primary_81bp.tsv",
        "strain": "TIGR4"
    },
}

GENOME_COORDS = {
    "D39V": {"len": 2046572, "ori": 1, "ter": 1023286},
    "TIGR4": {"len": 2160842, "ori": 1, "ter": 1080421},
}


def load_d39v_utrs() -> List[Tuple[float, str, int]]:
    gff_cds_path = ROOT / "data/reference/sequence.gff3"
    gff_tss_path = ROOT / "data/reference/D39V_annotation_TSS_Victor.gff"

    if not gff_cds_path.exists() or not gff_tss_path.exists():
        return []

    cds_dict = {}
    with open(gff_cds_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and parts[2] == "CDS":
                start, end, strand = int(parts[3]), int(parts[4]), parts[6]
                attr = dict(item.split("=") for item in parts[8].split(";") if "=" in item)
                locus = attr.get("locus_tag", "NA")
                cds_dict[locus] = {"start": start, "end": end, "strand": strand}

    valid_utrs = []
    with open(gff_tss_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9:
                t_pos = int(parts[3])
                t_strand = parts[6]

                if t_strand == "+":
                    matches = [(l, c["start"] - t_pos) for l, c in cds_dict.items() if c["strand"] == "+" and c["start"] >= t_pos]
                else:
                    matches = [(l, t_pos - c["end"]) for l, c in cds_dict.items() if c["strand"] == "-" and c["end"] <= t_pos]

                if matches:
                    matches.sort(key=lambda x: x[1])
                    min_utr, locus = matches[0][1], matches[0][0]
                    if 0 <= min_utr <= 1000:
                        valid_utrs.append((float(min_utr), locus, t_pos))

    return valid_utrs


def analyze_5utr_lengths(df: pd.DataFrame, dataset_info: Dict) -> Dict:
    valid_utrs = []

    if dataset_info.get("strain") == "D39V":
        valid_utrs = load_d39v_utrs()

    if not valid_utrs:
        utr_col = None
        for col in ["UTR5_Length", "5UTR_Length_bp", "5'-UTR_length", "UTR_length"]:
            if col in df.columns:
                utr_col = col
                break

        if utr_col:
            for _, row in df.iterrows():
                val = row[utr_col]
                try:
                    val_num = float(val)
                    if not math.isnan(val_num):
                        locus = str(row.get("Locus_Tag", row.get("Locus_tag", row.get("Sequence_ID", "NA"))))
                        pos = row.get("TSS_Position", row.get("TSS_Position_0based", row.get("TSS_POS", 0)))
                        valid_utrs.append((val_num, locus, pos))
                except (ValueError, TypeError):
                    continue

    if not valid_utrs:
        return {"available": False}

    lengths = [item[0] for item in valid_utrs]
    mean_val = float(np.mean(lengths))
    median_val = float(np.median(lengths))
    min_val = float(np.min(lengths))
    max_val = float(np.max(lengths))
    stdev_val = float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0

    leaderless = [item for item in valid_utrs if item[0] <= 5]
    ultra_long = [item for item in valid_utrs if item[0] >= 200]

    return {
        "available": True,
        "n_total": len(lengths),
        "mean": mean_val,
        "median": median_val,
        "min": min_val,
        "max": max_val,
        "stdev": stdev_val,
        "n_leaderless": len(leaderless),
        "pct_leaderless": (len(leaderless) / len(lengths)) * 100.0,
        "leaderless_examples": sorted(leaderless, key=lambda x: x[0])[:5],
        "n_ultra_long": len(ultra_long),
        "pct_ultra_long": (len(ultra_long) / len(lengths)) * 100.0,
        "ultra_long_examples": sorted(ultra_long, key=lambda x: -x[0])[:5],
    }


def analyze_tss_clustering(df: pd.DataFrame, dataset_info: Dict, max_dist: int = 100) -> Dict:
    pos_col = None
    for col in df.columns:
        if col.upper() in ["TSS_POSITION", "TSS_POSITION_0BASED", "TSS_POS", "TSS_POS_0", "POS"]:
            pos_col = col
            break

    if not pos_col:
        return {"available": False}

    df_clean = df.dropna(subset=[pos_col]).copy()
    df_clean[pos_col] = pd.to_numeric(df_clean[pos_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[pos_col])

    positions = sorted(df_clean[pos_col].astype(int).unique())
    if len(positions) < 2:
        return {"available": False}

    close_pairs = 0
    clusters = []
    current_cluster = [positions[0]]

    for i in range(1, len(positions)):
        dist = positions[i] - positions[i - 1]
        if dist <= max_dist:
            close_pairs += 1
            current_cluster.append(positions[i])
        else:
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            current_cluster = [positions[i]]

    if len(current_cluster) >= 3:
        clusters.append(current_cluster)

    return {
        "available": True,
        "total_unique_tss": len(positions),
        "close_pairs_under_100bp": close_pairs,
        "pct_tss_in_clusters": (close_pairs / len(positions)) * 100.0,
        "n_hotspot_clusters": len(clusters),
        "clusters": clusters[:5],
    }


def analyze_replicative_strand_bias(df: pd.DataFrame, strain_type: str = "TIGR4") -> Dict:
    pos_col = None
    for col in df.columns:
        if col.upper() in ["TSS_POSITION", "TSS_POSITION_0BASED", "TSS_POS", "POS"]:
            pos_col = col
            break

    strand_col = None
    for col in df.columns:
        if col.upper() in ["STRAND", "STRAND_ORIENTATION"]:
            strand_col = col
            break

    if not pos_col or not strand_col:
        return {"available": False}

    coords = GENOME_COORDS.get(strain_type, GENOME_COORDS["TIGR4"])
    genome_len = coords["len"]
    ter_pos = coords["ter"]

    leading_count = 0
    lagging_count = 0

    for _, row in df.iterrows():
        try:
            pos = int(row[pos_col])
            strand = str(row[strand_col]).strip()
        except (ValueError, TypeError):
            continue

        if pos <= ter_pos:
            if strand == "-":
                leading_count += 1
            elif strand == "+":
                lagging_count += 1
        else:
            if strand == "+":
                leading_count += 1
            elif strand == "-":
                lagging_count += 1

    total = leading_count + lagging_count
    if total == 0:
        return {"available": False}

    return {
        "available": True,
        "total_analyzed": total,
        "leading_count": leading_count,
        "pct_leading": (leading_count / total) * 100.0,
        "lagging_count": lagging_count,
        "pct_lagging": (lagging_count / total) * 100.0,
        "leading_to_lagging_ratio": leading_count / lagging_count if lagging_count > 0 else 0.0,
    }


def main():
    print("════════════════════════════════════════════════════════════════")
    print(" EXPLORATORY FEATURE ANALYSIS: 5'-UTR, CLUSTERING & REPLICATIVE BIAS")
    print("════════════════════════════════════════════════════════════════\n")

    for name, info in DATASETS.items():
        path = info["tsv"]
        if not path.exists():
            print(f"[SKIP] Path not found: {path}")
            continue

        df = pd.read_csv(path, sep="\t")
        strain_type = info.get("strain", "TIGR4")

        utr_res = analyze_5utr_lengths(df, info)
        cluster_res = analyze_tss_clustering(df, info, max_dist=100)
        rep_res = analyze_replicative_strand_bias(df, strain_type)

        print(f"▶ DATASET: {name} (N = {len(df):,} entries)")
        print("─" * 60)

        # 1. 5'-UTR
        if utr_res.get("available"):
            print("1. 5'-UTR LENGTH DISTRIBUTION:")
            print(f"   • Mean ± SD:     {utr_res['mean']:.1f} ± {utr_res['stdev']:.1f} bp (Median: {utr_res['median']:.0f} bp, Range: {utr_res['min']:.0f}-{utr_res['max']:.0f} bp)")
            print(f"   • Leaderless mRNAs (≤5 bp):    {utr_res['n_leaderless']} ({utr_res['pct_leaderless']:.1f}%)")
            for item in utr_res["leaderless_examples"]:
                print(f"       - Locus {item[1]} (TSS {item[2]}): UTR = {item[0]:.0f} bp")
            print(f"   • Ultra-Long 5'-UTRs (≥200 bp): {utr_res['n_ultra_long']} ({utr_res['pct_ultra_long']:.1f}%)")
            for item in utr_res["ultra_long_examples"]:
                print(f"       - Locus {item[1]} (TSS {item[2]}): UTR = {item[0]:.0f} bp")
        else:
            print("1. 5'-UTR LENGTH DISTRIBUTION: [Not Available]")

        print()

        # 2. Clustering
        if cluster_res.get("available"):
            print("2. GENOMIC DENSITY & TSS CLUSTERING (< 100 bp):")
            print(f"   • Close TSS Pairs (<100 bp):  {cluster_res['close_pairs_under_100bp']} ({cluster_res['pct_tss_in_clusters']:.1f}% of TSSs)")
            print(f"   • Dense Hotspot Clusters (≥3 TSSs): {cluster_res['n_hotspot_clusters']} clusters")
            for i, cl in enumerate(cluster_res["clusters"][:3], 1):
                span = cl[-1] - cl[0]
                print(f"       - Cluster {i}: {len(cl)} TSSs spanning {span} bp (Coords: {cl[0]} to {cl[-1]})")
        else:
            print("2. GENOMIC DENSITY & TSS CLUSTERING: [Not Available]")

        print()

        # 3. Replicative Bias
        if rep_res.get("available"):
            print("3. REPLICATIVE STRAND BIAS (oriC to terC):")
            print(f"   • Leading Strand (Co-directional with replication): {rep_res['leading_count']:,} ({rep_res['pct_leading']:.1f}%)")
            print(f"   • Lagging Strand (Head-on replication collisions):  {rep_res['lagging_count']:,} ({rep_res['pct_lagging']:.1f}%)")
            print(f"   • Co-directional Preference Ratio:                   {rep_res['leading_to_lagging_ratio']:.2f}x")
        else:
            print("3. REPLICATIVE STRAND BIAS: [Not Available]")

        print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
