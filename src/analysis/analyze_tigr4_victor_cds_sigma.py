#!/usr/bin/env python3
"""
Comprehensive Analysis of TIGR4 TSS & Promoter Overlaps with CDS Regions (40 nt Window Analysis).

Implements Victor's experimental directives for CDS overlaps:
1. Intergenic vs Intragenic TSS classification (inside vs outside CDSs).
2. 40 nt promoter window overlap analysis (UP element [-60, -20] vs TSS [+1]):
   - Overlap at CDS 5'-start, CDS internal body, and CDS 3'-end.
   - Separate analysis for + and - strands.
   - Frequency of TSS inside CDSs.
3. Comparative analysis across strains (D39V vs TIGR4).

Usage:
    pixi run python src/analysis/analyze_tigr4_victor_cds_sigma.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
TIGR4_GFF = ROOT / "data/reference/NC_003028.gff3"
TIGR4_FASTA = ROOT / "data/reference/NC_003028.fasta"
TIGR4_EXCEL = ROOT / "output/tigr4_data/S1_TSS.xlsx"

D39V_GFF_CDS = ROOT / "data/reference/sequence.gff3"
D39V_GFF_TSS = ROOT / "data/reference/D39V_annotation_TSS_Victor.gff"

OUTPUT_DIR = ROOT / "output/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIGR4_DATA_DIR = ROOT / "output/tigr4_data"


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


def analyze_cds_overlaps(
    tss_records: List[Dict], cds_list: List[Tuple[int, int, str, str]], window_size: int = 40
) -> Dict:
    n_total = len(tss_records)
    n_intragenic = 0
    n_intergenic = 0

    strand_counts = {"+": 0, "-": 0}
    up_overlap_counts = {"+": 0, "-": 0}
    tss_overlap_counts = {"+": 0, "-": 0}

    cds_position_types = {
        "5_prime_start": 0,
        "internal_body": 0,
        "3_prime_end": 0,
    }

    detailed_records = []

    for r in tss_records:
        tss_pos = r["TSS_Position"]
        strand = r["Strand"]
        locus = r["Locus_Tag"]
        strand_counts[strand] = strand_counts.get(strand, 0) + 1

        if strand == "+":
            up_start, up_end = tss_pos - window_size, tss_pos - 1
        else:
            up_start, up_end = tss_pos + 1, tss_pos + window_size

        is_tss_inside_cds = False
        is_up_inside_cds = False
        pos_type = "intergenic"
        matched_cds_locus = "none"

        for c_start, c_end, c_strand, c_locus in cds_list:
            if c_start <= tss_pos <= c_end:
                is_tss_inside_cds = True
                matched_cds_locus = c_locus
                if strand == c_strand:
                    tss_overlap_counts[strand] += 1

                dist_to_start = abs(tss_pos - c_start) if c_strand == "+" else abs(c_end - tss_pos)
                dist_to_end = abs(c_end - tss_pos) if c_strand == "+" else abs(tss_pos - c_start)

                if dist_to_start <= window_size:
                    pos_type = "5_prime_start"
                elif dist_to_end <= window_size:
                    pos_type = "3_prime_end"
                else:
                    pos_type = "internal_body"
                break

            if (c_start <= up_start <= c_end) or (c_start <= up_end <= c_end):
                is_up_inside_cds = True
                if strand == c_strand:
                    up_overlap_counts[strand] += 1

        if is_tss_inside_cds:
            n_intragenic += 1
            cds_position_types[pos_type] = cds_position_types.get(pos_type, 0) + 1
        else:
            n_intergenic += 1

        r_out = dict(r)
        r_out.update({
            "Is_Intragenic": is_tss_inside_cds,
            "CDS_Position_Type": pos_type if is_tss_inside_cds else "intergenic",
            "Matched_CDS": matched_cds_locus,
            "UP_Element_Overlap": is_up_inside_cds,
        })
        detailed_records.append(r_out)

    freq_intragenic = (n_intragenic / n_total * 100.0) if n_total > 0 else 0.0
    freq_intergenic = (n_intergenic / n_total * 100.0) if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_intragenic": n_intragenic,
        "n_intergenic": n_intergenic,
        "freq_intragenic": freq_intragenic,
        "freq_intergenic": freq_intergenic,
        "strand_counts": strand_counts,
        "tss_overlap_counts": tss_overlap_counts,
        "up_overlap_counts": up_overlap_counts,
        "cds_position_types": cds_position_types,
        "detailed_records": detailed_records,
    }


def update_dataset_tsv(tsv_path: Path, annotations_lookup: Dict[Tuple[int, str], Dict]):
    if not tsv_path.exists():
        return
    df = pd.read_csv(tsv_path, sep="\t")
    if "TSS_Position" in df.columns and "Strand" in df.columns:
        intra_list = []
        pos_type_list = []
        up_list = []
        for idx, row in df.iterrows():
            key = (int(row["TSS_Position"]), str(row["Strand"]))
            anno = annotations_lookup.get(key, {})
            intra_list.append(anno.get("Is_Intragenic", False))
            pos_type_list.append(anno.get("CDS_Position_Type", "intergenic"))
            up_list.append(anno.get("UP_Element_Overlap", False))

        # Drop Sigma_Factor column if present
        if "Sigma_Factor" in df.columns:
            df = df.drop(columns=["Sigma_Factor"])

        df["Is_Intragenic"] = intra_list
        df["CDS_Position_Type"] = pos_type_list
        df["UP_Element_Overlap"] = up_list
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"[UPDATED] Cleaned TSV (CDS overlaps only) ➔ {tsv_path.name}")


def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" VICTOR DIRECTIVE: TIGR4 & D39V CDS OVERLAP ANALYSIS (40 NT WINDOW)")
    print("═════════════════════════════════════════════════════════════════\n")

    print("[INFO] Loading TIGR4 reference GFF3...")
    tigr4_cds = load_cds_features(TIGR4_GFF)

    df_hconf = pd.read_excel(TIGR4_EXCEL, sheet_name="High Confidence (TSS_100.4)")
    tigr4_hconf_records = [
        {"Locus_Tag": str(r["Locus_tag"]), "TSS_Position": int(r["TSS_position"]), "Strand": str(r["Strand"])}
        for idx, r in df_hconf.iterrows()
    ]

    tigr4_overlap_res = analyze_cds_overlaps(tigr4_hconf_records, tigr4_cds, window_size=40)

    t4_lookup = {(r["TSS_Position"], r["Strand"]): r for r in tigr4_overlap_res["detailed_records"]}
    update_dataset_tsv(TIGR4_DATA_DIR / "positives_tigr4_high_conf_primary_81bp.tsv", t4_lookup)
    update_dataset_tsv(TIGR4_DATA_DIR / "positives_tigr4_extended_primary_81bp.tsv", t4_lookup)

    print("\n[INFO] Loading D39V reference GFF3...")
    d39v_cds = load_cds_features(D39V_GFF_CDS)

    d39v_tss_records = []
    with open(D39V_GFF_TSS) as f:
        for idx, line in enumerate(f):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and any(k in parts[2].lower() for k in ["transcription_start_site", "tss"]):
                d39v_tss_records.append({
                    "Locus_Tag": f"D39V_TSS_{idx}",
                    "TSS_Position": int(parts[3]),
                    "Strand": parts[6],
                })

    d39v_overlap_res = analyze_cds_overlaps(d39v_tss_records, d39v_cds, window_size=40)
    d39_lookup = {(r["TSS_Position"], r["Strand"]): r for r in d39v_overlap_res["detailed_records"]}
    update_dataset_tsv(ROOT / "data/benchmark/positives_81bp.tsv", d39_lookup)

    print("\n" + "═" * 90)
    print(" 1. INTERGENIC VS. INTRAGENIC TSS DISTRIBUTION & FREQUENCY")
    print("═" * 90)
    print(f"{'Strain / Dataset':<30} | {'Total TSS':<10} | {'Intergenic (%)':<18} | {'Intragenic (%)':<18}")
    print("─" * 90)
    print(
        f"{'TIGR4 High Conf (Core)':<30} | {tigr4_overlap_res['n_total']:<10,} | "
        f"{tigr4_overlap_res['n_intergenic']:,} ({tigr4_overlap_res['freq_intergenic']:.1f}%)   | "
        f"{tigr4_overlap_res['n_intragenic']:,} ({tigr4_overlap_res['freq_intragenic']:.1f}%)"
    )
    print(
        f"{'D39V Cappable-seq':<30} | {d39v_overlap_res['n_total']:<10,} | "
        f"{d39v_overlap_res['n_intergenic']:,} ({d39v_overlap_res['freq_intergenic']:.1f}%)   | "
        f"{d39v_overlap_res['n_intragenic']:,} ({d39v_overlap_res['freq_intragenic']:.1f}%)"
    )
    print("═" * 90)

    print("\n" + "═" * 90)
    print(" 2. 40 NT PROMOTER WINDOW & UP ELEMENT CDS OVERLAP ANALYSIS")
    print("═" * 90)
    print(f"TIGR4 CDS Overlap Breakdown (Window = 40 nt):")
    print(f" • Overlaps at CDS 5'-Start Region:  {tigr4_overlap_res['cds_position_types']['5_prime_start']:,}")
    print(f" • Overlaps inside CDS Internal Body: {tigr4_overlap_res['cds_position_types']['internal_body']:,}")
    print(f" • Overlaps at CDS 3'-End Region:    {tigr4_overlap_res['cds_position_types']['3_prime_end']:,}")
    print(f" • UP Element Overlaps on (+) Strand: {tigr4_overlap_res['up_overlap_counts']['+']:,}")
    print(f" • UP Element Overlaps on (-) Strand: {tigr4_overlap_res['up_overlap_counts']['-']:,}")
    print("═" * 90 + "\n")


if __name__ == "__main__":
    main()
