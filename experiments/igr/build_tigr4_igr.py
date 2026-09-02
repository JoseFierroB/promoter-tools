#!/usr/bin/env python3
"""
================================================================================
BUILD TIGR4 IGR-SPECIFIC BENCHMARK DATASETS
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Description:
    Constructs isolated, reproducible IGR-specific promoter datasets for
    Streptococcus pneumoniae TIGR4 (NC_003028.3) by mapping experimental
    TSSs (from S1_TSS.xlsx) into refined IGR intervals (TIGR4_igrs_refined.tsv).

Outputs generated in: data/benchmark_igr/tigr4/
    1. subset_1_high_conf_primary/   (Primary High-Confidence in IGRs)
    2. subset_2_high_conf_all/       (Primary + Secondary High-Confidence in IGRs)
    3. subset_3_all_primary/         (High + Low Confidence Primary in IGRs)
    4. subset_4_all_comprehensive/   (All 4 sheets TSSs in IGRs)
    5. summary_reports/              (TSV & Markdown documentation)
================================================================================
"""

import sys
import random
import bisect
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

GENOME_FASTA = ROOT_DIR / "data/reference/NC_003028.fasta"
if not GENOME_FASTA.exists():
    GENOME_FASTA = ROOT_DIR / "data/reference/TIGR4.fasta"

IGR_TSV = ROOT_DIR / "output/intergenic_refined/tigr4/TIGR4_igrs_refined.tsv"
S1_XLSX = ROOT_DIR / "data/tigr4/S1_TSS.xlsx"
OUT_BASE = ROOT_DIR / "data/benchmark_igr/tigr4"

def calculate_gc(seq_str: str) -> float:
    g = seq_str.upper().count("G")
    c = seq_str.upper().count("C")
    return round((g + c) / len(seq_str) * 100, 3) if len(seq_str) > 0 else 0.0

def extract_81bp_window(genome_seq: str, genome_len: int, pos_1based: int, strand: str) -> str:
    """Extracts 81 bp sequence with TSS +1 at index 60 (0-based) in 5'->3' sense orientation."""
    if strand == "+":
        start_0 = pos_1based - 1 - 60
        end_0 = pos_1based - 1 + 21
        if start_0 < 0:
            return genome_seq[start_0 % genome_len:] + genome_seq[:end_0]
        elif end_0 > genome_len:
            return genome_seq[start_0:] + genome_seq[:end_0 % genome_len]
        else:
            return genome_seq[start_0:end_0]
    else:
        start_0 = pos_1based - 1 - 20
        end_0 = pos_1based - 1 + 61
        if start_0 < 0:
            raw = genome_seq[start_0 % genome_len:] + genome_seq[:end_0]
        elif end_0 > genome_len:
            raw = genome_seq[start_0:] + genome_seq[:end_0 % genome_len]
        else:
            raw = genome_seq[start_0:end_0]
        return str(Seq(raw).reverse_complement())

def sample_igr_negatives(linear_igrs, all_tss_sorted, genome_seq, genome_len, num_needed, seed=42, tss_margin=50):
    """Samples non-promoter 81-mers strictly inside IGRs, avoiding any known TSS by >= tss_margin."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Candidate sampling intervals
    valid_slots = []
    for _, igr in linear_igrs.iterrows():
        igr_start = int(igr["start"])
        igr_end = int(igr["end"])
        igr_len = igr_end - igr_start + 1
        if igr_len >= 81:
            for s in range(igr_start, igr_end - 81 + 2):
                e = s + 81 - 1
                # Check TSS margin
                # Find if any TSS falls in [s - tss_margin, e + tss_margin]
                idx_left = bisect.bisect_left(all_tss_sorted, s - tss_margin)
                idx_right = bisect.bisect_right(all_tss_sorted, e + tss_margin)
                if idx_left == idx_right: # No TSS within margin!
                    valid_slots.append({
                        "igr_id": igr["igr_id"],
                        "orientation": igr["orientation_type"],
                        "start": s,
                        "end": e
                    })
                    
    print(f"    Available candidate IGR negative windows (margin >= {tss_margin}bp): {len(valid_slots)}")
    if len(valid_slots) < num_needed:
        print(f"    [WARNING] Not enough non-overlapping slots with margin {tss_margin}bp. Relaxing margin to 30bp...")
        return sample_igr_negatives(linear_igrs, all_tss_sorted, genome_seq, genome_len, num_needed, seed=seed, tss_margin=30)
        
    sampled_indices = random.sample(range(len(valid_slots)), num_needed)
    negatives = []
    
    for i, idx in enumerate(sampled_indices):
        slot = valid_slots[idx]
        strand = random.choice(["+", "-"])
        s_0 = slot["start"] - 1
        e_0 = slot["end"]
        seq_raw = genome_seq[s_0:e_0]
        if strand == "-":
            seq_final = str(Seq(seq_raw).reverse_complement())
        else:
            seq_final = seq_raw
            
        negatives.append({
            "Sequence_ID": f"NEG_TIGR4_IGR_{i+1:04d}_{slot['igr_id']}_{slot['start']}_{strand}",
            "IGR_ID": slot["igr_id"],
            "IGR_Orientation": slot["orientation"],
            "Genomic_Start": slot["start"],
            "Genomic_End": slot["end"],
            "Strand": strand,
            "Sequence": seq_final,
            "Length": len(seq_final),
            "GC_Percent": calculate_gc(seq_final)
        })
        
    return pd.DataFrame(negatives)

def main():
    print("=" * 80)
    print("BUILDING TIGR4 IGR-SPECIFIC BENCHMARK DATASETS")
    print("=" * 80)
    
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    
    # 1. Load genome
    genome_rec = list(SeqIO.parse(GENOME_FASTA, "fasta"))[0]
    genome_seq = str(genome_rec.seq).upper()
    genome_len = len(genome_seq)
    print(f"Loaded TIGR4 Genome: {genome_rec.id} ({genome_len} bp)")
    
    # 2. Load Refined IGRs
    df_igr = pd.read_csv(IGR_TSV, sep="\t")
    linear_igrs = df_igr[~df_igr["is_circular_origin_wrap"]].copy()
    linear_igrs["start_int"] = linear_igrs["start"].astype(int)
    linear_igrs["end_int"] = linear_igrs["end"].astype(int)
    print(f"Loaded Refined IGRs: {len(linear_igrs)} linear intergenic intervals")
    
    # 3. Parse all 4 sheets from S1_TSS.xlsx
    xl = pd.ExcelFile(S1_XLSX)
    
    def parse_sheet_tss(sheet_name, pos_col, source_tag):
        df = xl.parse(sheet_name)
        parsed = []
        for idx, r in df.iterrows():
            pos = int(r[pos_col])
            strand = str(r["Strand"]).strip()
            locus = str(r.get("Locus_tag", r.get("Locus", f"TIGR4_{pos}")))
            
            # Check if in IGR
            matches = linear_igrs[(linear_igrs["start_int"] <= pos) & (pos <= linear_igrs["end_int"])]
            in_igr = len(matches) > 0
            igr_id = matches.iloc[0]["igr_id"] if in_igr else None
            igr_type = matches.iloc[0]["orientation_type"] if in_igr else None
            igr_start = matches.iloc[0]["start_int"] if in_igr else None
            igr_end = matches.iloc[0]["end_int"] if in_igr else None
            
            seq_81bp = extract_81bp_window(genome_seq, genome_len, pos, strand)
            
            parsed.append({
                "Source_Sheet": source_tag,
                "Locus_Tag": locus,
                "TSS_Position": pos,
                "Strand": strand,
                "In_Refined_IGR": in_igr,
                "IGR_ID": igr_id,
                "IGR_Orientation": igr_type,
                "IGR_Start": igr_start,
                "IGR_End": igr_end,
                "Sequence": seq_81bp,
                "Length": len(seq_81bp),
                "GC_Percent": calculate_gc(seq_81bp)
            })
        return pd.DataFrame(parsed)

    df_hp = parse_sheet_tss("High Confidence (TSS_100.4)", "TSS_position", "High_Conf_Primary")
    df_hs = parse_sheet_tss("Secondary TSS, High confidence", "Secondary_TSS", "High_Conf_Secondary")
    df_lp = parse_sheet_tss("Low Confidence (TSS_2.1)", "TSS_position", "Low_Conf_Primary")
    df_ls = parse_sheet_tss("Secondary TSS, Low confidence", "Secondary_TSS", "Low_Conf_Secondary")
    
    df_all_tss = pd.concat([df_hp, df_hs, df_lp, df_ls], ignore_index=True)
    all_known_tss_sorted = sorted(df_all_tss["TSS_Position"].unique())
    print(f"Total Unique Known TSSs across all 4 sheets: {len(all_known_tss_sorted)}")

    # Define the 4 target subsets
    subsets = {
        "subset_1_high_conf_primary": {
            "name": "TIGR4 IGR High Confidence Primary",
            "df_pos": df_hp[df_hp["In_Refined_IGR"]].copy(),
            "desc": "Primary High-Confidence TSSs (TSS_100.4) located strictly inside refined IGRs."
        },
        "subset_2_high_conf_all": {
            "name": "TIGR4 IGR High Confidence Primary + Secondary",
            "df_pos": pd.concat([df_hp[df_hp["In_Refined_IGR"]], df_hs[df_hs["In_Refined_IGR"]]], ignore_index=True),
            "desc": "All High-Confidence TSSs (Primary + Secondary) located strictly inside refined IGRs."
        },
        "subset_3_all_primary": {
            "name": "TIGR4 IGR All Primary (High + Low Confidence)",
            "df_pos": pd.concat([df_hp[df_hp["In_Refined_IGR"]], df_lp[df_lp["In_Refined_IGR"]]], ignore_index=True),
            "desc": "All Primary TSSs (High and Low confidence) located strictly inside refined IGRs."
        },
        "subset_4_all_comprehensive": {
            "name": "TIGR4 IGR All Comprehensive (4 Sheets)",
            "df_pos": df_all_tss[df_all_tss["In_Refined_IGR"]].copy(),
            "desc": "Comprehensive set of all experimental TSSs from S1_TSS.xlsx located strictly inside refined IGRs."
        }
    }
    
    summary_records = []
    
    for sub_key, sub_info in subsets.items():
        sub_dir = OUT_BASE / sub_key
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        df_p = sub_info["df_pos"].drop_duplicates(subset=["TSS_Position", "Strand"]).reset_index(drop=True)
        # Assign unique Sequence_ID
        df_p["Sequence_ID"] = [f"POS_TIGR4_IGR_{i+1:04d}_{r['TSS_Position']}_{r['Strand']}" for i, r in df_p.iterrows()]
        
        n_pos = len(df_p)
        print(f"\nProcessing {sub_info['name']} (N={n_pos} positives)...")
        
        # Sample balanced 1:1 IGR negatives
        df_n = sample_igr_negatives(linear_igrs, all_known_tss_sorted, genome_seq, genome_len, num_needed=n_pos, seed=42, tss_margin=50)
        
        # Save Positives FASTA and TSV
        pos_fa = sub_dir / "positives_81bp.fasta"
        pos_tsv = sub_dir / "positives_metadata.tsv"
        with open(pos_fa, "w") as f:
            for _, r in df_p.iterrows():
                f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
        df_p.to_csv(pos_tsv, sep="\t", index=False)
        
        # Save Negatives FASTA and TSV
        neg_fa = sub_dir / "negatives_81bp.fasta"
        neg_tsv = sub_dir / "negatives_metadata.tsv"
        with open(neg_fa, "w") as f:
            for _, r in df_n.iterrows():
                f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
        df_n.to_csv(neg_tsv, sep="\t", index=False)
        
        print(f"  [SAVED] {pos_fa.name} (N={len(df_p)}) & {neg_fa.name} (N={len(df_n)})")
        
        summary_records.append({
            "Subset_Key": sub_key,
            "Subset_Name": sub_info["name"],
            "Positives_N": n_pos,
            "Negatives_N": len(df_n),
            "Total_N": n_pos + len(df_n),
            "Pos_GC_Mean": round(df_p["GC_Percent"].mean(), 3),
            "Neg_GC_Mean": round(df_n["GC_Percent"].mean(), 3),
            "Description": sub_info["desc"]
        })

    # Save summary report
    sum_df = pd.DataFrame(summary_records)
    sum_tsv = OUT_BASE / "TIGR4_IGR_DATASETS_SUMMARY.tsv"
    sum_df.to_csv(sum_tsv, sep="\t", index=False)
    
    print("\n" + "=" * 80)
    print("TIGR4 IGR DATASETS GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(sum_df.to_string(index=False))

if __name__ == "__main__":
    main()
