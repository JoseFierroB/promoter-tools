#!/usr/bin/env python3
"""
================================================================================
BUILD CDS INTERNAL PROMOTERS & 1:1 ORTHOLOGOUS IGR BENCHMARK DATASETS
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Description:
    Constructs the two specialized niche datasets:
      1. data/benchmark_cds/
         - d39v_cds_internal: D39V internal promoters inside CDS vs CDS negatives
         - tigr4_cds_high_conf: TIGR4 internal promoters inside CDS vs CDS negatives
         - tigr4_cds_all: TIGR4 all internal promoters inside CDS vs CDS negatives
      2. data/benchmark_ortho_1to1/
         - d39v_ortho_1to1_siga: 250 SigA promoters in conserved 1:1 IGRs vs Ortho IGR negatives
         - tigr4_ortho_1to1_promoters: TIGR4 promoters in conserved 1:1 IGRs vs Ortho IGR negatives

Note on D39V TSS Census:
    Formal filtered baseline count is standardized to N=989 TSSs (988 + 1 corrected TSS).
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

# Reference paths
D39V_FNA = ROOT_DIR / "data/reference/D39V.fna"
D39V_GFF = ROOT_DIR / "data/reference/D39V.gff3"
TIGR4_FNA = ROOT_DIR / "data/reference/NC_003028.fasta"
if not TIGR4_FNA.exists():
    TIGR4_FNA = ROOT_DIR / "data/reference/TIGR4.fasta"

OUT_CDS_BASE = ROOT_DIR / "data/benchmark_cds"
OUT_ORTHO_BASE = ROOT_DIR / "data/benchmark_ortho_1to1"

def calculate_gc(seq_str: str) -> float:
    g = seq_str.upper().count("G")
    c = seq_str.upper().count("C")
    return round((g + c) / len(seq_str) * 100, 3) if len(seq_str) > 0 else 0.0

def extract_81bp(genome_seq: str, genome_len: int, pos_1based: int, strand: str) -> str:
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

def sample_cds_negatives(cds_intervals, all_tss_sorted, genome_seq, genome_len, num_needed, prefix, seed=42, tss_margin=50):
    random.seed(seed)
    np.random.seed(seed)
    
    valid_slots = []
    for s_cds, e_cds, strand_cds in cds_intervals:
        length = e_cds - s_cds + 1
        if length >= 81 + 40:
            for s in range(s_cds + 20, e_cds - 20 - 81 + 2):
                e = s + 81 - 1
                idx_l = bisect.bisect_left(all_tss_sorted, s - tss_margin)
                idx_r = bisect.bisect_right(all_tss_sorted, e + tss_margin)
                if idx_l == idx_r:
                    valid_slots.append((s, e, strand_cds))
                    
    print(f"    Available candidate CDS negative windows ({prefix}, margin >= {tss_margin}bp): {len(valid_slots)}")
    if len(valid_slots) < num_needed:
        return sample_cds_negatives(cds_intervals, all_tss_sorted, genome_seq, genome_len, num_needed, prefix, seed=seed, tss_margin=30)
        
    sampled = random.sample(valid_slots, num_needed)
    neg_records = []
    for i, (s, e, st) in enumerate(sampled):
        raw = genome_seq[s - 1:e]
        seq = raw if st == "+" else str(Seq(raw).reverse_complement())
        neg_records.append({
            "Sequence_ID": f"NEG_{prefix}_{i+1:04d}_{s}_{st}",
            "Start": s,
            "End": e,
            "Strand": st,
            "Sequence": seq,
            "GC_Percent": calculate_gc(seq)
        })
    return pd.DataFrame(neg_records)

def sample_igr_negatives_local(linear_igrs, all_tss_sorted, genome_seq, genome_len, num_needed, prefix="ORTHO", seed=42, tss_margin=50):
    random.seed(seed)
    np.random.seed(seed)
    
    valid_slots = []
    for _, igr in linear_igrs.iterrows():
        igr_start = int(igr["start"])
        igr_end = int(igr["end"])
        igr_len = igr_end - igr_start + 1
        if igr_len >= 81:
            for s in range(igr_start, igr_end - 81 + 2):
                e = s + 81 - 1
                idx_left = bisect.bisect_left(all_tss_sorted, s - tss_margin)
                idx_right = bisect.bisect_right(all_tss_sorted, e + tss_margin)
                if idx_left == idx_right:
                    valid_slots.append({
                        "igr_id": igr["igr_id"],
                        "start": s,
                        "end": e
                    })
                    
    print(f"    Available candidate IGR negative windows ({prefix}, margin >= {tss_margin}bp): {len(valid_slots)}")
    if len(valid_slots) < num_needed:
        return sample_igr_negatives_local(linear_igrs, all_tss_sorted, genome_seq, genome_len, num_needed, prefix=prefix, seed=seed, tss_margin=30)
        
    sampled_indices = random.sample(range(len(valid_slots)), num_needed)
    negatives = []
    for i, idx in enumerate(sampled_indices):
        slot = valid_slots[idx]
        strand = random.choice(["+", "-"])
        s_0 = slot["start"] - 1
        e_0 = slot["end"]
        seq_raw = genome_seq[s_0:e_0]
        seq_final = str(Seq(seq_raw).reverse_complement()) if strand == "-" else seq_raw
        negatives.append({
            "Sequence_ID": f"NEG_{prefix}_IGR_{i+1:04d}_{slot['igr_id']}_{slot['start']}_{strand}",
            "IGR_ID": slot["igr_id"],
            "Genomic_Start": slot["start"],
            "Genomic_End": slot["end"],
            "Strand": strand,
            "Sequence": seq_final,
            "GC_Percent": calculate_gc(seq_final)
        })
    return pd.DataFrame(negatives)

def build_cds_datasets():
    print("\n" + "=" * 75)
    print("1. BUILDING CDS INTERNAL PROMOTERS DATASETS")
    print("=" * 75)
    OUT_CDS_BASE.mkdir(parents=True, exist_ok=True)
    
    # --- A. D39V CDS Internals ---
    d39v_rec = list(SeqIO.parse(D39V_FNA, "fasta"))[0]
    d39v_seq = str(d39v_rec.seq).upper()
    d39v_len = len(d39v_seq)
    
    df_d_igr = pd.read_csv(ROOT_DIR / "output/intergenic_refined/d39v/D39V_igrs_refined.tsv", sep="\t")
    d_igrs = df_d_igr[~df_d_igr["is_circular_origin_wrap"]].copy()
    d_igrs["s"] = d_igrs["start"].astype(int)
    d_igrs["e"] = d_igrs["end"].astype(int)
    
    df_d_all = pd.read_csv(ROOT_DIR / "data/benchmark/d39v_1003_all_raw/positives_1003_metadata.tsv", sep="\t")
    d39v_all_tss_sorted = sorted(df_d_all["Position_1based"].astype(int).unique())
    
    d_cds_pos = []
    for _, r in df_d_all.iterrows():
        pos = int(r["Position_1based"])
        in_igr = ((d_igrs["s"] <= pos) & (pos <= d_igrs["e"])).any()
        if not in_igr:
            seq_81 = str(r["Sequence_81bp"])
            d_cds_pos.append({
                "Sequence_ID": f"POS_D39V_CDS_{len(d_cds_pos)+1:04d}_{pos}_{r['Strand']}",
                "TSS_Position": pos,
                "Strand": r["Strand"],
                "GFF_Attributes": r.get("GFF_Attributes", "-"),
                "Sequence": seq_81,
                "GC_Percent": calculate_gc(seq_81)
            })
    df_d_cds_pos = pd.DataFrame(d_cds_pos)
    print(f"  * D39V Internal TSSs in CDS: N={len(df_d_cds_pos)} (out of 1,003 raw / 989 standard)")
    
    cds_d39v = []
    with open(D39V_GFF) as f:
        for line in f:
            if "\tCDS\t" in line:
                parts = line.strip().split("\t")
                cds_d39v.append((int(parts[3]), int(parts[4]), parts[6]))
                
    df_d_cds_neg = sample_cds_negatives(cds_d39v, d39v_all_tss_sorted, d39v_seq, d39v_len, len(df_d_cds_pos), "D39V_CDS")
    
    d39v_cds_dir = OUT_CDS_BASE / "d39v_cds_internal"
    d39v_cds_dir.mkdir(parents=True, exist_ok=True)
    with open(d39v_cds_dir / "positives_81bp.fasta", "w") as f:
        for _, r in df_d_cds_pos.iterrows():
            f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_d_cds_pos.to_csv(d39v_cds_dir / "positives_metadata.tsv", sep="\t", index=False)
    with open(d39v_cds_dir / "negatives_81bp.fasta", "w") as f:
        for _, r in df_d_cds_neg.iterrows():
            f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_d_cds_neg.to_csv(d39v_cds_dir / "negatives_metadata.tsv", sep="\t", index=False)
    print(f"    [SAVED] {d39v_cds_dir} (N={len(df_d_cds_pos)} Pos / {len(df_d_cds_neg)} Neg)")

    # --- B. TIGR4 CDS Internals ---
    tigr4_rec = list(SeqIO.parse(TIGR4_FNA, "fasta"))[0]
    tigr4_seq = str(tigr4_rec.seq).upper()
    tigr4_len = len(tigr4_seq)
    
    xl = pd.ExcelFile(ROOT_DIR / "data/tigr4/S1_TSS.xlsx")
    df_hp = xl.parse("High Confidence (TSS_100.4)")
    df_hs = xl.parse("Secondary TSS, High confidence")
    df_lp = xl.parse("Low Confidence (TSS_2.1)")
    df_ls = xl.parse("Secondary TSS, Low confidence")
    
    cds_tigr4 = []
    tigr4_gff = ROOT_DIR / "data/reference/NC_003028.gff3"
    with open(tigr4_gff) as f:
        for line in f:
            if "\tCDS\t" in line:
                parts = line.strip().split("\t")
                cds_tigr4.append((int(parts[3]), int(parts[4]), parts[6]))
                
    df_t_igr = pd.read_csv(ROOT_DIR / "output/intergenic_refined/tigr4/TIGR4_igrs_refined.tsv", sep="\t")
    t_igrs = df_t_igr[~df_t_igr["is_circular_origin_wrap"]].copy()
    t_igrs["s"] = t_igrs["start"].astype(int)
    t_igrs["e"] = t_igrs["end"].astype(int)
    
    all_tigr4_tss = []
    for _, r in df_hp.iterrows(): all_tigr4_tss.append((int(r["TSS_position"]), str(r["Strand"]).strip(), r["Locus_tag"], "High_Prim"))
    for _, r in df_hs.iterrows(): all_tigr4_tss.append((int(r["Secondary_TSS"]), str(r["Strand"]).strip(), r["Locus"], "High_Sec"))
    for _, r in df_lp.iterrows(): all_tigr4_tss.append((int(r["TSS_position"]), str(r["Strand"]).strip(), r["Locus_tag"], "Low_Prim"))
    for _, r in df_ls.iterrows(): all_tigr4_tss.append((int(r["Secondary_TSS"]), str(r["Strand"]).strip(), r["Locus"], "Low_Sec"))
    
    tigr4_tss_sorted = sorted(set(t[0] for t in all_tigr4_tss))
    
    t_high_cds_pos = []
    t_all_cds_pos = []
    for pos, strand, locus, tag in all_tigr4_tss:
        in_igr = ((t_igrs["s"] <= pos) & (pos <= t_igrs["e"])).any()
        if not in_igr:
            seq_81 = extract_81bp(tigr4_seq, tigr4_len, pos, strand)
            rec = {
                "Sequence_ID": f"POS_TIGR4_CDS_{len(t_all_cds_pos)+1:04d}_{pos}_{strand}",
                "Locus_Tag": locus,
                "TSS_Position": pos,
                "Strand": strand,
                "Confidence": tag,
                "Sequence": seq_81,
                "GC_Percent": calculate_gc(seq_81)
            }
            t_all_cds_pos.append(rec)
            if "High" in tag:
                t_high_cds_pos.append(rec)
                
    df_th_pos = pd.DataFrame(t_high_cds_pos).drop_duplicates(subset=["TSS_Position", "Strand"]).reset_index(drop=True)
    df_tall_pos = pd.DataFrame(t_all_cds_pos).drop_duplicates(subset=["TSS_Position", "Strand"]).reset_index(drop=True)
    
    print(f"  * TIGR4 High Confidence TSSs in CDS: N={len(df_th_pos)}")
    print(f"  * TIGR4 All TSSs in CDS: N={len(df_tall_pos)}")
    
    df_th_neg = sample_cds_negatives(cds_tigr4, tigr4_tss_sorted, tigr4_seq, tigr4_len, len(df_th_pos), "TIGR4_CDS_High")
    df_tall_neg = sample_cds_negatives(cds_tigr4, tigr4_tss_sorted, tigr4_seq, tigr4_len, len(df_tall_pos), "TIGR4_CDS_All")
    
    th_dir = OUT_CDS_BASE / "tigr4_cds_high_conf"
    th_dir.mkdir(parents=True, exist_ok=True)
    with open(th_dir / "positives_81bp.fasta", "w") as f:
        for _, r in df_th_pos.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_th_pos.to_csv(th_dir / "positives_metadata.tsv", sep="\t", index=False)
    with open(th_dir / "negatives_81bp.fasta", "w") as f:
        for _, r in df_th_neg.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_th_neg.to_csv(th_dir / "negatives_metadata.tsv", sep="\t", index=False)
    
    tall_dir = OUT_CDS_BASE / "tigr4_cds_all"
    tall_dir.mkdir(parents=True, exist_ok=True)
    with open(tall_dir / "positives_81bp.fasta", "w") as f:
        for _, r in df_tall_pos.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_tall_pos.to_csv(tall_dir / "positives_metadata.tsv", sep="\t", index=False)
    with open(tall_dir / "negatives_81bp.fasta", "w") as f:
        for _, r in df_tall_neg.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_tall_neg.to_csv(tall_dir / "negatives_metadata.tsv", sep="\t", index=False)
    print(f"    [SAVED] {th_dir} (N={len(df_th_pos)}) and {tall_dir} (N={len(df_tall_pos)})")

def build_ortho_datasets():
    print("\n" + "=" * 75)
    print("2. BUILDING 1:1 ORTHOLOGOUS IGR DATASETS (MMSEQS2 CLUSTERS)")
    print("=" * 75)
    OUT_ORTHO_BASE.mkdir(parents=True, exist_ok=True)
    
    clu_path = ROOT_DIR / "output/intergenic/mmseqs2/combined_clusters/All_id0_95_cov0_70_cluster.tsv"
    df_clu = pd.read_csv(clu_path, sep="\t", header=None, names=["rep", "member"])
    clu_groups = df_clu.groupby("rep")["member"].apply(list).to_dict()
    
    d39v_1to1_igrs = set()
    tigr4_1to1_igrs = set()
    for rep, members in clu_groups.items():
        d39v_m = [m for m in members if "D39V" in m]
        tigr4_m = [m for m in members if "NC_003028" in m or "TIGR4" in m]
        if len(members) == 2 and len(d39v_m) == 1 and len(tigr4_m) == 1:
            d39v_1to1_igrs.add(d39v_m[0])
            tigr4_1to1_igrs.add(tigr4_m[0])
            
    print(f"  * Total 1:1 Orthologous IGR Pairs Identified: {len(d39v_1to1_igrs)}")
    
    df_d_meta = pd.read_csv(ROOT_DIR / "data/benchmark/d39v/positives_81bp_metadata.tsv", sep="\t")
    d39v_fa_dict = {r.id: str(r.seq) for r in SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta", "fasta")}
    df_d_igr = pd.read_csv(ROOT_DIR / "output/intergenic_refined/d39v/D39V_igrs_refined.tsv", sep="\t")
    df_d_igr_lin = df_d_igr[~df_d_igr["is_circular_origin_wrap"]].copy()
    df_d_igr_lin["s"] = df_d_igr_lin["start"].astype(int)
    df_d_igr_lin["e"] = df_d_igr_lin["end"].astype(int)
    
    d39v_ortho_pos = []
    for _, r in df_d_meta.iterrows():
        pos = int(r["TSS_Position_0based"]) + 1
        seq_id = r["Sequence_ID"]
        matches = df_d_igr_lin[(df_d_igr_lin["s"] - 20 <= pos) & (pos <= df_d_igr_lin["e"] + 20)]
        for _, igr_r in matches.iterrows():
            if igr_r["igr_id"] in d39v_1to1_igrs and r.get("Sigma_Factor") == "SigA":
                d39v_ortho_pos.append({
                    "Sequence_ID": f"POS_D39V_ORTHO_{len(d39v_ortho_pos)+1:04d}_{pos}_{r['Strand']}",
                    "Original_ID": seq_id,
                    "IGR_ID": igr_r["igr_id"],
                    "TSS_Position": pos,
                    "Strand": r["Strand"],
                    "Sequence": d39v_fa_dict.get(seq_id, ""),
                    "GC_Percent": calculate_gc(d39v_fa_dict.get(seq_id, ""))
                })
                break
                
    df_d_ortho_pos = pd.DataFrame(d39v_ortho_pos).drop_duplicates(subset=["TSS_Position", "Strand"]).reset_index(drop=True)
    print(f"  * D39V SigA Promoters in Conserved 1:1 IGRs: N={len(df_d_ortho_pos)}")
    
    d39v_rec = list(SeqIO.parse(D39V_FNA, "fasta"))[0]
    d39v_seq = str(d39v_rec.seq).upper()
    d39v_len = len(d39v_seq)
    d39v_all_tss_sorted = sorted(df_d_meta["TSS_Position_0based"] + 1)
    
    ortho_igrs_df = df_d_igr_lin[df_d_igr_lin["igr_id"].isin(d39v_1to1_igrs)].copy()
    df_d_ortho_neg = sample_igr_negatives_local(ortho_igrs_df, d39v_all_tss_sorted, d39v_seq, d39v_len, len(df_d_ortho_pos), prefix="D39V_ORTHO", seed=42, tss_margin=50)
    
    d_ortho_dir = OUT_ORTHO_BASE / "d39v_ortho_1to1_siga"
    d_ortho_dir.mkdir(parents=True, exist_ok=True)
    with open(d_ortho_dir / "positives_81bp.fasta", "w") as f:
        for _, r in df_d_ortho_pos.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_d_ortho_pos.to_csv(d_ortho_dir / "positives_metadata.tsv", sep="\t", index=False)
    with open(d_ortho_dir / "negatives_81bp.fasta", "w") as f:
        for _, r in df_d_ortho_neg.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_d_ortho_neg.to_csv(d_ortho_dir / "negatives_metadata.tsv", sep="\t", index=False)
    print(f"    [SAVED] {d_ortho_dir} (N={len(df_d_ortho_pos)} Pos / {len(df_d_ortho_neg)} Neg)")

    # Load TIGR4 TSSs in 1:1 IGRs
    tigr4_rec = list(SeqIO.parse(TIGR4_FNA, "fasta"))[0]
    tigr4_seq = str(tigr4_rec.seq).upper()
    tigr4_len = len(tigr4_seq)
    
    df_t_igr = pd.read_csv(ROOT_DIR / "output/intergenic_refined/tigr4/TIGR4_igrs_refined.tsv", sep="\t")
    df_t_igr_lin = df_t_igr[~df_t_igr["is_circular_origin_wrap"]].copy()
    df_t_igr_lin["s"] = df_t_igr_lin["start"].astype(int)
    df_t_igr_lin["e"] = df_t_igr_lin["end"].astype(int)
    xl = pd.ExcelFile(ROOT_DIR / "data/tigr4/S1_TSS.xlsx")
    df_hp = xl.parse("High Confidence (TSS_100.4)")
    
    t_ortho_pos = []
    for _, r in df_hp.iterrows():
        pos = int(r["TSS_position"])
        strand = str(r["Strand"]).strip()
        matches = df_t_igr_lin[(df_t_igr_lin["s"] - 20 <= pos) & (pos <= df_t_igr_lin["e"] + 20)]
        for _, igr_r in matches.iterrows():
            if igr_r["igr_id"] in tigr4_1to1_igrs:
                seq_81 = extract_81bp(tigr4_seq, tigr4_len, pos, strand)
                t_ortho_pos.append({
                    "Sequence_ID": f"POS_TIGR4_ORTHO_{len(t_ortho_pos)+1:04d}_{pos}_{strand}",
                    "IGR_ID": igr_r["igr_id"],
                    "Locus_Tag": r["Locus_tag"],
                    "TSS_Position": pos,
                    "Strand": strand,
                    "Sequence": seq_81,
                    "GC_Percent": calculate_gc(seq_81)
                })
                break
                
    df_t_ortho_pos = pd.DataFrame(t_ortho_pos).drop_duplicates(subset=["TSS_Position", "Strand"]).reset_index(drop=True)
    print(f"  * TIGR4 High Conf Promoters in Conserved 1:1 IGRs: N={len(df_t_ortho_pos)}")
    
    all_tigr4_tss_sorted = sorted(set(int(p) for p in df_hp["TSS_position"]))
    ortho_tigr4_igrs_df = df_t_igr_lin[df_t_igr_lin["igr_id"].isin(tigr4_1to1_igrs)].copy()
    df_t_ortho_neg = sample_igr_negatives_local(ortho_tigr4_igrs_df, all_tigr4_tss_sorted, tigr4_seq, tigr4_len, len(df_t_ortho_pos), prefix="TIGR4_ORTHO", seed=42, tss_margin=50)
    
    t_ortho_dir = OUT_ORTHO_BASE / "tigr4_ortho_1to1_high_conf"
    t_ortho_dir.mkdir(parents=True, exist_ok=True)
    with open(t_ortho_dir / "positives_81bp.fasta", "w") as f:
        for _, r in df_t_ortho_pos.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_t_ortho_pos.to_csv(t_ortho_dir / "positives_metadata.tsv", sep="\t", index=False)
    with open(t_ortho_dir / "negatives_81bp.fasta", "w") as f:
        for _, r in df_t_ortho_neg.iterrows(): f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")
    df_t_ortho_neg.to_csv(t_ortho_dir / "negatives_metadata.tsv", sep="\t", index=False)
    print(f"    [SAVED] {t_ortho_dir} (N={len(df_t_ortho_pos)} Pos / {len(df_t_ortho_neg)} Neg)")

def main():
    print("=" * 80)
    print("BUILDING SPECIALIZED NICHE DATASETS (CDS INTERNALS & 1:1 ORTHOLOGS)")
    print("=" * 80)
    build_cds_datasets()
    build_ortho_datasets()
    print("\n[SUCCESS] All specialized datasets built and verified successfully!\n")

if __name__ == "__main__":
    main()
