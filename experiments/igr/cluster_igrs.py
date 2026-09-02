#!/usr/bin/env python3
"""
Compile the cross-strain IGR cluster dataset (D39V vs TIGR4, MMseqs2).

Consumes MMseqs2 combined clusters + refined IGR TSVs + TSS metadata to produce:
  - output/tables/igr_ortholog_pairs.tsv        (cross-strain 1:1 pairs)
  - output/tables/igr_singletons.tsv            (525 D39V + 599 TIGR4)
  - output/tables/igr_multihit_and_paralogs.tsv (multi-hit families / intra-strain paralogs)

Usage:
    python experiments/igr/cluster_igrs.py
"""

import sys
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner

ROOT = Path(__file__).resolve().parents[2]


def main():
    OUT_TABLES = ROOT / "output/tables"
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # 1. Load Cluster TSV (2,247 clusters)
    CLU_PATH = ROOT / "output/intergenic/mmseqs2/combined_clusters/All_id0_95_cov0_70_cluster.tsv"
    df_clu = pd.read_csv(CLU_PATH, sep="\t", header=None, names=["rep", "member"])

    df_d39v_igrs = pd.read_csv(ROOT / "output/intergenic/d39v/D39V_igrs.tsv", sep="\t").set_index("igr_id")
    df_tigr4_igrs = pd.read_csv(ROOT / "output/intergenic/tigr4/TIGR4_igrs.tsv", sep="\t").set_index("igr_id")

    # Load TSS data
    df_d39v_tss = pd.read_csv(ROOT / "data/benchmark/d39v/positives_81bp_metadata.tsv", sep="\t")
    df_tigr4_tss = pd.read_csv(ROOT / "output/tigr4_data/positives_tigr4_all_tss_81bp.tsv", sep="\t")

    # Map TSS to IGRs
    d39v_tss_by_igr = {}
    for _, row in df_d39v_tss.iterrows():
        pos = int(row["TSS_Position_0based"])
        strand = str(row["Strand"])
        tss_id = str(row["TSS_ID"])
        sigma = str(row["Sigma_Factor"]) if pd.notna(row["Sigma_Factor"]) else "None"
        downstream = str(row["Downstream_Gene"]) if pd.notna(row["Downstream_Gene"]) else "-"
    
        matches = df_d39v_igrs[(df_d39v_igrs["start"] - 20 <= pos) & (pos <= df_d39v_igrs["end"] + 20)]
        for igr_id, igr_row in matches.iterrows():
            if igr_id not in d39v_tss_by_igr:
                d39v_tss_by_igr[igr_id] = []
            d39v_tss_by_igr[igr_id].append({
                "tss_id": tss_id,
                "pos": pos,
                "strand": strand,
                "sigma": sigma,
                "downstream": downstream
            })

    tigr4_tss_by_igr = {}
    for _, row in df_tigr4_tss.iterrows():
        pos = int(row["TSS_Position"])
        strand = str(row["Strand"])
        seq_id = str(row["Sequence_ID"])
        locus = str(row["Locus_Tag"]) if pd.notna(row["Locus_Tag"]) else "-"
    
        matches = df_tigr4_igrs[(df_tigr4_igrs["start"] - 20 <= pos) & (pos <= df_tigr4_igrs["end"] + 20)]
        for igr_id, igr_row in matches.iterrows():
            if igr_id not in tigr4_tss_by_igr:
                tigr4_tss_by_igr[igr_id] = []
            tigr4_tss_by_igr[igr_id].append({
                "seq_id": seq_id,
                "pos": pos,
                "strand": strand,
                "locus": locus
            })

    # Setup Aligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1

    # 2. Process Clusters
    clu_groups = df_clu.groupby("rep")["member"].apply(list).to_dict()

    ortho_records = []
    singleton_records = []
    multihit_records = []

    for rep, members in clu_groups.items():
        d39v_m = [m for m in members if "D39V" in m]
        tigr4_m = [m for m in members if "NC_003028" in m or "TIGR4" in m]
    
        if len(members) == 2 and len(d39v_m) == 1 and len(tigr4_m) == 1:
            d_id = d39v_m[0]
            t_id = tigr4_m[0]
        
            row_d = df_d39v_igrs.loc[d_id]
            row_t = df_tigr4_igrs.loc[t_id]
        
            # Alignment stats
            seq_d = row_d["sequence"]
            seq_t = row_t["sequence"]
            score = aligner.score(seq_d, seq_t)
        
            min_len = min(len(seq_d), len(seq_t))
            ident_pct = min(100.0, max(0.0, (score / min_len) * 100.0))
        
            orient_d = row_d["orientation_type"]
            orient_t = row_t["orientation_type"]
            orient_concordant = (orient_d == orient_t)
        
            d_tss = d39v_tss_by_igr.get(d_id, [])
            t_tss = tigr4_tss_by_igr.get(t_id, [])
        
            has_d_tss = len(d_tss) > 0
            has_t_tss = len(t_tss) > 0
        
            if has_d_tss and has_t_tss:
                tss_cat = "Both Strains (Conserved Promoter)"
            elif has_d_tss:
                tss_cat = "D39V Only"
            elif has_t_tss:
                tss_cat = "TIGR4 Only"
            else:
                tss_cat = "Neither Strain"
            
            sigmas = list(set(str(t["sigma"]) for t in d_tss))
            sigmas_str = ",".join(sigmas) if sigmas else "None"
        
            ortho_records.append({
                "Cluster_ID": rep,
                "D39V_IGR_ID": d_id,
                "D39V_Start": row_d["start"],
                "D39V_End": row_d["end"],
                "D39V_Length_bp": row_d["length"],
                "D39V_Orientation": orient_d,
                "D39V_Left_CDS": f"{row_d['left_cds']}({row_d['left_strand']})",
                "D39V_Right_CDS": f"{row_d['right_cds']}({row_d['right_strand']})",
                "TIGR4_IGR_ID": t_id,
                "TIGR4_Start": row_t["start"],
                "TIGR4_End": row_t["end"],
                "TIGR4_Length_bp": row_t["length"],
                "TIGR4_Orientation": orient_t,
                "TIGR4_Left_CDS": f"{row_t['left_cds']}({row_t['left_strand']})",
                "TIGR4_Right_CDS": f"{row_t['right_cds']}({row_t['right_strand']})",
                "Orientation_Concordant": orient_concordant,
                "Length_Delta_bp": abs(row_d["length"] - row_t["length"]),
                "Length_Ratio": round(max(row_d["length"], row_t["length"]) / max(1, min(row_d["length"], row_t["length"])), 2),
                "Pairwise_Score": score,
                "Pairwise_Identity_Pct": round(ident_pct, 2),
                "D39V_Has_TSS": has_d_tss,
                "D39V_TSS_Count": len(d_tss),
                "D39V_TSS_IDs": ";".join(str(t["tss_id"]) for t in d_tss) if d_tss else "-",
                "D39V_Sigma_Factors": sigmas_str,
                "TIGR4_Has_TSS": has_t_tss,
                "TIGR4_TSS_Count": len(t_tss),
                "TIGR4_TSS_IDs": ";".join(str(t["seq_id"]) for t in t_tss) if t_tss else "-",
                "TSS_Conservation_Status": tss_cat
            })
        
        elif len(members) == 1:
            # Singleton
            m = members[0]
            if "D39V" in m:
                strain = "D39V"
                row = df_d39v_igrs.loc[m]
                t_list = d39v_tss_by_igr.get(m, [])
                sigmas = list(set(str(t["sigma"]) for t in t_list))
                sigmas_str = ",".join(sigmas) if sigmas else "None"
                tss_ids_str = ";".join(str(t["tss_id"]) for t in t_list) if t_list else "-"
            else:
                strain = "TIGR4"
                row = df_tigr4_igrs.loc[m]
                t_list = tigr4_tss_by_igr.get(m, [])
                sigmas_str = "-"
                tss_ids_str = ";".join(str(t["seq_id"]) for t in t_list) if t_list else "-"
            
            length = row["length"]
            if length < 50:
                size_cat = "Small Spacer (<50 bp)"
            elif length <= 150:
                size_cat = "Operon Spacer (50-150 bp)"
            else:
                size_cat = "Promoter IGR (>150 bp)"
            
            singleton_records.append({
                "Singleton_ID": m,
                "Strain": strain,
                "Chromosome": row["chrom"],
                "Start": row["start"],
                "End": row["end"],
                "Length_bp": length,
                "Orientation_Type": row["orientation_type"],
                "Left_CDS": f"{row['left_cds']}({row['left_strand']})",
                "Right_CDS": f"{row['right_cds']}({row['right_strand']})",
                "Size_Category": size_cat,
                "Has_TSS": len(t_list) > 0,
                "TSS_Count": len(t_list),
                "TSS_IDs": tss_ids_str,
                "Sigma_Factors": sigmas_str
            })
        
        else:
            # Multi-hit or intra-strain paralog
            if len(members) == 2 and len(d39v_m) == 2:
                clu_type = "Intra-Strain D39V Paralog Pair"
                mol_note = "Local tandem duplication / operon repeat in D39V"
            elif len(members) == 2 and len(tigr4_m) == 2:
                clu_type = "Intra-Strain TIGR4 Paralog Pair"
                mol_note = "Local tandem duplication / operon repeat in TIGR4"
            else:
                clu_type = f"Multi-Hit Family (Size {len(members)})"
                if len(members) >= 5 and len(d39v_m) >= 2 and len(tigr4_m) >= 2:
                    mol_note = "Ribosomal RNA Operon Promoters (rrnA-D) / BOX Conserved Repeats"
                elif "1399" in rep or "0383" in rep:
                    mol_note = "BOX / RUP Repeat Unit Pneumococcal Family"
                elif len(d39v_m) == 0:
                    mol_note = "TIGR4-Specific Transposase / Mobile Insertion Sequence"
                else:
                    mol_note = "Cross-strain Multi-copy Intergenic Family"
                
            multihit_records.append({
                "Cluster_ID": rep,
                "Cluster_Type": clu_type,
                "Total_Size": len(members),
                "D39V_Members_Count": len(d39v_m),
                "TIGR4_Members_Count": len(tigr4_m),
                "D39V_Members": ";".join(d39v_m) if d39v_m else "-",
                "TIGR4_Members": ";".join(tigr4_m) if tigr4_m else "-",
                "Molecular_Annotation": mol_note
            })

    # Create DataFrames
    df_ortho = pd.DataFrame(ortho_records)
    df_single = pd.DataFrame(singleton_records)
    df_multi = pd.DataFrame(multihit_records)

    # Save Master TSVs
    df_ortho.to_csv(OUT_TABLES / "igr_ortholog_pairs.tsv", sep="\t", index=False)
    df_single.to_csv(OUT_TABLES / "igr_singletons.tsv", sep="\t", index=False)
    df_multi.to_csv(OUT_TABLES / "igr_multihit_and_paralogs.tsv", sep="\t", index=False)

    print(f"Successfully compiled {len(df_ortho)} ortholog pairs to igr_ortholog_pairs.tsv")
    print(f"Successfully compiled {len(df_single)} singletons to igr_singletons.tsv")
    print(f"Successfully compiled {len(df_multi)} multi-hit/paralogs to igr_multihit_and_paralogs.tsv")
    print(f"Total Clusters: {len(df_ortho) + len(df_single) + len(df_multi)}")



if __name__ == "__main__":
    main()
