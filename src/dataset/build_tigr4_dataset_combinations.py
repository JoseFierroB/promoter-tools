#!/usr/bin/env python3
"""
Builds all 4 Combinations of TIGR4 Promoter Datasets:
1. tigr4_prim_high      : Primary High Confidence (N ≈ 738)
2. tigr4_prim_high_sec  : Primary + Secondary High Confidence (N ≈ 780-786)
3. tigr4_extended_prim  : Primary High Confidence + Primary Low Confidence (N ≈ 2,000)
4. tigr4_extended_all   : All High Confidence (Prim+Sec) + All Low Confidence (Prim+Sec) (N ≈ 2,120-2,150)

Each dataset includes:
- Positive FASTA (81-mer: -60 to +20)
- Positive TSV (rich metadata)
- GC-matched Negative FASTA (1:1 ratio, >=200 bp to any TSS, <25 bp steric conflict resolution)

Usage:
    pixi run python src/dataset/build_tigr4_dataset_combinations.py
"""

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parent.parent.parent
TIGR4_DIR = ROOT / "output/tigr4_data"
TIGR4_DIR.mkdir(parents=True, exist_ok=True)

XLSX_PATH = TIGR4_DIR / "S1_TSS.xlsx"
GENOME_FASTA = ROOT / "data/reference/NC_003028.fasta"


def load_genome() -> Tuple[str, str]:
    rec = list(SeqIO.parse(GENOME_FASTA, "fasta"))[0]
    return rec.id, str(rec.seq).upper()


def extract_81mer(seq: str, tss_pos: int, strand: str, up: int = 60, down: int = 20) -> str:
    pos_0 = tss_pos - 1
    if strand == "+":
        start = pos_0 - up
        end = pos_0 + down + 1
        if start < 0 or end > len(seq):
            return ""
        return seq[start:end]
    else:
        start = pos_0 - down
        end = pos_0 + up + 1
        if start < 0 or end > len(seq):
            return ""
        sub = seq[start:end]
        return str(Seq(sub).reverse_complement()).upper()


def filter_closeness_conflicts(records: List[Dict], threshold_bp: int = 25) -> List[Dict]:
    sorted_recs = sorted(records, key=lambda x: (x["Chromosome"], x["Strand"], x["TSS_Position"]))
    kept = []

    for r in sorted_recs:
        if not kept:
            kept.append(r)
            continue
        prev = kept[-1]
        if prev["Chromosome"] == r["Chromosome"] and prev["Strand"] == r["Strand"]:
            if abs(r["TSS_Position"] - prev["TSS_Position"]) < threshold_bp:
                continue
        kept.append(r)

    return kept


def load_xlsx_records(genome_id: str, genome_seq: str) -> Dict[str, List[Dict]]:
    xl = pd.ExcelFile(XLSX_PATH)
    sheets_dict = {}

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name)
        recs = []

        is_secondary = "Secondary" in sheet_name

        for idx, row in df.iterrows():
            if is_secondary and "Secondary_TSS" in row:
                pos_col = "Secondary_TSS"
            else:
                pos_col = "TSS_position" if "TSS_position" in row else ("Primary_TSS" if "Primary_TSS" in row else None)

            if pos_col is None or pd.isna(row[pos_col]):
                continue

            try:
                tss_pos = int(row[pos_col])
            except ValueError:
                continue

            strand = str(row["Strand"]).strip()
            locus = str(row["Locus_tag" if "Locus_tag" in row else ("Locus" if "Locus" in row else "UNKNOWN")]).strip()

            seq_81 = extract_81mer(genome_seq, tss_pos, strand)
            if len(seq_81) != 81:
                continue

            gc_pct = (seq_81.count("G") + seq_81.count("C")) / 81.0 * 100.0
            utr_len = row["5'-UTR_length" if "5'-UTR_length" in row else "Primary_5'-UTR_length"] if ("5'-UTR_length" in row or "Primary_5'-UTR_length" in row) and not pd.isna(row.get("5'-UTR_length", row.get("Primary_5'-UTR_length"))) else -1

            rec = {
                "Sequence_ID": f"TIGR4_{locus}_TSS_{tss_pos}_{strand}",
                "Locus_Tag": locus,
                "Chromosome": genome_id,
                "TSS_Position": tss_pos,
                "Strand": strand,
                "Window_Size": 81,
                "GC_Content": round(gc_pct, 2),
                "UTR5_Length": utr_len,
                "Confidence_Sheet": sheet_name,
                "Sequence_81bp": seq_81,
            }
            recs.append(rec)

        sheets_dict[sheet_name] = recs

    return sheets_dict


def sample_gc_matched_negatives(
    genome_id: str, genome_seq: str, pos_records: List[Dict], target_count: int, target_gc: float = 30.87
) -> List[Dict]:
    pos_tss_set = set((r["Chromosome"], r["Strand"], r["TSS_Position"]) for r in pos_records)
    random.seed(42)

    candidates = []
    for i in range(100, len(genome_seq) - 100, 10):
        for strand in ["+", "-"]:
            conflict = False
            for tss_pos in range(i - 200, i + 201):
                if (genome_id, strand, tss_pos) in pos_tss_set:
                    conflict = True
                    break
            if conflict:
                continue

            seq_81 = extract_81mer(genome_seq, i, strand)
            if len(seq_81) == 81:
                gc = (seq_81.count("G") + seq_81.count("C")) / 81.0 * 100.0
                candidates.append({"TSS_Position": i, "Strand": strand, "GC_Content": gc, "Sequence_81bp": seq_81})

    candidates.sort(key=lambda x: abs(x["GC_Content"] - target_gc))

    kept_negs = []
    for cand in candidates:
        if len(kept_negs) >= target_count:
            break
        too_close = False
        for k in kept_negs:
            if k["Strand"] == cand["Strand"] and abs(k["TSS_Position"] - cand["TSS_Position"]) < 25:
                too_close = True
                break
        if not too_close:
            cand["Sequence_ID"] = f"TIGR4_NEG_CDS_{cand['TSS_Position']}_{cand['Strand']}"
            cand["Chromosome"] = genome_id
            kept_negs.append(cand)

    return kept_negs


def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" BUILDING ALL 4 COMBINATIONS OF TIGR4 PROMOTER DATASETS")
    print("═════════════════════════════════════════════════════════════════\n")

    genome_id, genome_seq = load_genome()
    sheets_dict = load_xlsx_records(genome_id, genome_seq)

    high_prim = sheets_dict["High Confidence (TSS_100.4)"]
    high_sec = sheets_dict.get("Secondary TSS, High confidence", [])
    low_prim = sheets_dict["Low Confidence (TSS_2.1)"]
    low_sec = sheets_dict.get("Secondary TSS, Low confidence", [])

    print(f"[INFO] Excel records loaded:")
    print(f"       High Confidence Primary: {len(high_prim)}")
    print(f"       High Confidence Secondary: {len(high_sec)}")
    print(f"       Low Confidence Primary: {len(low_prim)}")
    print(f"       Low Confidence Secondary: {len(low_sec)}")

    combinations = {
        "tigr4_prim_high": {
            "name": "TIGR4 Primary High Confidence (Core)",
            "pos_recs": high_prim,
        },
        "tigr4_prim_high_sec": {
            "name": "TIGR4 Primary + Secondary High Confidence",
            "pos_recs": high_prim + high_sec,
        },
        "tigr4_extended_prim": {
            "name": "TIGR4 Extended Primary (Primary High + Primary Low)",
            "pos_recs": high_prim + low_prim,
        },
        "tigr4_extended_all": {
            "name": "TIGR4 Extended All (Prim+Sec High + Prim+Sec Low)",
            "pos_recs": high_prim + high_sec + low_prim + low_sec,
        },
    }

    summary_rows = []

    for tag, cfg in combinations.items():
        print(f"\n[INFO] Building dataset: {cfg['name']} ({tag})...")
        raw_pos = cfg["pos_recs"]
        clean_pos = filter_closeness_conflicts(raw_pos, threshold_bp=25)

        target_gc = float(np.mean([r["GC_Content"] for r in clean_pos]))
        clean_negs = sample_gc_matched_negatives(genome_id, genome_seq, clean_pos, len(clean_pos), target_gc=target_gc)

        # Export Positives FASTA & TSV
        pos_fasta = TIGR4_DIR / f"positives_{tag}_81bp.fasta"
        pos_tsv = TIGR4_DIR / f"positives_{tag}_81bp.tsv"

        with open(pos_fasta, "w") as f:
            for r in clean_pos:
                f.write(f">{r['Sequence_ID']}\n{r['Sequence_81bp']}\n")

        pd.DataFrame(clean_pos).to_csv(pos_tsv, sep="\t", index=False)

        # Export Negatives FASTA & TSV
        neg_fasta = TIGR4_DIR / f"negatives_{tag}_81bp.fasta"
        neg_tsv = TIGR4_DIR / f"negatives_{tag}_81bp.tsv"

        with open(neg_fasta, "w") as f:
            for r in clean_negs:
                f.write(f">{r['Sequence_ID']}\n{r['Sequence_81bp']}\n")

        pd.DataFrame(clean_negs).to_csv(neg_tsv, sep="\t", index=False)

        pos_gc_mean = float(np.mean([r["GC_Content"] for r in clean_pos]))
        neg_gc_mean = float(np.mean([r["GC_Content"] for r in clean_negs]))
        purines_pos = float(np.mean([1 if r["Sequence_81bp"][60] in ["A", "G"] else 0 for r in clean_pos]) * 100.0)

        tataat_count = sum(1 for r in clean_pos if "TATAAT" in r["Sequence_81bp"][40:57])
        tataat_pct = float(tataat_count / len(clean_pos) * 100.0)

        summary_rows.append({
            "Tag": tag,
            "Dataset Name": cfg["name"],
            "Positives (N)": len(clean_pos),
            "Negatives (N)": len(clean_negs),
            "Pos GC (%)": round(pos_gc_mean, 2),
            "Neg GC (%)": round(neg_gc_mean, 2),
            "GC Gap (%)": round(abs(neg_gc_mean - pos_gc_mean), 2),
            "Purine +1 (%)": round(purines_pos, 1),
            "Caja -10 Match (%)": round(tataat_pct, 1),
        })

    df_sum = pd.DataFrame(summary_rows)
    print("\n" + "═" * 105)
    print(" SUMMARY OF ALL 4 TIGR4 DATASET COMBINATIONS")
    print("═" * 105)
    print(df_sum.to_string(index=False))
    print("═" * 105 + "\n")

    summary_csv = TIGR4_DIR / "tigr4_dataset_combinations_summary.csv"
    df_sum.to_csv(summary_csv, index=False)
    print(f"[SUCCESS] All 4 TIGR4 dataset combinations built and summary saved ➔ {summary_csv}\n")


if __name__ == "__main__":
    main()
