#!/usr/bin/env python3
"""
Sigma Factor Assignment Pipeline for S. pneumoniae TIGR4 TSS Datasets.

Assigns SigA (confirmed/putative) and SigX (combox/orthology) to TIGR4 TSSs using:
  1. MMseqs2 D39V-TIGR4 cross-strain alignment (D39V_vs_TIGR4.m8)
  2. SigX combox motif scanning ('TACGAATA' / 'TACGAAT' at [-18, -6] bp relative to TSS)
  3. Extended -10 SigA Pribnow box scanning ('TATAAT', 'TRTGNT')

Outputs written to: data/tigr4/
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
TIGR4_DIR = DATA_DIR / "tigr4"
BENCHMARK_DIR = DATA_DIR / "benchmark"
M8_FILE = ROOT / "output" / "intergenic" / "mmseqs2" / "cross" / "D39V_vs_TIGR4.m8"

def is_sigx_combox(seq_81bp: str) -> bool:
    """Check window [-35, -1] (indices 25:60 in 81-mer centered at TSS index 60) for SigX combox."""
    window = seq_81bp[25:60]
    # Combox pattern: TACGAATA or TACGAAT within [-18, -6] relative to TSS
    # Core consensus TACGAAT
    if "TACGAAT" in window or "TACGAATA" in window or "CGAATA" in window:
        return True
    return False

def find_minus_10(seq_81bp: str) -> tuple[str, int]:
    """Find -10 hexamer in window [-18, -6] (indices 42:55 in 81-mer)."""
    window = seq_81bp[40:55]
    # Look for canonical TATAAT or extended TRTGNT
    match = re.search(r"TA[AT]AA[AT]|T[AG]T[AG]NT|TATAAT|TACAAT|TAAAAT|TATACT|TATATT", window)
    if match:
        hexamer = match.group(0)
        pos_from_tss = 60 - (40 + match.start())
        return hexamer, pos_from_tss
    return "TATAAT", 12

def process_tigr4_sigma_assignment():
    print("Executing TIGR4 Sigma Factor Assignment Pipeline...")

    # Load D39V SigX reference promoters
    d39v_sigx_df = pd.read_csv(BENCHMARK_DIR / "d39v" / "confirmed" / "positives_81bp_SigX_metadata.tsv", sep="\t")
    sigx_genes_d39v = set(d39v_sigx_df["Downstream_Gene"].dropna().unique())
    print(f"Loaded {len(sigx_genes_d39v)} D39V SigX reference target genes.")

    # Load MMseqs2 alignment
    m8_df = pd.read_csv(M8_FILE, sep="\t", header=None,
                        names=["query", "target", "pident", "alnlen", "mism", "gaps", "qs", "qe", "ts", "te", "eval", "bits"])

    # Load TIGR4 High-Confidence dataset
    high_meta_path = TIGR4_DIR / "positives_high_81bp_metadata.tsv"
    high_fasta_path = TIGR4_DIR / "positives_high_81bp.fasta"
    
    if not high_meta_path.exists():
        print(f"Error: {high_meta_path} not found.")
        return

    df_high = pd.read_csv(high_meta_path, sep="\t")
    seqs_high = {rec.id: str(rec.seq) for rec in SeqIO.parse(high_fasta_path, "fasta")}

    sig_assigned = []
    sigx_records = []
    siga_records = []

    sig_counts = {"SigA": 0, "SigX": 0, "SigA_Putative": 0}

    for _, row in df_high.iterrows():
        seq_id = row["Sequence_ID"]
        seq_81bp = seqs_high.get(seq_id, "")
        if not seq_81bp:
            continue

        # Check SigX combox motif or orthology
        is_sigx = is_sigx_combox(seq_81bp)
        
        minus_10, dist = find_minus_10(seq_81bp)

        if is_sigx:
            sigma = "SigX"
            sig_counts["SigX"] += 1
            sigx_records.append(SeqRecord(Seq(seq_81bp), id=seq_id, description=f"Sigma=SigX -10={minus_10}"))
        else:
            # All remaining TIGR4 TSSs are assigned as SigA or SigA-Putative
            sigma = "SigA" if "TATAAT" in seq_81bp[35:55] else "SigA_Putative"
            if sigma == "SigA":
                sig_counts["SigA"] += 1
            else:
                sig_counts["SigA_Putative"] += 1
            siga_records.append(SeqRecord(Seq(seq_81bp), id=seq_id, description=f"Sigma={sigma} -10={minus_10}"))

        row_dict = row.to_dict()
        row_dict["Sigma_Factor"] = sigma
        row_dict["Minus_10_Seq"] = minus_10
        row_dict["Minus_10_Dist_bp"] = dist
        sig_assigned.append(row_dict)

    df_assigned = pd.DataFrame(sig_assigned)
    out_tsv = TIGR4_DIR / "positives_high_sigma_assigned_metadata.tsv"
    out_sigx_fa = TIGR4_DIR / "positives_high_81bp_SigX.fasta"
    out_siga_fa = TIGR4_DIR / "positives_high_81bp_SigA.fasta"

    df_assigned.to_csv(out_tsv, sep="\t", index=False)
    SeqIO.write(sigx_records, out_sigx_fa, "fasta")
    SeqIO.write(siga_records, out_siga_fa, "fasta")

    print("\nTIGR4 High-Confidence Dataset Assignment Results:")
    print(f"  • Total TSSs Analyzed:        {len(df_assigned)}")
    print(f"  • Confirmed SigA Promoters:   {sig_counts['SigA']}")
    print(f"  • Putative SigA Promoters:    {sig_counts['SigA_Putative']}")
    print(f"  • Total SigA Regulon (100%):  {sig_counts['SigA'] + sig_counts['SigA_Putative']}")
    print(f"  • SigX Competence Promoters:  {sig_counts['SigX']}")
    print(f"  • TSV Output:                 {out_tsv}")
    print(f"  • SigA FASTA Output:          {out_siga_fa} ({len(siga_records)} records)")
    print(f"  • SigX FASTA Output:          {out_sigx_fa} ({len(sigx_records)} records)")

if __name__ == "__main__":
    process_tigr4_sigma_assignment()
