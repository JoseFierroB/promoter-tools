#!/usr/bin/env python3
"""
Extract 40 bp and 20 bp upstream windows for all 1,003 TSSs.
"""

from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parent.parent.parent
CURRENT_DIR = Path(__file__).resolve().parent

GENOME_FNA = ROOT / "data" / "reference" / "D39V.fna"
GFF_TSS = ROOT / "data" / "reference" / "D39V_annotation_TSS_Victor.gff"

FASTA_40BP = CURRENT_DIR / "d39v_tss_40bp_upstream.fasta"
FASTA_20BP = CURRENT_DIR / "d39v_tss_20bp_upstream.fasta"


def extract_windows():
    print("[STEP 2] Extracting upstream windows for 1,003 TSSs...")
    record = SeqIO.read(GENOME_FNA, "fasta")
    genome_seq = record.seq
    genome_len = len(genome_seq)

    tss_records = []
    with open(GFF_TSS) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if parts[2] == "transcription_start_site":
                pos = int(parts[3])
                strand = parts[6]
                attrs = dict(item.split("=", 1) for item in parts[8].split(";") if "=" in item)
                seq_id = attrs.get("ID", f"TSS_{pos}_{strand}")
                tss_records.append((seq_id, pos, strand))

    print(f"  Loaded {len(tss_records)} TSSs from {GFF_TSS.name}")

    with open(FASTA_40BP, "w") as f40, open(FASTA_20BP, "w") as f20:
        for seq_id, pos, strand in tss_records:
            # 1-based coordinates
            if strand == "+":
                # [-40, -1] relative to pos
                s40 = max(0, pos - 41)
                e40 = pos - 1
                seq40 = str(genome_seq[s40:e40]).upper()

                # [-20, -1]
                s20 = max(0, pos - 21)
                e20 = pos - 1
                seq20 = str(genome_seq[s20:e20]).upper()
            else:
                # For minus strand, upstream is in increasing coordinate direction: pos+1 to pos+40
                s40 = pos
                e40 = min(genome_len, pos + 40)
                raw40 = genome_seq[s40:e40]
                seq40 = str(raw40.reverse_complement()).upper()

                s20 = pos
                e20 = min(genome_len, pos + 20)
                raw20 = genome_seq[s20:e20]
                seq20 = str(raw20.reverse_complement()).upper()

            f40.write(f">{seq_id}\n{seq40}\n")
            f20.write(f">{seq_id}\n{seq20}\n")

    print(f"  Saved 40 bp FASTA ({len(tss_records)} entries): {FASTA_40BP}")
    print(f"  Saved 20 bp FASTA ({len(tss_records)} entries): {FASTA_20BP}")


if __name__ == "__main__":
    extract_windows()
