#!/usr/bin/env python3
"""
Step 3: Extract TSS Upstream Windows for D39V.

Extracts:
  - 40 bp upstream window ([-40, -1] relative to TSS +1) for RpoD and ComX scanning.
  - 20 bp upstream window ([-20, -1] relative to TSS +1) for isolated -10 scanning.
"""

from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STEPS_DIR = Path(__file__).resolve().parent

GENOME_PATH = ROOT / "data" / "reference" / "D39V.fna"
GFF_PATH = ROOT / "data" / "reference" / "D39V_annotation_TSS_Victor.gff"

OUT_FA_40BP = STEPS_DIR / "d39v_tss_40bp_upstream.fasta"
OUT_FA_20BP = STEPS_DIR / "d39v_tss_20bp_upstream.fasta"


def main():
    print("[STEP 3] Extracting 40 bp and 20 bp upstream windows for 1,003 TSSs...")
    if not GENOME_PATH.exists() or not GFF_PATH.exists():
        raise FileNotFoundError(f"Missing genome or GFF: {GENOME_PATH}, {GFF_PATH}")

    genome = str(list(SeqIO.parse(GENOME_PATH, "fasta"))[0].seq)

    tss_records = []
    with open(GFF_PATH) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9:
                pos = int(parts[3]) - 1
                strand = parts[6]
                attr = parts[8]
                tss_records.append((pos, strand, attr))

    with open(OUT_FA_40BP, "w") as f40, open(OUT_FA_20BP, "w") as f20:
        for i, (pos, strand, attr) in enumerate(tss_records):
            sid = f"TSS_{i+1}_D39V_{pos+1}_{strand}"
            if strand == "+":
                up40 = genome[max(0, pos - 40):pos]
                up20 = genome[max(0, pos - 20):pos]
            else:
                up40 = str(Seq(genome[pos + 1:pos + 41]).reverse_complement())
                up20 = str(Seq(genome[pos + 1:pos + 21]).reverse_complement())

            f40.write(f">{sid}\n{up40}\n")
            f20.write(f">{sid}\n{up20}\n")

    print(f"  Extracted {len(tss_records)} sequences to {OUT_FA_40BP.name} and {OUT_FA_20BP.name}")


if __name__ == "__main__":
    main()
