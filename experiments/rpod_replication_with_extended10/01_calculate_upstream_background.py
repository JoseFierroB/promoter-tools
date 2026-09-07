#!/usr/bin/env python3
"""
Step 1: Calculate 500 bp Upstream Background Nucleotide Distribution in D39V.

Extracts the [-500, -1] bp window relative to each TSS +1 on the proper strand
and calculates empirical background frequencies for A, C, G, T.
"""

import json
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parent.parent.parent
GENOME_PATH = ROOT / "data" / "reference" / "D39V.fna"
TSS_META_CANDIDATES = [
    ROOT / "data" / "reference" / "D39V_annotation_TSS_Victor.gff",
    ROOT.parent / "_proyectos_tools" / "experimentos" / "promoter-tools" / "data" / "reference" / "D39V_annotation_TSS_Victor.gff",
    ROOT / "data" / "benchmark" / "d39v" / "positives_81bp_metadata.tsv",
]
OUTPUT_JSON = Path(__file__).resolve().parent / "d39v_500bp_upstream_background.json"


def compute_background(genome_fasta: Path) -> dict:
    records = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(genome_fasta, "fasta")}
    genome_seq = list(records.values())[0]

    # Load 1003 TSS coordinates with true strand
    gff_path = next((p for p in TSS_META_CANDIDATES if p.exists()), None)
    if not gff_path:
        raise FileNotFoundError("Could not find TSS GFF file")

    tss_coords = []
    if str(gff_path).endswith(".gff"):
        with open(gff_path) as f:
            for line in f:
                if line.startswith("#") or not line.strip(): continue
                parts = line.strip().split("\t")
                if len(parts) >= 9:
                    pos = int(parts[3]) - 1
                    strand = parts[6]
                    tss_coords.append((pos, strand))
    else:
        df = pd.read_csv(gff_path, sep="\t")
        for _, row in df.iterrows():
            pos = int(row["TSS_Position_0based"])
            strand = row["Strand"]
            tss_coords.append((pos, strand))
    
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    total = 0

    for (pos, strand) in tss_coords:
        # Extract 500 bp upstream of TSS (+1)
        if strand == "+":
            up_seq = genome_seq[max(0, pos - 500):pos]
        else:
            up_seq = str(Seq(genome_seq[pos + 1:pos + 501]).reverse_complement())

        for base in up_seq:
            if base in counts:
                counts[base] += 1
                total += 1

    bg_freqs = {base: round(counts[base] / total, 4) for base in "ACGT"}
    return bg_freqs


def main():
    print("[STEP 1] Computing empirical 500 bp upstream background for D39V...")
    if GENOME_PATH.exists():
        bg = compute_background(GENOME_PATH)
    else:
        print("Check the path for genome file")
        return

    with open(OUTPUT_JSON, "w") as f:
        json.dump(bg, f, indent=2)

    print(f"  Background saved to: {OUTPUT_JSON}")
    print(f"  f(A)={bg['A']:.4f}, f(C)={bg['C']:.4f}, f(G)={bg['G']:.4f}, f(T)={bg['T']:.4f}")


if __name__ == "__main__":
    main()
