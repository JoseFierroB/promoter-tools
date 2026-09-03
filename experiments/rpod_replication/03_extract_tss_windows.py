#!/usr/bin/env python3
"""
Step 3: Extract TSS Upstream Windows.

Extracts:
  - 40 bp upstream window ([-40, -1] relative to TSS +1) for RpoD and ComX scanning.
"""

from pathlib import Path
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
CURRENT_DIR = Path(__file__).resolve().parent

# Check potential benchmark paths
FASTA_CANDIDATES = [
    ROOT / "data" / "benchmark" / "d39v" / "positives_81bp.fasta",
    ROOT / "data" / "benchmark" / "positives_81bp.fasta",
    ROOT / "data" / "benchmark" / "d39v_1003_all_raw" / "positives_1003_all_raw.fasta",
]

FASTA_PATH = next((p for p in FASTA_CANDIDATES if p.exists()), None)
OUT_FA_40BP = CURRENT_DIR / "d39v_tss_40bp_upstream.fasta"


def main():
    print("[STEP 3] Extracting 40 bp upstream windows ([-40, -1] relative to TSS +1)...")
    if not FASTA_PATH:
        raise FileNotFoundError(f"Could not find positives benchmark fasta in: {FASTA_CANDIDATES}")

    records = list(SeqIO.parse(FASTA_PATH, "fasta"))

    # In standard 81 bp benchmark sequences [-60, +20], TSS +1 is at 0-based index 60.
    # Window [-40, -1] corresponds to indices 20:60 (40 nt total).
    with open(OUT_FA_40BP, "w") as f:
        for rec in records:
            seq_40bp = str(rec.seq)[20:60]
            f.write(f">{rec.id}\n{seq_40bp}\n")

    print(f"  Extracted {len(records)} sequences from {FASTA_PATH.name} to: {OUT_FA_40BP}")


if __name__ == "__main__":
    main()
