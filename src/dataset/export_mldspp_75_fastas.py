#!/usr/bin/env python3
"""
Exporter for MLDSPP 75/25 Train/Test FASTA Splits.
Reads .npz index files in data/benchmark/ and exports exact 75% train and 25% test FASTAs.

The npz -> FASTA mapping is resolved by coverage: the split that covers
exactly the number of records in each known FASTA. Mis-mapped/absent
combinations (e.g. extended_high 776-index split vs a 2000-record FASTA)
are skipped with a warning instead of exporting a corrupted dataset.

Outputs written to: data/benchmark/splits/
"""

import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.backend.local import _count_seqs  # noqa: E402

BENCHMARK_DIR = ROOT / "data" / "benchmark"
TIGR4_DIR = ROOT / "data" / "tigr4"
SPLITS_DIR = BENCHMARK_DIR / "splits"

# Known positive FASTA datasets to export splits for
FASTA_SOURCES = [
    (BENCHMARK_DIR / "d39v" / "positives_81bp.fasta", "d39v"),
    (TIGR4_DIR / "positives_high_81bp.fasta", "tigr4_high"),
    (TIGR4_DIR / "positives_extended_81bp.fasta", "tigr4_extended"),
]


def find_split_for(fasta_path: Path) -> Path:
    """Return the split covering n_pos records; disambiguate by name overlap."""
    import re

    n_pos = _count_seqs(fasta_path)
    matches = []
    for npz in sorted(BENCHMARK_DIR.glob("mldspp_75_split_*.npz")):
        try:
            d = np.load(npz)
            covered = len(d["train_idx"]) + len(d["test_idx"])
        except Exception:
            continue
        if covered == n_pos:
            matches.append(npz)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    path_toks = set(re.split(r"[_\-.\\/]+", str(fasta_path).lower()))
    scored = []
    for npz in matches:
        sname = npz.name.replace("mldspp_75_split_", "").replace(".npz", "")
        s_toks = set(re.split(r"[_\-.]+", sname))
        overlap = len(s_toks & path_toks)
        if overlap > 0:
            scored.append((overlap, npz))
    if not scored:
        return None
    scored.sort(reverse=True)
    winners = [npz for ov, npz in scored if ov == scored[0][0]]
    return winners[0] if len(winners) == 1 else None


def export_splits():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting MLDSPP 75/25 Train/Test FASTA Splits...")

    for fasta_path, prefix in FASTA_SOURCES:
        if not fasta_path.exists():
            print(f"Skipping {prefix}: FASTA not found at {fasta_path}")
            continue

        n_pos = _count_seqs(fasta_path)
        npz_path = find_split_for(fasta_path)
        if npz_path is None:
            print(f"Skipping {prefix}: no unique split with coverage == {n_pos} "
                  f"(available: {[p.name for p in BENCHMARK_DIR.glob('mldspp_75_split_*.npz')]})")
            continue

        data = np.load(npz_path)
        train_idx = data["train_idx"]
        test_idx = data["test_idx"]
        if len(train_idx) + len(test_idx) != n_pos:
            print(f"Skipping {prefix}: split coverage mismatch")
            continue

        records = list(SeqIO.parse(fasta_path, "fasta"))
        if len(records) != n_pos:
            print(f"Skipping {prefix}: FASTA has {len(records)} records but split covers {n_pos}")
            continue

        train_records = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]

        out_train_fasta = SPLITS_DIR / f"{prefix}_train_75.fasta"
        out_test_fasta = SPLITS_DIR / f"{prefix}_test_25.fasta"

        SeqIO.write(train_records, out_train_fasta, "fasta")
        SeqIO.write(test_records, out_test_fasta, "fasta")

        print(f"\n{prefix.upper()} ({npz_path.name}):")
        print(f"  • Source FASTA:  {fasta_path.name} ({n_pos} records)")
        print(f"  • Train (75%):   {out_train_fasta} ({len(train_records)} records)")
        print(f"  • Test (25%):    {out_test_fasta} ({len(test_records)} records)")


if __name__ == "__main__":
    export_splits()