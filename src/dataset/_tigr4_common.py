"""Shared helpers for TIGR4 dataset extraction scripts (positive + negative)."""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parents[2]


def load_genome(fasta_path: Path) -> Tuple[str, Seq, int]:
    """Load the first record of a FASTA genome file."""
    if not fasta_path.exists():
        alt_fasta = fasta_path.parent / "TIGR4.fasta"
        if alt_fasta.exists():
            fasta_path = alt_fasta
        else:
            print(f"[ERROR] FASTA file not found at {fasta_path}", file=sys.stderr)
            sys.exit(1)

    genome_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    if not genome_dict:
        print(f"[ERROR] Could not parse FASTA from {fasta_path}", file=sys.stderr)
        sys.exit(1)

    chrom_id = list(genome_dict.keys())[0]
    seq = genome_dict[chrom_id].seq
    print(f"[INFO] Loaded genome '{chrom_id}' (length: {len(seq):,} bp) from {fasta_path.name}")
    return chrom_id, seq, len(seq)


def write_dataset_files(records: List[Dict], out_prefix: Path) -> Tuple[Path, Path]:
    """Write records as .fasta + .tsv (metadata without Sequence column)."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fasta_out = out_prefix.with_suffix(".fasta")
    tsv_out = out_prefix.with_suffix(".tsv")

    with open(fasta_out, "w") as f:
        for r in records:
            f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")

    df_meta = pd.DataFrame(records)
    if "Sequence" in df_meta.columns:
        df_meta = df_meta.drop(columns=["Sequence"])
    df_meta.to_csv(tsv_out, sep="\t", index=False)
    return fasta_out, tsv_out


def compute_gc_statistics(records: List[Dict], genome_seq: Seq) -> Dict:
    """Compute GC content statistics for sampled records vs genome background."""
    if not records:
        return {}

    sampled_gc = [r["GC_Content"] for r in records]
    mean_gc = float(np.mean(sampled_gc))
    stdev_gc = float(np.std(sampled_gc, ddof=1)) if len(sampled_gc) > 1 else 0.0

    gen_seq_str = str(genome_seq).upper()
    gc_gen_count = gen_seq_str.count("G") + gen_seq_str.count("C")
    gen_gc_mean = (gc_gen_count / len(gen_seq_str)) * 100.0

    import math
    n = len(sampled_gc)
    std_error = stdev_gc / math.sqrt(n) if n > 0 and stdev_gc > 0 else 1.0
    z_score = (mean_gc - gen_gc_mean) / std_error if std_error > 0 else 0.0
    cohen_d = (mean_gc - gen_gc_mean) / stdev_gc if stdev_gc > 0 else 0.0

    return {
        "n_samples": n, "mean_gc": mean_gc, "stdev_gc": stdev_gc,
        "genome_gc_mean": gen_gc_mean, "z_score": z_score, "cohen_d": cohen_d,
    }


import numpy as np  # noqa: E402  (needed for np.mean/np.std above)
