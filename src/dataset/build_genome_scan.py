#!/usr/bin/env python3
"""Build a semi-real timing benchmark: tile a whole genome into 81-mers.

Generates a scan FASTA with one 81-bp window per --stride positions
(default 10 -> ~200k windows for a 2 Mb genome; stride 1 -> ~2M).
The file is split in halves (scan_pos / scan_neg) ONLY so the standard
`cli run --pos/--neg` interface accepts it: labels are arbitrary and the
resulting metrics are meaningless. What counts is wall time per tool in
a semi-real full-genome regime (see fixed_fpr.py for the FP analysis).

Writes: <out>.fasta (all windows, IDs SCAN_<i>), plus the run command.

Usage:
    pixi run python src/dataset/build_genome_scan.py \
      --fasta data/reference/D39V.fna --stride 10 \
      -o /tmp/scan_d39v_s10/scan_81bp
    pixi run python src/cli.py run <tools> \
      --pos /tmp/scan_d39v_s10/scan_81bp_pos.fasta \
      --neg /tmp/scan_d39v_s10/scan_81bp_neg.fasta \
      --threads 1 --gpu --name scan_d39v_s10
"""
import argparse
from pathlib import Path

from Bio import SeqIO

WIN = 81


def main():
    ap = argparse.ArgumentParser(description="Tile a genome into 81-mers for timing.")
    ap.add_argument("--fasta", required=True, help="Reference genome FASTA")
    ap.add_argument("--stride", type=int, default=10,
                    help="Stride between window starts (default 10)")
    ap.add_argument("-o", "--output", required=True,
                    help="Output prefix (writes _pos.fasta + _neg.fasta)")
    a = ap.parse_args()
    rec = next(iter(SeqIO.parse(a.fasta, "fasta")))
    seq = str(rec.seq).upper()
    n = len(seq)
    ids, seqs = [], []
    for i, start in enumerate(range(0, n - WIN + 1, a.stride)):
        w = seq[start:start + WIN]
        if "N" in w:
            continue
        ids.append(f"SCAN_{i:07d}")
        seqs.append(w)
    half = len(ids) // 2
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, part in (("pos", (ids[:half], seqs[:half])),
                         ("neg", (ids[half:], seqs[half:]))):
        with open(f"{out}_{suffix}.fasta", "w") as fh:
            for i, s in zip(*part):
                fh.write(f">{i}\n{s}\n")
    print(f"genome={n}bp stride={a.stride} windows={len(ids)} "
          f"(pos={half}, neg={len(ids)-half})")
    print("run with:")
    print(f"  pixi run python src/cli.py run <tools> --pos {out}_pos.fasta "
          f"--neg {out}_neg.fasta --threads 1 --gpu --name {out.name}")


if __name__ == "__main__":
    main()
