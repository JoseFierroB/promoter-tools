#!/usr/bin/env python3
"""Build 17_d39v_tigr4_high_mix5050: dedup positives (1399) + 50/50 negatives.

Negatives: 1399 CDS (subsample seed 42 of canonical 1738) + 1399 IGR
(subsample seed 42 of the 2798 from 16_d39v_tigr4_high_igr_1to2).
Dedup within negatives + cross leakage vs positives must be 0.

Usage:
    pixi run python src/dataset/build_mix_5050.py [-o data/benchmark/datasets/17_d39v_tigr4_high_mix5050]
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
CANON = ROOT / "data/benchmark/datasets/2_d39v_tigr4_high_sigma"
IGR16 = ROOT / "data/benchmark/datasets/16_d39v_tigr4_high_igr_1to2"
SEED = 42


def load_fa(path):
    return [(r.id, str(r.seq).upper()) for r in SeqIO.parse(path, "fasta")]


def main():
    ap = argparse.ArgumentParser(description="Build 50/50 mixed-background dataset.")
    ap.add_argument("-o", "--output",
                    default="data/benchmark/datasets/17_d39v_tigr4_high_mix5050")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Positives: same 1399 keep-first dedup as build_igr_12.
    seen, pos_ids, pos_seqs = set(), [], []
    for i, s in load_fa(CANON / "positives_81bp.fasta"):
        if s not in seen:
            seen.add(s)
            pos_ids.append(i)
            pos_seqs.append(s)
    print(f"positives: {len(pos_ids)} unique")

    # CDS + IGR negatives: 1399 unique each via shuffled keep-first
    # (dedup within + across pools with top-up; deterministic seed).
    cds = load_fa(CANON / "negatives_81bp.fasta")
    igr = load_fa(IGR16 / "negatives_81bp.fasta")

    def take_unique(pool, n, taken):
        order = pool[:]
        rng.shuffle(order)
        out = []
        for i, s in order:
            if s not in taken:
                taken.add(s)
                out.append((i, s))
            if len(out) >= n:
                break
        return out

    taken = set()
    cds_sub = take_unique(cds, 1399, taken)
    igr_sub = take_unique(igr, 1399, taken)
    assert len(cds_sub) == 1399 and len(igr_sub) == 1399, "pool exhausted"

    neg_ids = [i for i, _ in cds_sub + igr_sub]
    neg_seqs = [s for _, s in cds_sub + igr_sub]
    assert len(set(neg_seqs)) == len(neg_seqs), "dup within negatives"
    assert not (set(neg_seqs) & set(pos_seqs)), "pos-neg leakage"
    src = (["CDS"] * len(cds_sub)) + (["IGR"] * len(igr_sub))
    print(f"negatives: {len(neg_ids)} ({src.count('CDS')} CDS + {src.count('IGR')} IGR)")

    def write_fa(path, ids, seqs):
        with open(path, "w") as fh:
            for i, s in zip(ids, seqs):
                fh.write(f">{i}\n{s}\n")
    write_fa(out / "positives_81bp.fasta", pos_ids, pos_seqs)
    write_fa(out / "negatives_81bp.fasta", neg_ids, neg_seqs)
    meta = pd.read_csv(CANON / "positives_81bp_metadata.tsv",
                       sep="\t", dtype=str).set_index("Sequence_ID")
    meta.loc[pos_ids].reset_index().to_csv(
        out / "positives_81bp_metadata.tsv", sep="\t", index=False)
    pd.DataFrame({"Sequence_ID": neg_ids, "Source": src}).to_csv(
        out / "negatives_81bp_metadata.tsv", sep="\t", index=False)
    print(f"WROTE {out}: {len(pos_ids)} pos + {len(neg_ids)} neg")


if __name__ == "__main__":
    main()
