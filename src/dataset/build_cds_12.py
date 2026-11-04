#!/usr/bin/env python3
"""Build 18_d39v_tigr4_high_cds_1to2: dedup positives (1399) + CDS negatives 1:2 (2798).

Negatives: merge canonical CDS (1738) + supplementary seed-43 sets, then
shuffled keep-first dedup to 2798, strain-proportional to canonical
(1000 D39V : 738 TIGR4). Cross leakage vs positives must be 0.

Usage:
    pixi run python src/dataset/build_cds_12.py [-o data/benchmark/datasets/18_d39v_tigr4_high_cds_1to2]
    # supplementary sets first:
    #   negatives_tss_d39v.py ... --limit 1500 --seed 43 -o /tmp/cdsplus/d39v_extra
    #   negatives_tss_tigr4.py ... --tier high_conf_primary --limit 1500 --seed 43 -o /tmp/cdsplus/tigr4_extra
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
CANON = ROOT / "data/benchmark/datasets/2_d39v_tigr4_high_sigma"
SEED = 42


def load_fa(path):
    return [(r.id, str(r.seq).upper()) for r in SeqIO.parse(path, "fasta")]


def strain_of(nid):
    return "d39v" if "D39V" in nid.split("_") else "tigr4"


def main():
    ap = argparse.ArgumentParser(description="Build CDS-negatives 1:2 dataset.")
    ap.add_argument("-o", "--output",
                    default="data/benchmark/datasets/18_d39v_tigr4_high_cds_1to2")
    ap.add_argument("--extra-dir", default="/tmp/cdsplus",
                    help="Dir with d39v_extra.fasta + tigr4_extra.fasta")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    extra = Path(args.extra_dir)

    # Positives: same 1399 keep-first dedup as build_igr_12.
    seen, pos_ids, pos_seqs = set(), [], []
    for i, s in load_fa(CANON / "positives_81bp.fasta"):
        if s not in seen:
            seen.add(s)
            pos_ids.append(i)
            pos_seqs.append(s)
    print(f"positives: {len(pos_ids)} unique")

    # Negatives pool: canonical 1738 + supplementary 3000.
    pool = (load_fa(CANON / "negatives_81bp.fasta")
            + load_fa(extra / "d39v_extra.fasta")
            + load_fa(extra / "tigr4_extra.fasta"))
    by_strain = {"d39v": [], "tigr4": []}
    for i, s in pool:
        by_strain[strain_of(i)].append((i, s))
    print(f"pool: D39V={len(by_strain['d39v'])} TIGR4={len(by_strain['tigr4'])}")

    # Quotas proportional to canonical strain ratio (1000:738).
    n_neg, taken = 2 * len(pos_ids), set(pos_seqs)
    want = {"d39v": round(n_neg * 1000 / 1738), "tigr4": 0}
    want["tigr4"] = n_neg - want["d39v"]
    neg_ids, neg_seqs, neg_src = [], [], []
    for strain in ("d39v", "tigr4"):
        order = by_strain[strain][:]
        rng.shuffle(order)
        got = 0
        for i, s in order:
            if s not in taken:
                taken.add(s)
                neg_ids.append(i)
                neg_seqs.append(s)
                neg_src.append("CDS")
                got += 1
            if got >= want[strain]:
                break
        print(f"{strain}: wanted {want[strain]}, got {got}")
    assert len(neg_ids) == n_neg, f"short: {len(neg_ids)}/{n_neg}"
    assert len(set(neg_seqs)) == n_neg and not (set(neg_seqs) & set(pos_seqs))

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
    strains = ["D39V" if strain_of(i) == "d39v" else "TIGR4" for i in neg_ids]
    pd.DataFrame({"Sequence_ID": neg_ids, "Source": neg_src,
                  "Strain": strains}).to_csv(
        out / "negatives_81bp_metadata.tsv", sep="\t", index=False)
    print(f"WROTE {out}: {len(pos_ids)} pos + {len(neg_ids)} neg")


if __name__ == "__main__":
    main()
