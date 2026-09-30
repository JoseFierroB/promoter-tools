#!/usr/bin/env python3
"""Build 16_d39v_tigr4_high_igr_1to2: dedup positives (1399) + IGR negatives 1:2 (2798).

Positives: order-preserving keep-first dedup of the canonical 1727 (same rule
as make_dedup_balanced.py). Negatives: 81-mers strictly inside refined IGRs
(D39V+TIGR4), >=50 bp from any known TSS, seed 42, deduped within negatives
and cross-checked against positives (leakage must be 0).

Usage:
    pixi run python src/dataset/build_igr_12.py [-o data/benchmark/datasets/16_d39v_tigr4_high_igr_1to2]
"""
import argparse
import bisect
import random
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent

CANON_POS = ROOT / "data/benchmark/datasets/2_d39v_tigr4_high_sigma/positives_81bp.fasta"
CANON_META = ROOT / "data/benchmark/datasets/2_d39v_tigr4_high_sigma/positives_81bp_metadata.tsv"
D39V_IGR = ROOT / "output/intergenic_refined/d39v/D39V_igrs_refined.tsv"
TIGR4_IGR = ROOT / "output/intergenic_refined/tigr4/TIGR4_igrs_refined.tsv"
D39V_GENOME = ROOT / "data/reference/D39V.fna"
TIGR4_GENOME = ROOT / "data/reference/NC_003028.fasta"

SEED = 42
MARGIN = 50
WIN = 81


def load_genome(path):
    rec = next(iter(SeqIO.parse(path, "fasta")))
    return str(rec.seq).upper()


def main():
    ap = argparse.ArgumentParser(description="Build IGR-negatives 1:2 dataset.")
    ap.add_argument("-o", "--output",
                    default="data/benchmark/datasets/16_d39v_tigr4_high_igr_1to2")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # 1. Positives: keep-first dedup of canonical 1727.
    pos_recs = list(SeqIO.parse(CANON_POS, "fasta"))
    seen, pos_ids, pos_seqs = set(), [], []
    for r in pos_recs:
        s = str(r.seq).upper()
        if s not in seen:
            seen.add(s)
            pos_ids.append(r.id)
            pos_seqs.append(s)
    print(f"positives: {len(pos_recs)} -> {len(pos_ids)} unique")
    meta = pd.read_csv(CANON_META, sep="\t", dtype=str).set_index("Sequence_ID")
    pos_meta = meta.loc[pos_ids].reset_index()

    # 2. TSS positions per strain (0-based) for exclusion.
    tss = {"D39V": sorted(int(v) for v in
                           meta[meta["Chromosome"] == "D39V"]["TSS_Position_0based"]),
           "TIGR4": sorted(int(v) for v in
                            meta[meta["Chromosome"] != "D39V"]["TSS_Position_0based"])}
    print(f"TSS for exclusion: D39V={len(tss['D39V'])} TIGR4={len(tss['TIGR4'])}")

    # 3. Candidate 81-mer slots strictly inside linear IGRs.
    genomes = {"D39V": load_genome(D39V_GENOME), "TIGR4": load_genome(TIGR4_GENOME)}
    slots = {"D39V": [], "TIGR4": []}
    skipped_wrap = 0
    for strain, path in (("D39V", D39V_IGR), ("TIGR4", TIGR4_IGR)):
        df = pd.read_csv(path, sep="\t")
        for _, r in df.iterrows():
            if bool(r.get("is_circular_origin_wrap", False)):
                skipped_wrap += 1
                continue
            s, e = int(r["start"]), int(r["end"])
            for w in range(s, e - WIN + 2):
                lo = bisect.bisect_left(tss[strain], w - MARGIN)
                hi = bisect.bisect_right(tss[strain], w + WIN - 1 + MARGIN)
                if lo == hi:
                    slots[strain].append((w, w + WIN))
    print(f"slots: D39V={len(slots['D39V'])} TIGR4={len(slots['TIGR4'])} "
          f"(wraps skipped: {skipped_wrap})")

    # 4. Sample 2x negatives proportional to slots, dedup, cross-check.
    n_neg = 2 * len(pos_ids)
    pos_set = set(pos_seqs)
    total_slots = len(slots["D39V"]) + len(slots["TIGR4"])
    want = {st: round(n_neg * len(slots[st]) / total_slots) for st in slots}
    want["D39V"] += n_neg - sum(want.values())  # fix rounding
    neg_ids, neg_seqs, neg_rows = [], [], []
    seen_n = set()
    counters = {"D39V": 0, "TIGR4": 0}
    for strain in ("D39V", "TIGR4"):
        pool = slots[strain][:]
        rng.shuffle(pool)
        got = 0
        genome = genomes[strain]
        i = 0
        for (w0, w1) in pool:
            seq = genome[w0:w1]
            if len(seq) != WIN or "N" in seq or seq in pos_set or seq in seen_n:
                continue
            seen_n.add(seq)
            i += 1
            counters[strain] += 1
            nid = f"NEG_IGR_{strain}_{counters[strain]:04d}"
            neg_ids.append(nid)
            neg_seqs.append(seq)
            neg_rows.append({"Sequence_ID": nid, "Strain": strain,
                             "Chrom_Start_0based": w0,
                             "GC_Content(%)": round(
                                 (seq.count("G") + seq.count("C")) / WIN * 100, 2)})
            got += 1
            if got >= want[strain]:
                break
        print(f"{strain}: wanted {want[strain]}, got {got}")
    assert len(neg_ids) == n_neg, f"short: {len(neg_ids)}/{n_neg}"
    assert len(set(neg_seqs)) == n_neg and not (set(neg_seqs) & pos_set)

    # 5. Write.
    def write_fa(path, ids, seqs):
        with open(path, "w") as fh:
            for i, s in zip(ids, seqs):
                fh.write(f">{i}\n{s}\n")
    write_fa(out / "positives_81bp.fasta", pos_ids, pos_seqs)
    write_fa(out / "negatives_81bp.fasta", neg_ids, neg_seqs)
    pos_meta.to_csv(out / "positives_81bp_metadata.tsv", sep="\t", index=False)
    pd.DataFrame(neg_rows).to_csv(out / "negatives_81bp_metadata.tsv",
                                  sep="\t", index=False)
    print(f"WROTE {out}: {len(pos_ids)} pos + {len(neg_ids)} neg (1:2)")


if __name__ == "__main__":
    main()
