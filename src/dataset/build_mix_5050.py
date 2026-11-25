#!/usr/bin/env python3
"""Build the deduped 1399 + 1399 CDS + 1399 IGR collection (sole assembler).

Recipe (all deterministic, seed 42):
  positives: keep-first dedup of the CANON positives pool (same 1399
      sequences from any complete pool: raw D39V 989 + TIGR4 738, or the
      1727 combined set). D39V headers carry the sigma-assigned family label
      (C1/C5 transferred to SigA); TIGR4 headers carry conf + family suffix.
  negatives: 1399 CDS (shuffled keep-first subsample of the canonical 1738:
      raw D39V 1000 + TIGR4 738) + 1399 IGR (same of the 2798 IGR pool).
      Dedup within + across pools, zero pos/neg leakage.
  --strain-match: per-strain quotas (D39V/TIGR4 ratio of the positives)
      inside EACH background via stratified quotas (loud top-up on short
      strata). This flag produces the vigente hc set (negatives metadata
      gains the Strain column).

Provenance of the committed collection
(data/benchmark/datasets/d39v_tigr4_hc_dedup_balanced, 1399+2798):
  built 2026-10-12 with this script, --strain-match, -o <hc path>.
  Verified facts:
  - positives: keep-first dedup gives the same 1399 sequences from the raw
    D39V 989 + TIGR4 738 pools; D39V headers carry the sigma-assigned family
    label (C1/C5 transferred to SigA), TIGR4 headers carry conf + family
    suffix (see split FASTAs + sigma metadata under data/tigr4/).
  - CDS negatives (1399): byte-exact reproduction of the 17_mix5050 sibling
    from raw D39V + TIGR4 negatives in file order (d39v then tigr4),
    random.Random(42).shuffle, keep-first. The hc set reuses that pool.
  - quotas: positives are 972 D39V + 427 TIGR4 (Chromosome column), and each
    negative background holds exactly 972 D39V + 427 TIGR4 rows — the
    --strain-match signature (negatives metadata gains the Strain column).
  - IGR pool (2798): sampled with the same seed-42 rule from the refined-IGR
    windows; the committed 16_d39v_tigr4_high_igr_1to2 snapshot in git
    history is a later resampling (same IDs, different windows), so reruns
    from current sources resample equivalently but not byte-identically.
  Pools at build time: CANON = 2_d39v_tigr4_high_sigma (positives with
  meta-key headers + 1727-row sigma metadata w/ Chromosome/Sigma_Factor;
  negatives = raw concat) + IGR16 (pool + strain metadata). Both restorable
  from git history (pre-purge commits).

Usage:
    pixi run python src/dataset/build_mix_5050.py [-o data/benchmark/datasets/17_d39v_tigr4_high_mix5050] [--prefer-sigx] [--strain-match]
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
    ap.add_argument("--prefer-sigx", action="store_true",
                    help="On duplicate-sequence conflicts, keep the SigX-family "
                         "(SigX/ComX) member over SigA/None (default: keep-first). "
                         "Same sequences, only the retained ID/metadata row changes.")
    ap.add_argument("--strain-match", action="store_true",
                    help="Match the positive strain ratio (D39V/TIGR4) inside EACH "
                         "negative background (CDS, IGR) via stratified quotas. "
                         "Short strata are topped up from the other strain of the "
                         "same background (logged). Off by default (legacy: "
                         "slot/pool-proportional sampling).")
    args = ap.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    meta_all = pd.read_csv(CANON / "positives_81bp_metadata.tsv",
                           sep="\t", dtype=str).set_index("Sequence_ID")

    def family(i):
        v = meta_all.loc[i, "Sigma_Factor"]
        return "SigX" if v in ("SigX", "ComX") else str(v)

    # Positives: same 1399 keep-first dedup as build_igr_12, optionally with
    # minority-family tie-break (position-stable: replacement keeps the slot,
    # so split indices stay valid).
    seen, pos_ids, pos_seqs = {}, [], []
    n_swapped = 0
    for i, s in load_fa(CANON / "positives_81bp.fasta"):
        if s not in seen:
            seen[s] = len(pos_ids)
            pos_ids.append(i)
            pos_seqs.append(s)
        elif args.prefer_sigx and family(pos_ids[seen[s]]) != "SigX" and family(i) == "SigX":
            pos_ids[seen[s]] = i
            n_swapped += 1
    print(f"positives: {len(pos_ids)} unique (sigx tie-break swaps: {n_swapped})")

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

    def strain_of_cds(i):
        if "TIGR4" in i:
            return "TIGR4"
        if "D39V" in i:
            return "D39V"
        raise SystemExit(f"ERROR: cannot infer strain from CDS id {i!r}")

    def take_stratified(labeled_pool, quotas, taken, background):
        """Sample per-strain quotas (dict strain -> n) from [(id, seq, strain)].

        Strata are shuffled in fixed D39V-then-TIGR4 order (seed 42). A short
        stratum is topped up from the other strain of the same background;
        the fallback is printed (loud, not silent). Returns
        ([(id, seq, strain)], {strain: n_taken}) with strains aligned.
        """
        got_ids = set()
        got = []
        for strain in ("D39V", "TIGR4"):
            stratum = [(i, s) for i, s, st in labeled_pool if st == strain]
            rng.shuffle(stratum)
            for i, s in stratum:
                if i not in got_ids and s not in taken:
                    taken.add(s)
                    got_ids.add(i)
                    got.append((i, s, strain))
                if sum(1 for _, _, st in got if st == strain) >= quotas[strain]:
                    break
        counts = {st: sum(1 for _, _, s in got if s == st) for st in ("D39V", "TIGR4")}
        if any(counts[st] < quotas[st] for st in ("D39V", "TIGR4")):
            rest = [(i, s, st) for i, s, st in labeled_pool if i not in got_ids]
            rng.shuffle(rest)
            for i, s, st in rest:
                if s not in taken:
                    taken.add(s)
                    got_ids.add(i)
                    got.append((i, s, st))
                if len(got) >= sum(quotas.values()):
                    break
            counts = {st: sum(1 for _, _, s in got if s == st) for st in ("D39V", "TIGR4")}
            print(f"  WARNING [{background}]: strain quota short "
                  f"(want {quotas}, got {counts}) — topped up within background")
        return got, counts

    taken = set()
    neg_strain = []
    if args.strain_match:
        n_d39v = sum(1 for i in pos_ids
                     if meta_all.loc[i, "Chromosome"] == "D39V")
        n_tig = len(pos_ids) - n_d39v
        quotas = {"D39V": n_d39v, "TIGR4": n_tig}
        print(f"strain quotas per background: {quotas} (from positives)")
        m16 = pd.read_csv(IGR16 / "negatives_81bp_metadata.tsv",
                          sep="\t", dtype=str).set_index("Sequence_ID")
        cds_labeled = [(i, s, strain_of_cds(i)) for i, s in cds]
        igr_labeled = [(i, s, str(m16.loc[i, "Strain"])) for i, s in igr]
        cds_got, cds_counts = take_stratified(cds_labeled, quotas, taken, "CDS")
        igr_got, igr_counts = take_stratified(igr_labeled, quotas, taken, "IGR")
        assert len(cds_got) == 1399 and len(igr_got) == 1399, "pool exhausted"
        cds_sub = [(i, s) for i, s, _ in cds_got]
        igr_sub = [(i, s) for i, s, _ in igr_got]
        neg_strain = [st for _, _, st in cds_got + igr_got]
        print(f"strain-matched negatives: CDS {cds_counts} + IGR {igr_counts}")
    else:
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
    neg_frame = pd.DataFrame({"Sequence_ID": neg_ids, "Source": src})
    if args.strain_match:
        neg_frame["Strain"] = neg_strain
    neg_frame.to_csv(out / "negatives_81bp_metadata.tsv", sep="\t", index=False)
    print(f"WROTE {out}: {len(pos_ids)} pos + {len(neg_ids)} neg")


if __name__ == "__main__":
    main()
