#!/usr/bin/env python3
"""Reproduce the canonical MMseqs2 cross-strain IGR clustering (2247 clusters).

Reconstructed canonical command (registered here; see notes on the residual gap):

    mmseqs easy-cluster ALL_IGRS_DEDUP.fasta OUT TMP \\
        --min-seq-id 0.95 -c 0.70 --cov-mode 0 --cluster-mode 1 \\
        -k 5 --mask 0 --threads 8

Parameter provenance:
  --min-seq-id 0.95 -c 0.70 --cov-mode 0 : archive/root_misc/MEMORIA.md:13 +
      output/tables/mmseqs2_params_sensitivity_note.md (run All_id0_95_cov0_70).
  --cluster-mode 1 (connected components) : MEMORIA.md:13.
      (mmseqs2_reversibility_test.tsv prose says mode 0, but mode 0 is greedy
      set-cover, not connected components; mode 1 matches MEMORIA.)
  -k 5 --mask 0 : same as the intra-genome runs in output/intergenic/README.md
      (small k, no low-complexity masking — IGRs are AT-rich for real).
  Input: output/intergenic/combined/All_IGRs.fasta (3454) exact-dedup keep-first
      -> 2659 sequences. (The reversibility note's "2659 input" = 3454 raw minus
      795 exact duplicates, verified: len(set(seqs)) == 2659. NOT a <20bp filter:
      >20bp gives 2608, >=20bp gives 2626.)
  Binary: mmseqs 15.6f452 (pixi.lock, default env).

Variant sweep (261005, mmseqs 15.6f452, 8 threads; canonical = 2247 clusters):

| variant                        | input | clusters |
|--------------------------------|-------|----------|
| mode1 k5 mask0 (this script)   | dedup | 2278     |
| mode0 k5 mask0                 | dedup | 2278     |
| mode1 defaults                 | dedup | 2282     |
| mode0 defaults                 | dedup | 2282     |
| mode1 k5 mask0 s7.5            | dedup | 2278     |
| mode1 k5 mask0 (full 3454)     | full  | 2422     |
| mode1 k4 mask0                 | dedup | 2290     |
| mode1 k5 mask0 e10 / reassign  | dedup | 2278     |
| mode1 k5 mask0 threads1        | dedup | 2278     |

Residual gap: best reproduction gives 2278 clusters vs canonical 2247
(member-id sets agree ~93%, ARI reported below). Not closed by mode/k/mask/
sensitivity/evalue/reassign/threads/input-full-vs-dedup. Likely causes:
original run used an older MMseqs2 version, or post-processing
(merge/split) after clustering. The verification below quantifies the gap
instead of hiding it.

Usage:
    pixi run python experiments/igr/run_mmseqs_canonical.py [-o output/intergenic/mmseqs2/repro_canonical] [--threads 8]

Outputs: <out>/All_repro_cluster.tsv + <out>/REPRODUCIBILITY.md with the
exact CLI, versions, counts and agreement metrics vs the canonical file.
Exit code: 0 always (gap documented); prints AGREE metrics to stdout.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent.parent
ALL_IGRS = ROOT / "output/intergenic/combined/All_IGRs.fasta"
CANON = ROOT / "output/intergenic/mmseqs2/combined_clusters/All_id0_95_cov0_70_cluster.tsv"

MIN_SEQ_ID = 0.95
COV = 0.70
COV_MODE = 0
CLUSTER_MODE = 1
K = 5
MASK = 0


def mmseqs_bin():
    for cand in [ROOT / ".pixi/envs/default/bin/mmseqs", shutil.which("mmseqs")]:
        if cand and Path(cand).exists():
            return str(cand)
    raise SystemExit("ERROR: mmseqs binary not found (pixi install first)")


def build_dedup_input(out_path: Path) -> int:
    seen, kept = set(), []
    for r in SeqIO.parse(ALL_IGRS, "fasta"):
        s = str(r.seq).upper()
        if s not in seen:
            seen.add(s)
            kept.append((r.id, s))
    with open(out_path, "w") as fh:
        for i, s in kept:
            fh.write(f">{i}\n{s}\n")
    return len(kept)


def run_easy_cluster(fasta: Path, prefix: Path, tmp: Path, threads: int) -> list[str]:
    cmd = [mmseqs_bin(), "easy-cluster", str(fasta), str(prefix), str(tmp),
           "--min-seq-id", str(MIN_SEQ_ID), "-c", str(COV),
           "--cov-mode", str(COV_MODE), "--cluster-mode", str(CLUSTER_MODE),
           "-k", str(K), "--mask", str(MASK), "--threads", str(threads)]
    log = prefix.parent / "easy_cluster_cmd.txt"
    log.write_text(" ".join(cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    (prefix.parent / "easy_cluster.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"ERROR: easy-cluster rc={r.returncode}\n{r.stderr[-2000:]}")
    return cmd


def agreement(new_tsv: Path) -> dict:
    new = pd.read_csv(new_tsv, sep="\t", header=None, names=["rep", "mem"])
    can = pd.read_csv(CANON, sep="\t", header=None, names=["rep", "mem"])
    m1 = dict(zip(new.mem.astype(str), new.rep.astype(str)))
    m0 = dict(zip(can.mem.astype(str), can.rep.astype(str)))
    shared = [i for i in m1 if i in m0]
    labs1 = [m1[i] for i in shared]
    labs0 = [m0[i] for i in shared]
    ari = adjusted_rand_score(labs0, labs1)
    from collections import defaultdict
    c0, c1 = defaultdict(set), defaultdict(set)
    for i in shared:
        c0[m0[i]].add(i)
        c1[m1[i]].add(i)
    s0 = {frozenset(v) for v in c0.values()}
    s1 = {frozenset(v) for v in c1.values()}
    return {"n_new": int(new.rep.nunique()), "n_canon": int(can.rep.nunique()),
            "shared_ids": len(shared), "identical_sets": len(s0 & s1),
            "n_canon_sets": len(s0), "ari": round(float(ari), 4)}


def main():
    ap = argparse.ArgumentParser(description="Reproduce canonical MMseqs2 clustering.")
    ap.add_argument("-o", "--output",
                    default="output/intergenic/mmseqs2/repro_canonical")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    out = ROOT / args.output
    (out / "tmp").mkdir(parents=True, exist_ok=True)

    n_in = build_dedup_input(out / "All_IGRs_dedup.fasta")
    print(f"input: {n_in} unique sequences")
    cmd = run_easy_cluster(out / "All_IGRs_dedup.fasta", out / "All_repro",
                            out / "tmp", args.threads)
    print("cmd:", " ".join(cmd))
    met = agreement(out / "All_repro_cluster.tsv")
    print("agreement:", met)

    ver = subprocess.run([mmseqs_bin(), "version"], capture_output=True, text=True)
    (out / "REPRODUCIBILITY.md").write_text(
        "# Canonical MMseqs2 clustering — reproduction record\n\n"
        f"mmseqs: {ver.stdout.strip()}\n\n"
        f"command: `{' '.join(cmd)}`\n\n"
        f"input: All_IGRs.fasta exact-dedup keep-first -> {n_in} seqs\n\n"
        f"result clusters: {met['n_new']} (canonical file: {met['n_canon']})\n\n"
        f"shared ids: {met['shared_ids']}, identical member sets: "
        f"{met['identical_sets']}/{met['n_canon_sets']}, ARI={met['ari']}\n\n"
        "Residual gap (31 clusters) documented in script docstring; not closed "
        "by mode/k/mask/sensitivity/evalue/reassign/threads/input variants.\n")
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
