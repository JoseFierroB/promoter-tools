#!/usr/bin/env python3
"""MEME runner — STREME de novo discovery + FIMO 2-fold CV."""
import argparse
import csv
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from Bio import SeqIO
import pandas as pd

# Ensure streme/fimo binaries are in PATH
_ENV_BIN = str(Path(__file__).resolve().parent.parent.parent / "tools/meme/.pixi/envs/default/bin")
if _ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_ENV_BIN}:{os.environ.get('PATH', '')}"


def main():
    p = argparse.ArgumentParser(description="MEME: STREME+FIMO 2-fold CV")
    p.add_argument("--pos", required=True, help="Positive FASTA")
    p.add_argument("--neg", required=True, help="Negative FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    args = p.parse_args()

    random.seed(42)

    pos_recs = list(SeqIO.parse(args.pos, "fasta"))
    neg_recs = list(SeqIO.parse(args.neg, "fasta"))
    random.shuffle(pos_recs)
    random.shuffle(neg_recs)
    mid_pos = len(pos_recs) // 2
    mid_neg = len(neg_recs) // 2

    t0 = time.perf_counter()
    all_scores = {}

    for fold in range(2):
        if fold == 0:
            train_pos, train_neg = pos_recs[:mid_pos], neg_recs[:mid_neg]
            test_pos, test_neg = pos_recs[mid_pos:], neg_recs[mid_neg:]
        else:
            train_pos, train_neg = pos_recs[mid_pos:], neg_recs[mid_neg:]
            test_pos, test_neg = pos_recs[:mid_pos], neg_recs[:mid_neg]

        tmpdir = Path(tempfile.mkdtemp(prefix="meme_cv_"))
        train_pf = tmpdir / "tp.fa"
        train_nf = tmpdir / "tn.fa"
        SeqIO.write(train_pos, train_pf, "fasta")
        SeqIO.write(train_neg, train_nf, "fasta")
        test_fa = tmpdir / "test.fa"
        with open(test_fa, "w") as f:
            for r in test_pos:
                SeqIO.write(r, f, "fasta")
            for r in test_neg:
                SeqIO.write(r, f, "fasta")

        res = subprocess.run(
            ["streme", "-oc", str(tmpdir / "streme"), "-dna", "-minw", "10", "-maxw", "20",
             "-p", str(train_pf), "-n", str(train_nf)],
            capture_output=True, text=True, timeout=300)
        if res.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            continue

        res = subprocess.run(
            ["fimo", "--text", "--skip-matched-sequence",
             str(tmpdir / "streme" / "streme.txt"), str(test_fa)],
            capture_output=True, text=True, timeout=300)

        for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
            try:
                pval = float(row["p-value"])
            except (ValueError, KeyError, TypeError):
                continue
            nl = 999.0 if pval <= 0 else -math.log10(pval)
            s = row["sequence_name"]
            if s not in all_scores or nl > all_scores[s]:
                all_scores[s] = nl

        shutil.rmtree(tmpdir, ignore_errors=True)

    for r in pos_recs + neg_recs:
        if r.id not in all_scores:
            all_scores[r.id] = 0.0

    elapsed = time.perf_counter() - t0
    n_total = len(pos_recs) + len(neg_recs)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PRED": [all_scores[r.id] for r in pos_recs]}).to_csv(
        out_dir / "meme_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": [all_scores[r.id] for r in neg_recs]}).to_csv(
        out_dir / "meme_neg.csv", sep="\t", index=False)

    print(f"MEME: {n_total} seqs in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
