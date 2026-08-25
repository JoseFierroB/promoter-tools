#!/usr/bin/env python3
"""MEME runner — STREME de novo discovery + FIMO 2-fold CV."""
import argparse
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
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in os.environ.get("PYTHONPATH", ""):
    import sys as _sys
    _sys.path.insert(0, str(_ROOT))

_ENV_BIN = str(_ROOT / "tools/meme/.pixi/envs/default/bin")
if _ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_ENV_BIN}:{os.environ.get('PATH', '')}"


def main():
    p = argparse.ArgumentParser(description="MEME: STREME+FIMO 2-fold CV")
    p.add_argument("--pos", required=True, help="Positive FASTA")
    p.add_argument("--neg", required=True, help="Negative FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    args = p.parse_args()

    random.seed(42)

    # Original FASTA order is preserved for output alignment; only the CV
    # fold assignment uses the shuffled copies.
    pos_recs = list(SeqIO.parse(args.pos, "fasta"))
    neg_recs = list(SeqIO.parse(args.neg, "fasta"))
    pos_order = [(r.id, str(r.seq)) for r in pos_recs]
    neg_order = [(r.id, str(r.seq)) for r in neg_recs]
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
             "-seed", "42", "-p", str(train_pf), "-n", str(train_nf)],
            capture_output=True, text=True,
            timeout=max(300, int((len(train_pos) + len(train_neg)) / 40.0 * 3)))
        if res.returncode != 0:
            shutil.rmtree(tmpdir, ignore_errors=True)
            continue

        res = subprocess.run(
            ["fimo", "--text", "--skip-matched-sequence",
             str(tmpdir / "streme" / "streme.txt"), str(test_fa)],
            capture_output=True, text=True,
            timeout=max(300, int(len(test_pos) + len(test_neg)) / 40.0 * 3))

        from src.runners._shared import fimo_score_merge
        for s, nl in fimo_score_merge(res.stdout).items():
            if s not in all_scores or nl > all_scores[s]:
                all_scores[s] = nl

        shutil.rmtree(tmpdir, ignore_errors=True)

    for r_id, _ in pos_order + neg_order:
        if r_id not in all_scores:
            all_scores[r_id] = 0.0

    elapsed = time.perf_counter() - t0
    n_total = len(pos_order) + len(neg_order)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write in original FASTA order so row i of each CSV corresponds to
    # sequence i of the input FASTA (positional alignment for downstream
    # stratified analyses).
    pd.DataFrame({"PRED": [all_scores[r_id] for r_id, _ in pos_order]}).to_csv(
        out_dir / "meme_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": [all_scores[r_id] for r_id, _ in neg_order]}).to_csv(
        out_dir / "meme_neg.csv", sep="\t", index=False)

    print(f"MEME: {n_total} seqs in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
