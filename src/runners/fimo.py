#!/usr/bin/env python3
"""FIMO zero-shot promoter prediction — unified runner (Prok or E. coli DB).

Invoked directly (defaults to Prok DB) or through the thin wrappers
fimo_prok.py / fimo_db.py, which inject --db/--label/--tag.
"""
import argparse
import concurrent.futures
import csv
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from Bio import SeqIO
import pandas as pd

_ENV_BIN = str(Path(__file__).resolve().parent.parent.parent / "tools/meme/.pixi/envs/default/bin")
if _ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_ENV_BIN}:{os.environ.get('PATH', '')}"

PROK_DB = Path(__file__).resolve().parent.parent.parent / "tools/meme/motif_databases/unified_prokaryote.meme"


def _run_fimo_chunk(db: str, fasta_path: Path, timeout: int) -> str:
    res = subprocess.run(
        ["fimo", "--text", "--skip-matched-sequence", db, str(fasta_path)],
        capture_output=True, text=True, timeout=timeout)
    return res.stdout


def main():
    p = argparse.ArgumentParser(description="FIMO zero-shot promoter prediction")
    p.add_argument("--pos", required=True, help="Positive FASTA")
    p.add_argument("--neg", required=True, help="Negative FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    p.add_argument("--db", default=str(PROK_DB), help="Motif DB path")
    p.add_argument("--label", default="fimo_prok", help="Output file prefix (e.g. fimo_prok)")
    p.add_argument("--tag", default="FIMO_PROK", help="Tag for the summary line")
    args = p.parse_args()

    pos = list(SeqIO.parse(args.pos, "fasta"))
    neg = list(SeqIO.parse(args.neg, "fasta"))

    tmpdir = Path(tempfile.mkdtemp(prefix=f"fimo_{args.label}_"))
    records = pos + neg
    n_chunks = max(1, int(os.environ.get("OMP_NUM_THREADS", "1") or 1))
    n_chunks = min(n_chunks, len(records))
    chunk_size = max(1, math.ceil(len(records) / n_chunks))
    chunk_paths = []
    for i in range(0, len(records), chunk_size):
        cp = tmpdir / f"chunk_{i // chunk_size}.fa"
        with open(cp, "w") as f:
            for r in records[i:i + chunk_size]:
                SeqIO.write(r, f, "fasta")
        chunk_paths.append(cp)

    t0 = time.perf_counter()
    n_total = len(records)
    fimo_timeout = max(900, int((n_total / max(n_chunks, 1)) / 40.0 * 3))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_chunks) as ex:
        outs = list(ex.map(lambda cp: _run_fimo_chunk(args.db, cp, fimo_timeout), chunk_paths))

    scores = {}
    for out in outs:
        for row in csv.DictReader(out.splitlines(), delimiter="\t"):
            try:
                pv = float(row["p-value"])
            except (ValueError, KeyError, TypeError):
                continue
            nl = 999.0 if pv <= 0 else -math.log10(pv)
            s = row["sequence_name"]
            if s not in scores or nl > scores[s]:
                scores[s] = nl

    for r in pos + neg:
        if r.id not in scores:
            scores[r.id] = 0.0

    elapsed = time.perf_counter() - t0
    n_total = len(pos) + len(neg)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PRED": [scores[r.id] for r in pos]}).to_csv(
        out_dir / f"{args.label}_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": [scores[r.id] for r in neg]}).to_csv(
        out_dir / f"{args.label}_neg.csv", sep="\t", index=False)

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"{args.tag}: {n_total} seqs in {elapsed:.3f}s")


if __name__ == "__main__":
    main()