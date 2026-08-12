#!/usr/bin/env python3
"""
Run FIMO with each motif database individually and save prediction CSVs.
Output: fimo_{db}_pos.csv / fimo_{db}_neg.csv in output/predictions/
"""
import subprocess, tempfile, shutil, csv, math
from pathlib import Path
import numpy as np, pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
POS = ROOT / "data/benchmark/d39v/positives_81bp.fasta"
NEG = ROOT / "data/benchmark/d39v/negatives_81bp.fasta"
OUT_DIR = ROOT / "output/predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

G1 = ROOT / "tools/meme/motif_databases/PROKARYOTE"
G2 = ROOT / "tools/meme/motif_databases/ECOLI"

DB_MATRIX = {
    # Group 1 — letter-probability matrix
    "prodoric_2021.9": G1 / "prodoric_2021.9.meme",
    "collectf":        G1 / "collectf.meme",
    "swissregulon":    G2 / "SwissRegulon_e_coli.meme",
    "fan_2020":        G1 / "fan2020.meme",
    # Group 2 — log-odds matrix
    "dpinteract":      G2 / "dpinteract.meme",
    "regtransbase":    G1 / "regtransbase.meme",
}


def fimo_score(db_path: Path, pos_recs, neg_recs) -> tuple:
    tmpdir = Path(tempfile.mkdtemp(prefix="fm_"))
    combined = tmpdir / "all.fa"
    with open(combined, "w") as f:
        for r in pos_recs:
            SeqIO.write(r, f, "fasta")
        for r in neg_recs:
            SeqIO.write(r, f, "fasta")

    res = subprocess.run(
        ["fimo", "--text", "--skip-matched-sequence", str(db_path), str(combined)],
        capture_output=True, text=True, timeout=120)

    scores = {}
    for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
        try:
            pv = float(row["p-value"])
        except (ValueError, KeyError, TypeError):
            continue
        nl = 999.0 if pv <= 0 else -math.log10(pv)
        s = row["sequence_name"]
        if s not in scores or nl > scores[s]:
            scores[s] = nl

    for r in pos_recs + neg_recs:
        if r.id not in scores:
            scores[r.id] = 0.0

    shutil.rmtree(tmpdir, ignore_errors=True)
    return scores


def main():
    pos = list(SeqIO.parse(POS, "fasta"))
    neg = list(SeqIO.parse(NEG, "fasta"))

    for name, path in DB_MATRIX.items():
        pos_file = OUT_DIR / f"fimo_{name}_pos.csv"
        neg_file = OUT_DIR / f"fimo_{name}_neg.csv"

        if pos_file.exists() and neg_file.exists():
            print(f"  {name}: skip (exists)")
            continue

        scores = fimo_score(path, pos, neg)
        pd.DataFrame({"PRED": [scores[r.id] for r in pos]}).to_csv(
            pos_file, sep="\t", index=False)
        pd.DataFrame({"PRED": [scores[r.id] for r in neg]}).to_csv(
            neg_file, sep="\t", index=False)
        print(f"  {name}: {len(pos)}+{len(neg)} seqs saved")


if __name__ == "__main__":
    main()
