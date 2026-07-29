#!/usr/bin/env python3
"""Pipeline A: FIMO + pre-existing E. coli motif DB (zero-shot, no training)."""
import subprocess, tempfile, shutil, csv, math, time
from pathlib import Path
from Bio import SeqIO
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
POS = ROOT / "data/benchmark/positives_81bp.fasta"
NEG = ROOT / "data/benchmark/negatives_81bp.fasta"

tmpdir = Path(tempfile.mkdtemp(prefix="fimo_db_"))
pos = list(SeqIO.parse(POS, "fasta"))
neg = list(SeqIO.parse(NEG, "fasta"))

combined = tmpdir / "all.fa"
with open(combined, "w") as f:
    for r in pos: SeqIO.write(r, f, "fasta")
    for r in neg: SeqIO.write(r, f, "fasta")

t0 = time.perf_counter()
res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
    str(ROOT / "tools/meme/motif_databases/ecoli_combined.meme"), str(combined)],
    capture_output=True, text=True, timeout=120)

scores = {}
for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
    try: pv = float(row["p-value"])
    except (ValueError, KeyError, TypeError): continue
    nl = 999.0 if pv <= 0 else -math.log10(pv)
    s = row["sequence_name"]
    if s not in scores or nl > scores[s]: scores[s] = nl

for r in pos + neg:
    if r.id not in scores: scores[r.id] = 0.0

y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
sc = np.array([scores[r.id] for r in pos + neg])
auc = roc_auc_score(y, sc)
print(f"FIMO+DB (E. coli, zero-shot): AUC={auc:.4f}, {len(pos)+len(neg)} seqs in {time.perf_counter()-t0:.1f}s")
shutil.rmtree(tmpdir)
