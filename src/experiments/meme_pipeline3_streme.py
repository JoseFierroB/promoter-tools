#!/usr/bin/env python3
"""
Pipeline 3: STREME → FIMO (discriminative predictor)
STREME uses BOTH positives and negatives to find discriminative motifs.
This is the recommended pipeline for benchmark use.
"""
import subprocess, tempfile, shutil, csv, math
from pathlib import Path
from Bio import SeqIO
import numpy as np
from sklearn.metrics import roc_auc_score

POS = "data/benchmark/positives_81bp.fasta"
NEG = "data/benchmark/negatives_81bp.fasta"

tmpdir = Path(tempfile.mkdtemp(prefix="m3_"))
pos = list(SeqIO.parse(POS, "fasta"))
neg = list(SeqIO.parse(NEG, "fasta"))

combined = tmpdir / "all.fa"
with open(combined, "w") as f:
    for r in pos: SeqIO.write(r, f, "fasta")
    for r in neg: SeqIO.write(r, f, "fasta")

# STREME: discriminative motif discovery (positives vs negatives)
print("STREME: discovering discriminative motifs...")
subprocess.run(["streme", "-oc", str(tmpdir/"streme"), "-dna", "-minw", "10", "-maxw", "20",
    "-p", POS, "-n", NEG],
    capture_output=True, text=True, timeout=120)

# FIMO: scan all sequences
print("FIMO: scanning...")
res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
    str(tmpdir/"streme"/"streme.txt"), str(combined)],
    capture_output=True, text=True, timeout=120)

scores = {}
for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
    try: pv = float(row["p-value"])
    except: continue
    nl = 999.0 if pv <= 0 else -math.log10(pv)
    s = row["sequence_name"]
    if s not in scores or nl > scores[s]: scores[s] = nl

for r in pos + neg:
    if r.id not in scores: scores[r.id] = 0.0

y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
sc = np.array([scores[r.id] for r in pos + neg])
auc = roc_auc_score(y, sc)

print(f"\n═══ Pipeline 3: STREME → FIMO ═══")
print(f"  Input:  {len(pos)} pos + {len(neg)} neg (both used for discovery)")
print(f"  AUC:    {auc:.4f}")
print(f"  Note:   Uses negatives to find discriminative motifs")
shutil.rmtree(tmpdir)
