#!/usr/bin/env python3
"""
Pipeline 1: MEME → FIMO (minimal predictor)
MEME discovers motifs from positives only, FIMO scores all sequences.
No negative set needed. No annotation.
"""
import subprocess, tempfile, shutil, csv, math
from pathlib import Path
from Bio import SeqIO
import numpy as np
from sklearn.metrics import roc_auc_score

POS = "data/benchmark/positives_81bp.fasta"
NEG = "data/benchmark/negatives_81bp.fasta"

tmpdir = Path(tempfile.mkdtemp(prefix="m1_"))
pos = list(SeqIO.parse(POS, "fasta"))
neg = list(SeqIO.parse(NEG, "fasta"))

# Combine pos + neg for FIMO scanning
combined = tmpdir / "all.fa"
with open(combined, "w") as f:
    for r in pos: SeqIO.write(r, f, "fasta")
    for r in neg: SeqIO.write(r, f, "fasta")

# MEME: discover motifs from positives only
print("MEME: discovering motifs...")
subprocess.run(["meme", POS, "-dna", "-mod", "zoops",
    "-minw", "10", "-maxw", "20", "-oc", str(tmpdir/"meme"), "-nostatus"],
    capture_output=True, text=True, timeout=300)

# FIMO: scan all sequences
print("FIMO: scanning...")
res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
    str(tmpdir/"meme"/"meme.xml"), str(combined)],
    capture_output=True, text=True, timeout=120)

# Score: max -log10(p-value)
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

print(f"\n═══ Pipeline 1: MEME → FIMO ═══")
print(f"  Input:  {len(pos)} pos (no negatives needed for MEME)")
print(f"  Score:  {len(pos)+len(neg)} sequences")
print(f"  AUC:    {auc:.4f}")
print(f"  Note:   No negative set used for motif discovery")
shutil.rmtree(tmpdir)
