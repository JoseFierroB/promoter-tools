#!/usr/bin/env python3
"""
Pipeline 2: MEME → FIMO → TOMTOM (annotated predictor)
Same as Pipeline 1 but adds Tomtom annotation against known databases.
Shows WHAT the discovered motifs are (sigma factors, TFs).
"""
import subprocess, tempfile, shutil, csv, math, re
from pathlib import Path
from Bio import SeqIO
import numpy as np
from sklearn.metrics import roc_auc_score

POS = "data/benchmark/positives_81bp.fasta"
NEG = "data/benchmark/negatives_81bp.fasta"
DB = "tools/meme/motif_databases/unified_prokaryote.meme"

tmpdir = Path(tempfile.mkdtemp(prefix="m2_"))
pos = list(SeqIO.parse(POS, "fasta"))
neg = list(SeqIO.parse(NEG, "fasta"))

# Combine
combined = tmpdir / "all.fa"
with open(combined, "w") as f:
    for r in pos: SeqIO.write(r, f, "fasta")
    for r in neg: SeqIO.write(r, f, "fasta")

# MEME: discover motifs
print("1. MEME: discovering motifs...")
subprocess.run(["meme", POS, "-dna", "-mod", "zoops",
    "-minw", "10", "-maxw", "20", "-oc", str(tmpdir/"meme"), "-nostatus"],
    capture_output=True, text=True, timeout=300)

# TOMTOM: annotate discovered motifs
print("2. TOMTOM: annotating against unified DB...")
res = subprocess.run(["tomtom", "-no-ssc", "-text", "-min-overlap", "4",
    str(tmpdir/"meme"/"meme.xml"), DB],
    capture_output=True, text=True, timeout=120)

# Show annotations
hits = []
for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
    try:
        q_val = float(row.get("q-value", 1))
    except (ValueError, TypeError):
        continue
    if q_val < 0.05 and row.get("Target_ID"):
        hits.append((row["Target_ID"], q_val))
        print(f"    → {row['Target_ID'][:50]} (q={q_val:.1e})")

# FIMO: scan all sequences
print("3. FIMO: scoring sequences...")
res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
    str(tmpdir/"meme"/"meme.xml"), str(combined)],
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

print(f"\n═══ Pipeline 2: MEME → FIMO → TOMTOM ═══")
print(f"  AUC:        {auc:.4f}")
print(f"  Annotations: {len(hits)} significant hits (q<0.05)")
print(f"  DB used:    {DB}")
shutil.rmtree(tmpdir)
