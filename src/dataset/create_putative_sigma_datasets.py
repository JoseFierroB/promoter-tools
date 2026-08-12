#!/usr/bin/env python3
"""
Generate putative sigma factor datasets from Slager-method assignments.

Reads output/tables/igr/d39v_sigma_assigned.tsv and creates:
  - positives_81bp_SigA_putative.fasta     (RpoD_complete from None)
  - positives_81bp_SigA_partial.fasta      (RpoD_partial from None)
  - positives_81bp_None_before.fasta       (all None TSS)
  - positives_81bp_None_after.fasta        (still unassigned after)
  + metadata TSVs for each.

Output: data/benchmark/
"""

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_FASTA = ROOT / "data/benchmark/d39v/positives_81bp.fasta"
SRC_META  = ROOT / "data/benchmark/d39v/positives_81bp_metadata.tsv"
SIGMA_TSV = ROOT / "output/tables/igr/d39v_sigma_assigned.tsv"
OUT_DIR   = ROOT / "data/benchmark/d39v"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load source sequences ──
seqs = {}
with open(SRC_FASTA) as f:
    header = ""
    seq = ""
    for line in f:
        if line.startswith(">"):
            if header: seqs[header.split()[0]] = seq
            header = line[1:].strip().split()[0]
            seq = ""
        else:
            seq += line.strip()
    if header: seqs[header.split()[0]] = seq

# ── Load source metadata ──
meta = {}
with open(SRC_META, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        meta[row["Sequence_ID"].strip()] = row

# ── Load sigma assignments ──
sigma = {}
with open(SIGMA_TSV, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        sid = row["tss_id"].strip()
        orig = row["original_sigma"].strip()
        pred = row["predicted_sigma"].strip()
        sigma[sid] = (orig, pred)

# ── Define output datasets ──
datasets = {
    "SigA_putative": {
        "fasta": "putative/positives_81bp_SigA_putative.fasta",
        "tsv":   "putative/positives_81bp_SigA_putative_metadata.tsv",
        "filter": lambda o, p: o == "None" and p == "RpoD_complete",
        "label": "putative_SigA",
    },
    "SigA_partial": {
        "fasta": "putative/positives_81bp_SigA_partial.fasta",
        "tsv":   "putative/positives_81bp_SigA_partial_metadata.tsv",
        "filter": lambda o, p: o == "None" and p == "RpoD_partial",
        "label": "putative_SigA_partial",
    },
    "None_before": {
        "fasta": "unassigned/positives_81bp_None_before.fasta",
        "tsv":   "unassigned/positives_81bp_None_before_metadata.tsv",
        "filter": lambda o, p: o == "None",
        "label": "None",
    },
    "None_after": {
        "fasta": "unassigned/positives_81bp_None_after.fasta",
        "tsv":   "unassigned/positives_81bp_None_after_metadata.tsv",
        "filter": lambda o, p: o == "None" and p == "unassigned",
        "label": "unassigned",
    },
    "ComE_putative": {
        "fasta": "putative/positives_81bp_ComE_putative.fasta",
        "tsv":   "putative/positives_81bp_ComE_putative_metadata.tsv",
        "filter": lambda o, p: o == "None" and p == "ComE",
        "label": "putative_ComE",
    },
}

# ── Generate each dataset ──
for name, ds in datasets.items():
    ids = [sid for sid, (o, p) in sigma.items() if ds["filter"](o, p)]
    
    if not ids:
        print(f"  {name:<20} 0 seqs → skipping")
        continue

    # Write FASTA
    fa_path = OUT_DIR / ds["fasta"]
    with open(fa_path, "w") as f:
        for sid in ids:
            seq = seqs.get(sid, "")
            label = ds["label"]
            header = sid.replace("_None", f"_{label}") if "_None" in sid else f"{sid}_{label}"
            f.write(f">{header}\n{seq}\n")
    
    # Write metadata TSV
    tsv_path = OUT_DIR / ds["tsv"]
    with open(tsv_path, "w", newline="") as f:
        if ids and ids[0] in meta:
            fieldnames = list(meta[ids[0]].keys())
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            w.writeheader()
            for sid in ids:
                if sid in meta:
                    w.writerow(meta[sid])

    print(f"  {name:<20} {len(ids):>4} seqs → {ds['fasta']}")

# ── Summary ──
print(f"\n  Done. Files in {OUT_DIR}/")
