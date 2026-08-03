#!/usr/bin/env python3
"""
Create and Evaluate Combined Top 4 Motif Database (Zero-Shot & Few-Shot).

Combines Top 3 AUC DBs + 1 Extra Compatible DB:
1. DPINTERACT (68 motifs)
2. PRODORIC 2021.9 (333 motifs)
3. CollecTF (84 motifs)
4. SwissRegulon (97 motifs)

Total: 582 unique prokaryotic & E. coli motifs.

Usage:
    pixi run --manifest-path tools/meme/pixi.toml python src/experiments/evaluate_top4_combined_db.py
"""

import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
POS_FASTA = ROOT / "data/benchmark/positives_81bp.fasta"
NEG_FASTA = ROOT / "data/benchmark/negatives_81bp.fasta"
DB_DIR = ROOT / "tools/meme/motif_databases"
OUT_PLOT_DIR = ROOT / "output" / "plots" / "meme"
OUT_TABLE_DIR = ROOT / "output" / "tables"
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_DBS = {
    "DPINTERACT": DB_DIR / "ECOLI/dpinteract.meme",
    "PRODORIC": DB_DIR / "PROKARYOTE/prodoric_2021.9.meme",
    "CollecTF": DB_DIR / "PROKARYOTE/collectf.meme",
    "SwissRegulon": DB_DIR / "ECOLI/SwissRegulon_e_coli.meme"
}

COMBINED_DB_PATH = DB_DIR / "top4_combined.meme"


def build_top4_combined_db() -> Path:
    """Concatenate and prefix motifs from 4 DBs to avoid ID collisions."""
    all_motifs = []
    
    base_header = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

"""

    for name, path in SOURCE_DBS.items():
        text = path.read_text()
        parts = re.split(r'(\nMOTIF\s+)', text)
        count = 0
        for i in range(1, len(parts), 2):
            motif_block = parts[i] + parts[i+1]
            lines = motif_block.strip().split('\n')
            first_tokens = lines[0].split()
            if len(first_tokens) >= 2:
                first_tokens[1] = f"{name}_{first_tokens[1]}"
            lines[0] = " ".join(first_tokens)
            all_motifs.append("\n".join(lines))
            count += 1
        print(f"  • Added {count:>3} motifs from {name}")

    with open(COMBINED_DB_PATH, "w") as f:
        f.write(base_header)
        f.write("\n\n".join(all_motifs))
        f.write("\n")

    print(f"[SUCCESS] Built combined DB: {COMBINED_DB_PATH} ({len(all_motifs)} total motifs)\n")
    return COMBINED_DB_PATH


def evaluate_fimo(motif_file: Path, pos_recs, neg_recs):
    tmpdir = Path(tempfile.mkdtemp(prefix="fimo_top4_"))
    combined = tmpdir / "all.fa"
    with open(combined, "w") as f:
        for r in pos_recs: SeqIO.write(r, f, "fasta")
        for r in neg_recs: SeqIO.write(r, f, "fasta")

    res = subprocess.run(["fimo", "--text", "--skip-matched-sequence", str(motif_file), str(combined)],
                         capture_output=True, text=True, timeout=180)

    scores = {}
    for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
        try: pv = float(row["p-value"])
        except (ValueError, KeyError, TypeError): continue
        nl = 999.0 if pv <= 0 else -math.log10(pv)
        s = row["sequence_name"]
        if s not in scores or nl > scores[s]: scores[s] = nl

    for r in pos_recs + neg_recs:
        if r.id not in scores: scores[r.id] = 0.0

    shutil.rmtree(tmpdir, ignore_errors=True)

    y_true = np.hstack([np.ones(len(pos_recs)), np.zeros(len(neg_recs))])
    y_scores = np.array([scores[r.id] for r in pos_recs + neg_recs])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return y_true, y_scores, fpr, tpr, roc_auc


def main():
    print("════════════════════════════════════════════════════════════════")
    print(" CONCATENATING TOP 3 + 1 EXTRA COMPATIBLE MOTIF DATABASE")
    print("════════════════════════════════════════════════════════════════")
    
    db_path = build_top4_combined_db()

    pos_recs = list(SeqIO.parse(POS_FASTA, "fasta"))
    neg_recs = list(SeqIO.parse(NEG_FASTA, "fasta"))

    print("Evaluating Zero-Shot performance on full S. pneumoniae dataset...")
    y_true, y_scores, fpr, tpr, roc_auc = evaluate_fimo(db_path, pos_recs, neg_recs)

    print(f"➔ Top 4 Combined DB Zero-Shot AUC-ROC: {roc_auc:.4f}\n")

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(fpr, tpr, lw=2.2, color="#0055ff", label=f"Top 4 Combined DB (582 motifs)  (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3, label="Random Guess (AUC = 0.500)")
    
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("MEME Suite — S. pneumoniae D39V\nTop 4 Concatenated Databases (DPINTERACT + PRODORIC + CollecTF + SwissRegulon)",
                 fontweight="bold", fontsize=11, pad=12)
    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.92)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    png_out = OUT_PLOT_DIR / "top4_combined_roc.png"
    svg_out = OUT_PLOT_DIR / "top4_combined_roc.svg"

    plt.savefig(png_out, dpi=300)
    plt.savefig(svg_out, dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved ROC plot to {png_out} and {svg_out}")


if __name__ == "__main__":
    main()
