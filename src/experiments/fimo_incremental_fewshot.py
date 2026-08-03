#!/usr/bin/env python3
"""
Incremental Few-Shot Motif Discovery & Transfer Benchmark.

Evaluates the top 3 individual motif databases (DPINTERACT, PRODORIC 2021.9, CollecTF)
incrementally augmented with S. pneumoniae local motifs discovered via STREME at
training fractions ranging from 0% (Pure Zero-Shot) to 75% (Full Train set).

Evaluation is strictly performed on a fixed 25% held-out Test Set (seed=42).

Usage:
    pixi run --manifest-path tools/meme/pixi.toml python src/experiments/fimo_incremental_fewshot.py
"""

import csv
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
POS_FASTA = ROOT / "data/benchmark/positives_81bp.fasta"
NEG_FASTA = ROOT / "data/benchmark/negatives_81bp.fasta"

OUT_TABLE_DIR = ROOT / "output" / "tables"
OUT_PLOT_DIR = ROOT / "output" / "plots" / "meme"
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Top 3 individual DBs
DATABASES = {
    "DPINTERACT (E. coli)": ROOT / "tools/meme/motif_databases/ECOLI/dpinteract.meme",
    "PRODORIC 2021.9": ROOT / "tools/meme/motif_databases/PROKARYOTE/prodoric_2021.9.meme",
    "CollecTF": ROOT / "tools/meme/motif_databases/PROKARYOTE/collectf.meme",
}

# Training fractions relative to the 75% training pool
TRAIN_FRACTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]
RANDOM_SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_and_split_data(test_ratio=0.25):
    """Load FASTA records and create a fixed train/test split."""
    set_seed(RANDOM_SEED)
    pos_recs = list(SeqIO.parse(POS_FASTA, "fasta"))
    neg_recs = list(SeqIO.parse(NEG_FASTA, "fasta"))

    random.shuffle(pos_recs)
    random.shuffle(neg_recs)

    n_pos_test = int(len(pos_recs) * test_ratio)
    n_neg_test = int(len(neg_recs) * test_ratio)

    pos_test = pos_recs[:n_pos_test]
    pos_train = pos_recs[n_pos_test:]

    neg_test = neg_recs[:n_neg_test]
    neg_train = neg_recs[n_neg_test:]

    print(f"[DATA SPLIT] Fixed Test Set (25%): {len(pos_test)} pos + {len(neg_test)} neg")
    print(f"[DATA SPLIT] Total Train Pool (75%): {len(pos_train)} pos + {len(neg_train)} neg\n")

    return (pos_train, neg_train), (pos_test, neg_test)


def run_streme_discovery(pos_recs, neg_recs, work_dir: Path) -> Path:
    """Run STREME on positive and negative records to extract local motifs."""
    train_pos_fa = work_dir / "streme_pos.fa"
    train_neg_fa = work_dir / "streme_neg.fa"
    streme_out_dir = work_dir / "streme_out"

    with open(train_pos_fa, "w") as f:
        SeqIO.write(pos_recs, f, "fasta")
    with open(train_neg_fa, "w") as f:
        SeqIO.write(neg_recs, f, "fasta")

    cmd = [
        "streme", "-oc", str(streme_out_dir),
        "-dna", "-minw", "10", "-maxw", "20",
        "-p", str(train_pos_fa),
        "-n", str(train_neg_fa)
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    streme_txt = streme_out_dir / "streme.txt"
    return streme_txt if streme_txt.exists() else None


def combine_meme_databases(db1_path: Path, db2_path: Path, output_path: Path):
    """Concatenate two MEME motif files into a single valid MEME file."""
    with open(output_path, "w") as fout:
        with open(db1_path, "r") as f1:
            fout.write(f1.read().strip() + "\n\n")
        
        if db2_path and db2_path.exists():
            with open(db2_path, "r") as f2:
                content = f2.read()
                # Remove redundant MEME version header from second file
                content_clean = re.sub(r"^MEME version \d+.*?\n\n", "", content, flags=re.DOTALL)
                fout.write(content_clean.strip() + "\n")


def evaluate_fimo_auc(motif_file: Path, pos_test, neg_test) -> float:
    """Run FIMO scanning on the test set using a given motif file and return ROC-AUC."""
    tmpdir = Path(tempfile.mkdtemp(prefix="fimo_eval_"))
    test_fa = tmpdir / "test_combined.fa"

    with open(test_fa, "w") as f:
        for r in pos_test:
            SeqIO.write(r, f, "fasta")
        for r in neg_test:
            SeqIO.write(r, f, "fasta")

    cmd = ["fimo", "--text", "--skip-matched-sequence", str(motif_file), str(test_fa)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

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

    for r in pos_test + neg_test:
        if r.id not in scores:
            scores[r.id] = 0.0

    shutil.rmtree(tmpdir, ignore_errors=True)

    y_true = np.hstack([np.ones(len(pos_test)), np.zeros(len(neg_test))])
    y_scores = np.array([scores[r.id] for r in pos_test + neg_test])
    return float(roc_auc_score(y_true, y_scores))


def main():
    t0 = time.time()
    (pos_train, neg_train), (pos_test, neg_test) = load_and_split_data(test_ratio=0.25)

    results_data = []

    # Include pure STREME baseline (no external DB)
    all_experiments = dict(DATABASES)
    all_experiments["STREME Local Only (No DB)"] = None

    print("════════════════════════════════════════════════════════════════")
    print(" STARTING INCREMENTAL FEW-SHOT EXPERIMENT")
    print("════════════════════════════════════════════════════════════════\n")

    for db_name, db_path in all_experiments.items():
        print(f"▶ Evaluating Model Pipeline: {db_name}")
        for frac in TRAIN_FRACTIONS:
            set_seed(RANDOM_SEED)
            
            n_pos_sub = int(len(pos_train) * frac)
            n_neg_sub = int(len(neg_train) * frac)

            sub_pos_train = pos_train[:n_pos_sub]
            sub_neg_train = neg_train[:n_neg_sub]

            work_dir = Path(tempfile.mkdtemp(prefix=f"fewshot_{frac:.2f}_"))

            streme_motif_file = None
            if frac > 0.0 and len(sub_pos_train) > 0:
                streme_motif_file = run_streme_discovery(sub_pos_train, sub_neg_train, work_dir)

            if db_name == "STREME Local Only (No DB)":
                if frac == 0.0 or not streme_motif_file:
                    auc_val = 0.5000  # Baseline random guessing when 0 train data used
                else:
                    auc_val = evaluate_fimo_auc(streme_motif_file, pos_test, neg_test)
            else:
                if frac == 0.0:
                    # Pure Zero-Shot using DB only
                    auc_val = evaluate_fimo_auc(db_path, pos_test, neg_test)
                else:
                    # Hybrid: External DB + Local STREME motifs
                    combined_motif_file = work_dir / "combined_hybrid.meme"
                    combine_meme_databases(db_path, streme_motif_file, combined_motif_file)
                    auc_val = evaluate_fimo_auc(combined_motif_file, pos_test, neg_test)

            shutil.rmtree(work_dir, ignore_errors=True)

            percent_train = int(frac * 75)
            print(f"   • Train Fraction: {frac*100:>5.1f}% ({percent_train}% of total data, n_pos={len(sub_pos_train)}) ➔ Test AUC: {auc_val:.4f}")

            results_data.append({
                "Pipeline": db_name,
                "Fraction_Train_Pool": frac,
                "Percent_Total_Data": percent_train,
                "N_Pos_Train": len(sub_pos_train),
                "N_Neg_Train": len(sub_neg_train),
                "AUC_Test": round(auc_val, 4)
            })
        print()

    # Save TSV Table
    df_results = pd.DataFrame(results_data)
    tsv_out = OUT_TABLE_DIR / "fimo_incremental_fewshot.tsv"
    df_results.to_csv(tsv_out, sep="\t", index=False)
    print(f"[SUCCESS] Saved results table to {tsv_out}")

    # Generate Professional Plot
    plot_results(df_results)
    print(f"[SUCCESS] Experiment completed in {time.time() - t0:.1f}s")


def plot_results(df: pd.DataFrame):
    """Plot publication-quality AUC trajectory curves."""
    plt.figure(figsize=(9.5, 6), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = {
        "DPINTERACT (E. coli)": "#1f77b4",       # Muted Blue
        "PRODORIC 2021.9": "#2ca02c",             # Emerald Green
        "CollecTF": "#ff7f0e",                    # Warm Amber/Orange
        "STREME Local Only (No DB)": "#d62728"   # Crimson Red
    }
    markers = {
        "DPINTERACT (E. coli)": "o",
        "PRODORIC 2021.9": "s",
        "CollecTF": "^",
        "STREME Local Only (No DB)": "D"
    }

    linestyles = {
        "DPINTERACT (E. coli)": "-",
        "PRODORIC 2021.9": "-",
        "CollecTF": "-",
        "STREME Local Only (No DB)": "--"
    }

    for pipeline in df["Pipeline"].unique():
        sub = df[df["Pipeline"] == pipeline].sort_values("Fraction_Train_Pool")
        x_vals = sub["Percent_Total_Data"].values
        y_vals = sub["AUC_Test"].values

        plt.plot(
            x_vals, y_vals,
            label=pipeline,
            color=colors.get(pipeline, "black"),
            marker=markers.get(pipeline, "o"),
            linestyle=linestyles.get(pipeline, "-"),
            linewidth=2.2,
            markersize=7,
            alpha=0.9
        )

    plt.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2, label="Random Guess (0.50)")
    
    plt.title("Incremental Few-Shot Motif Discovery: S. pneumoniae D39V", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Percentage of Total S. pneumoniae Dataset in Training (%)", fontsize=12, labelpad=10)
    plt.ylabel("ROC-AUC Score on Fixed Test Set (25%)", fontsize=12, labelpad=10)
    
    plt.xticks([0, 7.5, 18.75, 37.5, 56.25, 75], ["0% (Zero-Shot)", "7.5%", "18.75%", "37.5%", "56.25%", "75%"])
    plt.ylim(0.48, 0.90)
    plt.legend(frameon=True, framealpha=0.95, edgecolor="none", fontsize=10, loc="lower right")
    plt.tight_layout()

    png_out = OUT_PLOT_DIR / "fimo_incremental_fewshot_auc.png"
    svg_out = OUT_PLOT_DIR / "fimo_incremental_fewshot_auc.svg"
    
    plt.savefig(png_out, dpi=300)
    plt.savefig(svg_out)
    plt.close()
    print(f"[SUCCESS] Saved plots to {png_out} and {svg_out}")


if __name__ == "__main__":
    main()
