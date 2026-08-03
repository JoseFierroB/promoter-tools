#!/usr/bin/env python3
"""
ROC Curves (FPR vs TPR) for Incremental Few-Shot & Zero-Shot FIMO Pipelines.
Generates ROC curves matching the aesthetic style of meme_zero_shot_roc.png.

Usage:
    pixi run --manifest-path tools/meme/pixi.toml python src/experiments/plot_fewshot_roc_curves.py
"""

import csv
import math
import os
import random
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

from Bio import SeqIO
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
POS_FASTA = ROOT / "data/benchmark/positives_81bp.fasta"
NEG_FASTA = ROOT / "data/benchmark/negatives_81bp.fasta"
OUT_DIR = ROOT / "output" / "plots" / "meme"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATABASES = {
    "DPINTERACT": ROOT / "tools/meme/motif_databases/ECOLI/dpinteract.meme",
    "PRODORIC 2021.9": ROOT / "tools/meme/motif_databases/PROKARYOTE/prodoric_2021.9.meme",
    "CollecTF": ROOT / "tools/meme/motif_databases/PROKARYOTE/collectf.meme",
}

RANDOM_SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def load_and_split_data(test_ratio=0.25):
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

    return (pos_train, neg_train), (pos_test, neg_test)


def run_streme_discovery(pos_recs, neg_recs, work_dir: Path) -> Path:
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
    with open(output_path, "w") as fout:
        with open(db1_path, "r") as f1:
            fout.write(f1.read().strip() + "\n\n")
        
        if db2_path and db2_path.exists():
            with open(db2_path, "r") as f2:
                content = f2.read()
                content_clean = re.sub(r"^MEME version \d+.*?\n\n", "", content, flags=re.DOTALL)
                fout.write(content_clean.strip() + "\n")


def get_fimo_predictions(motif_file: Path, pos_test, neg_test):
    tmpdir = Path(tempfile.mkdtemp(prefix="fimo_roc_"))
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
    return y_true, y_scores


def main():
    (pos_train, neg_train), (pos_test, neg_test) = load_and_split_data(test_ratio=0.25)

    curves = []

    print("Generating ROC Curves for Zero-Shot & Few-Shot Pipelines...")

    # 1. Zero-Shot DBs (0% S. pneumoniae Train Data)
    for name, db_path in DATABASES.items():
        y_true, y_scores = get_fimo_predictions(db_path, pos_test, neg_test)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        curves.append((f"{name} (Zero-Shot)", fpr, tpr, roc_auc, "zero_shot"))

    # 2. Few-Shot 7.5% Train Data (74 promoters)
    work_dir_7 = Path(tempfile.mkdtemp(prefix="fewshot_7_"))
    set_seed(RANDOM_SEED)
    sub7_pos = pos_train[:int(len(pos_train) * 0.10)]
    sub7_neg = neg_train[:int(len(neg_train) * 0.10)]
    streme_7 = run_streme_discovery(sub7_pos, sub7_neg, work_dir_7)

    for name, db_path in DATABASES.items():
        comb_file = work_dir_7 / f"{name}_comb.meme"
        combine_meme_databases(db_path, streme_7, comb_file)
        y_true, y_scores = get_fimo_predictions(comb_file, pos_test, neg_test)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        curves.append((f"{name} + STREME 7.5%", fpr, tpr, roc_auc, "few_shot_7"))

    # 3. Few-Shot 37.5% Train Data (370 promoters - Saturation)
    work_dir_37 = Path(tempfile.mkdtemp(prefix="fewshot_37_"))
    set_seed(RANDOM_SEED)
    sub37_pos = pos_train[:int(len(pos_train) * 0.50)]
    sub37_neg = neg_train[:int(len(neg_train) * 0.50)]
    streme_37 = run_streme_discovery(sub37_pos, sub37_neg, work_dir_37)

    # Pure STREME local model
    y_true, y_scores = get_fimo_predictions(streme_37, pos_test, neg_test)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    curves.append(("STREME Local Only (37.5% Train)", fpr, tpr, roc_auc, "streme_local"))

    # Cleanup temp dirs
    shutil.rmtree(work_dir_7, ignore_errors=True)
    shutil.rmtree(work_dir_37, ignore_errors=True)

    # ── PLOT ROC CURVES ──
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=300)

    # Styling colors and linestyles
    color_map = {
        "DPINTERACT (Zero-Shot)": "#1f77b4",
        "DPINTERACT + STREME 7.5%": "#0055ff",
        "CollecTF (Zero-Shot)": "#ff7f0e",
        "CollecTF + STREME 7.5%": "#d95f02",
        "PRODORIC 2021.9 (Zero-Shot)": "#2ca02c",
        "PRODORIC 2021.9 + STREME 7.5%": "#006d2c",
        "STREME Local Only (37.5% Train)": "#d62728"
    }

    style_map = {
        "DPINTERACT (Zero-Shot)": ":",
        "DPINTERACT + STREME 7.5%": "-",
        "CollecTF (Zero-Shot)": ":",
        "CollecTF + STREME 7.5%": "-",
        "PRODORIC 2021.9 (Zero-Shot)": ":",
        "PRODORIC 2021.9 + STREME 7.5%": "-",
        "STREME Local Only (37.5% Train)": "--"
    }

    width_map = {
        "DPINTERACT (Zero-Shot)": 1.5,
        "DPINTERACT + STREME 7.5%": 2.2,
        "CollecTF (Zero-Shot)": 1.5,
        "CollecTF + STREME 7.5%": 2.2,
        "PRODORIC 2021.9 (Zero-Shot)": 1.5,
        "PRODORIC 2021.9 + STREME 7.5%": 2.2,
        "STREME Local Only (37.5% Train)": 2.0
    }

    for label, fpr, tpr, roc_auc, group in curves:
        c = color_map.get(label, "#333333")
        ls = style_map.get(label, "-")
        lw = width_map.get(label, 1.8)
        ax.plot(fpr, tpr, lw=lw, ls=ls, color=c, alpha=0.88,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3, label="Random Guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, labelpad=8)
    ax.set_title("MEME Suite — S. pneumoniae D39V\nROC Curves: Zero-Shot vs Few-Shot Transfer (Fixed 25% Test Set)",
                 fontweight="bold", fontsize=12, pad=12)
    
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92, frameon=True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    png_out = OUT_DIR / "fimo_fewshot_roc_curves.png"
    svg_out = OUT_DIR / "fimo_fewshot_roc_curves.svg"

    plt.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.savefig(svg_out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Saved ROC curves to {png_out} and {svg_out}")


if __name__ == "__main__":
    main()
