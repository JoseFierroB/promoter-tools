#!/usr/bin/env python3
"""
FIMO Zero-Shot Benchmark + STREME De Novo Discovery & TOMTOM Motif Annotation for TIGR4 & D39V.

Pipeline Steps:
1. FIMO Zero-Shot Cross-Strain Evaluation using concatenated 582-motif database (top4_combined.meme).
   - D39V (988 Pos vs 1,000 Neg)
   - TIGR4 High Conf Primary (738 Pos vs 738 Neg)
   - TIGR4 Extended Primary (2,000 Pos vs 2,000 Neg)
2. STREME De Novo Motif Discovery on TIGR4 High Conf Primary promoters.
3. TOMTOM Motif Matching of TIGR4 STREME motifs against top4_combined.meme DB to annotate TFs (SigA, SigX, CcpA, CodY, etc.).

Usage:
    pixi run --manifest-path tools/meme/pixi.toml python src/experiments/evaluate_fimo_streme_tomtom_tigr4.py
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent.parent
MEME_BIN_DIR = ROOT / "tools/meme/.pixi/envs/default/bin"
FIMO_BIN = MEME_BIN_DIR / "fimo"
STREME_BIN = MEME_BIN_DIR / "streme"
TOMTOM_BIN = MEME_BIN_DIR / "tomtom"

COMBINED_DB = ROOT / "tools/meme/motif_databases/top4_combined.meme"

OUTPUT_DIR = ROOT / "output/meme"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = ROOT / "output/plots/meme"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "D39V (Cappable-seq Primary)": {
        "pos": ROOT / "data/benchmark/positives_81bp.fasta",
        "neg": ROOT / "data/benchmark/negatives_81bp.fasta",
        "color": "#1f77b4",
    },
    "TIGR4 (High Conf Primary)": {
        "pos": ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.fasta",
        "neg": ROOT / "output/tigr4_data/negatives_tigr4_high_conf_primary_81bp.fasta",
        "color": "#2ca02c",
    },
    "TIGR4 (Extended Primary)": {
        "pos": ROOT / "output/tigr4_data/positives_tigr4_extended_primary_81bp.fasta",
        "neg": ROOT / "output/tigr4_data/negatives_tigr4_extended_primary_81bp.fasta",
        "color": "#ff7f0e",
    },
}


# ════════════════════════════════════════════════════════════════
# 1. FIMO Zero-Shot Cross-Strain Evaluator
# ════════════════════════════════════════════════════════════════

def run_fimo_on_dataset(
    pos_path: Path, neg_path: Path, out_dir: Path
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Runs FIMO using top4_combined.meme DB and extracts max log-odds score per sequence."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge pos and neg into temporary combined FASTA file
    combined_fasta = out_dir / "eval_combined.fasta"
    pos_recs = list(SeqIO.parse(pos_path, "fasta"))
    neg_recs = list(SeqIO.parse(neg_path, "fasta"))

    pos_ids = set()
    neg_ids = set()

    with open(combined_fasta, "w") as f:
        for r in pos_recs:
            sid = f"POS_{r.id}"
            pos_ids.add(sid)
            f.write(f">{sid}\n{str(r.seq).upper()}\n")
        for r in neg_recs:
            sid = f"NEG_{r.id}"
            neg_ids.add(sid)
            f.write(f">{sid}\n{str(r.seq).upper()}\n")

    fimo_out = out_dir / "fimo_output"
    cmd = [
        str(FIMO_BIN),
        "--oc",
        str(fimo_out),
        "--thresh",
        "1e-2",
        "--verbosity",
        "1",
        str(COMBINED_DB),
        str(combined_fasta),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] FIMO execution failed: {res.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse FIMO TSV output for max log-odds score per sequence
    fimo_tsv = fimo_out / "fimo.tsv"
    scores_dict = {}

    if fimo_tsv.exists():
        df_fimo = pd.read_csv(fimo_tsv, sep="\t", comment="#")
        if "sequence_name" in df_fimo.columns and "score" in df_fimo.columns:
            # Group by sequence_name and get maximum score
            df_scores = df_fimo.groupby("sequence_name")["score"].max()
            scores_dict = df_scores.to_dict()

    y_true = []
    y_scores = []

    # Default score for sequences with zero FIMO matches (min score - 1)
    min_score = min(scores_dict.values()) - 1.0 if scores_dict else -50.0

    for sid in pos_ids:
        y_true.append(1)
        y_scores.append(scores_dict.get(sid, min_score))

    for sid in neg_ids:
        y_true.append(0)
        y_scores.append(scores_dict.get(sid, min_score))

    y_true = np.array(y_true, dtype=np.int32)
    y_scores = np.array(y_scores, dtype=np.float32)

    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    return y_true, y_scores, float(roc_auc), float(pr_auc)


# ════════════════════════════════════════════════════════════════
# 2. STREME De Novo Motif Discovery on TIGR4
# ════════════════════════════════════════════════════════════════

def run_streme_tigr4(pos_path: Path, neg_path: Path, out_dir: Path) -> Path:
    """Runs STREME de novo motif discovery on TIGR4 promoters."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(STREME_BIN),
        "--p",
        str(pos_path),
        "--n",
        str(neg_path),
        "--dna",
        "--minw",
        "6",
        "--maxw",
        "15",
        "--oc",
        str(out_dir),
        "--verbosity",
        "1",
    ]

    print(f"[INFO] Running STREME de novo motif discovery on TIGR4...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] STREME failed: {res.stderr}", file=sys.stderr)

    streme_xml = out_dir / "streme.xml"
    streme_txt = out_dir / "streme.txt"
    target_file = streme_xml if streme_xml.exists() else streme_txt
    print(f"[SUCCESS] STREME finished ➔ {target_file}")
    return target_file


# ════════════════════════════════════════════════════════════════
# 3. TOMTOM Motif Matching & Transcription Factor Identification
# ════════════════════════════════════════════════════════════════

def run_tomtom_annotation(streme_file: Path, out_dir: Path) -> pd.DataFrame:
    """Runs TOMTOM to match TIGR4 STREME motifs against top4_combined.meme DB."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(TOMTOM_BIN),
        "-oc",
        str(out_dir),
        "-thresh",
        "0.05",
        "-verbosity",
        "1",
        str(streme_file),
        str(COMBINED_DB),
    ]

    print(f"[INFO] Running TOMTOM motif matching against top4_combined.meme DB...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] TOMTOM failed: {res.stderr}", file=sys.stderr)

    tomtom_tsv = out_dir / "tomtom.tsv"
    if not tomtom_tsv.exists():
        print(f"[WARNING] TOMTOM TSV output not found at {tomtom_tsv}")
        return pd.DataFrame()

    df_tomtom = pd.read_csv(tomtom_tsv, sep="\t", comment="#")
    print(f"[SUCCESS] TOMTOM identified {len(df_tomtom):,} motif matches ➔ {tomtom_tsv}")
    return df_tomtom


# ════════════════════════════════════════════════════════════════
# 4. Main Pipeline Orchestrator
# ════════════════════════════════════════════════════════════════

def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" FIMO + STREME + TOMTOM PIPELINE: TIGR4 & D39V CROSS-STRAIN BENCHMARK")
    print("═════════════════════════════════════════════════════════════════\n")

    if not COMBINED_DB.exists():
        print(f"[ERROR] Combined motif DB not found at {COMBINED_DB}", file=sys.stderr)
        sys.exit(1)

    # 1. Run FIMO Zero-Shot Benchmark
    fimo_results = []
    for name, paths in DATASETS.items():
        pos_path = paths["pos"]
        neg_path = paths["neg"]

        if not pos_path.exists() or not neg_path.exists():
            print(f"[SKIP] Missing dataset files for {name}")
            continue

        print(f"[FIMO EVALUATING] {name}...")
        out_fimo_dir = OUTPUT_DIR / f"fimo_{name.split()[0].lower()}"
        y_true, y_scores, roc_auc, pr_auc = run_fimo_on_dataset(pos_path, neg_path, out_fimo_dir)

        # Youden's J statistic for optimal threshold
        fpr, tpr, roc_thresh = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_thresh = float(roc_thresh[opt_idx])

        y_pred = (y_scores >= opt_thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        fimo_results.append({
            "Dataset": name,
            "N_Pos": len(y_true[y_true == 1]),
            "N_Neg": len(y_true[y_true == 0]),
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "Accuracy": acc,
            "Sensitivity": sens,
            "Specificity": spec,
            "F1_Score": f1,
            "MCC": mcc,
        })

    # Display FIMO Table
    df_fimo_summary = pd.DataFrame(fimo_results)
    print("\n" + "═" * 95)
    print(" FIMO ZERO-SHOT (top4_combined.meme 582 Motifs) PERFORMANCE TABLE")
    print("═" * 95)
    print(
        f"{'Dataset Tier / Strain':<30} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Accuracy':<8} | {'Sens(TPR)':<9} | {'Spec(TNR)':<9} | {'F1-Score':<8} | {'MCC':<6}"
    )
    print("─" * 95)
    for r in fimo_results:
        print(
            f"{r['Dataset']:<30} | {r['ROC_AUC']:<8.4f} | {r['PR_AUC']:<8.4f} | {r['Accuracy']:<8.4f} | {r['Sensitivity']:<9.4f} | {r['Specificity']:<9.4f} | {r['F1_Score']:<8.4f} | {r['MCC']:<6.4f}"
        )
    print("═" * 95 + "\n")

    # 2. Run STREME on TIGR4 High Conf Primary
    tigr4_pos = DATASETS["TIGR4 (High Conf Primary)"]["pos"]
    tigr4_neg = DATASETS["TIGR4 (High Conf Primary)"]["neg"]
    streme_out_dir = OUTPUT_DIR / "streme_tigr4_high_conf"
    streme_file = run_streme_tigr4(tigr4_pos, tigr4_neg, streme_out_dir)

    # 3. Run TOMTOM Motif Annotation
    tomtom_out_dir = OUTPUT_DIR / "tomtom_tigr4_high_conf"
    df_tomtom = run_tomtom_annotation(streme_file, tomtom_out_dir)

    if not df_tomtom.empty:
        print("\n" + "═" * 90)
        print(" TOP IDENTIFIED TRANSCRIPTION FACTOR MOTIFS IN TIGR4 (TOMTOM vs top4_combined.meme)")
        print("═" * 90)
        cols = [c for c in ["Query_ID", "Target_ID", "p-value", "E-value", "q-value", "Overlap"] if c in df_tomtom.columns]
        if len(cols) < 3:
            cols = df_tomtom.columns.tolist()[:6]
        print(df_tomtom[cols].head(10).to_string(index=False))
        print("═" * 90 + "\n")


if __name__ == "__main__":
    main()
