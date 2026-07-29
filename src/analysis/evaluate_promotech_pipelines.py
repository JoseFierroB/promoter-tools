#!/usr/bin/env python3
"""
Evaluate and Compare both PromoTech Pipelines (Direct 40bp vs. Sliding Window 81bp).
Designed for Streptococcus pneumoniae D39V promoter benchmarking.

Pipeline 1: Direct Mode (Corte 40pb)
  - Extracts the exact center 40bp of the 81bp sequence (bases 20 to 60).
  - Predicts directly using PromoTech's sequence prediction mode (-s).
  - Fast, memory-efficient, and easy to scale.

Pipeline 2: Sliding Window Mode (Parse Genome)
  - Slides a 40bp window across the 81bp sequence (42 windows per strand).
  - Predicts using PromoTech's genome parsing and prediction mode (-pg / -g).
  - Aggregates window scores using our optimal Gaussian Weighted Max (sigma=5).
  - Highly robust to coordinate shifts and spacer variations.

Author: Antigravity Code Assistant / José (TFM)
Date: July 15, 2026
"""

import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def get_temp_dir():
    """Pick the best temp directory: nobackup on compute nodes, research as fallback."""
    env_tmp = os.environ.get("TMPDIR", "")
    if env_tmp and os.path.isdir(os.path.dirname(env_tmp)):
        return Path(env_tmp)
    candidates = [
        "/hps/nobackup/jlees/fierro/tmp",
        "/nfs/research/jlees/fierro/tmp",
    ]
    for p in candidates:
        parent = os.path.dirname(p)
        if os.path.isdir(parent) and os.access(parent, os.W_OK):
            return Path(p)
    return Path("/nfs/research/jlees/fierro/tmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and compare both PromoTech prediction pipelines (Direct vs. Sliding Window)."
    )
    parser.add_argument(
        "-p", "--positives",
        type=str,
        default="data/benchmark/positives_81bp.fasta",
        help="Path to positive 81bp FASTA."
    )
    parser.add_argument(
        "-n", "--negatives",
        type=str,
        default="data/benchmark/negatives_81bp.fasta",
        help="Path to negative 81bp FASTA."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="/nfs/research/jlees/fierro/resultados",
        help="Output directory for reports and summaries."
    )
    parser.add_argument(
        "--promotech-dir",
        type=str,
        default="tools/Promotech",
        help="Path to PromoTech installation directory."
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not clean up temporary prediction files."
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Directory for heavy temporary files (default: auto-detect nobackup or research)."
    )
    return parser.parse_args()

def run_cmd(cmd, cwd):
    """Executes a shell command and raises an error if it fails."""
    print(f"[RUNNING] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with returncode={res.returncode}: {cmd}", flush=True)
        if res.returncode < 0:
            import signal
            sig = -res.returncode
            sig_name = signal.Signals(sig).name if hasattr(signal, 'Signals') else f"signal {sig}"
            print(f"[SIGNAL] Process was killed by {sig_name} ({sig})", flush=True)
        if res.stdout:
            print(f"[STDOUT] {res.stdout[-2000:]}", flush=True)
        if res.stderr:
            print(f"[STDERR] {res.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"Command failed (rc={res.returncode}): {cmd}\nStdout: {res.stdout[-1000:]}\nStderr: {res.stderr[-1000:]}")
    return res

def main():
    args = parse_args()
    
    # Resolve absolute paths
    root_dir = Path(os.getcwd()).resolve()
    promotech_dir = Path(args.promotech_dir).resolve()
    pos_fasta = Path(args.positives).resolve()
    neg_fasta = Path(args.negatives).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Create temporary directory – prefer nobackup on compute nodes
    if args.temp_dir:
        temp_dir = Path(args.temp_dir).resolve()
    else:
        temp_dir = get_temp_dir() / "temp_promotech_pipelines"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify file existence
    if not pos_fasta.exists():
        raise FileNotFoundError(f"Positives FASTA not found at {pos_fasta}")
    if not neg_fasta.exists():
        raise FileNotFoundError(f"Negatives FASTA not found at {neg_fasta}")
    if not (promotech_dir / "promotech.py").exists():
        raise FileNotFoundError(f"PromoTech entrypoint not found at {promotech_dir}/promotech.py")
        
    pos_records = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_records = list(SeqIO.parse(neg_fasta, "fasta"))
    pos_ids = [r.id for r in pos_records]
    neg_ids = [r.id for r in neg_records]
    num_pos = len(pos_records)
    num_neg = len(neg_records)
    
    print("==========================================================================")
    print("      EVALUATING BOTH PROMOTECH PIPELINES (DIRECT VS. SW SCANNING)")
    print("==========================================================================")
    print(f"Positives (81bp):  {pos_fasta.name} ({num_pos} sequences)")
    print(f"Negatives (81bp):  {neg_fasta.name} ({num_neg} sequences)")
    print(f"PromoTech Path:    {promotech_dir}")
    print("==========================================================================\n")
    
    # ==========================================================================
    # PIPELINE 1: Direct Mode (Corte 40pb)
    # ==========================================================================
    print("--- [PIPELINE 1] Processing Direct 40bp Cut Mode ---")
    pos_40_temp = temp_dir / "pos_40bp.fasta"
    neg_40_temp = temp_dir / "neg_40bp.fasta"
    
    # Slice the center 40bp [20:60] of each 81bp sequence
    print("  -> Slicing center [20:60] windows...")
    for records_81, out_path in [(pos_records, pos_40_temp), (neg_records, neg_40_temp)]:
        sliced_records = []
        for r in records_81:
            r_copy = r[:]
            r_copy.seq = r.seq[20:60]
            sliced_records.append(r_copy)
        SeqIO.write(sliced_records, out_path, "fasta")
        
    # Run sequence prediction mode (-s) in PromoTech
    print("  -> Predicting on sliced sequences using RF-HOT and RF-TETRA...")
    hot_s_pos = temp_dir / "hot_s_pos"
    hot_s_neg = temp_dir / "hot_s_neg"
    tetra_s_pos = temp_dir / "tetra_s_pos"
    tetra_s_neg = temp_dir / "tetra_s_neg"
    
    # RF-HOT
    run_cmd(f"pixi run python promotech.py -s -m RF-HOT -f {pos_40_temp} -o {hot_s_pos}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -s -m RF-HOT -f {neg_40_temp} -o {hot_s_neg}", promotech_dir)
    # RF-TETRA
    run_cmd(f"pixi run python promotech.py -s -m RF-TETRA -f {pos_40_temp} -o {tetra_s_pos}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -s -m RF-TETRA -f {neg_40_temp} -o {tetra_s_neg}", promotech_dir)

    # ==========================================================================
    # PIPELINE 2: Sliding Window Mode (Parse Genome)
    # ==========================================================================
    print("\n--- [PIPELINE 2] Processing Sliding Window Scanning (81bp) ---")
    hot_pg_pos = temp_dir / "hot_pg_pos"
    hot_pg_neg = temp_dir / "hot_pg_neg"
    tetra_pg_pos = temp_dir / "tetra_pg_pos"
    tetra_pg_neg = temp_dir / "tetra_pg_neg"
    
    # Parse and predict genomes on 81bp (step=1 sliding window)
    # Outputs all scores (-t 0.0) for complete mathematical aggregation
    print("  -> Scanning 81bp positive and negative collections...")
    # RF-HOT
    run_cmd(f"pixi run python promotech.py -pg -m RF-HOT -f {pos_fasta} -o {hot_pg_pos}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -g -m RF-HOT -t 0.0 -i {hot_pg_pos} -o {hot_pg_pos}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -pg -m RF-HOT -f {neg_fasta} -o {hot_pg_neg}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -g -m RF-HOT -t 0.0 -i {hot_pg_neg} -o {hot_pg_neg}", promotech_dir)
    # RF-TETRA
    run_cmd(f"pixi run python promotech.py -pg -m RF-TETRA -f {pos_fasta} -o {tetra_pg_pos}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -g -m RF-TETRA -t 0.0 -i {tetra_pg_pos} -o {tetra_pg_pos}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -pg -m RF-TETRA -f {neg_fasta} -o {tetra_pg_neg}", promotech_dir)
    run_cmd(f"pixi run python promotech.py -g -m RF-TETRA -t 0.0 -i {tetra_pg_neg} -o {tetra_pg_neg}", promotech_dir)

    # ==========================================================================
    # METRICS EVALUATION & ANALYSIS
    # ==========================================================================
    print("\n--- Parsing and Compiling Metrics ---")
    
    # Ground Truth labels
    y_true = np.array([1] * num_pos + [0] * num_neg)
    
    # 1. Load Pipeline 1 (Direct Mode) scores
    def get_direct_scores(pos_csv, neg_csv):
        df_pos = pd.read_csv(pos_csv, sep='\t')
        df_neg = pd.read_csv(neg_csv, sep='\t')
        return np.concatenate([df_pos['PRED'].values, df_neg['PRED'].values])
        
    scores_hot_direct = get_direct_scores(hot_s_pos / "sequences_predictions.csv", hot_s_neg / "sequences_predictions.csv")
    scores_tetra_direct = get_direct_scores(tetra_s_pos / "sequences_predictions.csv", tetra_s_neg / "sequences_predictions.csv")
    
    # 2. Load Pipeline 2 (Sliding Window) score matrices of shape (num_seqs, 42)
    # The modified PG (process_genome.py) emits per-sequence chrom IDs and
    # per-sequence start offsets (0-41), NOT concatenated genome offsets.
    def load_pt_matrix(csv_path, ids, window_max=42):
        id_to_idx = {seq_id: idx for idx, seq_id in enumerate(ids)}
        matrix = np.zeros((len(ids), window_max))
        if not os.path.exists(csv_path):
            return matrix
        df = pd.read_csv(csv_path, sep='\t')
        for _, row in df.iterrows():
            chrom = str(row['chrom'])
            if str(row['strand']) != '+':
                continue
            base_id = chrom.rsplit('_offset_', 1)[0]
            try:
                offset = int(chrom.rsplit('_offset_', 1)[1])
            except (IndexError, ValueError):
                continue
            if base_id in id_to_idx and offset < window_max:
                matrix[id_to_idx[base_id], offset] = float(row['score'])
        return matrix

    hot_pos_matrix = load_pt_matrix(hot_pg_pos / "genome_predictions.csv", pos_ids)
    hot_neg_matrix = load_pt_matrix(hot_pg_neg / "genome_predictions.csv", neg_ids)
    
    tetra_pos_matrix = load_pt_matrix(tetra_pg_pos / "genome_predictions.csv", pos_ids)
    tetra_neg_matrix = load_pt_matrix(tetra_pg_neg / "genome_predictions.csv", neg_ids)

    # 3. Apply Optimal Gaussian Weighted Max Aggregation (sigma = 5)
    def apply_gaussian_aggregation(matrix, sigma=5):
        offsets = np.arange(42)
        # Gaussian weights centered at index 20 (TSS promoter)
        weights = np.exp(-((offsets - 20) ** 2) / (2 * (sigma ** 2)))
        return np.max(matrix * weights, axis=1)

    scores_hot_sw = np.concatenate([
        apply_gaussian_aggregation(hot_pos_matrix),
        apply_gaussian_aggregation(hot_neg_matrix)
    ])
    scores_tetra_sw = np.concatenate([
        apply_gaussian_aggregation(tetra_pos_matrix),
        apply_gaussian_aggregation(tetra_neg_matrix)
    ])
    
    # 4. Calculate final ROC and PR metrics
    metrics = []
    for model_name, direct_scores, sw_scores in [
        ("RF-HOT", scores_hot_direct, scores_hot_sw),
        ("RF-TETRA", scores_tetra_direct, scores_tetra_sw)
    ]:
        # Direct Mode
        auc_dir = roc_auc_score(y_true, direct_scores)
        prec_dir, rec_dir, _ = precision_recall_curve(y_true, direct_scores)
        pr_dir = auc(rec_dir, prec_dir)
        
        # SW Mode
        auc_sw = roc_auc_score(y_true, sw_scores)
        prec_sw, rec_sw, _ = precision_recall_curve(y_true, sw_scores)
        pr_sw = auc(rec_sw, prec_sw)
        
        metrics.append({
            "Model": model_name,
            "Pipeline 1 (Direct 40bp) ROC AUC": round(auc_dir, 4),
            "Pipeline 1 (Direct 40bp) PR AUC": round(pr_dir, 4),
            "Pipeline 2 (SW Gaussian) ROC AUC": round(auc_sw, 4),
            "Pipeline 2 (SW Gaussian) PR AUC": round(pr_sw, 4)
        })
        
    # Output Table
    metrics_df = pd.DataFrame(metrics)
    print("\n==========================================================================")
    print("                    FINAL PIPELINES COMPARISON SUMMARY")
    print("==========================================================================")
    print(metrics_df.to_string(index=False))
    print("==========================================================================")
    
    # Save report
    report_path = output_dir / "promotech_pipelines_comparison_summary.tsv"
    metrics_df.to_csv(report_path, sep='\t', index=False)
    print(f"\n[SUCCESS] Final comparison report written to: {report_path}")
    
    # Clean up temporary predictions
    if not args.no_cleanup:
        print("\nCleaning up temporary workspace directory...")
        run_cmd(f"rm -rf {temp_dir}", root_dir)
        print("[CLEANUP SUCCESSFUL]")
    else:
        print("\nSkipping cleanup of temporary files as requested.")

if __name__ == "__main__":
    main()
