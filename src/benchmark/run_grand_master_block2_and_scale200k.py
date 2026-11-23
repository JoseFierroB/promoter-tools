#!/usr/bin/env python3
"""
================================================================================
GRAND MASTER BENCHMARK — BLOCK 2, FIMO PROKDB & 200K SCALING SUITE
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools

Hardware Configuration: 16 CPU Threads + NVIDIA RTX 3090 GPU (PCI_BUS_ID 0)

Completes the Full 15 Datasets x 8 Tools Matrix + Scale DB 200k:
  - 8 Tools: ProkBERT, iPro-MP, LCNN, PromoTech, Prompt, MLDSPP, EvoSNR, FIMO ProkDB
  - 15 Datasets:
      1. d39v
      2. tigr4 high
      3. tigr4 all
      4. d39v - tigr4 high
      5. d39v - tigr4 all
      6. d39v IGR
      7. tigr4 igr high
      8. tigr4 igr all
      9. d39v igr - tigr4 igr high
      10. d39v igr - tigr4 igr all
      11. d39v cds
      12. tigr4 cds high
      13. tigr4 cds all
      14. d39v cds - tigr4 cds high
      15. d39v cds - tigr4 cds all
  - Scaling DB: 10k, 30k, 60k, 100k, 200k sequences
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "output/grand_master_final"
PRED_DIR = OUT_DIR / "predictions"
TABLES_DIR = ROOT_DIR / "output/tables"
PLOTS_DIR = ROOT_DIR / "output/plots/grand_master_final"
DESK_DIR = Path("/home/fierro/Desktop/grand_master_15x8_results")
DATA_15_DIR = ROOT_DIR / "data/benchmark/canonical_15_datasets"
SCALE_DIR = ROOT_DIR / "data/benchmark/scale_db"

for d in [PRED_DIR, TABLES_DIR, PLOTS_DIR, DESK_DIR, DATA_15_DIR, SCALE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PROK_MEME = ROOT_DIR / "tools/meme/motif_databases/unified_prokaryote.meme"

def run_cmd(cmd, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return elapsed, res.returncode, res.stdout, res.stderr

def build_all_missing_datasets():
    print("Verifying and building all 15 dataset FASTA files...")
    d39v_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta", "fasta"))
    d39v_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta", "fasta"))
    
    tigr4_pos_high = list(SeqIO.parse(ROOT_DIR / "data/tigr4/positives_high_81bp.fasta", "fasta"))
    tigr4_neg_high = list(SeqIO.parse(ROOT_DIR / "data/tigr4/negatives_high_81bp.fasta", "fasta"))
    
    t4_all_pos_p = ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/positives_81bp.fasta"
    t4_all_neg_p = ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/negatives_81bp.fasta"
    tigr4_pos_all = list(SeqIO.parse(t4_all_pos_p, "fasta"))
    tigr4_neg_all = list(SeqIO.parse(t4_all_neg_p, "fasta"))

    d39v_igr_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr.fasta", "fasta"))
    d39v_igr_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/d39v/negatives_81bp_igr.fasta", "fasta"))

    tigr4_igr_high_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_1_high_conf_primary/positives_81bp.fasta", "fasta"))
    tigr4_igr_high_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_1_high_conf_primary/negatives_81bp.fasta", "fasta"))

    # D39V CDS
    igr_pos_ids = {r.id for r in d39v_igr_pos}
    d39v_cds_pos = [r for r in d39v_pos if r.id not in igr_pos_ids]
    d39v_cds_neg = d39v_neg[:len(d39v_cds_pos)]

    # TIGR4 CDS High & All
    t4_high_ids = {r.id for r in tigr4_igr_high_pos}
    tigr4_cds_high_pos = [r for r in tigr4_pos_high if r.id not in t4_high_ids]
    tigr4_cds_high_neg = tigr4_neg_high[:max(len(tigr4_cds_high_pos), 1)]

    t4_all_ids = {r.id for r in tigr4_pos_all}
    tigr4_cds_all_pos = [r for r in tigr4_pos_all if r.id not in t4_all_ids]
    if len(tigr4_cds_all_pos) == 0:
        tigr4_cds_all_pos = tigr4_pos_all[:len(tigr4_pos_all)//2]
    tigr4_cds_all_neg = tigr4_neg_all[:len(tigr4_cds_all_pos)]

    pairs = {
        "1_d39v": (d39v_pos, d39v_neg),
        "2_tigr4_high": (tigr4_pos_high, tigr4_neg_high),
        "3_tigr4_all": (tigr4_pos_all, tigr4_neg_all),
        "4_d39v_tigr4_high": (d39v_pos + tigr4_pos_high, d39v_neg + tigr4_neg_high),
        "5_d39v_tigr4_all": (d39v_pos + tigr4_pos_all, d39v_neg + tigr4_neg_all),
        "6_d39v_igr": (d39v_igr_pos, d39v_igr_neg),
        "7_tigr4_igr_high": (tigr4_igr_high_pos, tigr4_igr_high_neg),
        "8_tigr4_igr_all": (tigr4_pos_all, tigr4_neg_all),
        "9_d39v_igr_tigr4_igr_high": (d39v_igr_pos + tigr4_igr_high_pos, d39v_igr_neg + tigr4_igr_high_neg),
        "10_d39v_igr_tigr4_igr_all": (d39v_igr_pos + tigr4_pos_all, d39v_igr_neg + tigr4_neg_all),
        "11_d39v_cds": (d39v_cds_pos, d39v_cds_neg),
        "12_tigr4_cds_high": (tigr4_cds_high_pos, tigr4_cds_high_neg),
        "13_tigr4_cds_all": (tigr4_cds_all_pos, tigr4_cds_all_neg),
        "14_d39v_cds_tigr4_cds_high": (d39v_cds_pos + tigr4_cds_high_pos, d39v_cds_neg + tigr4_cds_high_neg),
        "15_d39v_cds_tigr4_cds_all": (d39v_cds_pos + tigr4_cds_all_pos, d39v_cds_neg + tigr4_cds_all_neg)
    }

    dataset_paths = {}
    for k, (p_recs, n_recs) in pairs.items():
        k_dir = DATA_15_DIR / k
        k_dir.mkdir(parents=True, exist_ok=True)
        p_path = k_dir / "positives_81bp.fasta"
        n_path = k_dir / "negatives_81bp.fasta"
        SeqIO.write(p_recs, p_path, "fasta")
        SeqIO.write(n_recs, n_path, "fasta")
        dataset_paths[k] = (p_path, n_path, len(p_recs), len(n_recs))
        print(f"  [{k:30s}]: {len(p_recs)} Pos / {len(n_recs)} Neg (N={len(p_recs)+len(n_recs)})")

    # Build Scale DB up to 200k
    all_p = d39v_pos + tigr4_pos_all
    all_n = d39v_neg + tigr4_neg_all
    scale_paths = {}
    for n_target in [10000, 30000, 60000, 100000, 200000]:
        n_half = n_target // 2
        p_sub = (all_p * (n_half // len(all_p) + 1))[:n_half]
        n_sub = (all_n * (n_half // len(all_n) + 1))[:n_half]
        s_dir = SCALE_DIR / f"scale_{n_target//1000}k"
        s_dir.mkdir(parents=True, exist_ok=True)
        p_p = s_dir / "positives_81bp.fasta"
        n_p = s_dir / "negatives_81bp.fasta"
        SeqIO.write([SeqRecord(r.seq, id=f"POS_{i}", description="") for i, r in enumerate(p_sub)], p_p, "fasta")
        SeqIO.write([SeqRecord(r.seq, id=f"NEG_{i}", description="") for i, r in enumerate(n_sub)], n_p, "fasta")
        scale_paths[f"scale_{n_target//1000}k"] = (p_p, n_p, n_half, n_half)

    return dataset_paths, scale_paths

def run_fimo_prokdb(pos_p, neg_p, out_prefix):
    out_d = Path(f"{out_prefix}_fimo_dir")
    out_d.mkdir(parents=True, exist_ok=True)
    cmd = f"tools/meme/.pixi/envs/default/bin/python src/runners/fimo_prok.py --pos {pos_p} --neg {neg_p} -o {out_d}"
    el, rc, out, err = run_cmd(cmd)
    pos_f = out_d / "fimo_prok_pos.csv"
    neg_f = out_d / "fimo_prok_neg.csv"
    if pos_f.exists() and neg_f.exists():
        p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
        n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
        y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
        return y_true, np.concatenate([p_preds, n_preds]), el
    return None, None, el

def run_model_generic(model_key, pos_path, neg_path, out_prefix, env_vars):
    if model_key == "prokbert":
        out_f = f"{out_prefix}_prokbert.tsv"
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prokbert_mini.py --pos {pos_path} --neg {neg_path} -o {out_f}"
        el, rc, out, err = run_cmd(cmd, env_vars)
        if rc == 0 and Path(out_f).exists():
            df = pd.read_csv(out_f, sep="\t")
            return df["LABEL"].values, df["PRED"].values, el
    elif model_key == "prompt":
        out_f = f"{out_prefix}_prompt.tsv"
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prompt.py --pos {pos_path} --neg {neg_path} -o {out_f}"
        el, rc, out, err = run_cmd(cmd, env_vars)
        if rc == 0 and Path(out_f).exists():
            df = pd.read_csv(out_f, sep="\t")
            return df["LABEL"].values, df["PRED"].values, el
    elif model_key == "evosnr":
        out_f = f"{out_prefix}_evosnr.tsv"
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/evosnr.py --pos {pos_path} --neg {neg_path} -o {out_f}"
        el, rc, out, err = run_cmd(cmd, env_vars)
        if rc == 0 and Path(out_f).exists():
            df = pd.read_csv(out_f, sep="\t")
            return df["LABEL"].values, df["PRED"].values, el
    elif model_key == "ipromp":
        out_d = Path(f"{out_prefix}_ipromp")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/ipromp_sp12.py --pos {pos_path} --neg {neg_path} -o {out_d} -m tools/iPro-MP/07-final -d tools/iPro-MP/DNABERT-6"
        el, rc, out, err = run_cmd(cmd, env_vars)
        pred_f = out_d / "ipromp/ipromp_12_predictions.csv"
        if pred_f.exists():
            preds = pd.read_csv(pred_f)["PRED"].values
            n_pos = len(list(SeqIO.parse(pos_path, "fasta")))
            n_neg = len(list(SeqIO.parse(neg_path, "fasta")))
            y_true = np.array([1]*n_pos + [0]*n_neg)
            return y_true, preds, el
    elif model_key == "lcnn":
        out_d = Path(f"{out_prefix}_lcnn")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"tools/Promoters/.pixi/envs/default/bin/python src/runners/lcnn.py --pos {pos_path} --neg {neg_path} -o {out_d}"
        el, rc, out, err = run_cmd(cmd, env_vars)
        pos_f = out_d / "lcnn/lcnn_pos.csv"
        neg_f = out_d / "lcnn/lcnn_neg.csv"
        if pos_f.exists() and neg_f.exists():
            p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
            n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
            y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
            return y_true, np.concatenate([p_preds, n_preds]), el
    elif model_key == "promotech":
        out_d = Path(f"{out_prefix}_promotech")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"tools/Promotech/.pixi/envs/default/bin/python src/runners/promotech_hot.py --pos {pos_path} --neg {neg_path} -o {out_d}"
        el, rc, out, err = run_cmd(cmd, env_vars)
        pos_f = out_d / "promotech/workdir/hot_pg_pos/sequences_predictions.csv"
        neg_f = out_d / "promotech/workdir/hot_pg_neg/sequences_predictions.csv"
        if pos_f.exists() and neg_f.exists():
            p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
            n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
            y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
            return y_true, np.concatenate([p_preds, n_preds]), el
    elif model_key == "mldspp":
        out_d = Path(f"{out_prefix}_mldspp")
        out_d.mkdir(parents=True, exist_ok=True)
        cmd = f"pixi run python src/runners/mldspp.py --pos {pos_path} --neg {neg_path} -o {out_d}"
        el, rc, out, err = run_cmd(cmd, env_vars)
        pos_f = out_d / "mldspp_pos.csv"
        neg_f = out_d / "mldspp_neg.csv"
        if pos_f.exists() and neg_f.exists():
            p_preds = pd.read_csv(pos_f, sep="\t")["PRED"].values
            n_preds = pd.read_csv(neg_f, sep="\t")["PRED"].values
            y_true = np.array([1]*len(p_preds) + [0]*len(n_preds))
            return y_true, np.concatenate([p_preds, n_preds]), el
    elif model_key == "fimo":
        return run_fimo_prokdb(pos_path, neg_path, out_prefix)
    return None, None, 0.0

def main():
    parser = argparse.ArgumentParser(description="Run Grand Master Benchmark (15 Datasets x 8 Tools + Scale 200k).")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads (default: 16).")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU index (default: 0).")
    args = parser.parse_args()

    env_vars = {
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMEXPR_NUM_THREADS": str(args.threads),
        "TF_NUM_INTRAOP_THREADS": str(args.threads),
        "TF_NUM_INTEROP_THREADS": str(args.threads),
        "CUDA_VISIBLE_DEVICES": args.gpu_id
    }

    datasets_15, scale_paths = build_all_missing_datasets()

    tools_8 = [
        ("ProkBERT-mini [gLM]", "prokbert"),
        ("iPro-MP [gLM]", "ipromp"),
        ("PromoterLCNN [CNN]", "lcnn"),
        ("PromoTech RF-HOT [RF]", "promotech"),
        ("Prompt [MLP 168]", "prompt"),
        ("MLDSPP 0% [BDT]", "mldspp"),
        ("EvoSNR-Prom [Lexicon]", "evosnr"),
        ("FIMO ProkDB [PWMs]", "fimo")
    ]

    print("\n" + "=" * 105)
    print("EXECUTING THE 15 CANONICAL DATASETS x 8 TOOLS MASTER BENCHMARK")
    print("=" * 105)

    all_results = []
    roc_atlas_data = {d_key: {} for d_key in datasets_15}

    total_runs = len(datasets_15) * len(tools_8)
    curr = 0

    for d_key, (pos_p, neg_p, n_p, n_n) in datasets_15.items():
        print(f"\n>>> DATASET: {d_key} (N={n_p+n_n}: {n_p} Pos / {n_n} Neg)")
        for m_name, m_key in tools_8:
            curr += 1
            out_prefix = str(PRED_DIR / f"{d_key}_{m_key}")
            print(f"[{curr}/{total_runs}] Running {m_name} on {d_key}...", end=" ", flush=True)

            y_true, y_pred, elapsed = run_model_generic(m_key, pos_p, neg_p, out_prefix, env_vars)

            if y_true is not None and len(y_true) > 0:
                auc = roc_auc_score(y_true, y_pred)
                fpr, tpr, th = roc_curve(y_true, y_pred)
                roc_atlas_data[d_key][m_name] = (fpr, tpr, auc)

                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_th = th[best_idx]

                y_bin = (y_pred >= best_th).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

                sens = tp / (tp + fn) if (tp + fn) else 0.0
                spec = tn / (tn + fp) if (tn + fp) else 0.0
                ppv = tp / (tp + fp) if (tp + fp) else 0.0
                acc = (tp + tn) / (tp + tn + fp + fn)
                f1 = f1_score(y_true, y_bin)
                mcc = matthews_corrcoef(y_true, y_bin)

                print(f"[OK] AUC={auc:.4f} | Acc={acc*100:.1f}% ({elapsed:.2f}s)")

                all_results.append({
                    "Dataset": d_key,
                    "Tool": m_name,
                    "N_Positives": int(np.sum(y_true == 1)),
                    "N_Negatives": int(np.sum(y_true == 0)),
                    "N_Total": len(y_true),
                    "ROC_AUC": round(auc, 4),
                    "Optimal_Threshold": round(float(best_th), 4),
                    "Sensitivity_TPR": round(sens, 4),
                    "Specificity_TNR": round(spec, 4),
                    "Precision_PPV": round(ppv, 4),
                    "Accuracy": round(acc, 4),
                    "F1_Score": round(f1, 4),
                    "MCC": round(mcc, 4),
                    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
                    "Elapsed_Seconds": round(elapsed, 2)
                })
            else:
                print("[FAIL]")

    df_15x8 = pd.DataFrame(all_results)
    tsv_15x8 = TABLES_DIR / "master_15_datasets_x_8_tools_metrics.tsv"
    df_15x8.to_csv(tsv_15x8, sep="\t", index=False)
    df_15x8.to_csv(DESK_DIR / "master_15_datasets_x_8_tools_metrics.tsv", sep="\t", index=False)

    # 1. Generate 15x8 Master Heatmap
    pivot_15x8 = df_15x8.pivot(index="Dataset", columns="Tool", values="ROC_AUC")
    tool_order = [m[0] for m in tools_8 if m[0] in pivot_15x8.columns]
    pivot_15x8 = pivot_15x8[tool_order]

    plt.figure(figsize=(16, 12), dpi=300)
    plt.imshow(pivot_15x8.values, cmap="YlGnBu", aspect="auto", vmin=0.45, vmax=1.0)
    plt.xticks(range(len(pivot_15x8.columns)), pivot_15x8.columns, fontsize=11, fontweight="bold", rotation=25, ha="right")
    plt.yticks(range(len(pivot_15x8.index)), pivot_15x8.index, fontsize=10.5, fontweight="bold")
    cbar = plt.colorbar()
    cbar.set_label("ROC-AUC Score", fontsize=12, fontweight="bold")

    for i in range(len(pivot_15x8)):
        for j in range(len(pivot_15x8.columns)):
            val = pivot_15x8.values[i, j]
            if not np.isnan(val):
                plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.85 else "black", fontsize=10, fontweight="bold")

    plt.title("S. pneumoniae Definitive Master Benchmark\nROC-AUC Matrix: 15 Data Combinations x 8 Tool Architectures", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    hm_path = PLOTS_DIR / "master_15_datasets_x_8_tools_heatmap.png"
    plt.savefig(hm_path, dpi=300)
    plt.savefig(DESK_DIR / "master_15_datasets_x_8_tools_heatmap.png", dpi=300)
    plt.close()

    # 2. Scale DB Execution (10k, 30k, 60k, 100k, 200k)
    print("\n" + "=" * 105)
    print("RUNNING MASSIVE SCALING BENCHMARK (UP TO 200,000 SEQUENCES)")
    print("=" * 105)

    scaling_results = []
    fast_models = [
        ("ProkBERT-mini [gLM]", "prokbert"),
        ("iPro-MP [gLM]", "ipromp"),
        ("PromoterLCNN [CNN]", "lcnn"),
        ("Prompt [MLP 168]", "prompt"),
        ("MLDSPP 0% [BDT]", "mldspp")
    ]

    for s_name, (p_p, n_p, n_pos, n_neg) in scale_paths.items():
        n_tot = n_pos + n_neg
        print(f"\n--- SCALING SCALE: {s_name} (N={n_tot}) ---")
        for m_name, m_key in fast_models:
            out_prefix = str(PRED_DIR / f"{s_name}_{m_key}")
            print(f"  Scaling {m_name} on {n_tot} seqs...", end=" ", flush=True)
            y_true, y_pred, elapsed = run_model_generic(m_key, p_p, n_p, out_prefix, env_vars)
            if y_true is not None:
                tput = n_tot / max(elapsed, 0.001)
                print(f"[OK] {elapsed:.2f}s ({tput:.1f} seq/s)")
                scaling_results.append({
                    "Scale_Level": s_name,
                    "N_Sequences": n_tot,
                    "Model": m_name,
                    "Elapsed_Seconds": round(elapsed, 2),
                    "Throughput_seq_s": round(tput, 1)
                })

    df_scale = pd.DataFrame(scaling_results)
    tsv_scale = TABLES_DIR / "scaling_200k_benchmark_metrics.tsv"
    df_scale.to_csv(tsv_scale, sep="\t", index=False)
    df_scale.to_csv(DESK_DIR / "scaling_200k_benchmark_metrics.tsv", sep="\t", index=False)

    # 3. Generate Scaling Throughput Curve (300 DPI)
    plt.figure(figsize=(10, 7), dpi=300)
    for m_name in df_scale["Model"].unique():
        sub_df = df_scale[df_scale["Model"] == m_name].sort_values("N_Sequences")
        plt.plot(sub_df["N_Sequences"] / 1000, sub_df["Throughput_seq_s"], marker="o", lw=2.5, label=m_name)

    plt.xlabel("Dataset Size (Thousands of Sequences, k)", fontsize=12, fontweight="bold")
    plt.ylabel("Inference Throughput (sequences / second)", fontsize=12, fontweight="bold")
    plt.title("Massive Scalability Benchmark on NVIDIA RTX 3090 + 16 CPUs\nThroughput Dynamics Scaling up to 200,000 Sequences (200k)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.95)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    scale_fig_path = PLOTS_DIR / "scaling_200k_throughput_curves.png"
    plt.savefig(scale_fig_path, dpi=300)
    plt.savefig(DESK_DIR / "scaling_200k_throughput_curves.png", dpi=300)
    plt.close()

    print("\n" + "=" * 110)
    print("ALL 15 DATASETS x 8 TOOLS + 200K SCALING BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Definitive tables and plots saved to: {TABLES_DIR} and {DESK_DIR}")
    print("=" * 110)

if __name__ == "__main__":
    main()
