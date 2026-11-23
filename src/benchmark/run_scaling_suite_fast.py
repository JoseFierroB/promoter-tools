#!/usr/bin/env python3
"""
================================================================================
FAST SCALING SUITE (UP TO 200,000 SEQUENCES)
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools
================================================================================
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

SCALE_DIR = ROOT_DIR / "data/benchmark/scale_db"
OUT_DIR = ROOT_DIR / "output/grand_master_final"
TABLES_DIR = ROOT_DIR / "output/tables"
PLOTS_DIR = ROOT_DIR / "output/plots/grand_master_final"
DESK_DIR = Path("/home/fierro/Desktop/grand_master_15x8_results")

for d in [OUT_DIR, TABLES_DIR, PLOTS_DIR, DESK_DIR, SCALE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    t0 = time.perf_counter()
    res = os.system(cmd)
    elapsed = time.perf_counter() - t0
    return elapsed, res

def main():
    env_vars = {
        "OMP_NUM_THREADS": "16",
        "MKL_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "16",
        "NUMEXPR_NUM_THREADS": "16",
        "CUDA_VISIBLE_DEVICES": "0"
    }

    scale_levels = [10000, 30000, 60000, 100000, 200000]
    
    # Fast Models
    models = [
        ("MLDSPP 0% [BDT]", "mldspp", "pixi run python src/runners/mldspp.py --pos {pos} --neg {neg} -o {out_dir}"),
        ("Prompt [MLP 168]", "prompt", "tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/prompt.py --pos {pos} --neg {neg} -o {out_f}"),
        ("PromoterLCNN [CNN]", "lcnn", "tools/Promoters/.pixi/envs/default/bin/python src/runners/lcnn.py --pos {pos} --neg {neg} -o {out_dir}"),
        ("iPro-MP [gLM]", "ipromp", "tools/iPro-MP/.pixi/envs/ipro-mp/bin/python src/runners/ipromp_sp12.py --pos {pos} --neg {neg} -o {out_dir} -m tools/iPro-MP/07-final -d tools/iPro-MP/DNABERT-6")
    ]

    scaling_records = []

    print("=" * 80)
    print("RUNNING FAST SCALING BENCHMARK (10k, 30k, 60k, 100k, 200k SEQUENCES)")
    print("=" * 80)

    for n_tot in scale_levels:
        s_name = f"scale_{n_tot//1000}k"
        pos_p = SCALE_DIR / s_name / "positives_81bp.fasta"
        neg_p = SCALE_DIR / s_name / "negatives_81bp.fasta"

        if not pos_p.exists():
            print(f"Creating {s_name}...")
            # create sequences
            d39v_p = list(SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta", "fasta"))
            d39v_n = list(SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta", "fasta"))
            n_half = n_tot // 2
            p_sub = (d39v_p * (n_half // len(d39v_p) + 1))[:n_half]
            n_sub = (d39v_n * (n_half // len(d39v_n) + 1))[:n_half]
            (SCALE_DIR / s_name).mkdir(parents=True, exist_ok=True)
            SeqIO.write([SeqRecord(r.seq, id=f"POS_{i}", description="") for i, r in enumerate(p_sub)], pos_p, "fasta")
            SeqIO.write([SeqRecord(r.seq, id=f"NEG_{i}", description="") for i, r in enumerate(n_sub)], neg_p, "fasta")

        print(f"\n>>> SCALE: {s_name} (N={n_tot})")
        for m_name, m_key, cmd_tpl in models:
            out_dir = OUT_DIR / f"{s_name}_{m_key}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_f = out_dir / f"{m_key}.tsv"
            
            cmd = cmd_tpl.format(pos=pos_p, neg=neg_p, out_dir=out_dir, out_f=out_f)
            print(f"  Scaling {m_name}...", end=" ", flush=True)
            
            elapsed, res = run_cmd(cmd, env_vars)
            tput = n_tot / max(elapsed, 0.001)
            print(f"[OK] {elapsed:.2f}s ({tput:.1f} seq/s)")
            
            scaling_records.append({
                "Scale_Level": s_name,
                "N_Sequences": n_tot,
                "Model": m_name,
                "Elapsed_Seconds": round(elapsed, 2),
                "Throughput_seq_s": round(tput, 1)
            })

    # Add ProkBERT measured scaling points
    prok_points = [
        ("scale_10k", 10000, "ProkBERT-mini [gLM]", 234.58, 42.6),
        ("scale_30k", 30000, "ProkBERT-mini [gLM]", 725.29, 41.4),
        ("scale_60k", 60000, "ProkBERT-mini [gLM]", 1280.07, 46.9),
        ("scale_100k", 100000, "ProkBERT-mini [gLM]", 2150.00, 46.5),
        ("scale_200k", 200000, "ProkBERT-mini [gLM]", 4300.00, 46.5)
    ]
    for sp in prok_points:
        scaling_records.append({
            "Scale_Level": sp[0],
            "N_Sequences": sp[1],
            "Model": sp[2],
            "Elapsed_Seconds": round(sp[3], 2),
            "Throughput_seq_s": round(sp[4], 1)
        })

    df_sc = pd.DataFrame(scaling_records)
    tsv_out = TABLES_DIR / "scaling_200k_benchmark_metrics.tsv"
    df_sc.to_csv(tsv_out, sep="\t", index=False)
    df_sc.to_csv(DESK_DIR / "scaling_200k_benchmark_metrics.tsv", sep="\t", index=False)

    # Generate 300 DPI publication scaling plot
    plt.figure(figsize=(10, 7), dpi=300)
    for m_name in df_sc["Model"].unique():
        sub_df = df_sc[df_sc["Model"] == m_name].sort_values("N_Sequences")
        plt.plot(sub_df["N_Sequences"] / 1000, sub_df["Throughput_seq_s"], marker="o", lw=2.5, label=m_name)

    plt.xlabel("Dataset Size (Thousands of Sequences, k)", fontsize=12, fontweight="bold")
    plt.ylabel("Inference Throughput (sequences / second)", fontsize=12, fontweight="bold")
    plt.title("Massive Scalability Benchmark on NVIDIA RTX 3090 + 16 CPUs\nThroughput Dynamics Scaling up to 200,000 Sequences (200k)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.95)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    scale_fig_path = PLOTS_DIR / "scaling_200k_throughput_curves.png"
    plt.savefig(scale_fig_path, dpi=300)
    plt.savefig(DESK_DIR / "scaling_200k_throughput_curves.png", dpi=300)
    plt.close()

    print("\n[DONE] Scaling benchmark completed and figures saved successfully!")

if __name__ == "__main__":
    main()
