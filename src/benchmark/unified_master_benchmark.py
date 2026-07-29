#!/usr/bin/env python3
"""
Pipeline maestro de benchmark de promotores.
Evalúa todos los modelos sobre el dataset D39V de S. pneumoniae:
  1. MLDSPP (XGBoost, SVM, Random Forest) - 5-fold CV out-of-fold.
  2. PromoterLCNN - inferencia con modelo TF local.
  3. PromoTech (RF-HOT/RF-TETRA) - Modo Directo (corte 40bp) y Sliding Window (Gaussiana).
  4. iPro-MP (all 23 species) - parallel dynamic predictions.

Genera tabla comparativa (ROC AUC, PR AUC, MCC, Sensitivity, Specificity)
y gráfico ROC general.

Flow: submit_master_benchmark.sh → unified_master_benchmark.py → generate_master_plots.py
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No display backend, required for headless servers
import matplotlib.pyplot as plt
from pathlib import Path
from Bio import SeqIO
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef
import argparse
import concurrent.futures  # Parallel iPro-MP execution

def run_cmd(cmd, cwd):
    """Run a shell command in cwd; on failure print stdout/stderr and raise."""
    print(f"[RUNNING] {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        if res.stdout:
            print(f"[STDOUT] {res.stdout[-2000:]}")
        if res.stderr:
            print(f"[STDERR] {res.stderr[-2000:]}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res


def get_temp_dir():
    """Pick the best temp dir: nobackup on compute nodes, research as fallback."""
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
    """Argumentos del pipeline: paths a datasets, modelos, y config de CPUs."""
    parser = argparse.ArgumentParser(
        description="Unified Master Benchmark Pipeline for Promoter Prediction."
    )
    parser.add_argument('-p', '--positives', type=str,
        default="data/benchmark/positives_81bp.fasta",
        help="Path to positive 81bp FASTA.")
    parser.add_argument('-n', '--negatives', type=str,
        default="data/benchmark/negatives_81bp.fasta",
        help="Path to negative 81bp FASTA.")
    parser.add_argument('-o', '--output-dir', type=str,
        default="output",
        help="Output directory.")
    parser.add_argument('--promotech-dir', type=str, default="tools/Promotech",
        help="Path to PromoTech.")
    parser.add_argument('--mldspp-dir', type=str,
        default="tools/MLDSPP-Promoter-prediction", help="Path to MLDSPP.")
    parser.add_argument('--lcnn-dir', type=str, default="tools/Promoters",
        help="Path to PromoterLCNN.")
    parser.add_argument('--ipromp-dir', type=str, default="tools/iPro-MP",
        help="Path to iPro-MP.")
    
    # Detección dinámica del path de modelos iPro-MP
    # Prioriza paths absolutos del cluster, fallback a local.
    default_model_dir = "tools/iPro-MP/07-final"
    for path in ["/nfs/research/jlees/fierro/models/07-final",
                  "/hps/nobackup/jlees/fierro/models/07-final",
                  "/hps/software/users/jlees/fierro/promoter-tools/models",
                  "models", "iPro-MP/07-final"]:
        if os.path.exists(path):
            try:
                files = os.listdir(path)
                if any(f.endswith("_fold_1.pth") for f in files):
                    default_model_dir = str(Path(path).resolve())
                    print(f"[INFO] Selected model directory: {default_model_dir}")
                    break
            except Exception as e:
                print(f"[DEBUG] Model path '{path}' exists but check failed: {e}")
        else:
            print(f"[DEBUG] Model path '{path}' does not exist.")
                
    default_dnabert_dir = "tools/iPro-MP/DNABERT-6"
    for path in ["/nfs/research/jlees/fierro/DNABERT-6",
                  "/hps/nobackup/jlees/fierro/DNABERT-6",
                  "/hps/software/users/jlees/fierro/promoter-tools/DNABERT-6",
                  "DNABERT-6", "iPro-MP/DNABERT-6"]:
        if os.path.exists(path):
            has_file = (os.path.exists(os.path.join(path, "pytorch_model.bin")) or
                        os.path.exists(os.path.join(path, "model.safetensors")))
            if has_file:
                default_dnabert_dir = str(Path(path).resolve())
                print(f"[INFO] Selected DNABERT directory: {default_dnabert_dir}")
                break
                
    parser.add_argument('--ipromp-model-dir', type=str,
        default=default_model_dir,
        help="Path to iPro-MP species models folder (07-final).")
    parser.add_argument('--ipromp-dnabert-dir', type=str,
        default=default_dnabert_dir,
        help="Path to pre-trained DNABERT-6 directory.")
    parser.add_argument('--temp-dir', type=str, default=None,
        help="Directory for heavy temporary files (default: auto-detect).")
    
    # CPUs disponibles respetando límite de Slurm (cpus-per-task)
    try:
        default_threads = len(os.sched_getaffinity(0))
    except AttributeError:
        default_threads = os.cpu_count() or 8
    default_threads = min(default_threads, 23)  # max 23 iPro-MP species
    
    parser.add_argument('-t', '--threads', type=int,
        default=default_threads,
        help=f"Parallel workers (auto: {default_threads}).")
    return parser.parse_args()

def calculate_optimal_metrics(y_true, y_scores):
    """
    Compute ROC AUC, PR AUC, and metrics at the optimal Youden threshold.
    Youden = TPR - FPR; picks the threshold that maximises this difference.
    """
    from sklearn.metrics import confusion_matrix, precision_score, f1_score

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_arr, precision_arr)
    
    youden = tpr - fpr
    valid_n = len(thresholds)
    opt_idx = np.argmax(youden[:valid_n])
    opt_thresh = thresholds[opt_idx]
    y_pred = (y_scores >= opt_thresh).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Opt_Threshold": opt_thresh,
        "Sensitivity": tpr[opt_idx],
        "Specificity": 1 - fpr[opt_idx],
        "Precision": precision_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

def main():
    """Main pipeline: runs 4 models sequentially and compiles results."""
    args = parse_args()
    
    # ── Resolve absolute paths and load sequences ──
    root_dir = Path(os.getcwd()).resolve()
    resultados_dir = Path(args.output_dir).resolve()
    promotech_dir = Path(args.promotech_dir).resolve()
    mldspp_env_dir = Path(args.mldspp_dir).resolve()
    lcnn_env_dir = Path(args.lcnn_dir).resolve()
    ipromp_dir = Path(args.ipromp_dir).resolve()
    
    pos_fasta = Path(args.positives).resolve()
    neg_fasta = Path(args.negatives).resolve()
    
    pos_records = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_records = list(SeqIO.parse(neg_fasta, "fasta"))
    pos_ids = [r.id for r in pos_records]
    neg_ids = [r.id for r in neg_records]
    y_true = np.array([1] * len(pos_ids) + [0] * len(neg_ids))
    
    # ── Detect available CPUs ──
    try:
        allocated_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        allocated_cpus = os.cpu_count() or 8
    threads_per_proc = max(1, allocated_cpus // args.threads)
    
    print("="*74)
    print("               UNIFIED MASTER BENCHMARK")
    print("="*74)
    print(f"Positives: {len(pos_ids)} | Negatives: {len(neg_ids)} | "
          f"CPUs: {allocated_cpus} | Workers: {args.threads}")
    print("="*74 + "\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: MLDSPP (XGBoost, SVM, Random Forest) — 5-fold CV
    # ══════════════════════════════════════════════════════════════════════
    # Uses SantaLucia dinucleotide stability features + pixi env
    print("\n─── STEP 1: MLDSPP 5-fold CV ───")
    run_cmd(
        f"pixi run --manifest-path {mldspp_env_dir}/pixi.toml python src/benchmark/run_mldspp_cv_predictions.py "
        f"-p {pos_fasta} -n {neg_fasta} -o {resultados_dir}",
        root_dir
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: PromoterLCNN (CNN with pre-trained model)
    # ══════════════════════════════════════════════════════════════════════
    # Model: IsPromoter_fold_5 (TensorFlow)
    print("\n─── STEP 2: PromoterLCNN ───")
    run_cmd(
        f"pixi run --manifest-path {lcnn_env_dir}/pixi.toml python src/benchmark/predict_lcnn.py "
        f"-p {pos_fasta} -n {neg_fasta} -o {resultados_dir} -m {lcnn_env_dir}/weights/PromoterLCNN/IsPromoter_fold_5",
        root_dir
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: PromoTech — Direct (40bp corte) + Sliding Window (Gaussian)
    # ══════════════════════════════════════════════════════════════════════
    # evaluate_promotech_pipelines.py runs both modes and outputs CSVs
    print("\n─── STEP 3: PromoTech ───")
    if args.temp_dir:
        temp_dir = Path(args.temp_dir).resolve()
    else:
        temp_dir = get_temp_dir() / "temp_promotech_pipelines"
    temp_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        f"pixi run python src/analysis/evaluate_promotech_pipelines.py "
        f"-p {pos_fasta} -n {neg_fasta} -o {resultados_dir} --promotech-dir {promotech_dir} "
        f"--temp-dir {temp_dir} --no-cleanup",
        root_dir
    )
    sw_dir = temp_dir  # Reuse CSVs generated by evaluate_promotech_pipelines
 
    def load_pt_matrix(csv_path, ids, window_max=42):
        """Load per-sequence genome_predictions.csv into [n_seqs × window_max] matrix."""
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
 
    # Cargar matrices 988×42 de scores sliding window
    hot_pos_mat = load_pt_matrix(sw_dir / "hot_pg_pos/genome_predictions.csv", pos_ids)
    hot_neg_mat = load_pt_matrix(sw_dir / "hot_pg_neg/genome_predictions.csv", neg_ids)
    tetra_pos_mat = load_pt_matrix(sw_dir / "tetra_pg_pos/genome_predictions.csv", pos_ids)
    tetra_neg_mat = load_pt_matrix(sw_dir / "tetra_pg_neg/genome_predictions.csv", neg_ids)
 
    def aggregate_gaussian(matrix, sigma):
        """
        Aggregate 42 window scores into a single score per sequence.
        Weights: Gaussian centred at position 20 (centre of 81bp sequence).
        Takes the MAX weighted score (not the average).
        """
        offsets = np.arange(42)
        weights = np.exp(-((offsets - 20) ** 2) / (2 * (sigma ** 2)))
        return np.max(matrix * weights, axis=1)
 
    # Sigma: 5 for HOT (narrower, centre-focused), 10 for TETRA (wider)
    scores_hot_sw = np.concatenate([aggregate_gaussian(hot_pos_mat, 5), aggregate_gaussian(hot_neg_mat, 5)])
    scores_tetra_sw = np.concatenate([aggregate_gaussian(tetra_pos_mat, 10), aggregate_gaussian(tetra_neg_mat, 10)])
 
    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: iPro-MP (23 species, parallel)
    # ══════════════════════════════════════════════════════════════════════
    # Combines pos+neg into a single FASTA (one line per seq, no BioPython wrap)
    # and runs iPro-MP_predict.py for each species in parallel.
    print(f"\n─── STEP 4: iPro-MP (23 species, {args.threads} workers) ───")
    
    combined_fasta = resultados_dir / "combined_81bp_ipromp.fasta"
    with open(combined_fasta, "w") as f:
        for r in pos_records + neg_records:
            f.write(f">{r.id}\n{str(r.seq)}\n")  # No wrap (BioPython breaks lines at 60 cols)
        
    ipromp_out_dir = resultados_dir / "ipromp_out"
    ipromp_out_dir.mkdir(parents=True, exist_ok=True)
    
    def run_sp_prediction(sp_id):
        """Corre iPro-MP para una especie. Los threads se limitan con vars de entorno."""
        output_csv = ipromp_out_dir / f"ipromp_{sp_id}_predictions.csv"
        
        # Limit threads per PyTorch process to prevent CPU thrashing
        env = os.environ.copy()
        for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
                     "TORCH_NUM_THREADS"]:
            env[var] = str(threads_per_proc)
        
        cmd = [
            "pixi", "run", "-e", "ipro-mp", "python", "iPro-MP_predict.py",
            "-i", str(combined_fasta), "-s", str(sp_id), "-o", str(output_csv),
            "-m", str(Path(args.ipromp_model_dir).resolve()),
            "-d", str(Path(args.ipromp_dnabert_dir).resolve())
        ]
        
        print(f"    [iPro-MP] Species {sp_id}...", end=" ", flush=True)
        res = subprocess.run(cmd, cwd=str(ipromp_dir), env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print("FALLÓ")
            print(f"Stderr: {res.stderr[-1000:]}")
            raise RuntimeError(f"iPro-MP sp {sp_id} failed.")
        print("OK")
        return sp_id

    # Parallelise the 23 species with ThreadPoolExecutor
    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(run_sp_prediction, sp): sp for sp in range(1, 24)}
        for fut in concurrent.futures.as_completed(futures):
            sp = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] Species {sp} failed: {e}", file=sys.stderr)
                failed = True
                for f in futures:
                    f.cancel()
                
    if combined_fasta.exists():
        combined_fasta.unlink()
    if failed:
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Compile predictions from all models
    # ══════════════════════════════════════════════════════════════════════
    # Each model outputs scores ordered [pos_1..pos_N, neg_1..neg_M]
    print("\n─── STEP 5: Compiling predictions ───")
    
    models_dict = {}
    
    # ── MLDSPP: 3 variants ──
    for name, pos_f, neg_f in [
        ("MLDSPP (XGBoost)",        "mldspp_pos.csv", "mldspp_neg.csv"),
        ("MLDSPP (SVM)",            "mldspp_svm_pos.csv", "mldspp_svm_neg.csv"),
        ("MLDSPP (Random Forest)",  "mldspp_rf_pos.csv", "mldspp_rf_neg.csv")
    ]:
        pos_df = pd.read_csv(resultados_dir / pos_f, sep='\t').set_index("CHROM")
        neg_df = pd.read_csv(resultados_dir / neg_f, sep='\t').set_index("CHROM")
        models_dict[name] = np.concatenate([
            pos_df.loc[pos_ids, "PRED"].values,
            neg_df.loc[neg_ids, "PRED"].values
        ])
        
    # ── PromoterLCNN ──
    lcnn_pos = pd.read_csv(resultados_dir / "lcnn_pos.csv", sep='\t').set_index("CHROM")
    lcnn_neg = pd.read_csv(resultados_dir / "lcnn_neg.csv", sep='\t').set_index("CHROM")
    models_dict["PromoterLCNN"] = np.concatenate([
        lcnn_pos.loc[pos_ids, "PRED"].values,
        lcnn_neg.loc[neg_ids, "PRED"].values
    ])
    
    # ── PromoTech Direct (40bp corte) ──
    pt_hot_s_pos = pd.read_csv(sw_dir / "hot_s_pos/sequences_predictions.csv", sep='\t').set_index("CHROM")
    pt_hot_s_neg = pd.read_csv(sw_dir / "hot_s_neg/sequences_predictions.csv", sep='\t').set_index("CHROM")
    models_dict["PromoTech RF-HOT (Direct 40bp)"] = np.concatenate([
        pt_hot_s_pos.loc[pos_ids, "PRED"].values,
        pt_hot_s_neg.loc[neg_ids, "PRED"].values
    ])
    
    pt_tetra_s_pos = pd.read_csv(sw_dir / "tetra_s_pos/sequences_predictions.csv", sep='\t').set_index("CHROM")
    pt_tetra_s_neg = pd.read_csv(sw_dir / "tetra_s_neg/sequences_predictions.csv", sep='\t').set_index("CHROM")
    models_dict["PromoTech RF-TETRA (Direct 40bp)"] = np.concatenate([
        pt_tetra_s_pos.loc[pos_ids, "PRED"].values,
        pt_tetra_s_neg.loc[neg_ids, "PRED"].values
    ])
    
    # ── PromoTech Sliding Window (Gaussian) ──
    models_dict["PromoTech RF-HOT (SW Gaussian)"] = scores_hot_sw
    models_dict["PromoTech RF-TETRA (SW Gaussian)"] = scores_tetra_sw
 
    # ── iPro-MP (23 species) ──
    print("  -> Compiling iPro-MP...")
    for sp_id in range(1, 24):
        csv_path = ipromp_out_dir / f"ipromp_{sp_id}_predictions.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            models_dict[f"iPro-MP (sp {sp_id})"] = df["Probability"].values

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Compute metrics for all models
    # ══════════════════════════════════════════════════════════════════════
    print("\n─── STEP 6: Computing metrics ───")
    results = []
    for name, y_scores in models_dict.items():
        m = calculate_optimal_metrics(y_true, y_scores)
        results.append({
            "Model": name,
            "ROC_AUC": round(m["ROC_AUC"], 4),
            "PR_AUC": round(m["PR_AUC"], 4),
            "Opt_Threshold": round(m["Opt_Threshold"], 4),
            "Sensitivity": round(m["Sensitivity"], 4),
            "Specificity": round(m["Specificity"], 4),
            "Precision": round(m["Precision"], 4),
            "F1": round(m["F1"], 4),
            "MCC": round(m["MCC"], 4),
            "TP": int(m["TP"]), "TN": int(m["TN"]),
            "FP": int(m["FP"]), "FN": int(m["FN"]),
        })
        
    df_results = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
    
    out_tsv = resultados_dir / "unified_master_benchmark_summary.tsv"
    df_results.to_csv(out_tsv, sep='\t', index=False)
    print(f"Report saved: {out_tsv}")

    # ── Bootstrap CIs and DeLong tests ──
    try:
        from src.analysis.statistics import bootstrap_auc_ci, delong_test
        print("\n─── STEP 6b: Bootstrap CIs & DeLong pairwise tests ───")
        stats_rows = []
        for name, y_scores in models_dict.items():
            ci = bootstrap_auc_ci(y_true, y_scores, n_bootstrap=2000)
            stats_rows.append({
                "Model": name,
                "AUC": ci["auc"],
                "CI_lower": ci["ci_lower"],
                "CI_upper": ci["ci_upper"],
            })
            print(f"  {name:<40} AUC={ci['auc']:.4f}  95% CI=[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

        names = list(models_dict.keys())
        print(f"\n  DeLong pairwise (p-values):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = delong_test(y_true, models_dict[names[i]], models_dict[names[j]])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if sig:
                    print(f"    {names[i]:<35} vs {names[j]:<35} p={p:.4f} {sig}")

        df_stats = pd.DataFrame(stats_rows).sort_values(by="AUC", ascending=False)
        stats_out = resultados_dir / "benchmark_statistics.tsv"
        df_stats.to_csv(stats_out, sep='\t', index=False)
        print(f"  Statistics saved: {stats_out}")
    except ImportError:
        print("  [WARNING] statistics module not available, skipping CIs")
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Full statistics table (Precision, F1, Confusion Matrix)
    # ══════════════════════════════════════════════════════════════════════
    print("\n─── STEP 7: Full statistics table ───")
    full_cols = ["Model", "ROC_AUC", "PR_AUC", "Sensitivity", "Specificity",
                 "Precision", "F1", "MCC", "TP", "TN", "FP", "FN"]
    df_full = df_results[full_cols].sort_values(by="ROC_AUC", ascending=False)
    full_out = resultados_dir / "full_statistics.tsv"
    df_full.to_csv(full_out, sep='\t', index=False)
    print(f"  Full statistics saved: {full_out}")
    print(f"\n  Top 5 by F1:")
    for _, row in df_full.head(5).iterrows():
        print(f"    {row['Model']:<35} F1={row['F1']:.4f}  MCC={row['MCC']:.4f}  AUC={row['ROC_AUC']:.4f}")
    
    print("\n" + "="*80)
    print("           CONSOLIDATED REPORT (TOP 10)")
    print("="*80)
    print(df_results.head(10).to_string(index=False))
    print("="*80 + "\n")
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Overall ROC plot
    # ══════════════════════════════════════════════════════════════════════
    # Top 6 models in colour, remainder in light grey
    print("─── STEP 7: Generating ROC plot ───")
    plt.figure(figsize=(6.5, 5.0), dpi=300)
    
    top_n = 6
    top_colors = ["#002D62", "#D95319", "#7E2F8E", "#00A087", "#3C5488", "#1f77b4"]
    top_n_names = df_results["Model"].head(top_n).tolist()
    color_map = dict(zip(top_n_names, top_colors))
    
    plt.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=0.8)
    
    for name, y_scores in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        if name in top_n_names:
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})",
                     color=color_map[name], lw=1.3)
        else:
            plt.plot(fpr, tpr, color="#DDDDDD", lw=0.4, alpha=0.4)
            
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC General - Benchmark de Promotores")
    plt.legend(frameon=False, loc="lower right", fontsize=6.5)
    
    plot_p = resultados_dir / "unified_master_benchmark_roc.png"
    plt.savefig(plot_p, bbox_inches="tight")
    print(f"ROC plot: {plot_p}")
    plt.close()

if __name__ == "__main__":
    main()
