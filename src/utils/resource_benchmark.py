#!/usr/bin/env python3
"""
Resource Usage Benchmarking Script (Single-Threaded Resource Monitor).
----------------------------------------------------------------------
This script executes promoter prediction models (PromoTech RF-HOT/RF-TETRA or tools/iPro-MP species)
under a strict 1-thread constraint, monitors their physical (RSS) and virtual peak memory usage 
in real-time via /proc, logs execution time, and saves a resource consumption report.

It runs independently of unified_benchmark.py to avoid breaking the core pipeline.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("resource_benchmark")

# Default Configurations
DEFAULT_POS_81 = "data/benchmark/positives_81bp.fasta"
DEFAULT_NEG_81 = "data/benchmark/negatives_81bp.fasta"
DEFAULT_PROMOTECH_DIR = "tools/Promotech"
DEFAULT_IPROMP_DIR = "tools/iPro-MP"
DEFAULT_OUTDIR = "benchmark_outputs"

DEFAULT_IPROMP_MODEL_DIR = os.environ.get("IPROMP_MODEL_DIR", "/nfs/research/jlees/fierro/models/07-final")

default_dnabert = "DNABERT-6"
local_dnabert = "tools/iPro-MP/DNABERT-6"
if not (os.path.exists(os.path.join(local_dnabert, "pytorch_model.bin")) or os.path.exists(os.path.join(local_dnabert, "model.safetensors"))):
    for path in ["/nfs/research/jlees/fierro/DNABERT-6", "/hps/nobackup/jlees/fierro/DNABERT-6"]:
        if os.path.exists(os.path.join(path, "pytorch_model.bin")) or os.path.exists(os.path.join(path, "model.safetensors")):
            default_dnabert = path
            break
DEFAULT_IPROMP_DNABERT_DIR = os.environ.get("IPROMP_DNABERT_DIR", default_dnabert)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resource Usage Benchmarking Script (Single-Threaded).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="The model to benchmark (forces 1-thread execution, e.g. RF-HOT, ipromp-13).")
    return parser.parse_args()

def get_proc_memory_stats(pid: int) -> dict:
    """Reads /proc/{pid}/status to extract VmHWM (peak RSS) and VmPeak (peak Virtual Memory)."""
    stats = {"peak_rss_kb": 0, "peak_virt_kb": 0}
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    stats["peak_rss_kb"] = int(line.split()[1])
                elif line.startswith("VmPeak:"):
                    stats["peak_virt_kb"] = int(line.split()[1])
    except FileNotFoundError:
        # Process has finished
        pass
    return stats

def main():
    args = parse_arguments()
    
    root_dir = Path(__file__).resolve().parent
    out_dir = Path(DEFAULT_OUTDIR).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pos_81 = Path(DEFAULT_POS_81).resolve()
    neg_81 = Path(DEFAULT_NEG_81).resolve()
    
    # 1. Prepare input files if not present
    if args.model.startswith("RF-"):
        # PromoTech needs 40bp combined FASTA (optimal slice seq[20:60])
        combined_40 = out_dir / "combined_input_40bp.fasta"
        if not combined_40.exists():
            logger.info("Generating combined 40bp input sequence for PromoTech...")
            # Inline slice and combine
            with open(combined_40, "w") as fout:
                for path in [pos_81, neg_81]:
                    with open(path, "r") as fin:
                        header = None
                        seq_lines = []
                        for line in fin:
                            line = line.strip()
                            if line.startswith(">"):
                                if header:
                                    seq = "".join(seq_lines)
                                    if len(seq) == 81:
                                        fout.write(f"{header}\n{seq[20:60]}\n")
                                header = line
                                seq_lines = []
                            else:
                                seq_lines.append(line)
                        if header:
                            seq = "".join(seq_lines)
                            if len(seq) == 81:
                                fout.write(f"{header}\n{seq[20:60]}\n")
                                
        cwd = Path(DEFAULT_PROMOTECH_DIR).resolve()
        out_subdir = out_dir / f"resource_promotech_{args.model.lower()}"
        cmd = [
            "pixi", "run", "python", "promotech.py",
            "-s", "-m", args.model, "-f", str(combined_40), "-o", str(out_subdir)
        ]
    else:
        # tools/iPro-MP needs 81bp combined FASTA
        combined_81 = out_dir / "combined_input_81bp.fasta"
        if not combined_81.exists():
            logger.info("Generating combined 81bp input sequence for tools/iPro-MP...")
            with open(combined_81, "w") as fout:
                for path in [pos_81, neg_81]:
                    with open(path, "r") as fin:
                        for line in fin:
                            if line.strip():
                                fout.write(line)
                                if not line.endswith("\n"):
                                    fout.write("\n")
                                    
        cwd = Path(DEFAULT_IPROMP_DIR).resolve()
        sp = args.model.split("-")[1]
        out_csv = out_dir / f"resource_ipromp_{sp}_predictions.csv"
        cmd = [
            "pixi", "run", "-e", "ipro-mp", "python", "tools/iPro-MP_predict.py",
            "-i", str(combined_81), "-s", str(sp), "-o", str(out_csv),
            "-m", DEFAULT_IPROMP_MODEL_DIR, "-d", DEFAULT_IPROMP_DNABERT_DIR
        ]

    logger.info(f"Command to execute: {' '.join(cmd)}")
    logger.info(f"CWD: {cwd}")
    
    # 2. Configure Environment variables to strictly force 1 thread
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TF_NUM_INTEROP_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    
    # Cache redirection
    env["PIXI_CACHE_DIR"] = str(root_dir / ".pixi-cache" / "pixi")
    env["UV_CACHE_DIR"] = str(root_dir / ".pixi-cache" / "uv")
    
    logger.info("Launching subprocess under 1-thread limit constraints...")
    
    # 3. Start Subprocess and Monitor
    start_time = time.time()
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    peak_rss = 0
    peak_virt = 0
    
    # Monitor loop
    try:
        while p.poll() is None:
            stats = get_proc_memory_stats(p.pid)
            if stats["peak_rss_kb"] > peak_rss:
                peak_rss = stats["peak_rss_kb"]
            if stats["peak_virt_kb"] > peak_virt:
                peak_virt = stats["peak_virt_kb"]
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.warning("Monitoring interrupted by user. Terminating process...")
        p.terminate()
        sys.exit(1)
        
    elapsed_time = time.time() - start_time
    stdout, stderr = p.communicate()
    
    if p.returncode != 0:
        logger.error(f"Execution failed with exit code {p.returncode}")
        if stderr.strip():
            logger.error(f"Subprocess Stderr:\n{stderr}")
        sys.exit(1)
        
    # Convert KB to MB
    peak_rss_mb = peak_rss / 1024.0
    peak_virt_mb = peak_virt / 1024.0
    
    logger.info("=====================================================")
    logger.info(f"RESOURCE BENCHMARK RESULTS FOR: {args.model}")
    logger.info("=====================================================")
    logger.info(f"Execution Time : {elapsed_time:.2f} seconds")
    logger.info(f"Peak RSS Memory: {peak_rss_mb:.2f} MB")
    logger.info(f"Peak VIRT Mem  : {peak_virt_mb:.2f} MB")
    logger.info("=====================================================")
    
    # 4. Save results to a model-specific TSV to avoid parallel write conflicts
    report_file = out_dir / f"resource_usage_{args.model.lower()}.tsv"
    with open(report_file, "w") as rf:
        rf.write("Model\tExecution_Time_Seconds\tPeak_RSS_MB\tPeak_VIRT_MB\n")
        rf.write(f"{args.model}\t{elapsed_time:.2f}\t{peak_rss_mb:.2f}\t{peak_virt_mb:.2f}\n")
        
    logger.info(f"Resource statistics successfully saved to: {report_file}")

if __name__ == "__main__":
    main()
