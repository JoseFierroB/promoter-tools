#!/usr/bin/env python3
"""
PromoTech PG runner — sliding window genome mode with max aggregation.

Runs PromoTech -pg + -g on positives and negatives separately.
Produces genome_predictions.csv (PromoTech native) +
sequences_predictions.csv (PRED column, pipeline-compatible).

Usage:
    pixi run python src/benchmark/run_promotech.py -m RF-HOT \
        -p data/benchmark/positives_81bp.fasta \
        -n data/benchmark/negatives_81bp.fasta \
        -o output/predictions/promotech/workdir
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path


def run_cmd(cmd, cwd, timeout=None):
    print(f"  [RUN] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd,
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print(f"  [ERROR] rc={res.returncode}", flush=True)
        if res.stdout:
            print(res.stdout[-1500:])
        if res.stderr:
            print(res.stderr[-1500:])
        raise RuntimeError(f"Command failed: {cmd}")
    return res


def pg_predict(model, fasta, out_dir, promotech_dir):
    """Run -pg + -g on a single FASTA, then convert output to PRED format."""
    print(f"\n  === PG mode: {model} on {fasta.name} ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        f"pixi run python promotech.py -pg -m {model} -f {fasta} -o {out_dir}",
        promotech_dir, timeout=300)
    run_cmd(
        f"pixi run python promotech.py -g -m {model} -t 0.0 -i {out_dir} -o {out_dir}",
        promotech_dir, timeout=600)

    genome_csv = out_dir / "genome_predictions.csv"
    if not genome_csv.exists():
        raise FileNotFoundError(f"Expected output not found: {genome_csv}")

    df = pd.read_csv(genome_csv, sep="\t")
    seq_csv = out_dir / "sequences_predictions.csv"
    df_seq = pd.DataFrame({"PRED": df["score"].values})
    df_seq.to_csv(seq_csv, sep="\t", index=False)

    n_seqs = len(df_seq)
    print(f"    -> {n_seqs} sequences predicted")
    return n_seqs


def parse_args():
    p = argparse.ArgumentParser(description="PromoTech PG runner for benchmark data.")
    p.add_argument("-m", "--model", required=True, choices=["RF-HOT", "RF-TETRA"],
                   help="PromoTech model")
    p.add_argument("-p", "--positives", default="data/benchmark/positives_81bp.fasta",
                   help="Positive 81bp FASTA")
    p.add_argument("-n", "--negatives", default="data/benchmark/negatives_81bp.fasta",
                   help="Negative 81bp FASTA")
    p.add_argument("-o", "--output-dir", default="output/predictions/promotech/workdir",
                   help="Output directory")
    p.add_argument("--promotech-dir", default="tools/Promotech",
                   help="PromoTech directory")
    return p.parse_args()


def main():
    args = parse_args()
    promotech_dir = Path(args.promotech_dir).resolve()
    out_base = Path(args.output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    tag = "hot" if args.model == "RF-HOT" else "tetra"

    for label, fasta_path in [("pos", args.positives), ("neg", args.negatives)]:
        fasta = Path(fasta_path).resolve()
        if not fasta.exists():
            raise FileNotFoundError(f"FASTA not found: {fasta}")
        pg_predict(args.model, fasta, out_base / f"{tag}_pg_{label}", promotech_dir)

    print(f"\nDone. Output: {out_base}")


if __name__ == "__main__":
    main()
