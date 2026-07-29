#!/usr/bin/env python3
"""
Run PromoTech predictions on benchmark datasets.
Generates Direct (40bp) and Sliding Window (PG) predictions
for RF-HOT and RF-TETRA models.

Usage:
  pixi run python run_promotech.py -p data/benchmark/positives_81bp.fasta \
    -n data/benchmark/negatives_81bp.fasta -o output/predictions/promotech/workdir
"""

import os, sys, argparse, subprocess, time
from pathlib import Path
from Bio import SeqIO


def run_cmd(cmd, cwd, timeout=None):
    print(f"  [RUN] {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd,
                         capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        print(f"  [ERROR] rc={res.returncode}", flush=True)
        if res.stdout: print(res.stdout[-1500:])
        if res.stderr: print(res.stderr[-1500:])
        raise RuntimeError(f"Command failed: {cmd}")
    return res


def parse_args():
    p = argparse.ArgumentParser(description="Run PromoTech predictions on benchmark data.")
    p.add_argument('-p', '--positives', default="data/benchmark/positives_81bp.fasta",
                   help="Positive 81bp FASTA")
    p.add_argument('-n', '--negatives', default="data/benchmark/negatives_81bp.fasta",
                   help="Negative 81bp FASTA")
    p.add_argument('-o', '--output-dir', default="output/predictions/promotech/workdir",
                   help="Working directory for predictions")
    p.add_argument('--promotech-dir', default="tools/Promotech",
                   help="Path to PromoTech directory")
    p.add_argument('--temp-dir', default=None,
                   help="Temp directory (default: auto)")
    p.add_argument('--skip-direct', action='store_true',
                   help="Skip direct mode (-s)")
    p.add_argument('--skip-pg', action='store_true',
                   help="Skip sliding window mode (-pg/-g)")
    p.add_argument('--no-cleanup', action='store_true',
                   help="Keep temp files")
    return p.parse_args()


def main():
    args = parse_args()
    root_dir = Path(os.getcwd()).resolve()
    promotech_dir = Path(args.promotech_dir).resolve()
    pos_fasta = Path(args.positives).resolve()
    neg_fasta = Path(args.negatives).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_records = list(SeqIO.parse(pos_fasta, "fasta"))
    neg_records = list(SeqIO.parse(neg_fasta, "fasta"))
    print(f"Positives: {len(pos_records)} | Negatives: {len(neg_records)}")

    # Temp dir
    if args.temp_dir:
        temp_dir = Path(args.temp_dir).resolve()
    else:
        temp_dir = out_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    # MODE 1: Direct (40bp center)
    # ════════════════════════════════════════════════════════════════
    if not args.skip_direct:
        print("\n=== Direct 40bp mode ===")
        pos_40 = temp_dir / "pos_40bp.fasta"
        neg_40 = temp_dir / "neg_40bp.fasta"
        for records_81, out_path in [(pos_records, pos_40), (neg_records, neg_40)]:
            sliced = []
            for r in records_81:
                rc = r[:]; rc.seq = r.seq[20:60]; sliced.append(rc)
            SeqIO.write(sliced, out_path, "fasta")

        for model in ["RF-HOT", "RF-TETRA"]:
            tag = "hot" if model == "RF-HOT" else "tetra"
            for label, fasta in [("pos", pos_40), ("neg", neg_40)]:
                od = out_dir / f"{tag}_s_{label}"
                od.mkdir(parents=True, exist_ok=True)
                run_cmd(
                    f"pixi run python promotech.py -s -m {model} -f {fasta} -o {od}",
                    promotech_dir, timeout=600)
        print("  Direct mode done.")

    # ════════════════════════════════════════════════════════════════
    # MODE 2: Sliding Window (-pg + -g)
    # ════════════════════════════════════════════════════════════════
    if not args.skip_pg:
        print("\n=== Sliding Window mode ===")
        for model in ["RF-HOT", "RF-TETRA"]:
            tag = "hot" if model == "RF-HOT" else "tetra"
            for label, fasta in [("pos", pos_fasta), ("neg", neg_fasta)]:
                od = out_dir / f"{tag}_pg_{label}"
                od.mkdir(parents=True, exist_ok=True)
                run_cmd(
                    f"pixi run python promotech.py -pg -m {model} -f {fasta} -o {od}",
                    promotech_dir, timeout=300)
                run_cmd(
                    f"pixi run python promotech.py -g -m {model} -t 0.0 -i {od} -o {od}",
                    promotech_dir, timeout=600)
        print("  PG mode done.")

    # Cleanup
    if not args.no_cleanup and temp_dir.exists() and temp_dir != out_dir:
        run_cmd(f"rm -rf {temp_dir}", root_dir)

    print(f"\nAll predictions saved to: {out_dir}")


if __name__ == "__main__":
    main()
