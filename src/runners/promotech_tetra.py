#!/usr/bin/env python3
"""PromoTech RF-TETRA runner — PG mode (sliding window + max aggregation)."""
import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pandas as pd

PROMOTECH_DIR = Path(__file__).resolve().parent.parent.parent / "tools/Promotech"


def get_promotech_python():
    from src.config import config
    return str(config.get_env_python(PROMOTECH_DIR / "pixi.toml"))


def main():
    p = argparse.ArgumentParser(description="PromoTech RF-TETRA PG prediction")
    p.add_argument("--pos", required=True, help="Positive test FASTA")
    p.add_argument("--neg", required=True, help="Negative test FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    args = p.parse_args()

    pos_count = sum(1 for l in open(args.pos) if l.startswith(">"))
    neg_count = sum(1 for l in open(args.neg) if l.startswith(">"))

    tmpdir = Path(tempfile.mkdtemp(prefix="pt_tetra_"))
    pt_python = get_promotech_python()
    pt_bin = str(PROMOTECH_DIR / ".pixi" / "envs" / "default" / "bin")
    run_env = {**os.environ, "PATH": f"{pt_bin}:{os.environ['PATH']}"}

    t0 = time.perf_counter()
    for label, fasta in [("pos", args.pos), ("neg", args.neg)]:
        od = tmpdir / f"tetra_pg_{label}"
        od.mkdir(parents=True, exist_ok=True)
        fasta_abs = str(Path(fasta).resolve())
        subprocess.run(
            [pt_python, "promotech.py", "-pg", "-m", "RF-TETRA", "-f", fasta_abs, "-o", str(od)],
            cwd=str(PROMOTECH_DIR), env=run_env,
            capture_output=True, text=True, timeout=300, check=True)
        subprocess.run(
            [pt_python, "promotech.py", "-g", "-m", "RF-TETRA",
             "-t", "0.0", "-i", str(od), "-o", str(od)],
            cwd=str(PROMOTECH_DIR), env=run_env,
            capture_output=True, text=True, timeout=600, check=True)

        df = pd.read_csv(od / "genome_predictions.csv", sep="\t")
        pd.DataFrame({"PRED": df["score"].values}).to_csv(
            od / "sequences_predictions.csv", sep="\t", index=False)

        out_dest = Path(args.output) / "promotech/workdir" / od.name
        out_dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(od, out_dest, dirs_exist_ok=True)

    elapsed = time.perf_counter() - t0
    shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"PromoTech RF-TETRA: {pos_count + neg_count} seqs in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
