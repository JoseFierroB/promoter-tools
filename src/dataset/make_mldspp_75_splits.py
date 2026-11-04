#!/usr/bin/env python3
"""Generate MLDSPP 75% S. pneumoniae splits (seed=42, ratio=0.75).

Replicates the protocol of src/runners/mldspp_75.py exactly so any
n_positives size has a pre-built split for LocalRunner:

    idx = np.random.RandomState(42).permutation(n_pos)
    train_idx, test_idx = idx[:int(n*0.75)], idx[int(n*0.75):]

Usage:
    pixi run python src/dataset/make_mldspp_75_splits.py --n-pos 1976 4940 9880
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "data" / "benchmark"
SEED = 42
RATIO = 0.75


def make_split(n_pos: int, out_dir: Path, name: str = None) -> Path:
    idx = np.random.RandomState(SEED).permutation(n_pos)
    n_train = int(n_pos * RATIO)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    out = out_dir / (name or f"mldspp_75_split_{n_pos}.npz")
    np.savez(out, train_idx=train_idx, test_idx=test_idx, seed=np.array(SEED))
    return out


def main():
    p = argparse.ArgumentParser(description="Generate mldspp_75 splits for scale_db sizes")
    p.add_argument("--n-pos", type=int, nargs="+", required=True,
                   help="Positive counts to build splits for")
    p.add_argument("--name", type=str, default=None,
                   help="Exact output filename (single --n-pos only); "
                        "default: mldspp_75_split_<n>.npz")
    args = p.parse_args()
    if args.name and len(set(args.n_pos)) != 1:
        p.error("--name requires exactly one --n-pos value")

    for n_pos in sorted(set(args.n_pos)):
        if n_pos <= 0 or int(n_pos * RATIO) * 4 != n_pos * 3:
            print(f"WARNING: {n_pos} not divisible by 4 — protocol assumes n*0.75 exact")
        out = make_split(n_pos, OUT_DIR, args.name)
        print(f"  {out.name}: train={int(n_pos*RATIO)} test={n_pos-int(n_pos*RATIO)}")


if __name__ == "__main__":
    main()