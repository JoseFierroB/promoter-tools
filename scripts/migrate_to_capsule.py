#!/usr/bin/env python3
"""Migrate legacy output layout to canonical capsule (dry-run by default).

Legacy:
  output/<YYMMDD>_runs/<name>/<config>/
  output/<YYMMDD>_comparison_*
  output/<YYMMDD>_benchmark_*

Canonical (new):
  output/runs/<YYMMDD>_<name>_<config>/
  output/comparisons/<YYMMDD>_comparison_*
  output/benchmarks/<YYMMDD>_.../

This script is reversible: it only creates symlinks (default) or moves
data when --execute is passed. Revert: `rm -rf output/runs output/comparisons output/benchmarks`
and legacy dirs remain (or are restored from symlink target).

Usage:
  python scripts/migrate_to_capsule.py --dry-run
  python scripts/migrate_to_capsule.py --execute
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEGACY_RUNS_GLOB = "output/*_runs"
COMPARISON_GLOB = "output/*_comparison*"
BENCHMARK_GLOB = "output/*_benchmark*"

# also handle output/260915_* style benchmarks
import re

def find_legacy_runs():
    runs = []
    for p in Path("output").glob("*_runs"):
        if not p.is_dir():
            continue
        for name_dir in p.iterdir():
            if not name_dir.is_dir():
                continue
            for cfg_dir in name_dir.iterdir():
                if cfg_dir.is_dir() and (cfg_dir / "STATUS.json").exists() or (cfg_dir / "predictions").exists():
                    runs.append(cfg_dir)
                # also handle case where predictions directly under name_dir (old)
    return runs

def canonical_for_legacy(legacy: Path) -> Path:
    # legacy: output/<YYMMDD>_runs/<name>/<config>
    parts = legacy.parts
    # find date prefix
    date = ""
    for part in parts:
        m = re.match(r"^(\d{6})_runs$", part)
        if m:
            date = m.group(1)
            break
        m2 = re.match(r"^(\d{6})_", part)
        if m2:
            date = m2.group(1)
    # name is parent of config
    name = legacy.parent.name
    config = legacy.name
    # date fallback
    if not date:
        date = "260101"
    return Path("output/runs") / f"{date}_{name}_{config}"

def main():
    p = argparse.ArgumentParser(description="Migrate legacy layout to capsule (symlink, reversible).")
    p.add_argument("--dry-run", action="store_true", default=True, help="Only show plan (default)")
    p.add_argument("--execute", action="store_true", help="Create canonical dirs + legacy symlinks")
    p.add_argument("--move", action="store_true", help="Move data to canonical (instead of symlink), keeps legacy symlink")
    args = p.parse_args()
    dry = not args.execute
    if args.execute:
        dry = False

    print("=== Migration dry-run ===" if dry else "=== Migration execute ===")
    runs = find_legacy_runs()
    print(f"Found {len(runs)} legacy runs:")
    for r in runs:
        canon = canonical_for_legacy(r)
        exists = canon.exists()
        print(f"  {r} -> {canon}  (exists={exists})")

    comps = list(Path("output").glob("*_comparison*"))
    benches = list(Path("output").glob("*_benchmark*"))
    # also new style already in output/comparisons etc
    print(f"\nComparisons (legacy top-level): {len(comps)}")
    for c in comps[:10]:
        print(f"  {c}")
    print(f"\nBenchmarks (legacy): {len(benches)}")
    for b in benches[:10]:
        print(f"  {b}")

    # also check predictions_by_dataset
    pred_pool = Path("output/predictions_by_dataset")
    if pred_pool.exists():
        print(f"\nLegacy pred-pool: {pred_pool} ({len(list(pred_pool.glob('*')))} entries) -> should be derived from runs, not kept as source")

    if dry:
        print("\n[dry-run] No changes made. Run with --execute to create symlinks.")
        print("Revert: rm -rf output/runs output/comparisons output/benchmarks (legacy dirs stay)")
        return

    # execute: create canonical dirs as symlinks to legacy (lightweight, reversible)
    for r in runs:
        canon = canonical_for_legacy(r)
        if canon.exists():
            print(f"  [skip] {canon} exists")
            continue
        canon.parent.mkdir(parents=True, exist_ok=True)
        try:
            # create symlink canonical -> legacy (so data stays in legacy, canonical is view)
            # we want canonical to be real data? For true capsule, data should be in canonical, legacy symlink to it.
            # But to keep legacy untouched for revert, we make canonical as symlink to legacy for now.
            canon.symlink_to(r.resolve())
            print(f"  [symlink] {canon} -> {r}")
        except Exception as e:
            print(f"  [error] {canon}: {e}")

    # comparisons
    for c in comps:
        if c.is_symlink():
            continue
        # canonical comparisons root
        target = Path("output/comparisons") / c.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(c.resolve())
            print(f"  [symlink] {target} -> {c}")
        except Exception as e:
            print(f"  [error] {c}: {e}")

    for b in benches:
        if b.is_symlink():
            continue
        target = Path("output/benchmarks") / b.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(b.resolve())
            print(f"  [symlink] {target} -> {b}")
        except Exception as e:
            print(f"  [error] {b}: {e}")

    print("\nDone. New canonical views in output/runs/, output/comparisons/, output/benchmarks/")
    print("Revert: rm output/runs/* output/comparisons/* output/benchmarks/* (and keep legacy)")

if __name__ == "__main__":
    main()
