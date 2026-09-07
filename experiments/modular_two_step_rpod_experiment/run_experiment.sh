#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Running Step 1: Build Modular Motifs ==="
pixi run python 01_build_modular_meme.py

echo "=== Running Step 2: Extract TSS Windows ==="
pixi run python 02_extract_tss_windows.py

echo "=== Running Step 3: Modular Two-Step Scan ==="
pixi run python 03_run_modular_two_step_scan.py

echo "=== Verifying Pipeline Integrity Tests ==="
cd ../..
pixi run python negatives_tss_test.py --run-tests
