#!/usr/bin/env bash
# ==============================================================================
# S. pneumoniae D39V Promoter & Sigma Factor Architecture Replication Pipeline
# Replicates Slager et al. (2018) methodology on 1,003 TSSs
# ==============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"

echo "========================================================================"
echo "   REPLICATING SLAGER ET AL. (2018) PROMOTER PIPELINE (D39V 1,003 TSSs)"
echo "========================================================================"

echo "[1/4] Calculating 500 bp Upstream Empirical Background Distribution..."
pixi run --manifest-path "${ROOT}/pixi.toml" python "${DIR}/steps/01_calculate_upstream_background.py"

echo "[2/4] Assembling Composite Bipartite MEME Matrices (SP15-19, Ext-10, ComX)..."
pixi run --manifest-path "${ROOT}/pixi.toml" python "${DIR}/steps/02_build_composite_motifs.py"

echo "[3/4] Extracting Upstream 40 bp & 20 bp Coordinate Windows..."
pixi run --manifest-path "${ROOT}/pixi.toml" python "${DIR}/steps/03_extract_tss_windows.py"

echo "[4/4] Executing Two-Tier Hierarchical FIMO Scan with Spatial Constraints..."
pixi run --manifest-path "${ROOT}/pixi.toml" python "${DIR}/steps/04_run_spatial_assignment.py"

echo "========================================================================"
echo "   REPLICATION COMPLETED SUCCESSFULLY"
echo "   Results saved to: pipeline/slager_promoter_assignment/reference_results/"
echo "========================================================================"
