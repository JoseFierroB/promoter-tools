#!/usr/bin/env bash
# ==============================================================================
# S. pneumoniae TIGR4 Promoter & Sigma Factor Assignment Runner
# Runs the strain-agnostic assignment tool on TIGR4 High-Confidence dataset (N=738)
# ==============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DIR}/../.." && pwd)"

TIGR4_FASTA="${ROOT}/data/tigr4/positives_high_81bp.fasta"
OUT_DIR="${DIR}/reference_results"

echo "========================================================================"
echo "   RUNNING PROMOTER & SIGMA ASSIGNMENT ON TIGR4 (High-Confidence N=738)"
echo "========================================================================"

pixi run --manifest-path "${ROOT}/pixi.toml" python "${DIR}/assign_strepto_sigma_promoters.py" \
    --input-fasta "${TIGR4_FASTA}" \
    --output-dir "${OUT_DIR}" \
    --output-prefix "tigr4_738_slager_assignments" \
    --tss-pos 60

echo "========================================================================"
echo "   TIGR4 ASSIGNMENT COMPLETED SUCCESSFULLY"
echo "========================================================================"
