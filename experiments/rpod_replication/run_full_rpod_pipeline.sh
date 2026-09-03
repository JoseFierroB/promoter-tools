#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== EXECUTING REPLICATION PIPELINE FOR RPOD / COMX REGULATORY MOTIFS ==="

echo "--> Step 1: Background Nucleotide Distribution"
python "${DIR}/01_calculate_upstream_background.py"

echo "--> Step 2: Assemble Composite Bipartite MEME Motifs"
python "${DIR}/02_build_shimada_composite_meme.py"

echo "--> Step 3: Extract TSS Upstream Windows"
python "${DIR}/03_extract_tss_windows.py"

echo "--> Step 4: Run FIMO Scanning with Spatial Constraints"
python "${DIR}/04_run_fimo_spatial_scan.py"

echo "=== PIPELINE COMPLETED SUCCESSFULLY ==="
