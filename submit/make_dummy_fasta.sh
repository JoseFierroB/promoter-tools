#!/bin/bash
set -euo pipefail

SRC="${1:-data/benchmark/d39v/positives_81bp.fasta}"
N="${2:-1}"
OUT="${3:-data/benchmark/dummy_pos_$(basename "$SRC" .fasta)x${N}.fasta}"

awk -v n="$N" '/^>/{h=$0; getline s; for(i=1;i<=n;i++){print h"_rep"i; print s}}' "$SRC" > "$OUT"

total=$(grep -c '^>' "$OUT")
src_n=$(grep -c '^>' "$SRC")
echo "OK: $total seqs ($src_n x $N) -> $OUT"