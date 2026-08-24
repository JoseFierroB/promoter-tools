#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# pipeline/run_pipeline.sh — end-to-end orchestrator (stages 2-6).
# Lives OUTSIDE the canonical src/ tree; nothing here is imported by the
# pipeline. Stages: datasets | splits | benchmark | analysis | all
#
# Usage:
#   ./pipeline/run_pipeline.sh datasets          # regenerate canonical datasets
#   ./pipeline/run_pipeline.sh datasets --gc     # + GC-matched negative sets
#   ./pipeline/run_pipeline.sh splits
#   ./pipeline/run_pipeline.sh benchmark         # 7 canonical tools, d39v
#   ./pipeline/run_pipeline.sh benchmark --threads 16 --runs 3
#   ./pipeline/run_pipeline.sh analysis
#   ./pipeline/run_pipeline.sh all
#
# Safety: dataset stages refuse to overwrite existing outputs unless --overwrite.
# Overrides (env): PYTHON, POS_FASTA, NEG_FASTA, DATA_DIR, THREADS, RUNS.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/.pixi/envs/default/bin/python}"
DATA_DIR="${DATA_DIR:-$ROOT/data}"
THREADS="${THREADS:-1}"
RUNS="${RUNS:-1}"
OVERWRITE=0
INCLUDE_GC=0

[ -x "$PY" ] || { echo "ERROR: python no encontrado en $PY (¿pixi install?)"; exit 1; }

stage_datasets() {
  local gff_tss="$DATA_DIR/reference/D39V_annotation_TSS_Victor.gff"
  local genome="$DATA_DIR/reference/D39V.fna"
  local gff_cds="$DATA_DIR/reference/D39V.gff3"
  local tigr_xlsx="$DATA_DIR/tigr4/S1_TSS.xlsx"
  local tigr_fa="$DATA_DIR/reference/NC_003028.fasta"

  ensure_new "$DATA_DIR/benchmark/d39v/positives_81bp.fasta" \
    "$PY src/dataset/positive_tss_d39v.py --gff $gff_tss --fasta $genome --gff-cds $gff_cds -o $DATA_DIR/benchmark/d39v/positives_81bp"
  ensure_new "$DATA_DIR/benchmark/d39v/negatives_81bp.fasta" \
    "$PY src/dataset/negatives_tss_d39v.py --gff-cds $gff_cds --fasta $genome --gff-tss $gff_tss --dedup-rc --limit 1000 -o $DATA_DIR/benchmark/d39v/negatives_81bp"
  ensure_new "$DATA_DIR/tigr4/positives_high_81bp.fasta" \
    "$PY src/dataset/positive_tss_tigr4.py --xlsx $tigr_xlsx --fasta $tigr_fa --tier high_conf_primary -o $DATA_DIR/tigr4/positives_high_81bp"
  ensure_new "$DATA_DIR/tigr4/negatives_high_81bp.fasta" \
    "$PY src/dataset/negatives_tss_tigr4.py --xlsx $tigr_xlsx --fasta $tigr_fa --tier high_conf_primary --limit 738 --dedup-rc -o $DATA_DIR/tigr4/negatives_high_81bp"
  if [ "$INCLUDE_GC" = "1" ]; then
    mkdir -p "$DATA_DIR/benchmark/d39v_gc" "$DATA_DIR/tigr4_gc"
    ensure_new "$DATA_DIR/benchmark/d39v_gc/negatives_81bp_gc30.fasta" \
      "$PY src/dataset/negatives_tss_d39v.py --gff-cds $gff_cds --fasta $genome --gff-tss $gff_tss --dedup-rc --limit 1000 --target-gc 30 --gc-tolerance 5 -o $DATA_DIR/benchmark/d39v_gc/negatives_81bp_gc30"
    ensure_new "$DATA_DIR/benchmark/d39v_gc/negatives_81bp_gc33.fasta" \
      "$PY src/dataset/negatives_tss_d39v.py --gff-cds $gff_cds --fasta $genome --gff-tss $gff_tss --dedup-rc --limit 1000 --target-gc 33 --gc-tolerance 5 -o $DATA_DIR/benchmark/d39v_gc/negatives_81bp_gc33"
    ensure_new "$DATA_DIR/tigr4_gc/negatives_high_81bp_gc31.fasta" \
      "$PY src/dataset/negatives_tss_tigr4.py --xlsx $tigr_xlsx --fasta $tigr_fa --tier high_conf_primary --limit 738 --dedup-rc --target-gc 31 --gc-tolerance 5 -o $DATA_DIR/tigr4_gc/negatives_high_81bp_gc31"
  fi
  echo "[OK] stage datasets"
}

stage_splits() {
  local n_sizes="988 1976 4940 9880 19760 29640 49400 98800 197600"
  for n in $n_sizes; do
    local f="$DATA_DIR/benchmark/mldspp_75_split_scale_db_$n.npz"
    [ -f "$f" ] && [ "$OVERWRITE" != "1" ] && continue
    "$PY" src/dataset/make_mldspp_75_splits.py --n-pos "$n"
  done
  echo "[OK] stage splits"
}

stage_benchmark() {
  local pos="${POS_FASTA:-$DATA_DIR/benchmark/d39v/positives_81bp.fasta}"
  local neg="${NEG_FASTA:-$DATA_DIR/benchmark/d39v/negatives_81bp.fasta}"
  local tools="${TOOLS:-meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12}"
  "$PY" src/cli.py run $tools \
    --pos "$pos" --neg "$neg" --threads "$THREADS" --runs "$RUNS"
  echo "[OK] stage benchmark ($tools, n=$(grep -c '>' "$pos" 2>/dev/null || echo ?) pos)"
}

stage_analysis() {
  "$PY" src/analysis/benchmark_statistics.py
  "$PY" src/analysis/benchmark_confusion.py
  "$PY" src/analysis/generate_master_roc.py
  "$PY" src/analysis/resource_plots.py
  "$PY" src/analysis/scaling_analysis.py --scale-db "$DATA_DIR/../scale_db_16cpu" || true
  echo "[OK] stage analysis"
}

ensure_new() {
  local out="$1"; shift
  if [ -f "$out" ] && [ "$OVERWRITE" != "1" ]; then
    echo "  skip (existe, usa --overwrite): $out"
    return
  fi
  echo "  >> $*"
  eval "$*"
}

usage() {
  sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed -n '3,14p'
  echo ""
  echo "Stages: datasets [--gc] | splits | benchmark | analysis | all"
  exit 0
}

[ $# -ge 1 ] || usage
STAGE="$1"; shift
case "$STAGE" in
  datasets)  while [ $# -gt 0 ]; do case "$1" in --gc) INCLUDE_GC=1;; --overwrite) OVERWRITE=1;; *) ;; esac; shift; done; stage_datasets ;;
  splits)    [ $# -gt 0 ] && OVERWRITE=1; stage_splits ;;
  benchmark) while [ $# -gt 0 ]; do case "$1" in --threads) THREADS="$2"; shift 2;; --runs) RUNS="$2"; shift 2;; *) shift;; esac; done; stage_benchmark ;;
  analysis)  stage_analysis ;;
  all)       stage_datasets; stage_splits; stage_benchmark; stage_analysis ;;
  *)         usage ;;
esac