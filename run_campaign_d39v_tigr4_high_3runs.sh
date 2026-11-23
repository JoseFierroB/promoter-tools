#!/bin/bash
set -eo pipefail

cd /home/fierro/promoter-tools

OUT_DIR="output/d39v_tigr4_high_3runs"
mkdir -p "${OUT_DIR}/logs"
mkdir -p "output/tables"

LOG_FILE="${OUT_DIR}/logs/execution.log"
METRICS_FILE="output/tables/resource_metrics_d39v_tigr4_high_3runs.tsv"

POS_FASTA="data/benchmark/canonical_15_datasets/4_d39v_tigr4_high/positives_81bp.fasta"
NEG_FASTA="data/benchmark/canonical_15_datasets/4_d39v_tigr4_high/negatives_81bp.fasta"

echo "================================================================" | tee -a "${LOG_FILE}"
echo "STARTING BENCHMARK CAMPAIGN (3 RUNS) ON WORKSTATION" | tee -a "${LOG_FILE}"
echo "Host: $(hostname)" | tee -a "${LOG_FILE}"
echo "Date: $(date)" | tee -a "${LOG_FILE}"
echo "Positives: ${POS_FASTA}" | tee -a "${LOG_FILE}"
echo "Negatives: ${NEG_FASTA}" | tee -a "${LOG_FILE}"
echo "Runs: 3" | tee -a "${LOG_FILE}"
echo "================================================================" | tee -a "${LOG_FILE}"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0

pixi run python src/cli.py run   mldspp   mldspp_75   ipromp_sp12   promotech_hot   lcnn   prokbert   prompt   fimo_prok   meme   --pos "${POS_FASTA}"   --neg "${NEG_FASTA}"   --runs 3   --output-dir "${OUT_DIR}/predictions"   -o "${METRICS_FILE}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?
echo "================================================================" | tee -a "${LOG_FILE}"
echo "CAMPAIGN FINISHED WITH EXIT CODE: ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "End Date: $(date)" | tee -a "${LOG_FILE}"
echo "Metrics saved at: ${METRICS_FILE}" | tee -a "${LOG_FILE}"
echo "================================================================" | tee -a "${LOG_FILE}"
