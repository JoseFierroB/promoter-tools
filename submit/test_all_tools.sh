#!/bin/bash
#SBATCH --job-name=test_all_tools
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=production
#SBATCH --output=slurm-%j-test-all.out
#SBATCH --error=slurm-%j-test-all.err

set -e

export PIXI_HOME="/hps/software/users/jlees/fierro/pixi/global"
export PATH="$PIXI_HOME/bin:$PATH"
# Cache handled by ~/.config/pixi/config.toml [cache] root

cd /hps/software/users/jlees/fierro/promoter-tools
OUT="/nfs/research/jlees/fierro/resultados"
mkdir -p "$OUT"

echo "============================================"
echo " TEST 1/5: MLDSPP (XGBoost + SVM + RF)"
echo "============================================"
pixi run --manifest-path tools/MLDSPP-Promoter-prediction/pixi.toml \
  python src/benchmark/run_mldspp_cv_predictions.py \
  -p data/benchmark/positives_81bp.fasta \
  -n data/benchmark/negatives_81bp.fasta \
  -o "$OUT"
echo "=> MLDSPP OK"

echo ""
echo "============================================"
echo " TEST 2/5: PromoterLCNN"
echo "============================================"
pixi run --manifest-path tools/Promoters/pixi.toml \
  python src/benchmark/predict_lcnn.py \
  -p data/benchmark/positives_81bp.fasta \
  -n data/benchmark/negatives_81bp.fasta \
  -o "$OUT" \
  -m tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5
echo "=> LCNN OK"

echo ""
echo "============================================"
echo " TEST 3/5: PromoTech (RF-HOT + RF-TETRA)"
echo "============================================"
pixi run python src/analysis/evaluate_promotech_pipelines.py \
  -p data/benchmark/positives_81bp.fasta \
  -n data/benchmark/negatives_81bp.fasta \
  -o "$OUT" \
  --promotech-dir tools/Promotech
echo "=> PromoTech OK"

echo ""
echo "============================================"
echo " TEST 4/5: iPro-MP (sp 12)"
echo "============================================"
cat data/benchmark/positives_81bp.fasta data/benchmark/negatives_81bp.fasta > /tmp/test_combined.fasta
pixi run -e ipro-mp --manifest-path tools/iPro-MP/pixi.toml \
  python tools/iPro-MP/iPro-MP_predict.py \
  -i /tmp/test_combined.fasta \
  -s 12 \
  -o "$OUT/ipromp_sp12_test.csv" \
  -m tools/iPro-MP/07-final \
  -d tools/iPro-MP/DNABERT-6
rm -f /tmp/test_combined.fasta
echo "=> iPro-MP OK"

echo ""
echo "============================================"
echo " TEST 5/5: MEME (STREME + FIMO)"
echo "============================================"
pixi run --manifest-path tools/meme/pixi.toml python -c "
from src.runner.local import LocalRunner
from src.benchmark.tools import PROMOTER_TOOLS, _load_toml_tools
_load_toml_tools()
r = LocalRunner(n_runs=1, warmup=False)
result = r.run(PROMOTER_TOOLS['meme'])
print(f'MEME: {result[\"success\"]}, {result[\"wall_seconds\"]:.1f}s')
"
echo "=> MEME OK"

echo ""
echo "============================================"
echo " ALL 5 TOOLS TESTED SUCCESSFULLY"
echo "============================================"
