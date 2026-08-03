#!/bin/bash
#SBATCH --job-name=test_all_tools
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=production
#SBATCH --output=slurm-%j-test-all.out
#SBATCH --error=slurm-%j-test-all.err

set -e

export PIXI_HOME="${PIXI_HOME:-$HOME/.pixi}"
export PATH="$PIXI_HOME/bin:$PATH"

cd "$(dirname "$0")/.."

echo "============================================"
echo " TEST 1/8: MEME (STREME + FIMO)"
echo "============================================"
pixi run python src/cli.py run meme
echo "=> MEME OK"

echo ""
echo "============================================"
echo " TEST 2/8: MLDSPP (XGBoost)"
echo "============================================"
pixi run python src/cli.py run mldspp
echo "=> MLDSPP OK"

echo ""
echo "============================================"
echo " TEST 3/8: PromoterLCNN"
echo "============================================"
pixi run python src/cli.py run lcnn
echo "=> LCNN OK"

echo ""
echo "============================================"
echo " TEST 4/8: FIMO + E. coli DB"
echo "============================================"
pixi run python src/cli.py run fimo_db
echo "=> FIMO_DB OK"

echo ""
echo "============================================"
echo " TEST 5/8: FIMO + Prokaryote DB"
echo "============================================"
pixi run python src/cli.py run fimo_prok
echo "=> FIMO_PROK OK"

echo ""
echo "============================================"
echo " TEST 6/8: PromoTech RF-HOT"
echo "============================================"
pixi run python src/cli.py run promotech_hot
echo "=> PromoTech RF-HOT OK"

echo ""
echo "============================================"
echo " TEST 7/8: PromoTech RF-TETRA"
echo "============================================"
pixi run python src/cli.py run promotech_tetra
echo "=> PromoTech RF-TETRA OK"

echo ""
echo "============================================"
echo " TEST 8/8: iPro-MP (sp 12, H. pylori)"
echo "============================================"
pixi run python src/cli.py run ipromp_sp12
echo "=> iPro-MP OK"

echo ""
echo "============================================"
echo " ALL 8 TOOLS TESTED SUCCESSFULLY"
echo "============================================"
