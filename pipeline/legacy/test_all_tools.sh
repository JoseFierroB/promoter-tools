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
echo " TEST 1/9: MEME (STREME + FIMO)"
echo "============================================"
pixi run python src/cli.py run meme
echo "=> MEME OK"

echo ""
echo "============================================"
echo " TEST 2/9: MLDSPP (XGBoost)"
echo "============================================"
pixi run python src/cli.py run mldspp
echo "=> MLDSPP OK"

echo ""
echo "============================================"
echo " TEST 3/9: MLDSPP (75% spn)"
echo "============================================"
pixi run python src/cli.py run mldspp_75
echo "=> MLDSPP_75 OK"

echo ""
echo "============================================"
echo " TEST 4/9: PromoterLCNN"
echo "============================================"
pixi run python src/cli.py run lcnn
echo "=> LCNN OK"

echo ""
echo "============================================"
echo " TEST 5/9: FIMO + E. coli DB"
echo "============================================"
pixi run python src/cli.py run fimo_db
echo "=> FIMO_DB OK"

echo ""
echo "============================================"
echo " TEST 6/9: FIMO + Prokaryote DB"
echo "============================================"
pixi run python src/cli.py run fimo_prok
echo "=> FIMO_PROK OK"

echo ""
echo "============================================"
echo " TEST 7/9: PromoTech RF-HOT"
echo "============================================"
pixi run python src/cli.py run promotech_hot
echo "=> PromoTech RF-HOT OK"

echo ""
echo "============================================"
echo " TEST 8/9: PromoTech RF-TETRA"
echo "============================================"
pixi run python src/cli.py run promotech_tetra
echo "=> PromoTech RF-TETRA OK"

echo ""
echo "============================================"
echo " TEST 9/9: iPro-MP (sp 12, H. pylori)"
echo "============================================"
pixi run python src/cli.py run ipromp_sp12
echo "=> iPro-MP OK"

echo ""
echo "============================================"
echo " ALL 9 TOOLS TESTED SUCCESSFULLY"
echo "============================================"