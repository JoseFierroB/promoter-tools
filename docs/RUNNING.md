# Running Instructions — promoter-tools

## Individual Tools

```bash
# MEME: STREME + FIMO 2-fold CV (de novo discovery)
pixi run python src/cli.py run meme

# MEME: FIMO + E. coli DB (zero-shot, comparable to PromoTech)
pixi run --manifest-path tools/meme/pixi.toml python src/experiments/fimo_db_pipeline.py

# MEME: All plots (logos, optimization, genome scan)
pixi run --manifest-path tools/meme/pixi.toml python src/experiments/meme_all_plots.py

# MLDSPP (cross-species, no local training)
pixi run --manifest-path tools/MLDSPP-Promoter-prediction/pixi.toml \
  python src/benchmark/run_mldspp_cv_predictions.py \
  -p data/benchmark/positives_81bp.fasta -n data/benchmark/negatives_81bp.fasta \
  -o OUT_DIR

# PromoterLCNN
pixi run --manifest-path tools/Promoters/pixi.toml \
  python src/benchmark/predict_lcnn.py \
  -p data/benchmark/positives_81bp.fasta -n data/benchmark/negatives_81bp.fasta \
  -o OUT_DIR -m tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5

# PromoTech RF-HOT
pixi run python src/benchmark/run_promotech.py -m RF-HOT \
  -p data/benchmark/positives_81bp.fasta -n data/benchmark/negatives_81bp.fasta \
  -o OUT_DIR

# PromoTech RF-TETRA
pixi run python src/benchmark/run_promotech.py -m RF-TETRA \
  -p data/benchmark/positives_81bp.fasta -n data/benchmark/negatives_81bp.fasta \
  -o OUT_DIR

# iPro-MP sp12 (needs GPU)
TMPDIR="${TMPDIR:-/tmp}"
cat data/benchmark/positives_81bp.fasta data/benchmark/negatives_81bp.fasta > "$TMPDIR/all.fa"
pixi run -e ipro-mp --manifest-path tools/iPro-MP/pixi.toml \
  python tools/iPro-MP/iPro-MP_predict.py \
  -i "$TMPDIR/all.fa" -s 12 -o OUT_DIR/ipromp_sp12.csv \
  -m tools/iPro-MP/07-final -d tools/iPro-MP/DNABERT-6
```

## Batch (Local)

```bash
# One or more tools
python submit/run_benchmark.py local meme,mldspp,lcnn,fimo_db

# All tools
python submit/run_benchmark.py local all
```

## Complete Benchmark (Slurm / HPC)

```bash
# All tools in parallel via Slurm
python submit/run_benchmark.py slurm all

# Single tool via Slurm
python submit/run_benchmark.py slurm meme
```

## MEME Pipeline Variants

| Command | Method | AUC | Notes |
|---------|--------|-----|-------|
| `src/cli.py run meme` | STREME + FIMO (2-fold CV) | ~0.85 | De novo, needs negatives |
| `src/experiments/fimo_db_pipeline.py` | FIMO + E. coli DB (zero-shot) | ~0.74 | No training, E. coli motifs only |
| `src/experiments/meme_all_plots.py` | All MEME plots | — | Logos, position, optimization, genome scan |

## Plots

```bash
# Master ROC (individual)
pixi run python src/analysis/generate_master_roc.py --tool meme

# Master ROC (combined)
pixi run python src/analysis/generate_master_roc.py

# Stability profiles
pixi run python src/experiments/stability_profiles_ci.py

# All MEME plots
pixi run --manifest-path tools/meme/pixi.toml python src/experiments/meme_all_plots.py
```

## Output Files

| File | Content |
|------|---------|
| `output/tables/resource_metrics.tsv` | Time, RAM per tool (auto-generated) |
| `output/tables/motif_stats.tsv` | STREME motifs with Tomtom annotation |
| `output/plots/benchmark/master_benchmark_roc.{svg,png}` | 7-curve ROC |
| `output/plots/meme/` | All MEME plots |
| `OUT_DIR/` | Per-tool prediction CSVs |
