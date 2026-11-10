# Promoter Prediction Benchmark — *S. pneumoniae* D39V & TIGR4

Modular benchmark suite for bacterial promoter prediction. Extracts datasets from genomic annotations (GFF3 + FASTA), runs 9 prediction tools (deep learning, machine learning, and motif-based), and generates comparative metrics (AUC, F1, confusion matrix).

**Tools:** iPro-MP (DNABERT-6), PromoterLCNN (CNN), PromoTech RF-HOT (Random Forest), MLDSPP (XGBoost, 0% and 75% spn variants), MEME Suite (STREME + FIMO), FIMO zero-shot against a unified prokaryote motif database, prompt (MLP), ProkBERT-mini (gLM).

**Everything runs without admin rights** — isolated pixi environments (one per tool: `tools/<tool>/pixi.toml`), using conda-forge (CUDA-enabled torch installed via pip from download.pytorch.org).

## Quick Start

```bash
# Install all environments
pixi install && for d in tools/*/; do (cd "$d" && pixi install); done

# Download models (one-time)
# See docs/DOWNLOADS.md

# Run a single tool on the benchmark
pixi run python src/cli.py run mldspp

# Run on Slurm cluster (one job per tool)
pixi run python src/cli.py run --slurm meme fimo_prok mldspp lcnn promotech_hot ipromp_sp12 prompt prokbert
```

## Benchmark

The default dataset is `d39v_tigr4_hc_dedup_balanced`: 1399 deduplicated
D39V+TIGR4 high-confidence TSS positives (SigA 520 / SigX-family 27 / None 853)
+ 2798 negatives 1:2 (1399 CDS-background + 1399 IGR-background), seed 42.

Latest 7-tool results on the default set (1cpu-gpu, `output/261027_runs/d39v_tigr4_hc_1_to2_5050_dedup/`, AUC):

| Tool | AUC |
|---|---|---|---|
| ProkBERT-mini [gLM] | 0.9280 |
| iPro-MP [gLM] | 0.9166 |
| PromoTech RF-HOT [RF] | 0.8979 |
| PromoterLCNN [CNN] | 0.8927 |
| MLDSPP [BDT] | 0.8373 |
| prompt [NN] | 0.8219 |
| FIMO ProkDB [PWMs] | 0.6871 |

Commands and full metrics in `docs/WORKFLOW.md` and `docs/REPORT.md`.

```bash
# Run the benchmark locally (default dataset auto-selected).
# No tool list = main 7 tools; --all-tools runs all 9; --only/-subset per script.
pixi run python src/cli.py run \
  --pos data/benchmark/datasets/d39v_tigr4_hc_dedup_balanced/positives_81bp.fasta \
  --neg data/benchmark/datasets/d39v_tigr4_hc_dedup_balanced/negatives_81bp.fasta

# Explicit 9-tool run, or drop tools from a comparison without touching run dirs:
pixi run python src/cli.py run --all-tools [...]
pixi run python src/analysis/compare_runs.py <run1> <run2> --exclude "MLDSPP 75pct [BDT]" -o <out>
```

## Project Structure

```
src/
├── cli.py              # Unified CLI (run/compare) — project + experimental datasets via --pos/--neg
├── runners/            # 9 tool runners + shared helpers (_shared.py)
├── benchmark/          # Tool registry + configuration (tools.d/*.toml)
├── backend/            # Local + Slurm execution backends
├── analysis/           # Metrics, ROC, statistics, resource & IGR plots
└── dataset/            # TSS extraction, dataset builders, splits, annotation_conflicts

pipeline/               # Shell entrypoints + legacy scripts (run_pipeline.sh)

data/
├── benchmark/          # Default hc_dedup_balanced set + splits + annotation_conflicts.tsv
├── benchmark/datasets/ # Versioned datasets (default: d39v_tigr4_hc_dedup_balanced)
├── benchmark_igr/      # Experimental: IGR benchmark (D39V 723 + TIGR4 subsets)
├── tigr4/              # TIGR4 TSS datasets (multiple tiers, S1_TSS.tsv)
└── reference/          # Reference genomes + annotations (GFF3 + FASTA)

experiments/igr/         # Experimental IGR modules — self-contained (builders,
                         #   clustering, IGR metrics, MMseqs2 reproduction) + README

tools/                   # External tool packages (each with OWN pixi.toml)
├── meme/                # MEME Suite
├── fimo/                # FIMO (own env, split from meme)
├── iPro-MP/             # DNABERT-6 models
├── MLDSPP-Promoter-prediction/
├── Promotech/           # Random Forest promoter predictor
├── Promoters/           # PromoterLCNN (Keras CNN)
├── Prompt/              # Prompt MLP
├── prompt/              # Prompt runner env (split from ipro-mp)
└── prokbert/            # ProkBERT-mini (own env, split from ipro-mp)

output/                  # Generated outputs (gitignored) — see docs/REPO_LAYOUT.md
├── <YYMMDD>_runs/<name>/<config>/  # one run = capsule (MANIFEST+STATUS+RUN.log,
│                                   #   1_inference/ 2_resources/ 3_tables/)
├── comparisons/         # cross-run figures + TSVs (no prediction copies)
├── experiments/         # A/B prefetch TSVs
├── tables/              # era reports (IGR/pangenome, read by experiments/igr)
├── intergenic*/ mmseqs2/  # IGR pipeline data (live inputs, do not move)
└── _archive/            # retired outputs, reversible
```

## Requirements

- **Pixi** (package manager) — committed `pixi.lock` files ensure reproducible environments
- **Python** per environment via pixi (root 3.10; tool envs 3.9-3.12 — e.g. PromoterLCNN needs 3.9 for TF 2.6)
- **GPU** optional — DL tools benefit from CUDA (LCNN pinned to sm_86 GPU: TF 2.6 has no sm_120 kernels and JIT-compiles ~10 min on RTX 5090; see docs/RUNNING.md for GPU assignment)
- For non-pixi users: `requirements.txt` lists core dependencies

## Dependencies note

- `scikit-bio` declares `numpy>=2.0`; the environment pins `numpy 1.26` for TensorFlow 2.6 compatibility. Works with warnings.
