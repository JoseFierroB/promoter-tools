# Promoter Prediction Benchmark — *S. pneumoniae* D39V & TIGR4

Modular benchmark suite for bacterial promoter prediction. Extracts datasets from genomic annotations (GFF3 + FASTA), runs 9 prediction tools (deep learning, machine learning, and motif-based), and generates comparative metrics (AUC, F1, confusion matrix) with bootstrap confidence intervals.

**Tools:** iPro-MP (DNABERT-6), PromoterLCNN (CNN), PromoTech (Random Forest), MLDSPP (XGBoost), MEME Suite (STREME + FIMO), and FIMO zero-shot against *E. coli* and prokaryote motif databases.

**Everything runs without admin rights** — isolated pixi environments using conda-forge exclusively.

## Quick Start

```bash
# Install all environments
pixi install && for d in tools/*/; do (cd "$d" && pixi install); done

# Download models (one-time)
# See docs/DOWNLOADS.md

# Run a single tool
pixi run python src/cli.py run mldspp

# Run the full benchmark locally
python submit/run_benchmark.py local all

# Run on Slurm cluster
python submit/run_benchmark.py slurm all
```

## Project Structure

```
src/
├── cli.py              # Unified CLI (run, results, dataset)
├── runners/             # 9 independent tool runners
├── benchmark/           # Tool registry + configuration
├── backend/             # Local + Slurm execution backends
├── analysis/            # Metrics, ROC, statistics, resource plots
│   └── igr/             # IGR conservation clustering + sigma assignment
└── dataset/             # TSS extraction from GFF3 + FASTA

data/
├── benchmark/           # D39V TSS datasets (FASTA + metadata)
├── tigr4/               # TIGR4 TSS datasets (multiple tiers)
└── reference/           # Reference genomes + annotations

tools/                   # External tool packages (each with pixi.toml)
├── meme/                # MEME Suite
├── iPro-MP/             # DNABERT-6 models
├── MLDSPP-Promoter-prediction/
├── Promotech/           # Random Forest promoter predictor
└── Promoters/           # PromoterLCNN (Keras CNN)

output/                  # Generated outputs (gitignored)
├── predictions/         # Per-tool prediction scores
├── tables/              # AUC tables, confusion matrix, stats
├── plots/               # ROC curves, resource charts
└── intergenic/          # IGR extraction + MMseqs2 clustering
```

## Documentation

- `docs/RUNNING.md` — detailed CLI usage, batch benchmarks, analysis scripts
- `docs/DOWNLOADS.md` — model download instructions + verification

## Requirements

- **Pixi** (package manager) — `pixi.lock` ensures reproducible environments
- **Python 3.10** via pixi environments
- **GPU** optional — only iPro-MP and LCNN benefit from CUDA
- For non-pixi users: `requirements.txt` lists core dependencies

## Citation

If you use this in your research, please cite the original tools:
- **iPro-MP**: Lin et al. (2024)
- **LCNN**: PromoterLCNN
- **PromoTech**: Londoño et al. (2023)
- **MLDSPP**: Dinucleotide stability + XGBoost
- **MEME Suite**: Bailey et al. (2009)
