# Promoter Prediction Benchmark — *S. pneumoniae* D39V & TIGR4

Modular benchmark suite for bacterial promoter prediction. Extracts datasets from genomic annotations (GFF3 + FASTA), runs 7 prediction tools (deep learning, machine learning, and motif-based), and generates comparative metrics (AUC, F1, confusion matrix) with bootstrap confidence intervals.

**Tools:** iPro-MP (DNABERT-6), PromoterLCNN (CNN), PromoTech RF-HOT (Random Forest), MLDSPP (XGBoost, 0% and 75% spn variants), MEME Suite (STREME + FIMO), and FIMO zero-shot against a unified prokaryote motif database.

**Everything runs without admin rights** — isolated pixi environments using conda-forge (CUDA-enabled torch for iPro-MP is installed via pip).

## Quick Start

```bash
# Install all environments
pixi install && for d in tools/*/; do (cd "$d" && pixi install); done

# Download models (one-time)
# See docs/DOWNLOADS.md

# Run a single tool
pixi run python src/cli.py run mldspp

# Run the full 7-tool benchmark locally
pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12

# Run on Slurm cluster (one job per tool)
pixi run python src/cli.py run --slurm meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12
```

## Project Structure

```
src/
├── cli.py              # Unified CLI (run)
├── runners/             # 9 independent tool runners
├── benchmark/           # Tool registry + configuration
├── backend/             # Local + Slurm execution backends
├── analysis/            # Metrics, ROC, statistics, resource plots
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

- `docs/WORKFLOW.md` — step-by-step pipeline diagram with commands (quick start)
- `docs/RUNNING.md` — detailed CLI usage, batch benchmarks, analysis scripts
- `docs/DOWNLOADS.md` — model download instructions + verification

## Requirements

- **Pixi** (package manager) — committed `pixi.lock` files ensure reproducible environments
- **Python** per environment via pixi (root 3.10; tool envs 3.9-3.12 — e.g. PromoterLCNN needs 3.9 for TF 2.6)
- **GPU** optional — iPro-MP and LCNN benefit from CUDA (see `docs/RUNNING.md` for GPU assignment)
- For non-pixi users: `requirements.txt` lists core dependencies

## Citation

If you use this in your research, please cite the original tools:
- **iPro-MP**: Su et al., *Genome Biology* (2025) — doi:10.1186/s13059-025-03819-9
- **PromoterLCNN**: Hernández et al., *Genes* (2022) — doi:10.3390/genes13071126
- **PromoTech**: Chevez-Guardado & Peña-Castillo, *Genome Biology* (2021) — doi:10.1186/s13059-021-02514-9
- **MLDSPP**: Paul et al., *J. Chem. Inf. Model.* (2024) — doi:10.1021/acs.jcim.3c02017
- **MEME Suite**: Bailey et al., *Nucleic Acids Research* (2009)

## Dependencies note

- `pygenometracks` (Circos-style plotting, not used by the prediction pipeline) declares `matplotlib==3.1.1`; the environment pins `matplotlib>=3.10.9` — `pip check` reports this conflict. Safe: no pipeline code imports pygenometracks.
- `scikit-bio` declares `numpy>=2.0`; the environment pins `numpy 1.26` for TensorFlow 2.6 compatibility. Works with warnings.
- Analysis scripts default data dir via `PROMOTER_DATA_DIR` env (fallback `~/Desktop`).

## Tests

```bash
pixi run python tests/smoke_benchmark.py   # integration smoke: lcnn + mldspp on d39v
```
