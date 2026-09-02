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

# Run a single tool on the canonical benchmark
pixi run python src/cli.py run mldspp

# Run on Slurm cluster (one job per tool)
pixi run python src/cli.py run --slurm meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12
```

## Canonical Benchmark

The canonical, consolidated evaluation: D39V TSS datasets (989 curated promoters
/ 1000 CDS negatives) and TIGR4 tiers, evaluated with all 7 tools via the
unified CLI. Commands and metrics in `docs/WORKFLOW.md` and `docs/REPORT.md`.

```bash
# Run the full 7-tool canonical benchmark locally
pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12
```

## Experimental Extensions (parallel, same resources)

> **Status: experimental** — preliminary datasets and results, subject to change.
> Reuses the same runners, CLI, and analysis modules; only the dataset changes.
> No dedicated orchestration code — pure CLI configuration (see `docs/RUNNING.md`).

- **IGR benchmark** — promoters inside refined intergenic regions: D39V (723/723) and 4 TIGR4 subsets (553/578/971/1009). Versioned in `data/benchmark_igr/`.
- **Specialized niches** — CDS-internal promoters and 1:1 ortholog pairs (`data/benchmark_cds/`, `data/benchmark_ortho_1to1/`, generated).
- **Cross-strain IGR clustering** — 2,247 MMseqs2 clusters between D39V/TIGR4 (1,074 1:1 pairs).

```bash
# 1. Build all IGR/niche datasets (see docs/RUNNING.md for each step)
python src/dataset/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v --circular
python src/dataset/build_igr_benchmark_dataset.py
python src/dataset/build_tigr4_igr_datasets.py
python src/dataset/build_cds_and_ortho_datasets.py

# 2. Benchmark IGR (same CLI, different dataset) — works with or without pixi
python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 \
    --pos data/benchmark_igr/d39v/positives_81bp_igr.fasta \
    --neg data/benchmark_igr/d39v/negatives_81bp_igr.fasta \
    --output-dir output/predictions_igr/d39v

# Full command reference
# See docs/RUNNING.md → "IGR Benchmark & Specialized Niches"
```

## Project Structure

```
src/
├── cli.py              # Unified CLI (run) — canonical + experimental datasets via --pos/--neg
├── runners/            # 9 tool runners + shared helpers (_shared.py)
├── benchmark/          # Tool registry + configuration (tools.d/*.toml)
├── backend/            # Local + Slurm execution backends
├── analysis/           # Metrics, ROC, statistics, resource & IGR plots
└── dataset/            # TSS extraction, IGR/niche dataset builders

pipeline/               # Shell entrypoints + legacy scripts (run_pipeline.sh)

data/
├── benchmark/          # Canonical: D39V TSS (989 pos / 1000 neg) + mldspp_75 splits
├── benchmark_igr/      # Experimental: IGR benchmark (D39V 723 + TIGR4 subsets)
├── tigr4/              # Canonical: TIGR4 TSS datasets (multiple tiers)
└── reference/          # Reference genomes + annotations (GFF3 + FASTA)

tools/                   # External tool packages (each with pixi.toml)
├── meme/                # MEME Suite
├── iPro-MP/             # DNABERT-6 models
├── MLDSPP-Promoter-prediction/
├── Promotech/           # Random Forest promoter predictor
└── Promoters/           # PromoterLCNN (Keras CNN)

output/                  # Generated outputs (gitignored; canonical summary
├── predictions/         #   tables versioned under output/tables/)
├── predictions_igr/     # IGR benchmark scores (experimental)
├── tables/              # AUC tables, confusion matrix, stats
├── plots/               # ROC curves, resource charts, master atlas
└── intergenic/          # IGR extraction + MMseqs2 clustering
```

## Documentation

- `docs/WORKFLOW.md` — step-by-step pipeline diagram with commands (quick start)
- `docs/RUNNING.md` — detailed CLI usage, batch benchmarks, IGR & niches (experimental), analysis scripts
- `docs/DOWNLOADS.md` — model download instructions + verification
- `docs/REPORT.md` — results summary and key findings
- `docs/GC_EXPERIMENT_REPORT.md` — GC-matched negatives experiment (controls for composition bias)

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
# Dataset integrity unit tests
python negatives_tss_test.py --run-tests

# Integration smoke (lcnn + mldspp on d39v)
python tests/smoke_benchmark.py
```

> Commands work identically with or without pixi: `pixi run python ...` inside
> the pixi environment == plain `python ...` with an activated environment.
> There are no pixi-specific tasks — every command is documented explicitly in
> `docs/RUNNING.md` so non-pixi users have full parity.
