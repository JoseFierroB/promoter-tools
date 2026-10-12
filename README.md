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
pixi run python src/cli.py run --slurm meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert
```

## Benchmark

The default dataset is `d39v_tigr4_hc_dedup_balanced`: 1399 deduplicated
D39V+TIGR4 high-confidence TSS positives (SigA 519 / SigX-family 27 / None 853)
+ 2798 negatives 1:2 (1399 CDS-background + 1399 IGR-background), seed 42,
zero pos/neg leakage. Builder: `src/dataset/build_mix_5050.py`; split:
`data/benchmark/mldspp_75_split_d39v_tigr4_hc_dedup_balanced.npz`
(train 1049 / test 350). Cross-strain annotation conflicts (7 exact-duplicate
groups) documented in `data/benchmark/annotation_conflicts.tsv`.

Latest 8-tool results on the default set (16cpu-gpu, `output/comparisons/261012_5050_8tools/`):

| Tool | AUC |
|---|---|
| MLDSPP 75% [BDT] | 0.9301 |
| ProkBERT-mini [gLM] | 0.9253 |
| iPro-MP [gLM] | 0.9152 |
| PromoterLCNN [CNN] | 0.8955 |
| MLDSPP 0% [BDT] | 0.8351 |
| prompt [NN] | 0.8194 |
| STREME+FIMO [motif] | 0.8057 |
| FIMO ProkDB [PWMs] | 0.6802 |

Commands and full metrics in `docs/WORKFLOW.md` and `docs/REPORT.md`.

```bash
# Run the full 9-tool benchmark locally (default dataset auto-selected)
pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert \
  --pos data/benchmark/datasets/d39v_tigr4_hc_dedup_balanced/positives_81bp.fasta \
  --neg data/benchmark/datasets/d39v_tigr4_hc_dedup_balanced/negatives_81bp.fasta
```

## Experimental Extensions (parallel, same resources)

> **Status: experimental** — preliminary datasets and results, subject to change.
> Reuses the same runners, CLI, and analysis modules; only the dataset changes.
> All IGR/niche code is self-contained in `experiments/igr/` (with its own README).

- **IGR benchmark** — promoters inside refined intergenic regions: D39V (723/723) and 4 TIGR4 subsets (553/578/971/1009). Versioned in `data/benchmark_igr/`.
- **Specialized niches** — CDS-internal promoters and 1:1 ortholog pairs (`data/benchmark_cds/`, `data/benchmark_ortho_1to1/`, generated).
- **Cross-strain IGR clustering** — 2,247 MMseqs2 clusters between D39V/TIGR4 (1,074 1:1 pairs). Reproduction script: `experiments/igr/run_mmseqs_canonical.py`.

```bash
# 1. Build all IGR/niche datasets (see docs/RUNNING.md for each step)
python experiments/igr/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v --circular
python experiments/igr/build_d39v_igr.py
python experiments/igr/build_tigr4_igr.py
python experiments/igr/build_cds_ortho.py

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
├── cli.py              # Unified CLI (run/compare) — project + experimental datasets via --pos/--neg
├── runners/            # 9 tool runners + shared helpers (_shared.py)
├── benchmark/          # Tool registry + configuration (tools.d/*.toml)
├── backend/            # Local + Slurm execution backends
├── analysis/           # Metrics, ROC, statistics, resource & IGR plots
└── dataset/            # TSS extraction, dataset builders, splits, annotation_conflicts

pipeline/               # Shell entrypoints + legacy scripts (run_pipeline.sh)

data/
├── benchmark/          # Default hc_dedup_balanced set + splits + annotation_conflicts.tsv
├── benchmark/datasets/ # All versioned datasets (16/17/18 IGR/mix/CDS + canonical 2_)
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

## Documentation

- `docs/WORKFLOW.md` — step-by-step pipeline diagram with commands (quick start)
- `docs/RUNNING.md` — detailed CLI usage, batch benchmarks, IGR & niches (experimental), analysis scripts
- `docs/DOWNLOADS.md` — model download instructions + verification
- `docs/REPORT.md` — results summary and key findings
- `docs/RUNNER_CONTRACT.md` — runner interface contract (flags, outputs, timing, seeds)
- `docs/REPO_LAYOUT.md` — repository map

## Requirements

- **Pixi** (package manager) — committed `pixi.lock` files ensure reproducible environments
- **Python** per environment via pixi (root 3.10; tool envs 3.9-3.12 — e.g. PromoterLCNN needs 3.9 for TF 2.6)
- **GPU** optional — DL tools benefit from CUDA (LCNN pinned to sm_86 GPU: TF 2.6 has no sm_120 kernels and JIT-compiles ~10 min on RTX 5090; see docs/RUNNING.md for GPU assignment)
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

## Tests

```bash
# Full contract + layout suite (no models, no GPU)
pixi run python -m pytest tests/ -q
```
