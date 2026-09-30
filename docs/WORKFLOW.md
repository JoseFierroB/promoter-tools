# Workflow — how to run the promoter benchmark end-to-end

This document is the step-by-step guide to reproduce the full benchmark.
All commands run from the repository root. Detailed references:
`docs/RUNNING.md` (commands), `docs/DOWNLOADS.md` (models).

```
┌────────────────────────────────────────────────────────────────────┐
│ 0. SETUP (once)                                                    │
│    git clone https://github.com/JoseFierroB/promoter-tools         │
│    cd promoter-tools                                               │
│    pixi install && for d in tools/*/; do (cd "$d" && pixi install); done
│    # models: see docs/DOWNLOADS.md (PromoTech, iPro-MP, DNABERT-6) │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 1. DATASETS (regenerate project sets; skips existing files)           │
│    ./pipeline/run_pipeline.sh datasets                             │
│    # individual commands: docs/RUNNING.md §Dataset Generation      │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 2. MLDSPP_75 SPLITS (75/25, seed 42)                               │
│    ./pipeline/run_pipeline.sh splits                               │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 3. BENCHMARK                                                       │
│    pixi run python src/cli.py run <tools> [flags]                  │
│                                                                     │
│    flags: --threads N (CPUs) · --cpu-only (no GPU) · --runs N      │
│           --pos/--neg · --output-dir · -o (metrics TSV)            │
│                                                                     │
│    # 9 tools · 1 CPU · 3 runs                                      │
│    pixi run python src/cli.py run meme fimo_prok mldspp \          │
│      mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert --threads 1 --runs 3 │
│                                                                     │
│    # 9 tools · 16 CPU (+GPU for lcnn / ipromp / prokbert)          │
│    pixi run python src/cli.py run meme fimo_prok mldspp \          │
│      mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert --threads 16 --runs 3│
│                                                                     │
│    # GPU tools only, CPU-only fallback                             │
│    pixi run python src/cli.py run lcnn ipromp_sp12 --cpu-only \    │
│      --threads 8                                                    │
│                                                                     │
│    outputs: <output-dir>/<tool>/*_pos.csv · *_neg.csv              │
│             + resource_metrics_<tool>.tsv                          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 4. ANALYSIS (all read run dirs; write into 1_inference/ + 3_tables/)│
│    ROC + metrics per run:                                          │
│      pixi run python src/cli.py run ... --plots  (or standalone:)  │
│      pixi run python src/analysis/analyze_run.py \                 │
│        --pred-dir <run>/1_inference/predictions --db <name> -o <run>│
│    Compute plots per run:                                          │
│      pixi run python src/analysis/generate_compute_plots.py <run>  │
│    Sigma stratification (once per dataset):                        │
│      pixi run python src/analysis/sigma_stratify.py \              │
│        --run-dir <run> --name <ds> --metadata <meta.tsv>            │
│    Operating points (Youden, global/sigma/strain):                 │
│      pixi run python src/analysis/operating_points.py \            │
│        --run-dir <run> --name <ds> [--metadata ...] [--neg ...]     │
│    Compare runs (time/ram/speedup/vram + TSV):                     │
│      pixi run python src/cli.py compare <run1> <run2> ... \         │
│        [-o <out>] [--dataset "Title"]                              │
│    Scaling ladder (single tool or --all):                          │
│      pixi run python src/analysis/plot_scaling.py <runs...> \       │
│        --tool "<Registry Name>" -o <out>   (NOT the display label)  │
│    3090 vs 5090 (3 DL tools):                                      │
│      pixi run python src/analysis/plot_gpu_compare.py \            │
│        --a <run-3090> --b <run-5090> -o <out>                       │
│    Contract + tests: docs/RUNNER_CONTRACT.md, tests/test_*.py      │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ 5. ALL-IN-ONE (stages 1-4)                                         │
│    ./pipeline/run_pipeline.sh all [--threads N] [--runs N]         │
└────────────────────────────────────────────────────────────────────┘
```

## Cheatsheet

| Task | Command |
|------|---------|
| Install envs | `pixi install && for d in tools/*/; do (cd "$d" && pixi install); done` |
| Datasets | `./pipeline/run_pipeline.sh datasets` |
| Splits | `./pipeline/run_pipeline.sh splits` |
| Benchmark (9 tools, 1 CPU) | `pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert --threads 1 --runs 3` |
| Benchmark (16 CPU + GPU) | same with `--threads 16 --gpu` |
| Single tool | `pixi run python src/cli.py run lcnn [flags]` |
| ROC plot | `pixi run python src/cli.py run ... --plots` (or standalone `src/analysis/analyze_run.py --pred-dir <pred> --db <name> -o <run>`) |
| AUC + CI + DeLong | `pixi run python src/analysis/compute_metrics.py <run_dir>` |
| Confusion matrices | `pixi run python src/analysis/compute_metrics.py <run_dir>` |
| Resource plots | `pixi run python src/analysis/generate_compute_plots.py <run_dir>` |
| Scaling analysis + plots | `pixi run python src/analysis/plot_scaling.py <run_dirs...> --tool ... -o <out>` |
| ROC/AUC curves | `pixi run python src/analysis/generate_auc_plots.py <run_dir>` |
| Everything | `./pipeline/run_pipeline.sh all` |

## Notes for replication

- **CPU vs GPU**: `--threads` controls CPUs only; GPU-capable tools (`lcnn`,
  `ipromp_sp12`) use GPU 0 automatically unless `--cpu-only` is given.
- **Models are not in git**: run `docs/DOWNLOADS.md` steps once before the
  first benchmark.
- **`time_s` is pure compute time** (model loading and training excluded);
  MLDSPP reports training time separately. `wall_seconds` is the full wall time.
- **Plots are regenerated** from the predictions and metrics produced by the
  CLI — nothing is committed to git under `output/`.
- The orchestrator (`pipeline/run_pipeline.sh`) is safe: dataset stages skip
  files that already exist unless `--overwrite` is passed.
