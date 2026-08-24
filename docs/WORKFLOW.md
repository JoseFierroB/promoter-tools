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
│ 1. DATASETS (regenerate canonical; skips existing files)           │
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
│    # 7 tools · 1 CPU · 3 runs                                      │
│    pixi run python src/cli.py run meme fimo_prok mldspp \          │
│      mldspp_75 lcnn promotech_hot ipromp_sp12 --threads 1 --runs 3 │
│                                                                     │
│    # 7 tools · 16 CPU (+GPU for lcnn / ipromp)                     │
│    pixi run python src/cli.py run meme fimo_prok mldspp \          │
│      mldspp_75 lcnn promotech_hot ipromp_sp12 --threads 16 --runs 3│
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
│ 4. ANALYSIS                                                        │
│    ROC:                                                            │
│      pixi run python src/analysis/generate_master_roc.py           │
│      pixi run python src/analysis/benchmark_statistics.py          │
│      pixi run python src/analysis/benchmark_confusion.py           │
│    Resources:                                                      │
│      pixi run python src/analysis/resource_plots.py --iter 9880    │
│      pixi run python src/analysis/resource_plots.py --by-iter      │
│      pixi run python src/analysis/scaling_analysis.py --scale-db DIR│
│      pixi run python src/analysis/generate_master_plots.py --mode all
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
| Benchmark (7 tools, 1 CPU) | `pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 --threads 1 --runs 3` |
| Benchmark (16 CPU + GPU) | same with `--threads 16` |
| Single tool | `pixi run python src/cli.py run lcnn [flags]` |
| ROC plot | `pixi run python src/analysis/generate_master_roc.py` |
| AUC + CI + DeLong | `pixi run python src/analysis/benchmark_statistics.py` |
| Confusion matrices | `pixi run python src/analysis/benchmark_confusion.py` |
| Resource plots | `pixi run python src/analysis/resource_plots.py [--by-iter]` |
| Scaling analysis + plots | `pixi run python src/analysis/scaling_analysis.py --scale-db DIR` |
| Canonical figure suite (119 fig. PNG/SVG/PDF) | `pixi run python src/analysis/generate_benchmark_plots.py` |
| ROC/AUC curves (N=1,976 y N=59,280) | `pixi run python src/analysis/generate_auc_plots.py` |
| Master plots suite (legacy) | `pixi run python src/analysis/generate_master_plots.py --mode all` |
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
