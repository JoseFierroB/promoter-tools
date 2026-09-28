# Running Instructions — promoter-tools

## Quick Setup

```bash
git clone <repo-url> promoter-tools
cd promoter-tools

# Install all environments (root + tools)
pixi install
(cd tools/meme                         && pixi install)
(cd tools/MLDSPP-Promoter-prediction   && pixi install)
(cd tools/Promoters                    && pixi install)
(cd tools/Promotech                    && pixi install)
(cd tools/iPro-MP                      && pixi install)
```

> **iPro-MP note**: the iPro-MP environment installs `torch` via pip (the
> `pytorch` conda channel stops at 2.5.1, which has no Blackwell/sm_120
> support). The PyPI wheel bundles CUDA and is ~2 GB. GPU support requires
> torch >= 2.9, so the lock file pins a recent version.

**HPC cluster setup**: if pixi is not in your PATH, add it:

```bash
export PIXI_HOME=~/.pixi               # or cluster-specific path
export PATH="$PIXI_HOME/bin:$PATH"
```

Then run the `pixi install` commands above. The lock file is committed — environments will be resolved from cache.

## Individual Tools

All tools are run via the unified CLI:

```bash
# MEME: STREME + FIMO 2-fold CV (de novo discovery)
pixi run python src/cli.py run meme

# FIMO + Prokaryote DB (zero-shot, 838 motifs)
pixi run python src/cli.py run fimo_prok

# MLDSPP XGBoost (0% spn — cross-species, no leakage)
pixi run python src/cli.py run mldspp

# MLDSPP XGBoost (75% spn — 75% of positives in training, reference only)
pixi run python src/cli.py run mldspp_75

# PromoterLCNN
pixi run python src/cli.py run lcnn

# PromoTech RF-HOT (PG mode, sliding window)
pixi run python src/cli.py run promotech_hot

# iPro-MP sp12 (H. pylori, DNABERT-6) — GPU strongly recommended
pixi run python src/cli.py run ipromp_sp12

# Prompt MLP (B. subtilis 168)
pixi run python src/cli.py run prompt

# ProkBERT-mini promoter gLM
pixi run python src/cli.py run prokbert
```

These 9 tools form the final benchmark. Two extra runners were retired to
`archive/legacy_dead/` (reconstructible from git history if needed):

- FIMO + E. coli DB (`fimo_db`) — excluded: no S. pneumoniae training data.
- PromoTech RF-TETRA (`promotech_tetra`) — excluded: RF-HOT performs better.

Runners are at `src/runners/{tool}.py` and can also be executed standalone:

```bash
pixi run --manifest-path tools/meme/pixi.toml python src/runners/meme.py \
  --pos data/benchmark/d39v/positives_81bp.fasta \
  --neg data/benchmark/d39v/negatives_81bp.fasta \
  -o output/predictions
```

## Dataset Generation (project commands)

Regenerating the project datasets from the reference annotations. All commands
run from the repo root with `pixi run python`. Reference inputs live in
`data/reference/` (D39V) and `data/tigr4/` (TIGR4).

### D39V positives (989 curated TSS from 1003 raw, 81 bp, [-60,+20])

```bash
pixi run python src/dataset/positive_tss_d39v.py \
  --gff data/reference/D39V_annotation_TSS_Victor.gff \
  --fasta data/reference/D39V.fna \
  --gff-cds data/reference/D39V.gff3 \
  -o data/benchmark/d39v/positives_81bp
```

### D39V negatives (1000 CDS-internal windows)

```bash
pixi run python src/dataset/negatives_tss_d39v.py \
  --gff-cds data/reference/D39V.gff3 \
  --fasta data/reference/D39V.fna \
  --gff-tss data/reference/D39V_annotation_TSS_Victor.gff \
  --dedup-rc --limit 1000 \
  -o data/benchmark/d39v/negatives_81bp
```

### TIGR4 high-confidence (738/738)

```bash
pixi run python src/dataset/positive_tss_tigr4.py \
  --tsv data/tigr4/S1_TSS.tsv --fasta data/reference/NC_003028.fasta \
  --tier high_conf_primary -o data/tigr4/positives_high_81bp

pixi run python src/dataset/negatives_tss_tigr4.py \
  --tsv data/tigr4/S1_TSS.tsv --fasta data/reference/NC_003028.fasta \
  --tier high_conf_primary --limit 738 --dedup-rc \
  -o data/tigr4/negatives_high_81bp
```

### MLDSPP 75% splits (stage 4)

Pre-built 75/25 train/test index splits (seed 42) matched by positive count:

| Split file | n_pos | Dataset |
|---|---|---|
| `mldspp_75_split_1_d39v.npz` | 989 | `datasets/1_d39v` |
| `mldspp_75_split_2_d39v_tigr4_high.npz` | 1727 | `datasets/2_d39v_tigr4_high` (+ sigma variant) |
| `mldspp_75_split_3_d39v_tigr4_all.npz` | 1998 | `datasets/3_d39v_tigr4_all` |
| `mldspp_75_split_scale_{10k,30k,60k,100k,200k}.npz` | 5000/15000/30000/50000/100000 | `scale_db` ladder |
| `mldspp_75_split_benchmark_igr.npz` | 723 | IGR experiment |

Regenerate any missing size with an explicit name:

```bash
pixi run python src/dataset/make_mldspp_75_splits.py --n-pos 5000 --name mldspp_75_split_scale_10k.npz
```

The runner (`mldspp_75`) matches splits to the FASTA by size; if none matches,
the tool is skipped with a message listing the available splits.

## End-to-end workflow (all stages)

```bash
# 1. Setup (once): pixi install in root + 5 tool envs (see Quick Setup), models (DOWNLOADS.md)
# 2. Datasets: commands above (or pipeline/run_pipeline.sh datasets)
# 3. Intergenic regions + cross-strain conservation: output/intergenic/README.md
# 4. MLDSPP splits: make_mldspp_75_splits.py (npz index) +
#    export_mldspp_75_fastas.py (npz → data/benchmark/splits/*.fasta)
# 5. Benchmark: pixi run python src/cli.py run <tools> [--threads N] [--runs N]
# 6. Analysis (per run dir): generate_auc_plots / generate_compute_plots /
#              compute_metrics (legacy predecessors in archive/analysis_legacy/)
# 7. Experiments (consensus / features): src/analysis/experiments/*.py
```

An orchestrator for stages 2-6 lives in `pipeline/run_pipeline.sh` (separate
folder, does not touch project source).

## Measurement methodology

Definitions of every column in `2_resources/resource_metrics.tsv`
(code: `src/utils/metrics.py`, sampler: `src/backend/local.py`):

| Column | Meaning | Method | Caveats |
|---|---|---|---|
| `wall_seconds` | Full process wall (load + compute) | `perf_counter` around the tool subprocess | Includes model loading (~3 s ProkBERT, ~40 s PromoTech forest) |
| `time_s` | Runner self-reported compute | Regex over runner stdout (`"N seqs in Xs"`) | Definition varies per runner; `None` → falls back to wall. Excludes load by convention |
| `cpu_seconds` | CPU time consumed | Derived: `mean_cpu_pct/100 × wall` (same semantics as Slurm `CPUTimeRAW`) | Not sampled directly |
| `mean_cpu_pct` | Mean CPU load | 0.2 s samples of summed per-process `cpu_percent()` over the process tree, arithmetic mean | 100% = 1 core (1600% possible); startup dilutes; blind to steal/iowait |
| `peak_ram_mb` | Peak host RAM | Max PSS over 0.2 s samples (`/proc/pid/smaps_rollup`) | High-water mark (allocators retain freed pages); misses <0.2 s spikes; RSS-sum fallback double-counts shared libs |
| `peak_vram_mb` | Peak device VRAM attributed to our PIDs | NVML per-process accounting filtered to our process tree; `nvidia-smi pid` fallback | Still includes own CUDA context + allocator cache; `gpu_util_pct` stays device-level |
| `model_size_mb` | Weight bytes on disk | File sizes (`Tool.model_size_mb()`) | Disk size, not VRAM residency |
| `throughput_seq_s` | Throughput | `n / time_s` from runner output | Inherits `time_s` fragility |

Conventions: single runs (`--runs 1`, no dispersion); `success=False` rows
(timeouts write wall=timeout and zeros) are excluded from comparisons;
per-run `harness` (threads, cpu-only/gpu/no-timeout flags, per-tool
`gpu_id` + effective device) is recorded in `STATUS.json`. Runner batch
sizes and worker counts live with each runner: ProkBERT 64 serial,
iPro-MP batch 128 + `ProcessPool(threads)` tokenize-ahead, prompt batch 128
serial (always CPU), LCNN batch 10000 (TF session), MLDSPP all-at-once
(`n_jobs=threads`), PromoTech 4 sequential subprocesses (RF `n_jobs` patch),
FIMO chunks=threads + thread pool over the binary, MEME 2 serial folds.

## Non-default Datasets (e.g. TIGR4)

Pass `--pos` / `--neg` FASTA files and a separate output dir. The number of
sequences in the metrics TSV is auto-detected from the FASTA files:

```bash
pixi run python src/cli.py run lcnn \
  --pos data/tigr4/positives_high_81bp.fasta \
  --neg data/tigr4/negatives_high_81bp.fasta \
  --output-dir output/tigr4/predictions
```

> **mldspp_75 splits**: the runner uses pre-built 75/25 splits from
> `data/benchmark/mldspp_75_split_*.npz`, matched to the FASTA by size.
> If the positive FASTA has no matching split (e.g. SigA/SigX), the tool is
> skipped with a message listing the available splits.

## IGR Benchmark & Specialized Niches (experimental)

> **Status: experimental** — datasets and results are preliminary and subject
> to change. Same runners and pipeline as the project benchmark; only the
> dataset changes. No dedicated code — everything below reuses the unified CLI
> (`src/cli.py`) plus `src/dataset/` builders and `src/analysis/`.

### 1. Build IGR datasets (once, ~1 min)

```bash
# Step 0 — Refined intergenic regions (input for all IGR datasets)
python experiments/igr/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v --circular

# Step 1-3 — the four dataset families (D39V IGR + TIGR4 IGR subsets + CDS + ortho 1:1)
python experiments/igr/build_d39v_igr.py      # D39V IGR 723/723 → data/benchmark_igr/d39v
python experiments/igr/build_tigr4_igr.py         # TIGR4 4 subsets (553/578/971/1009) → data/benchmark_igr/tigr4
python experiments/igr/build_cds_ortho.py     # CDS internal + ortho 1:1 → data/benchmark_cds, data/benchmark_ortho_1to1
```

> Commands shown without the `pixi run` prefix work identically inside the pixi
> environment (`pixi run python ...`) or with any activated environment — no
> pixi-specific tasks exist, so pixi and non-pixi users have full parity.

### 2. Run the 7-tool benchmark on IGR (pure CLI configuration)

```bash
pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 \
    --pos data/benchmark_igr/d39v/positives_81bp_igr.fasta \
    --neg data/benchmark_igr/d39v/negatives_81bp_igr.fasta \
    --output-dir output/predictions_igr/d39v \
    -o output/tables/resource_metrics_igr_d39v.tsv
```

Same command with different `--pos/--neg` covers every dataset family:

| Dataset | `--pos` | `--neg` |
|---|---|---|
| D39V IGR | `data/benchmark_igr/d39v/positives_81bp_igr.fasta` | `data/benchmark_igr/d39v/negatives_81bp_igr.fasta` |
| TIGR4 IGR subset_1 (high-conf primary) | `data/benchmark_igr/tigr4/subset_1_high_conf_primary/positives_81bp.fasta` | `.../negatives_81bp.fasta` |
| TIGR4 IGR subset_2 (high-conf all) | `data/benchmark_igr/tigr4/subset_2_high_conf_all/...` | `...` |
| TIGR4 IGR subset_3 (all primary) | `data/benchmark_igr/tigr4/subset_3_all_primary/...` | `...` |
| TIGR4 IGR subset_4 (all comprehensive) | `data/benchmark_igr/tigr4/subset_4_all_comprehensive/...` | `...` |
| CDS internal (D39V) | `data/benchmark_cds/d39v_cds_internal/positives_81bp.fasta` | `...` |
| Ortho 1:1 SigA | `data/benchmark_ortho_1to1/d39v_ortho_1to1_siga/positives_81bp.fasta` | `...` |

> **mldspp_75 split for IGR**: `data/benchmark/mldspp_75_split_benchmark_igr.npz`
> (seed 42, 542 train / 181 test over 723 positives).

### 3. Metrics, plots and cross-strain IGR clustering

```bash
python experiments/igr/process_results.py        # AUC/ACC/MCC + ROC from predictions_igr
python experiments/igr/cluster_igrs.py                         # cross-strain IGR clusters (2,247) tables
python experiments/igr/sigma_roc.py                 # ROC stratified by SigA/None/SigX

# Per-run metrics + plots (replaces legacy benchmark_confusion.py and
# the archived 119-figure suite; originals in archive/analysis_legacy/)
python src/analysis/compute_metrics.py <run_dir>            # confusion @Youden + bootstrap CI + DeLong
python src/analysis/generate_auc_plots.py <run_dir>         # title-free ROC overlay
python src/analysis/generate_compute_plots.py <run_dir>     # title-free time/RAM bars
```

**Dataset lineage (D39V)**: GFF 1003 TSS (Victor + Axel) → 989 curated
(proximity filter <25 bp) → 723 inside refined IGRs (266 CDS-internal excluded).

## Batch Benchmarks

```bash
# The full 9-tool benchmark, locally
pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert

# The full 9-tool benchmark on Slurm (one job per tool)
pixi run python src/cli.py run --slurm meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 prompt prokbert

# Single tool on Slurm
pixi run python src/cli.py run --slurm lcnn
```

## Common flags (CPU / GPU / runs)

```bash
pixi run python src/cli.py run <tools> [flags]
```

| Flag | Effect |
|------|--------|
| `--threads N` | Number of CPU threads: sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TF_NUM_INTRAOP_THREADS`, `TF_NUM_INTEROP_THREADS` to N. |
| *(default)* | GPU used automatically for GPU-capable tools (`lcnn`, `ipromp_sp12` → `gpu_id="0"`). |
| `--cpu-only` | Force CPU for all tools (`CUDA_VISIBLE_DEVICES=""`); LCNN/iPro-MP fall back to CPU. |
| `--runs N` | Independent runs; N≥3 recommended (reports mean ± SD). N=1 returns the raw run. |
| `--pos / --neg` | Custom FASTA pair (default: d39v confirmed positives/negatives). |
| `--output-dir` | Where per-tool prediction CSVs are written. |
| `-o` | Additional metrics TSV export; the run keeps its own primary metrics. |
| `--no-timeout` | Disable per-tool timeouts. |

Environment extras:

```bash
PROMOTER_TOOLS_LCNN_BATCH=0|1 pixi run python src/cli.py run lcnn ...   # LCNN inference batch (default 10000; 0 = all at once, 1 = one by one)
PROMOTER_TOOLS_PROKBERT_BATCH=128 PROMOTER_TOOLS_PROKBERT_PREFETCH=16 pixi run python src/cli.py run prokbert ...   # ProkBERT batch (default 64) + prefetch workers (default 0 = serial)
IPROMP_SPECIES=23             pixi run python src/cli.py run ipromp_sp12 ...   # iPro-MP species (default 12 = H. pylori; 23 = B. subtilis)
```

> **`time_s` semantics**: runners report *pure compute time* — model loading,
> session init and (for MLDSPP) training are excluded. MLDSPP prints the
> training time separately (`... in Xs (train Ys)`). `wall_seconds` in the
> metrics TSV is the full process wall time (load + compute).

## GPU Usage

| Tool | GPU | Why |
|------|-----|-----|
| `lcnn` (TF 2.6) | GPU 0 (`gpu_id="0"`) | TF 2.6 has no sm_120 kernels → must use an sm_86 card (e.g. RTX 3090) |
| `ipromp_sp12` (torch) | GPU 0 (`gpu_id="0"`) | Same GPU as LCNN; torch >= 2.9 supports sm_120 too |

- The local runner sets `CUDA_VISIBLE_DEVICES` and
  `TF_FORCE_GPU_ALLOW_GROWTH=true` automatically per tool.
- On a machine without the assigned GPU, the tool falls back to CPU
  (iPro-MP ~200 s vs ~15 s on GPU).
- Peak VRAM / GPU util are sampled during the run and reported in
  `resource_metrics.tsv` (requires `nvidia-ml-py`, installed in the root env).

## Analysis

```bash
# Automatically export ROC, AUC rows, time and RAM after scoring
pixi run python src/cli.py run prompt lcnn --input-dir <dataset_dir> --threads 1 --gpu --plots

# Regenerate plots from an existing run (no model execution)
pixi run python src/analysis/generate_auc_plots.py <run_dir>
pixi run python src/analysis/generate_compute_plots.py <run_dir>

# Optional confusion matrices, bootstrap CIs and DeLong tests
pixi run python src/analysis/compute_metrics.py <run_dir>

# Compare selected runs; each metric is a separate figure, with a title
pixi run python src/cli.py compare <run_dir1> <run_dir2> --dataset "D39V+TIGR4 high"

# Per-run compute plots (opt-in, not executed without --plots):
pixi run python src/analysis/generate_compute_plots.py <run_dir>
# (via harness: src/cli.py run ... --plots [--no-compute-plots to skip these])
```

## Output Files

Individual run figures (ROC, sigma ROC, time and RAM) have no title; axes,
legends, tool colors and method order remain. Comparison figures keep titles.
The per-run/compare workflow exports PNG and PDF only, one plot per file.

The CLI stores a run in `output/<YYMMDD>_runs/<dataset>/<config>/` and exposes
a primary alias in `output/runs/`. A completed repeat gets a separate `_02`
directory and alias, without changing its logical dataset/prediction prefix.

| File | Content |
|------|---------|
| `<run_dir>/1_inference/predictions/` | Per-tool predictions |
| `<run_dir>/1_inference/roc_auc_{name}.{png,pdf}` | Title-free ROC overlay |
| `<run_dir>/1_inference/sigma/roc_{class}_{name}.{png,pdf}` | Optional sigma ROC (`sigma_stratify.py`) |
| `<run_dir>/2_resources/resource_metrics.tsv` | Recorded time, RAM and VRAM per tool |
| `<run_dir>/2_resources/{compute_time,peak_ram}.{png,pdf}` | Title-free resource bars |
| `<run_dir>/3_tables/resource_metrics.tsv` | Link to the actual resource metrics |
| `<run_dir>/3_tables/metrics_rows.tsv` | AUC per tool |
| `<run_dir>/3_tables/metrics_table.tsv` | Optional confusion, CI and DeLong table |
| `<run_dir>/STATUS.json`, `MANIFEST.json` | Dataset, configuration, inputs and run identity |
| `output/comparisons/<comparison>/2_resources/compare_{time,ram,speedup,vram}.{png,pdf}` | Comparison figures (available metrics only) |
| `output/comparisons/<comparison>/3_tables/resources_compare.tsv` | Successful measurements with distinct source-run identities |

Readers accept legacy locations, but new exports do not add root-level
`plots/`, `resources/` or metric copies. `--output-dir` explicitly overrides
the prediction destination and is recorded in the manifest. Existing legacy
files are not deleted or moved automatically. VRAM is attributed to our
process tree when PIDs are known, else device-global (see table above).

> **Legacy scripts**: `pipeline/legacy/` (`make_scale_fastas.sh`,
> `test_all_tools.sh`) was moved out of the repo
> (`~/Desktop/promoter-attic/pipeline_legacy/`). Superseded by the unified CLI
> and `pipeline/run_pipeline.sh`.
