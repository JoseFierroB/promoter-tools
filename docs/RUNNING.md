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
```

These 7 tools form the final benchmark. Two extra tools are registered but
**excluded from the analysis** (kept for reference only):

```bash
# FIMO + E. coli DB (zero-shot) — excluded: no S. pneumoniae training data
pixi run python src/cli.py run fimo_db

# PromoTech RF-TETRA — excluded: RF-HOT performs better
pixi run python src/cli.py run promotech_tetra
```

Runners are at `src/runners/{tool}.py` and can also be executed standalone:

```bash
pixi run --manifest-path tools/meme/pixi.toml python src/runners/meme.py \
  --pos data/benchmark/d39v/positives_81bp.fasta \
  --neg data/benchmark/d39v/negatives_81bp.fasta \
  -o output/predictions
```

## Dataset Generation (canonical commands)

Regenerating the canonical datasets from the reference annotations. All commands
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
  --xlsx data/tigr4/S1_TSS.xlsx --fasta data/reference/NC_003028.fasta \
  --tier high_conf_primary -o data/tigr4/positives_high_81bp

pixi run python src/dataset/negatives_tss_tigr4.py \
  --xlsx data/tigr4/S1_TSS.xlsx --fasta data/reference/NC_003028.fasta \
  --tier high_conf_primary --limit 738 --dedup-rc \
  -o data/tigr4/negatives_high_81bp
```

### MLDSPP 75% splits (stage 4)

Pre-built 75/25 train/test index splits (seed 42) matched by positive count.
The canonical d39v split and all scale_db sizes are already committed; regenerate
any missing size with:

```bash
pixi run python src/dataset/make_mldspp_75_splits.py --n-pos 989 1976 4940 9880 19760 29640 49400 98800 197600
```

> The 989 split (`mldspp_75_split_scale_db_989.npz`, seed 42, 741 train / 248 test)
> matches the current canonical D39V dataset (TSS 1801133 included). The 988 npz
> is historical (pre-Axel dataset).

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
# 6. Analysis: benchmark_statistics / benchmark_confusion / generate_master_roc /
#              resource_plots / scaling_analysis / generate_master_plots
# 7. Experiments (consensus / features): src/analysis/experiments/*.py
```

An orchestrator for stages 2-6 lives in `pipeline/run_pipeline.sh` (separate
folder, does not touch canonical source).

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
> to change. Same runners and pipeline as the canonical benchmark; only the
> dataset changes. No dedicated code — everything below reuses the unified CLI
> (`src/cli.py`) plus canonical `src/dataset/` builders and `src/analysis/`.

### 1. Build IGR datasets (once, ~1 min)

```bash
# Step 0 — Refined intergenic regions (input for all IGR datasets)
python src/dataset/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v --circular

# Step 1-3 — the four dataset families (D39V IGR + TIGR4 IGR subsets + CDS + ortho 1:1)
python src/dataset/build_igr_benchmark_dataset.py      # D39V IGR 723/723 → data/benchmark_igr/d39v
python src/dataset/build_tigr4_igr_datasets.py         # TIGR4 4 subsets (553/578/971/1009) → data/benchmark_igr/tigr4
python src/dataset/build_cds_and_ortho_datasets.py     # CDS internal + ortho 1:1 → data/benchmark_cds, data/benchmark_ortho_1to1
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
python src/analysis/process_igr_benchmark_results.py        # AUC/ACC/MCC + ROC from predictions_igr
python src/analysis/cluster_igrs.py                         # cross-strain IGR clusters (2,247) tables
python src/analysis/sigma_stratified_roc.py                 # ROC stratified by SigA/None/SigX

# Confusion matrices for the canonical benchmark (reuse existing analysis code)
python src/analysis/benchmark_confusion.py
python src/analysis/generate_benchmark_plots.py             # canonical 119-figure suite
```

**Dataset lineage (D39V)**: GFF 1003 TSS (Victor + Axel) → 989 curated
(proximity filter <25 bp) → 723 inside refined IGRs (266 CDS-internal excluded).

## Batch Benchmarks

```bash
# The full 7-tool benchmark, locally
pixi run python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12

# The full 7-tool benchmark on Slurm (one job per tool)
pixi run python src/cli.py run --slurm meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12

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
| `-o` | Path of the metrics TSV (time, RAM, VRAM, CPU%, GPU%). |
| `--no-timeout` | Disable per-tool timeouts. |

Environment extras:

```bash
PROMOTER_TOOLS_LCNN_BATCH=0|1 pixi run python src/cli.py run lcnn ...   # LCNN inference batch (default 10000; 0 = all at once, 1 = one by one)
IPROMP_SPECIES=23             pixi run python src/cli.py run ipromp_sp12 ...   # iPro-MP species (default 12 = H. pylori; 23 = B. subtilis)
PROMOTER_DATA_DIR=/path       pixi run python src/analysis/*.py ...      # base dir for analysis campaign folders
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
# Master ROC (combined, all tools)
pixi run python src/analysis/generate_master_roc.py

# Bootstrap CIs + DeLong pairwise tests
pixi run python src/analysis/benchmark_statistics.py

# Confusion matrices (D39V + TIGR4)
pixi run python src/analysis/benchmark_confusion.py

# Resource plots (compute time + peak RAM + VRAM if available)
pixi run python src/analysis/resource_plots.py

# Canonical publication figure suite — 119 figures (time/RAM) across hardware
# regimes (1_cpu, 16_cpu, gpu_vram, combined, by_scale x 9) in PNG/SVG/PDF
pixi run python src/analysis/generate_benchmark_plots.py

# ROC/AUC curves for N = 1,976 and N = 59,280 (PNG/SVG/PDF)
pixi run python src/analysis/generate_auc_plots.py
```

## Output Files

| File | Content |
|------|---------|
| `output/tables/resource_metrics.tsv` | Time, RAM, VRAM per tool (auto-generated by CLI) |
| `output/tables/benchmark_statistics.tsv` | AUC + 95% CI + DeLong tests |
| `output/plots/benchmark/master_benchmark_roc.{svg,png}` | 7-curve ROC |
| `output/plots/benchmark/compute_time.{svg,png}` | Resource bar chart |
| `output/plots/benchmark/ram.{svg,png}` | RAM bar chart |
| `output/plots/benchmark/vram.{svg,png}` | VRAM bar chart (GPU tools) |
| `output/plots/meme/` | All MEME plots |
| `output/predictions/` | Per-tool prediction CSVs (incl. `ipromp/`, `promotech/`, `mldspp_75spn_*`) |
>
> **Legacy scripts** (unused, kept for reference): `pipeline/legacy/`
> (`make_scale_fastas.sh` — scale-db FASTA generation by duplication;
> `test_all_tools.sh` — early smoke loop). Superseded by the unified CLI
> and `pipeline/run_pipeline.sh`.
