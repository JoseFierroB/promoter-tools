# Promoter Prediction Benchmark — *S. pneumoniae* D39V

Suite unificada para extracción de datasets genómicos, evaluación comparativa de herramientas de predicción de promotores, y monitoreo de recursos computacionales (CPU/GPU/RAM).

**TFM** — José Miguel Fierro Bustos · EMBL-EBI · Lees Group  
Supervisores: John Lees, PhD · Víctor Rodríguez Bouza, PhD

---

## Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ 1. DATASET   │───▶│ 2. BENCHMARK      │───▶│ 3. ANALYSIS          │
│  generation  │    │  (per-tool pixi)  │    │  + monitoring        │
└─────────────┘    └──────────────────┘    └─────────────────────┘
       │                    │                        │
  positive_tss.py     PromoTech RF-HOT        master_benchmark_roc.svg
  negatives_tss.py    PromoTech RF-TETRA      master_benchmark_siga.svg
       │              PromoterLCNN (TF)       master_benchmark_sigx.svg
       ▼              iPro-MP (23 spp)        resource_comparison.svg
  81bp FASTA files    MLDSPP XGBoost          resource_scalability.svg
  (988 pos + 1000 neg)                        resource_metrics.tsv
```

---

## Quick Start

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and install
git clone https://github.com/JoseFierroB/promoter-tools.git
cd promoter-tools
pixi install

# Generate datasets
pixi run python src/dataset/positive_tss.py \
  --gff data/reference/D39V_annotation_TSS_Victor.gff \
  --fasta data/reference/D39V.fna \
  --gff-cds data/reference/sequence.gff3 \
  -u 60 -d 20 -o output/

pixi run python src/dataset/negatives_tss_master.py \
  --gff-cds data/reference/sequence.gff3 \
  --fasta data/reference/D39V.fna \
  --gff-tss data/reference/D39V_annotation_TSS_Victor.gff \
  --window 81 --limit 1000 -o output/

# Run benchmark with monitoring
pixi run python -c "
from src.benchmark.orchestrator import Orchestrator
Orchestrator(monitor=True).run_all()
"
```

---

## Project Structure

```
├── src/
│   ├── benchmark/     tools.py (registry), orchestrator.py (execution)
│   ├── monitoring/    profiler.py (GPU/CPU/VRAM), reporter.py (TSV+plots)
│   ├── dataset/       positive_tss.py, negatives_tss_master.py
│   ├── experiments/   mldspp_full_evaluation.py, benchmark_resources.py
│   └── genome/        genome_wide_scanner.py
│
├── tools/             (each with own pixi.toml)
│   ├── Promotech/       RF-HOT + RF-TETRA (3.6 GB models)
│   ├── Promoters/       PromoterLCNN (TensorFlow, 4 MB)
│   ├── iPro-MP/         DNABERT-6, 23 species (440 MB each)
│   ├── MLDSPP-Promoter-prediction/  SantaLucia features + XGBoost
│   └── (evaluated & rejected: bTSSFinder, CNNProm — see MEMORIA.md §4.6)
│
├── data/
│   ├── benchmark/      positives_81bp.fasta (988), negatives_81bp.fasta (1000)
│   └── reference/      D39V.fna (2 Mbp), GFF annotations
│
├── output/
│   ├── plots/          master_benchmark_roc.{svg,png}, stability_profiles_comparison.*
│   ├── tables/         resource_metrics.tsv, mldspp_metrics.tsv
│   └── predictions/    per-tool prediction CSVs
│
├── submit/             Slurm job submission scripts
├── MEMORIA.md          Living development log & decision record
└── pixi.toml           Root dependencies
```

---

## Benchmark Results

| # | Model | AUC | Type | GPU |
|---|-------|-----|------|-----|
| 1 | MLDSPP XGBoost (retrained) | **0.970** | ML | — |
| 2 | iPro-MP (H. pylori) | **0.962** | DL | ✓ |
| 3 | iPro-MP (C. jejuni) | **0.962** | DL | ✓ |
| 4 | iPro-MP (S. aureus) | **0.961** | DL | ✓ |
| 5 | PromoterLCNN | 0.953 | DL | ✓ |
| 6 | PromoTech RF-HOT (PG Max) | 0.948 | ML | — |
| 7 | PromoTech RF-TETRA (PG Max) | 0.925 | ML | — |
| 8 | MEME/FIMO (motif baseline) | 0.915 | Other | — |
| — | MLDSPP cross-species (no local training) | 0.80–0.83 | ML | — |

*MLDSPP retrained with real CDS negatives: 5-fold CV AUC 0.9725, independent 4k-negative test 0.9715 ± 0.0074. Pure cross-species transfer (zero S. pneumoniae data in training) only reaches 0.80–0.83 — local data is required.*
*Descartados: bTSSFinder (incompatible), CNNProm (no reproducible), iProL (superseded).*

---

## Resource Benchmark (CPU, 1988 seqs, 81bp)

| Tool | Time | RAM | Model | GPU speedup |
|------|------|-----|-------|-------------|
| MLDSPP XGBoost | 0.01s | <1 MB | <1 MB | — |
| CNNProm | 0.72s | 147 MB | 12 MB | — |
| PromoterLCNN | 1.3s | 187 MB | 4 MB | — |
| PromoTech RF-HOT | 101s | ~200 MB | 3.6 GB | — |
| iPro-MP (1 sp) | 27 min | 2.6 GB | 440 MB | **14× (A100)** |

*GPU benchmark: Lambda A100-SXM4-40GB. 23 especies GPU: ~8 min vs ~10h CPU.*

---

## Adding/Removing Tools

```python
from src.benchmark.tools import PROMOTER_TOOLS, enable, disable

# Run only specific tools
enable(["lcnn", "promotech_hot"])
disable(["ipromp_sp12"])

# All registered tools
for k, t in PROMOTER_TOOLS.items():
    print(f"{k}: {t.name} | GPU: {t.gpu_capable}")
```

---

## GPU Deployment

Deploy on any cloud GPU instance (Lambda, AWS, GCP):

```bash
# Clone the clean deployment
git clone https://github.com/JoseFierroB/promoter-tools.git
cd promoter-tools

# Download iPro-MP species models (Zenodo) + DNABERT-6 base
# → https://doi.org/10.5281/zenodo.15180138  (place under tools/iPro-MP/07-final and tools/iPro-MP/DNABERT-6)

# Install CUDA-enabled torch in the iPro-MP pixi env
pixi run --manifest-path tools/iPro-MP/pixi.toml -e ipro-mp pip install torch --index-url https://download.pytorch.org/whl/cu126

# Run GPU prediction (species 12 = H. pylori, best for S. pneumoniae)
pixi run --manifest-path tools/iPro-MP/pixi.toml -e ipro-mp \
  python tools/iPro-MP/iPro-MP_predict.py -i data/benchmark/positives_81bp.fasta -s 12 -o ipromp_12.csv
```

---

## Key Decisions Log (see MEMORIA.md for full details)

| Tool | Decision | Reason |
|------|----------|--------|
| MLDSPP | Adopted (retrained) | Real CDS negatives → AUC 0.97; pure cross-species only 0.80–0.83 |
| bTSSFinder | Rejected | Requires ≥300bp, 0 TSS at 81bp |
| CNNProm | Rejected | Web-only, no downloadable model |
| iProL | Dropped | Heavy Longformer; superseded by iPro-MP |
| PromoTech PG | Adopted | Max over 42 sliding windows → AUC 0.95 |
| MEME/FIMO | Done | Motif baseline: STREME motif + FIMO scan → AUC 0.915 |
