# IGR Experiments — Intergenic Regions & Specialized Niches

**Location:** `experiments/igr/` · **Status:** experimental (not consolidated)
**Datasets:** `data/benchmark_igr/` (versioned in Git)

Self-contained modules for analyzing promoters in refined intergenic regions
(IGRs) of *Streptococcus pneumoniae* (D39V and TIGR4) and specialized niches
(CDS-internal promoters and 1:1 ortholog pairs). These modules reuse the same
runners and CLI as the canonical benchmark; only the dataset changes.

---

## 1. Modules

```
experiments/igr/
├── extract_intergenic_regions_refined.py  # Refined IGRs (11 feature types, circular genome)
├── build_d39v_igr.py                      # D39V IGR benchmark: 723 pos / 723 neg
├── build_tigr4_igr.py                     # TIGR4: 4 subsets (553/578/971/1009)
├── build_cds_ortho.py                     # Niches: CDS-internal + 1:1 orthologs
├── cluster_igrs.py                        # Cross-strain MMseqs2 clusters (2,247)
├── process_results.py                     # AUC/ACC/MCC metrics + ROC from predictions_igr
└── sigma_roc.py                           # Sigma-stratified ROC (SigA/None/SigX)
```

---

## 2. Execution

### Step 0 — Refined IGRs (input for all builders)

```bash
python experiments/igr/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v --circular
```

### Step 1 — Build datasets

```bash
python experiments/igr/build_d39v_igr.py    # → data/benchmark_igr/d39v (723/723 + SigA/SigX)
python experiments/igr/build_tigr4_igr.py   # → data/benchmark_igr/tigr4 (4 subsets)
python experiments/igr/build_cds_ortho.py   # → data/benchmark_cds, data/benchmark_ortho_1to1
```

### Step 2 — Benchmark (canonical CLI, configuration only)

```bash
python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 \
    --pos data/benchmark_igr/d39v/positives_81bp_igr.fasta \
    --neg data/benchmark_igr/d39v/negatives_81bp_igr.fasta \
    --output-dir output/predictions_igr/d39v \
    -o output/tables/resource_metrics_igr_d39v.tsv
```

Same `--pos/--neg` flags for the other datasets (TIGR4 subsets, CDS, ortho —
full table in `docs/RUNNING.md`).

### Step 3 — Metrics and analysis

```bash
python experiments/igr/process_results.py   # AUC/ACC/MCC + ROC
python experiments/igr/cluster_igrs.py      # cross-strain cluster tables
python experiments/igr/sigma_roc.py         # ROC by sigma factor
```

---

## 3. Datasets (canonical IGR benchmark)

| Dataset | Pos/Neg | Content |
|---|---|---|
| `data/benchmark_igr/d39v/` | 723 / 723 | D39V promoters in refined IGRs vs intergenic background |
| `.../tigr4/subset_1_high_conf_primary/` | 553 / 553 | TIGR4 high-conf primary |
| `.../tigr4/subset_2_high_conf_all/` | 578 / 578 | TIGR4 high-conf all |
| `.../tigr4/subset_3_all_primary/` | 971 / 971 | TIGR4 all primary |
| `.../tigr4/subset_4_all_comprehensive/` | 1009 / 1009 | TIGR4 all comprehensive |

MLDSPP 75/25 split (seed 42): `data/benchmark/mldspp_75_split_benchmark_igr.npz`
(542 train / 181 test).

**D39V lineage:** GFF 1003 TSS (Victor + Axel) → 989 curated (proximity <25bp)
→ 723 inside refined IGRs (266 CDS-internal excluded).

---

## 4. Conventions

- Commands work identically with or without pixi (`pixi run python ...` inside
  the environment == `python ...` with the environment activated).
- Generated outputs (not versioned): `output/predictions_igr/`,
  `output/intergenic_refined/`, `output/tables/igr_*.tsv`.
- The canonical benchmark (D39V 989 / TIGR4 tiers) lives in `data/benchmark/`
  and `data/tigr4/` — this module is a parallel experimental extension.
