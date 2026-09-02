# Experimentos IGR — Regiones Intergénicas y Nichos Especializados

**Ubicación:** `experiments/igr/` · **Estado:** experimental (no consolidado)
**Datasets:** `data/benchmark_igr/` (versionados en Git)

Módulos autocontenidos para el análisis de promotores en regiones intergénicas
(IGRs) de *Streptococcus pneumoniae* (D39V y TIGR4) y nichos especializados
(promotores internos de CDS y pares de ortólogos 1:1). Reutilizan los mismos
runners y CLI del benchmark canónico; solo cambia el dataset.

---

## 1. Módulos

```
experiments/igr/
├── extract_intergenic_regions_refined.py  # IGRs refinadas (11 tipos de features, genoma circular)
├── build_d39v_igr.py                      # D39V IGR benchmark: 723 pos / 723 neg
├── build_tigr4_igr.py                     # TIGR4: 4 subsets (553/578/971/1009)
├── build_cds_ortho.py                     # Nichos: CDS interno + ortólogos 1:1
├── cluster_igrs.py                        # Clusters cross-strain MMseqs2 (2,247)
├── process_results.py                     # Métricas AUC/ACC/MCC + ROC desde predictions_igr
└── sigma_roc.py                           # ROC estratificado por sigma (SigA/None/SigX)
```

---

## 2. Ejecución

### Paso 0 — IGRs refinadas (input de todos los builders)

```bash
python experiments/igr/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v --circular
```

### Paso 1 — Construir datasets

```bash
python experiments/igr/build_d39v_igr.py    # → data/benchmark_igr/d39v (723/723 + SigA/SigX)
python experiments/igr/build_tigr4_igr.py   # → data/benchmark_igr/tigr4 (4 subsets)
python experiments/igr/build_cds_ortho.py   # → data/benchmark_cds, data/benchmark_ortho_1to1
```

### Paso 2 — Benchmark (CLI canónico, solo configuración)

```bash
python src/cli.py run meme fimo_prok mldspp mldspp_75 lcnn promotech_hot ipromp_sp12 \
    --pos data/benchmark_igr/d39v/positives_81bp_igr.fasta \
    --neg data/benchmark_igr/d39v/negatives_81bp_igr.fasta \
    --output-dir output/predictions_igr/d39v \
    -o output/tables/resource_metrics_igr_d39v.tsv
```

Mismos flags `--pos/--neg` para los demás datasets (TIGR4 subsets, CDS, ortho —
tabla completa en `docs/RUNNING.md`).

### Paso 3 — Métricas y análisis

```bash
python experiments/igr/process_results.py   # AUC/ACC/MCC + ROC
python experiments/igr/cluster_igrs.py      # tablas de clusters cross-strain
python experiments/igr/sigma_roc.py         # ROC por factor sigma
```

---

## 3. Datasets (benchmark canónico IGR)

| Dataset | Pos/Neg | Contenido |
|---|---|---|
| `data/benchmark_igr/d39v/` | 723 / 723 | Promotores D39V en IGRs refinadas vs fondo intergénico |
| `.../tigr4/subset_1_high_conf_primary/` | 553 / 553 | TIGR4 high-conf primary |
| `.../tigr4/subset_2_high_conf_all/` | 578 / 578 | TIGR4 high-conf all |
| `.../tigr4/subset_3_all_primary/` | 971 / 971 | TIGR4 all primary |
| `.../tigr4/subset_4_all_comprehensive/` | 1009 / 1009 | TIGR4 all comprehensive |

Partición MLDSPP 75/25 (seed 42): `data/benchmark/mldspp_75_split_benchmark_igr.npz`
(542 train / 181 test).

**Linaje D39V:** GFF 1003 TSS (Victor + Axel) → 989 curados (proximidad <25bp)
→ 723 dentro de IGRs refinadas (266 internos de CDS excluidos).

---

## 4. Convenciones

- Los comandos funcionan igual con o sin pixi (`pixi run python ...` dentro del
  entorno == `python ...` con el entorno activado).
- Outputs generados (no versionados): `output/predictions_igr/`,
  `output/intergenic_refined/`, `output/tables/igr_*.tsv`.
- El benchmark canónico (D39V 989 / TIGR4 tiers) vive en `data/benchmark/` y
  `data/tigr4/` — este módulo es una extensión experimental en paralelo.
