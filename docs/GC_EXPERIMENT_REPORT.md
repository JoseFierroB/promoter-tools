# Experimento GC — Reporte de composición y análisis de robustez

> Proyecto: Benchmark de herramientas de predicción de promotores en S. pneumoniae (D39V y TIGR4 high)
> Fecha: 21 Agosto 2026
> Objetivo: cuantificar el efecto del confundidor de composición GC (positivos 29.9% vs negativos CDS 41.2% en D39V; 30.9% vs 42.0% en TIGR4) mediante sets de negativos emparejados por GC.

---

## 1. DATASETS DEL EXPERIMENTO

| Set | Positivos | Negativos | GC neg (media±SD) | n |
|-----|-----------|-----------|--------------------|---|
| D39V cds (canónico) | d39v/positives_81bp.fasta | d39v/negatives_81bp.fasta | 41.2 ± 6.3 | 988/1000 |
| D39V gc30 | ídem | d39v_gc/negatives_81bp_gc30.fasta | **31.7 ± 2.5** | 988/1000 |
| D39V gc33 | ídem | d39v_gc/negatives_81bp_gc33.fasta | **34.0 ± 2.6** | 988/1000 |
| TIGR4 cds (canónico) | tigr4/positives_high_81bp.fasta | tigr4/negatives_high_81bp.fasta | 42.0 ± 6.3 | 738/738 |
| TIGR4 gc31 | ídem | tigr4_gc/negatives_high_81bp_gc31.fasta | **33.0 ± 2.5** | 738/738 |

- Generados con `negatives_tss_d39v.py --target-gc` y `negatives_tss_tigr4.py --target-gc` (flag aditivo nuevo, verificado byte-identical sin flags).
- TIGR4 gc31 regenerado con `--dedup-rc` (canónico TIGR4 verificado sin RC-dupes: byte-identical con y sin el flag).
- Sin solape de secuencias con positivos en ningún set. Positivos canónicos intactos.
- iPro-MP: ensemble 5-fold canónico (12_fold_1..5, promedio de softmax), split pos/neg exacto.
- MEME: re-corrido con STREME `-seed 42` (determinista; d39v cds AUC pasa de 0.8414 a 0.8617).

## 2. AUC POR SET — EFECTO COMPOSICIÓN

| Tool | D39V cds | D39V gc30 | D39V gc33 | Δgc30 | Δgc33 | TIGR4 cds | TIGR4 gc31 | Δgc31 |
|------|----------|-----------|-----------|-------|-------|-----------|------------|-------|
| iPro-MP | 0.9600 | 0.9550 | 0.9566 | −0.005 | −0.003 | 0.9037 | 0.9009 | −0.003 |
| MLDSPP75 | 0.9567 | 0.9484 | 0.9490 | −0.008 | −0.008 | 0.9234 | 0.9016 | −0.022 |
| PromoTech | 0.9431 | 0.9316 | 0.9300 | −0.012 | −0.013 | 0.9079 | 0.8788 | −0.029 |
| LCNN | 0.9487 | 0.9241 | 0.9321 | −0.025 | −0.017 | 0.9027 | 0.8636 | −0.039 |
| MEME (seed) | 0.8617 | 0.8263 | 0.8551 | −0.035 | −0.007 | 0.8326 | 0.8005 | −0.032 |
| MLDSPP | 0.8651 | 0.8608 | 0.8475 | −0.004 | −0.018 | 0.8182 | 0.7971 | −0.021 |
| FIMO | 0.7592 | 0.6486 | 0.6946 | **−0.111** | −0.065 | 0.7469 | 0.6256 | **−0.121** |

**Lectura**: el confundidor GC infla el AUC de TODAS las tools; FIMO es el más sensible (hasta −0.12: su fondo uniforme A/C/G/T premia regiones AT-ricas), LCNN pierde 2-4 puntos, e iPro-MP es el más robusto (Δ ≈ 0.003-0.005, señal motívica/posicional).

**Ranking con GC-matched (set honesto)**: iPro-MP > MLDSPP75 > PromoTech > LCNN > MEME > MLDSPP > FIMO.

## 3. CALIBRACIÓN (Brier)

| Tool | D39V cds | D39V gc30 | TIGR4 cds | TIGR4 gc31 |
|------|----------|-----------|-----------|------------|
| iPro-MP | 0.063 | 0.076 | 0.103 | 0.111 |
| PromoTech | 0.137 | 0.156 | 0.142 | 0.166 |
| LCNN | 0.162 | 0.210 | 0.169 | 0.216 |
| MLDSPP75 | 0.353 | 0.358 | 0.371 | 0.387 |
| MLDSPP | 0.365 | 0.362 | 0.365 | 0.367 |
| MEME | 9.08 | 9.34 | 8.15 | 9.88 |
| FIMO | 26.4 | 28.0 | 25.5 | 27.4 |

Los scores de FIMO (−log10 p-valor) y MEME (max sobre motivos) **no son probabilidades** (Brier ≫ 1); los DL/ML (iPro-MP, PromoTech, LCNN) son los mejor calibrados. Para uso probabilístico real, solo iPro-MP/PromoTech/LCNN son interpretables tras calibración isotónica.

## 4. CONSERVACIÓN CON GC-MATCHED (D39V, clases canónicas: 647/157/184)

| Tool | cds Cons | cds Intra | cds NonC | gc30 Cons | gc30 Intra | gc30 NonC |
|------|----------|-----------|----------|-----------|------------|-----------|
| iPro-MP | 0.9709 | 0.9125 | 0.9704 | 0.9670 | 0.9022 | 0.9673 |
| MLDSPP75 | 0.9668 | 0.9217 | 0.9564 | 0.9592 | 0.9101 | 0.9488 |
| LCNN | 0.9630 | 0.8833 | 0.9665 | 0.9448 | 0.8288 | 0.9508 |
| PromoTech | 0.9370 | 0.9485 | 0.9619 | 0.9246 | 0.9369 | 0.9541 |
| MLDSPP | 0.8811 | 0.7958 | 0.8799 | 0.8768 | 0.7932 | 0.8743 |
| MEME | 0.8584 | 0.8695 | 0.8661 | 0.8037 | 0.8144 | 0.8301 |
| FIMO | 0.7792 | 0.6915 | 0.7563 | 0.6745 | 0.5581 | 0.6479 |

El patrón se mantiene con GC-matched: PromoTech sigue siendo el más robusto a intragénicos (Δcons-intra pequeño); la caída intragénica de las demás persiste pero se atenúa para iPro-MP/MLDSPP.

## 5. CONFUSIÓN (Youden) — D39V canónico vs gc30

| Tool | cds TP/FN/FP/TN | cds F1 | gc30 TP/FN/FP/TN | gc30 F1 |
|------|-----------------|--------|-------------------|---------|
| iPro-MP | 892/96/31/969 | 0.934 | 875/119/42/952 | 0.916 |
| PromoTech | 868/120/64/936 | 0.904 | 847/141/54/946 | 0.897 |
| LCNN | 838/150/35/965 | 0.901 | 819/169/49/951 | 0.883 |
| MLDSPP75 | 878/110/111/889 | 0.888 | 881/107/126/874 | 0.883 |
| MEME | 788/200/133/867 | 0.826 | 750/238/214/786 | 0.768 |
| MLDSPP | 821/167/246/754 | 0.799 | 810/178/249/751 | 0.791 |
| FIMO | 665/323/286/714 | 0.686 | 496/492/271/729 | 0.565 |

## 6. DELONG PAREADO (d39v cds vs gc30) — Holm corregido

Significativos tras Holm (p<0.05): diferencias de ΔAUC de **FIMO vs todos** y de **MEME vs todos**; LCNN vs PromoTech (p=0.020), LCNN vs MLDSPP75 (p=0.022), LCNN vs MEME (p=0.030). iPro-MP vs PromoTech (p=0.296) y el resto del top NO difieren significativamente entre sí.

## 7. DUPLICADOS EN POSITIVOS (d39v: 988 entradas, 972 únicas, 16 dupes en 9 grupos)

| Tool | AUC 988 | AUC 972 únicas | Δ |
|------|---------|----------------|---|
| LCNN | 0.9487 | 0.9534 | +0.005 |
| iPro-MP | 0.9600 | 0.9637 | +0.004 |
| MLDSPP | 0.8651 | 0.8684 | +0.003 |
| MLDSPP75 | 0.9567 | 0.9576 | +0.001 |
| PromoTech | 0.9431 | 0.9426 | −0.001 |
| FIMO | 0.7592 | 0.7608 | +0.002 |
| MEME | 0.8617 | 0.8618 | +0.000 |

Impacto despreciable (≤0.005). La deduplicación canónica es por posición (chrom,pos,strand), no por identidad de secuencia.

## 8. AUC POR SIGMA FACTOR (D39V)

| Tool | cds None(570) | cds SigA(397) | cds SigX(21) | gc30 None | gc30 SigA | gc30 SigX |
|------|---------------|---------------|--------------|-----------|-----------|-----------|
| iPro-MP | 0.9424 | 0.9877 | 0.9133 | 0.9369 | 0.9857 | 0.8645 |
| MLDSPP75 | 0.9453 | 0.9737 | 0.9482 | 0.9360 | 0.9667 | 0.9388 |
| LCNN | 0.9233 | 0.9894 | 0.8705 | 0.8896 | 0.9822 | 0.7625 |
| PromoTech | 0.9382 | 0.9503 | 0.9390 | 0.9256 | 0.9410 | 0.9171 |
| MLDSPP | 0.8487 | 0.8885 | 0.8648 | 0.8450 | 0.8836 | 0.8617 |
| MEME | 0.8459 | 0.8675 | 0.8422 | 0.7980 | 0.8289 | 0.7733 |
| FIMO | 0.7405 | 0.7882 | 0.7196 | 0.6209 | 0.6900 | 0.6152 |

SigA es la clase más fácil (AUC 0.95-0.99); SigX (n=21) es ruidosa. Con GC-matched las caídas son pequeñas para iPro-MP/MLDSPP75.

## 9. INTERMEDIOS EN DISCO (workdirs)

| Set | PromoTech workdir | iPro-MP dir |
|-----|-------------------|-------------|
| D39V cds (1988 seqs) | 143 MB | 1 MB |
| TIGR4 cds (1476 seqs) | 105 MB | 1 MB |

PromoTech domina: ~54 KB/secuencia (matrices joblib 42n×160 float64 ×2 hebras + genome_predictions.csv) → **~42 GB estimados @197600** (n=395,200). El resto de tools no dejan intermedios significativos.

## 10. DECISIONES Y CAMBIOS DE ESTA SESIÓN

- **Canónico intacto**: data/benchmark/d39v/*, data/tigr4/*, output/predictions, REPORT.md no modificados.
- **Cambios de código** (pendientes de commit):
  1. `src/runners/ipromp_sp12.py` — ensemble 5-fold (canónico iPro-MP)
  2. `src/runners/meme.py` — STREME `-seed 42` (determinista)
  3. `src/dataset/negatives_tss_tigr4.py` — flags aditivos `--target-gc`/`--gc-tolerance`/`--dedup-rc` (byte-identical sin flags)
  4. `src/backend/slurm.py` — fix imports NameError latente
  5. `src/analysis/statistics.py` — aggregate_runs conserva VRAM/CPU/GPU
  6. `src/analysis/scaling_analysis.py` — CIs bootstrap de proyección (time_s_ci_low/high en extrapolation.tsv)
  7. `data/benchmark/mldspp_75_split_scale_db_988.npz` — split 988
- **Excluidos de reportes futuros**: PromoTech TETRA, FIMO E. coli DB, validación Shimada (por decisión).
- **Pendiente**: benchmark sp12 vs s23 (modelo s23 en WS/NFS); protocolo de medición uniforme (time_s puro) + re-runs de escalado.
- **Escalado**: los fits de LCNN (n^0.19) y MLDSPP (speedup 1.74×@988) son artefactos de load/training incluidos en time_s — pendiente re-medición con protocolo uniforme (los modelos honestos: LCNN ≈ 1.8 s + 9 µs·n; MLDSPP speedup real de inferencia ≈ 1.06×).

## 11. ARCHIVOS GENERADOS

- Datasets: `data/benchmark/d39v_gc/`, `data/tigr4_gc/`
- Predicciones: `output/d39v_gc/{cds,gc30,gc33}/predictions/`, `output/tigr4_gc/{cds,gc31}/predictions/` (WS) y `~/Desktop/{d39v_gc,tigr4_gc}/` (espejo local)
- Análisis: `output/gc_analysis/full_analysis.log`, `output/gc_analysis/calibration_brier.tsv`
- CIs de proyección: `output/tables/extrapolation.tsv` (+time_s_ci_low/high)
## 12. ERROR ANALYSIS Y CONSENSO (añadido)

### Consenso (media de scores: ranks / voto / isotónico)

| Set | Mejor individual | CONS_rank | CONS_vote | CONS_iso | Lift (iso) |
|-----|------------------|-----------|-----------|----------|------------|
| D39V cds | iPro-MP 0.9600 | 0.9913 | 0.9888 | **0.9946** | +0.035 |
| D39V gc30 | iPro-MP 0.9550 | 0.9840 | 0.9835 | **0.9918** | +0.037 |
| D39V gc33 | iPro-MP 0.9566 | 0.9862 | 0.9851 | **0.9928** | +0.036 |
| TIGR4 cds | MLDSPP75 0.9234 | 0.9679 | 0.9739 | **0.9859** | +0.063 |
| TIGR4 gc31 | iPro-MP 0.9009 | 0.9430 | 0.9566 | **0.9768** | +0.075 |

**El consenso de las 7 tools supera a la mejor individual en +0.03 a +0.08 AUC** (el consenso iso es in-sample, optimista; CONS_rank es la cota honesta: +0.03-0.04).

### Correlación de errores (phi, D39V cds)

Los errores son mayormente **independientes entre tools** (phi 0.02-0.62): LCNN-iPro-MP 0.62 (ambas DL), MLDSPP-MLDSPP75 0.49, LCNN-MLDSPP75 0.31; la mayoría de pares < 0.1 → las tools fallan en casos distintos → complementariedad que explica el consenso.

### Casos difíciles (positivos que fallan TODAS las tools)

- D39V (cds/gc30/gc33): **1 solo positivo** (conservado, sigma None, GC 43.2% vs media 29.9% — un promotor GC-rico atípico que confunde a las tools composicionales)
- TIGR4: 5-6 positivos
- **Ningún negativo** es falso positivo de todas las tools

### Fix de gráficos (hardcodes eliminados)

- `generate_master_plots.py`: tiempos/RAM/VRAM/pesos ahora se leen de las métricas de campaña (antes hardcodeados y obsoletos: LCNN RAM 337.1 vs real 1839.9; PromoTech 96.4 vs 106.0; VRAM LCNN 497.2 vs real 2072-5169); `n_pos` desde config; paths parametrizables (`--data-dir`, `--desktop-out`)
- `resource_plots.py`: `--data-dir`/`--out-dir` parametrizables
- `generate_master_roc.py`: etiqueta FIMO sin hardcode "838"
