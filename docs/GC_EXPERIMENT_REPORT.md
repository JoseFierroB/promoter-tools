# GC Experiment — Composition report and robustness analysis

> Project: Benchmark of promoter prediction tools in S. pneumoniae (D39V and TIGR4 high)
> Fecha: 21 Agosto 2026
> Objective: quantify the effect of the GC composition confounder (positives 29.9% vs CDS negatives 41.2% on D39V; 30.9% vs 42.0% on TIGR4) using GC-matched negative sets.

---

## 1. EXPERIMENT DATASETS

| Set | Positivos | Negativos | GC neg (media±SD) | n |
|-----|-----------|-----------|--------------------|---|
| D39V cds (canonical) | d39v/positives_81bp.fasta | d39v/negatives_81bp.fasta | 41.2 ± 6.3 | 988/1000 |
| D39V gc30 | same | d39v_gc/negatives_81bp_gc30.fasta | **31.7 ± 2.5** | 988/1000 |
| D39V gc33 | same | d39v_gc/negatives_81bp_gc33.fasta | **34.0 ± 2.6** | 988/1000 |
| TIGR4 cds (canonical) | tigr4/positives_high_81bp.fasta | tigr4/negatives_high_81bp.fasta | 42.0 ± 6.3 | 738/738 |
| TIGR4 gc31 | same | tigr4_gc/negatives_high_81bp_gc31.fasta | **33.0 ± 2.5** | 738/738 |

- Generados con `negatives_tss_d39v.py --target-gc` y `negatives_tss_tigr4.py --target-gc` (flag aditivo nuevo, verificado byte-identical sin flags).
- TIGR4 gc31 regenerated with `--dedup-rc` (canonical TIGR4 verified RC-dupe-free: byte-identical with and without the flag).
- No sequence overlap with positives in any set. Canonical positives untouched.
- iPro-MP: canonical 5-fold ensemble (12_fold_1..5, softmax average), exact pos/neg split.
- MEME: re-corrido con STREME `-seed 42` (determinista; d39v cds AUC pasa de 0.8414 a 0.8617).

## 2. AUC PER SET — COMPOSITION EFFECT

| Tool | D39V cds | D39V gc30 | D39V gc33 | Δgc30 | Δgc33 | TIGR4 cds | TIGR4 gc31 | Δgc31 |
|------|----------|-----------|-----------|-------|-------|-----------|------------|-------|
| iPro-MP | 0.9600 | 0.9550 | 0.9566 | −0.005 | −0.003 | 0.9037 | 0.9009 | −0.003 |
| MLDSPP75 | 0.9567 | 0.9484 | 0.9490 | −0.008 | −0.008 | 0.9234 | 0.9016 | −0.022 |
| PromoTech | 0.9431 | 0.9316 | 0.9300 | −0.012 | −0.013 | 0.9079 | 0.8788 | −0.029 |
| LCNN | 0.9487 | 0.9241 | 0.9321 | −0.025 | −0.017 | 0.9027 | 0.8636 | −0.039 |
| MEME (seed) | 0.8617 | 0.8263 | 0.8551 | −0.035 | −0.007 | 0.8326 | 0.8005 | −0.032 |
| MLDSPP | 0.8651 | 0.8608 | 0.8475 | −0.004 | −0.018 | 0.8182 | 0.7971 | −0.021 |
| FIMO | 0.7592 | 0.6486 | 0.6946 | **−0.111** | −0.065 | 0.7469 | 0.6256 | **−0.121** |

**Reading**: the GC confounder inflates AUC for ALL tools; FIMO is the most sensitive (up to −0.12: its uniform A/C/G/T background favors AT-rich regions), LCNN loses 2-4 points, and iPro-MP is the most robust (Δ ≈ 0.003-0.005, motif/positional signal).

**Ranking con GC-matched (set honesto)**: iPro-MP > MLDSPP75 > PromoTech > LCNN > MEME > MLDSPP > FIMO.

## 3. CALIBRATION (Brier)

| Tool | D39V cds | D39V gc30 | TIGR4 cds | TIGR4 gc31 |
|------|----------|-----------|-----------|------------|
| iPro-MP | 0.063 | 0.076 | 0.103 | 0.111 |
| PromoTech | 0.137 | 0.156 | 0.142 | 0.166 |
| LCNN | 0.162 | 0.210 | 0.169 | 0.216 |
| MLDSPP75 | 0.353 | 0.358 | 0.371 | 0.387 |
| MLDSPP | 0.365 | 0.362 | 0.365 | 0.367 |
| MEME | 9.08 | 9.34 | 8.15 | 9.88 |
| FIMO | 26.4 | 28.0 | 25.5 | 27.4 |

FIMO (−log10 p-value) and MEME (max over motifs) scores **are not probabilities** (Brier ≫ 1); DL/ML tools (iPro-MP, PromoTech, LCNN) are the best calibrated. For real probabilistic use, only iPro-MP/PromoTech/LCNN are interpretable after isotonic calibration.

## 4. CONSERVATION WITH GC-MATCHED (D39V, canonical classes: 647/157/184)

| Tool | cds Cons | cds Intra | cds NonC | gc30 Cons | gc30 Intra | gc30 NonC |
|------|----------|-----------|----------|-----------|------------|-----------|
| iPro-MP | 0.9709 | 0.9125 | 0.9704 | 0.9670 | 0.9022 | 0.9673 |
| MLDSPP75 | 0.9668 | 0.9217 | 0.9564 | 0.9592 | 0.9101 | 0.9488 |
| LCNN | 0.9630 | 0.8833 | 0.9665 | 0.9448 | 0.8288 | 0.9508 |
| PromoTech | 0.9370 | 0.9485 | 0.9619 | 0.9246 | 0.9369 | 0.9541 |
| MLDSPP | 0.8811 | 0.7958 | 0.8799 | 0.8768 | 0.7932 | 0.8743 |
| MEME | 0.8584 | 0.8695 | 0.8661 | 0.8037 | 0.8144 | 0.8301 |
| FIMO | 0.7792 | 0.6915 | 0.7563 | 0.6745 | 0.5581 | 0.6479 |

The pattern holds with GC-matched: PromoTech remains the most robust to intragenic (small Δcons-intra); the intragenic drop of the others persists but is attenuated for iPro-MP/MLDSPP.

## 5. CONFUSION (Youden) — D39V canonical vs gc30

| Tool | cds TP/FN/FP/TN | cds F1 | gc30 TP/FN/FP/TN | gc30 F1 |
|------|-----------------|--------|-------------------|---------|
| iPro-MP | 892/96/31/969 | 0.934 | 875/119/42/952 | 0.916 |
| PromoTech | 868/120/64/936 | 0.904 | 847/141/54/946 | 0.897 |
| LCNN | 838/150/35/965 | 0.901 | 819/169/49/951 | 0.883 |
| MLDSPP75 | 878/110/111/889 | 0.888 | 881/107/126/874 | 0.883 |
| MEME | 788/200/133/867 | 0.826 | 750/238/214/786 | 0.768 |
| MLDSPP | 821/167/246/754 | 0.799 | 810/178/249/751 | 0.791 |
| FIMO | 665/323/286/714 | 0.686 | 496/492/271/729 | 0.565 |

## 6. PAIRED DELONG (d39v cds vs gc30) — Holm-corrected

Significant after Holm (p<0.05): ΔAUC differences of **FIMO vs all** and **MEME vs all**; LCNN vs PromoTech (p=0.020), LCNN vs MLDSPP75 (p=0.022), LCNN vs MEME (p=0.030). iPro-MP vs PromoTech (p=0.296) and the rest of the top do NOT differ significantly from each other.

## 7. DUPLICATES IN POSITIVES (d39v: 988 entries, 972 unique, 16 dupes in 9 groups)

| Tool | AUC 988 | AUC 972 unique | Δ |
|------|---------|----------------|---|
| LCNN | 0.9487 | 0.9534 | +0.005 |
| iPro-MP | 0.9600 | 0.9637 | +0.004 |
| MLDSPP | 0.8651 | 0.8684 | +0.003 |
| MLDSPP75 | 0.9567 | 0.9576 | +0.001 |
| PromoTech | 0.9431 | 0.9426 | −0.001 |
| FIMO | 0.7592 | 0.7608 | +0.002 |
| MEME | 0.8617 | 0.8618 | +0.000 |

Negligible impact (≤0.005). Canonical deduplication is by position (chrom,pos,strand), not by sequence identity.

## 8. AUC BY SIGMA FACTOR (D39V)

| Tool | cds None(570) | cds SigA(397) | cds SigX(21) | gc30 None | gc30 SigA | gc30 SigX |
|------|---------------|---------------|--------------|-----------|-----------|-----------|
| iPro-MP | 0.9424 | 0.9877 | 0.9133 | 0.9369 | 0.9857 | 0.8645 |
| MLDSPP75 | 0.9453 | 0.9737 | 0.9482 | 0.9360 | 0.9667 | 0.9388 |
| LCNN | 0.9233 | 0.9894 | 0.8705 | 0.8896 | 0.9822 | 0.7625 |
| PromoTech | 0.9382 | 0.9503 | 0.9390 | 0.9256 | 0.9410 | 0.9171 |
| MLDSPP | 0.8487 | 0.8885 | 0.8648 | 0.8450 | 0.8836 | 0.8617 |
| MEME | 0.8459 | 0.8675 | 0.8422 | 0.7980 | 0.8289 | 0.7733 |
| FIMO | 0.7405 | 0.7882 | 0.7196 | 0.6209 | 0.6900 | 0.6152 |

SigA is the easiest class (AUC 0.95-0.99); SigX (n=21) is noisy. With GC-matched the drops are small for iPro-MP/MLDSPP75.

## 9. ON-DISK INTERMEDIATES (workdirs)

| Set | PromoTech workdir | iPro-MP dir |
|-----|-------------------|-------------|
| D39V cds (1988 seqs) | 143 MB | 1 MB |
| TIGR4 cds (1476 seqs) | 105 MB | 1 MB |

PromoTech domina: ~54 KB/secuencia (matrices joblib 42n×160 float64 ×2 hebras + genome_predictions.csv) → **~42 GB estimados @197600** (n=395,200). El resto de tools no dejan intermedios significativos.

## 10. DECISIONS AND CHANGES THIS SESSION

- **Canonical untouched**: data/benchmark/d39v/*, data/tigr4/*, output/predictions, REPORT.md unmodified.
- **Code changes** (pending commit):
  1. `src/runners/ipromp_sp12.py` — ensemble 5-fold (canonical iPro-MP)
  2. `src/runners/meme.py` — STREME `-seed 42` (determinista)
  3. `src/dataset/negatives_tss_tigr4.py` — flags aditivos `--target-gc`/`--gc-tolerance`/`--dedup-rc` (byte-identical sin flags)
  4. `src/backend/slurm.py` — fix imports NameError latente
  5. `src/analysis/statistics.py` — aggregate_runs conserva VRAM/CPU/GPU
  6. `src/analysis/scaling_analysis.py` — projection bootstrap CIs (time_s_ci_low/high in extrapolation.tsv)
  7. `data/benchmark/mldspp_75_split_scale_db_988.npz` — split 988
- **Excluded from future reports**: PromoTech TETRA, FIMO E. coli DB, Shimada validation (by decision).
- **Pending**: sp12 vs s23 benchmark (s23 model on WS/NFS); uniform measurement protocol (pure time_s) + scaling re-runs.
- **Scaling**: LCNN (n^0.19) and MLDSPP (speedup 1.74×@988) fits are load/training artifacts included in time_s — pending re-measurement with a uniform protocol (honest models: LCNN ≈ 1.8 s + 9 µs·n; MLDSPP true inference speedup ≈ 1.06×).

## 11. GENERATED FILES

- Datasets: `data/benchmark/d39v_gc/`, `data/tigr4_gc/`
- Predicciones: `output/d39v_gc/{cds,gc30,gc33}/predictions/`, `output/tigr4_gc/{cds,gc31}/predictions/` (WS) y `~/Desktop/{d39v_gc,tigr4_gc}/` (espejo local)
- Analysis: `output/gc_analysis/full_analysis.log`, `output/gc_analysis/calibration_brier.tsv`
- Projection CIs: `output/tables/extrapolation.tsv` (+time_s_ci_low/high)
## 12. ERROR ANALYSIS AND CONSENSUS (added)

### Consensus (score averaging: ranks / vote / isotonic)

| Set | Mejor individual | CONS_rank | CONS_vote | CONS_iso | Lift (iso) |
|-----|------------------|-----------|-----------|----------|------------|
| D39V cds | iPro-MP 0.9600 | 0.9913 | 0.9888 | **0.9946** | +0.035 |
| D39V gc30 | iPro-MP 0.9550 | 0.9840 | 0.9835 | **0.9918** | +0.037 |
| D39V gc33 | iPro-MP 0.9566 | 0.9862 | 0.9851 | **0.9928** | +0.036 |
| TIGR4 cds | MLDSPP75 0.9234 | 0.9679 | 0.9739 | **0.9859** | +0.063 |
| TIGR4 gc31 | iPro-MP 0.9009 | 0.9430 | 0.9566 | **0.9768** | +0.075 |

**El consenso de las 7 tools supera a la mejor individual en +0.03 a +0.08 AUC** (el consenso iso es in-sample, optimista; CONS_rank es la cota honesta: +0.03-0.04).

### Error correlation (phi, D39V cds)

Errors are mostly **independent across tools** (phi 0.02-0.62): LCNN-iPro-MP 0.62 (both DL), MLDSPP-MLDSPP75 0.49, LCNN-MLDSPP75 0.31; most pairs < 0.1 → tools fail on different cases → complementarity explaining the consensus.

### Hard cases (positives missed by ALL tools)

- D39V (cds/gc30/gc33): **a single positive** (conserved, sigma None, GC 43.2% vs 29.9% mean — an atypical GC-rich promoter that confuses composition-based tools)
- TIGR4: 5-6 positivos
- **No negative** is a false positive for all tools

### Plot fixes (hardcodes removed)

- `generate_master_plots.py`: time/RAM/VRAM/weights now read from campaign metrics (previously hardcoded and stale: LCNN RAM 337.1 vs actual 1839.9; PromoTech 96.4 vs 106.0; LCNN VRAM 497.2 vs actual 2072-5169); `n_pos` from config; parameterizable paths (`--data-dir`, `--desktop-out`)
- `resource_plots.py`: `--data-dir`/`--out-dir` parametrizables
- `generate_master_roc.py`: etiqueta FIMO sin hardcode "838"
