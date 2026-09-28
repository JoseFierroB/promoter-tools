# Promoter-Tools — Memory & Key Findings

> Project: Benchmark of 9 tools for promoter prediction in S. pneumoniae D39V+TIGR4 high
> Report date: September 2026 (260928). Canonical dataset N=3,465 (1,727 pos + 1,738 neg),
> canonical regime 16cpu-gpu (RTX 5090), run 260917. All AUCs re-verified from prediction TSVs.
> Operating points: per-tool Youden thresholds (`3_tables/operating_point_metrics{,_sigma,_strain}.tsv`).
> Historical sections (§4–§6b, August 2026 IGR era) kept unchanged below.
> Estado: benchmark consolidado; pendiente deck rebuild + scale_200k.

---

## 1. BENCHMARK: AUC per tool (D39V+TIGR4 high, N=3,465, 16cpu-gpu 260917)

| Tool | AUC | F1 (Youden) | MCC (Youden) |
|------|-----|-------------|--------------|
| ProkBERT-mini [gLM] | 0.9493 | 0.9226 | 0.8531 |
| MLDSPP 75% [BDT] | 0.9410 | 0.8773 | 0.7590 |
| iPro-MP [gLM] | 0.9358 | 0.9121 | 0.8381 |
| PromoterLCNN [CNN] | 0.9286 | 0.8918 | 0.7961 |
| PromoTech RF-HOT [RF] | 0.9267 | 0.8904 | 0.7960 |
| STREME+FIMO [motif] | 0.8544 | 0.8141 | 0.6510 |
| prompt [NN] | 0.8516 | 0.7655 | 0.5647 |
| MLDSPP 0% [BDT] | 0.8447 | 0.7746 | 0.5437 |
| FIMO ProkDB [PWMs] | 0.7533 | 0.6727 | 0.3754 |

4-regime congruence (1cpu/16cpu/1cpu-gpu/16cpu-gpu): AUC spread 0.000000 on all 9 tools.

### By sigma (16cpu-gpu 260917)

| Tool | SigA (675) | ComX (27) | None (1025) |
|------|-----------|-----------|-------------|
| ProkBERT-mini [gLM] | 0.9911 | 0.9419 | 0.9220 |
| PromoterLCNN [CNN] | 0.9906 | 0.8822 | 0.8890 |
| iPro-MP [gLM] | 0.9884 | 0.9011 | 0.9021 |
| MLDSPP 75% [BDT] | 0.9758 | 0.9002 | 0.9191 |
| PromoTech RF-HOT [RF] | 0.9287 | 0.8725 | 0.9268 |

ComX n=27: wide confidence intervals apply.

### By strain (Youden OP cuts, 16cpu-gpu 260917)

| Tool | D39V AUC (989/1000) | TIGR4 AUC (738/738) |
|------|--------------------|--------------------|
| ProkBERT-mini [gLM] | 0.9700 | 0.9209 |
| MLDSPP 75% [BDT] | 0.9533 | 0.9246 |
| iPro-MP [gLM] | 0.9594 | 0.9037 |
| PromoterLCNN [CNN] | 0.9481 | 0.9027 |
| PromoTech RF-HOT [RF] | 0.9095 | 0.9481 |

Note: PromoTech numbers differ from the August snapshot below (0.943/0.908) —
model/code changed since; current values verified from prediction TSVs.

### August 2026 snapshot (superseded, kept for reference)

Global AUC (6 datasets, 6 main tools)

| Tool | D39V | T4 hi | T4+sec | T4 ext | T4 all | MIX |
|------|------|-------|--------|--------|--------|-----|
| **MLDSPP 75%*** | **0.9567** | **0.9234** | **0.9112** | **0.8077** | **0.8062** | **0.9436** |
| iPro-MP | 0.9526 | 0.905 | 0.8897 | 0.7127 | 0.7045 | 0.9324 |
| LCNN | 0.9487 | 0.9027 | 0.8873 | 0.7116 | 0.7037 | 0.929 |
| PromoTech | 0.9431 | 0.9079 | 0.8953 | 0.7145 | 0.7049 | 0.9272 |
| MLDSPP 0% | 0.8651 | 0.8182 | 0.8056 | 0.6763 | 0.6686 | 0.8449 |
| FIMO PROK | 0.7592 | 0.7469 | 0.7398 | 0.6483 | 0.6402 | 0.7535 |

*MLDSPP 75% uses 75% S. pneumoniae in training → data leakage. MLDSPP 0% is the true value.

---

## 2. CONFUSION MATRIX — Best threshold (Youden's J), 260928 regeneration

Current: `output/260917_runs/4_d39v_tigr4_high_sigma/16cpu-gpu/3_tables/`
`operating_point_metrics.tsv` (global) + `_sigma.tsv` (27 rows) + `_strain.tsv`
(18 rows: D39V 989/1000, TIGR4 738/738). All thresholds re-verified
(recomputed F1/MCC from the stored threshold match exactly). See §1 tables above.
August snapshot below kept for reference.

### D39V (989 pos, 1000 neg) — August snapshot

| Tool | AUC | TP | FN | FP | TN | Sens | Spec | F1 | Balance |
|------|-----|----|----|----|----|------|------|-----|---------|
| **iPro-MP** | 0.960 | 892 | 96 | **31** | 969 | 0.903 | 0.969 | **0.934** | ✅ Very balanced |
| **LCNN** | 0.949 | 838 | 150 | 35 | 965 | 0.848 | 0.965 | 0.901 | ✅ Balanced |
| **PromoTech HOT** | 0.943 | 868 | 120 | 64 | 936 | 0.879 | 0.936 | 0.904 | ✅ Balanced |
| PromoTech TETRA | 0.917 | 820 | 168 | 115 | 885 | 0.830 | 0.885 | 0.853 | |
| MLDSPP | 0.865 | 821 | 167 | **246** | 754 | 0.831 | 0.754 | 0.799 | ⚠️ FP excess |
| MEME | 0.841 | 751 | 237 | 139 | 861 | 0.760 | 0.861 | 0.800 | |
| FIMO PROK | 0.759 | 665 | 323 | 286 | 714 | 0.673 | 0.714 | 0.686 | ⚠️ FP+FN excess |
| FIMO DB | 0.736 | 651 | 337 | **302** | 698 | 0.659 | 0.698 | 0.671 | ⚠️ Worst balance |

### TIGR4 high (738 pos, 738 neg)

| Tool | AUC | TP | FN | FP | TN | Sens | Spec | F1 |
|------|-----|----|----|----|----|------|------|-----|
| LCNN | 0.903 | 598 | 140 | 28 | 710 | 0.810 | 0.962 | 0.877 |
| PromoTech HOT | 0.908 | 600 | 138 | 37 | 701 | 0.813 | 0.950 | 0.873 |
| PromoTech TETRA | 0.889 | 582 | 156 | 75 | 663 | 0.789 | 0.898 | 0.834 |
| MLDSPP | 0.818 | 505 | 233 | 131 | 607 | 0.684 | 0.822 | 0.735 |

### Key conclusions
- **iPro-MP and PromoTech HOT**: balanced (FP and FN even) — most reliable
- **FIMO DB/FIMO PROK**: FP excess (30% false alarm) — low specificity
- **MLDSPP**: identifies TSS well but at an FP cost (24.6% on D39V)
- **TIGR4 is harder for all tools**: lower sensitivity, more missed TSS

---

## 3. RESOURCES (time, RAM)

| Tool | Time | RAM |
|------|------|-----|
| PromoTech RF-HOT | 76.5s | 7.36 GB |
| PromoTech RF-TETRA | 58.4s | 7.36 GB |
| PromoterLCNN | 2.62s | 7.36 GB |
| MLDSPP XGBoost | 1.73s | 7.36 GB |

---

## 4. AUC BY IGR CONSERVATION

### D39V — AUC split by conservation

| Tool | Global | Conserved | Non-Cons | Intragenic | Δ (Cons-Intra) |
|------|--------|-----------|----------|------------|----------------|
| LCNN | 0.9487 | 0.9630 | 0.9665 | 0.8833 | **+0.080** |
| iPro-MP | 0.9600 | 0.9709 | 0.9704 | 0.9125 | **+0.058** |
| MLDSPP | 0.8651 | 0.8811 | 0.8799 | 0.7958 | **+0.085** |
| PromoTech HOT | 0.9431 | 0.9370 | 0.9619 | 0.9485 | −0.017 |
| MEME | 0.8414 | 0.8377 | 0.8388 | 0.8567 | −0.019 |
| FIMO DB | 0.7364 | 0.7566 | 0.7459 | 0.6574 | **+0.099** |
| FIMO PROK | 0.7592 | 0.7792 | 0.7563 | 0.6915 | **+0.088** |

**Conclusion**: All tools drop AUC on intragenic TSS. PromoTech HOT is the most robust (no drop).

---

## 5. IGR ALIGNMENT D39V ↔ TIGR4 (MMseqs2)

### Cross-strain
- **1,323 pairs** conserved D39V→TIGR4 (971/1670 IGRs, 58.1%)
- **2,240 pairs** reverse TIGR4→D39V
- **823 intersection** (reciprocal best by direction)
- Identidad media: **96.2%**, 525 pares (39.7%) al 100%
- **81.9%** syntenic by gene architecture
- **643 pairs** with orthologous flanking genes (SPV_* ↔ SP_RS*)
- **209 pairs** 100% identical with TSS in both strains

### Intra-strain
- D39V: 1,670 IGRs → 1,580 clusters (98.4% singletons)
- TIGR4: 1,784 IGRs → 1,584 clusters (97.9% singletons)
- 31 repetitive IGRs (BOX/RUP/IS) = 28.9% of cross-strain pairs

### nucmer
- 1,025 collinear blocks, 86% in genomic order
- 52.1% of MMseqs2 pairs validated by nucmer

---

## 6. TSS — POSITION AND FEASIBILITY

### D39V (989 TSS curados de 1003 brutos)
| Category | N | % |
|-----------|---|---|
| In IGR (usable) | 804 | 81.4% |
| └─ Window 100% in IGR | 307 | 31.1% |
| └─ <50% CDS overlap | 490 | 49.6% |
| In CDS | 184 | 18.6% |
| └─ Deep internal (>50bp) | 95 | 9.6% |
| **TSS usable for IGR** | **891** | **90.2%** |
| **TSS missed** | **97** | **9.8%** |

### TIGR4 (738 TSS)
| Category | N | % |
|-----------|---|---|
| En IGR (usables) | 558 | 75.6% |
| En CDS | 180 | 24.4% |
| **TSS usables** | **627** | **85.0%** |
| **TSS missed** | **111** | **15.0%** |

### Sigma factors D39V
| Sigma | Total | En IGRs conservadas | % |
|-------|-------|--------------------|----|
| SigA | 397 | 271 | 68.3% |
| SigX | 21 | 11 | 52.4% |
| None | 570 | — | — |

---

## 6b. IGR BENCHMARK — EXPERIMENTAL (not consolidated)

> **Status: experimental.** Parallel extension of the project benchmark; same
> runners and CLI, only the dataset changes. Preliminary results subject to change.

| Dataset | Pos/Neg | Content |
|---|---|---|
| D39V IGR | 723 / 723 | Promoters in refined IGRs vs intergenic background |
| TIGR4 IGR subset_1 | 553 / 553 | high-conf primary |
| TIGR4 IGR subset_2 | 578 / 578 | high-conf all |
| TIGR4 IGR subset_3 | 971 / 971 | all primary |
| TIGR4 IGR subset_4 | 1009 / 1009 | all comprehensive |

- **D39V lineage**: GFF 1003 TSS (Victor + Axel) → 989 curated (proximity <25bp) → 723 in refined IGRs.
- **Cross-strain IGR clusters (2 strains)**: 2,247 MMseqs2 clusters (1,074 1:1 pairs, 1,124 singletons, 49 multi-hit).
- **Execution**: pure CLI configuration (`--pos/--neg`), no dedicated code — see `docs/RUNNING.md`.
- **Versioned datasets**: `data/benchmark_igr/` + MLDSPP 723 split (seed 42).

---

## 7. PENDING (260928)

| # | Tarea | Estado |
|---|-------|--------|
| 1 | Deck rebuild sobre outputs verificados | pendiente |
| 2 | scale_200k (valida proyeccion ~5 min GPU) | pendiente (datos generados, seed 42) |
| 3 | Bootstrap CIs (fase 2, tras limpieza) | aparcado |
| 4 | `--runs 3` en futuras campanas (backend lo soporta) | decision: seguir en n=1 |

Old August list below kept for reference.

| # | Tarea | Esfuerzo |
|---|-------|----------|
| 1 | Assign putative sigma to 570 D39V "None" TSS | 30 min |
| 2 | AUC per sigma factor (using cached scores) | 15 min |
| 3 | Run MEME/FIMO on TIGR4 locally | 5 min |
| 4 | iPro-MP on TIGR4 (requires GPU/Codon) | Slurm |
| 5 | Document TIGR4 sigma limitation | 5 min |

---

## 8. TOOLS AND THEIR STATUS (260928: 9/9 con scores D39V+TIGR4 high)

| # | Tool | D39V+TIGR4 high | Runner | Type |
|---|------|----------------|--------|------|
| 1 | MEME (STREME+FIMO) | ✅ 0.8544 | `meme.py` | Motif (2-fold CV, seed 42, train/infer split) |
| 2 | FIMO + Prok DB | ✅ 0.7533 | `fimo.py` | Motif (838 PWMs, `--norc`) |
| 3 | MLDSPP XGBoost (0%) | ✅ 0.8447 | `mldspp.py` | ML |
| 4 | MLDSPP XGBoost (75%) | ✅ 0.9410 | `mldspp_75.py` | ML (split versionado seed 42) |
| 5 | PromoterLCNN | ✅ 0.9286 | `lcnn.py` | DL (Keras/TF 2.6) |
| 6 | PromoTech RF-HOT | ✅ 0.9267 | `promotech_hot.py` | ML |
| 7 | prompt (MLP) | ✅ 0.8516 | `prompt.py` | NN |
| 8 | ProkBERT-mini | ✅ 0.9493 | `prokbert_mini.py` | gLM (kmer 6, prefetch128 disponible) |
| 9 | iPro-MP (DNABERT-6) | ✅ 0.9358 | `ipromp_sp12.py` | gLM |

Contract: `docs/RUNNER_CONTRACT.md`. Tests: `tests/test_runners_contract.py` (7 passed + 9 subtests).

### August snapshot (kept for reference)

| # | Tool | D39V scores | TIGR4 scores | Runner | Type |
|---|------|------------|-------------|--------|------|
| 1 | MEME (STREME+FIMO) | ✅ | ❌ | `meme.py` | Motif |
| 2 | FIMO + E. coli DB | ✅ | ❌ | `fimo_db.py` | Motif |
| 3 | FIMO + Prok DB | ✅ | ❌ | `fimo_prok.py` | Motif |
| 4 | MLDSPP XGBoost (0%) | ✅ | ✅ | `mldspp.py` | ML |
| 5 | MLDSPP XGBoost (75%)* | ✅ | ✅ | `mldspp_75.py` | ML* |
| 6 | PromoterLCNN | ✅ | ✅ | `lcnn.py` | DL |
| 7 | PromoTech RF-HOT | ✅ | ✅ | `promotech_hot.py` | ML |
| 8 | PromoTech RF-TETRA | ✅ | ✅ | `promotech_tetra.py` | ML |
| 9 | iPro-MP (DNABERT-6) | ✅ | ❌ | `ipromp_sp12.py` | DL |

*MLDSPP 75% = data leakage (75% S. pneumoniae in training). True value: MLDSPP 0%.
