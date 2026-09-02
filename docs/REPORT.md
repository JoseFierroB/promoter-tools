# Promoter-Tools — Memory & Key Findings

> Proyecto: Benchmark de 7 herramientas (9 runners) de predicción de promotores en S. pneumoniae D39V vs TIGR4
> Fecha del reporte: Agosto 2026
> Estado: ~85% completado

---

## 1. BENCHMARK: AUC por herramienta y dataset

### AUC Global (6 datasets, 6 tools principales)

| Tool | D39V | T4 hi | T4+sec | T4 ext | T4 all | MIX |
|------|------|-------|--------|--------|--------|-----|
| **MLDSPP 75%*** | **0.9567** | **0.9234** | **0.9112** | **0.8077** | **0.8062** | **0.9436** |
| iPro-MP | 0.9526 | 0.905 | 0.8897 | 0.7127 | 0.7045 | 0.9324 |
| LCNN | 0.9487 | 0.9027 | 0.8873 | 0.7116 | 0.7037 | 0.929 |
| PromoTech | 0.9431 | 0.9079 | 0.8953 | 0.7145 | 0.7049 | 0.9272 |
| MLDSPP 0% | 0.8651 | 0.8182 | 0.8056 | 0.6763 | 0.6686 | 0.8449 |
| FIMO PROK | 0.7592 | 0.7469 | 0.7398 | 0.6483 | 0.6402 | 0.7535 |

*MLDSPP 75% usa 75% S. pneumoniae en training → data leakage. MLDSPP 0% es el valor real.

---

## 2. CONFUSION MATRIX — Mejor umbral (Youden's J)

### D39V (989 pos, 1000 neg)

| Tool | AUC | TP | FN | FP | TN | Sens | Spec | F1 | Balance |
|------|-----|----|----|----|----|------|------|-----|---------|
| **iPro-MP** | 0.960 | 892 | 96 | **31** | 969 | 0.903 | 0.969 | **0.934** | ✅ Muy armónico |
| **LCNN** | 0.949 | 838 | 150 | 35 | 965 | 0.848 | 0.965 | 0.901 | ✅ Armónico |
| **PromoTech HOT** | 0.943 | 868 | 120 | 64 | 936 | 0.879 | 0.936 | 0.904 | ✅ Armónico |
| PromoTech TETRA | 0.917 | 820 | 168 | 115 | 885 | 0.830 | 0.885 | 0.853 | |
| MLDSPP | 0.865 | 821 | 167 | **246** | 754 | 0.831 | 0.754 | 0.799 | ⚠️ Exceso FP |
| MEME | 0.841 | 751 | 237 | 139 | 861 | 0.760 | 0.861 | 0.800 | |
| FIMO PROK | 0.759 | 665 | 323 | 286 | 714 | 0.673 | 0.714 | 0.686 | ⚠️ Exceso FP+FN |
| FIMO DB | 0.736 | 651 | 337 | **302** | 698 | 0.659 | 0.698 | 0.671 | ⚠️ Peor balance |

### TIGR4 high (738 pos, 738 neg)

| Tool | AUC | TP | FN | FP | TN | Sens | Spec | F1 |
|------|-----|----|----|----|----|------|------|-----|
| LCNN | 0.903 | 598 | 140 | 28 | 710 | 0.810 | 0.962 | 0.877 |
| PromoTech HOT | 0.908 | 600 | 138 | 37 | 701 | 0.813 | 0.950 | 0.873 |
| PromoTech TETRA | 0.889 | 582 | 156 | 75 | 663 | 0.789 | 0.898 | 0.834 |
| MLDSPP | 0.818 | 505 | 233 | 131 | 607 | 0.684 | 0.822 | 0.735 |

### Conclusión clave
- **iPro-MP y PromoTech HOT**: armónicas (FP y FN balanceados) — las más confiables
- **FIMO DB/FIMO PROK**: exceso de FP (30% falsa alarma) — poca especificidad
- **MLDSPP**: identifica bien TSS pero con costo en FP (24.6% en D39V)
- **TIGR4 es más difícil para todas**: menor sensibilidad, más TSS perdidos

---

## 3. RECURSOS (tiempo, RAM)

| Tool | Time | RAM |
|------|------|-----|
| PromoTech RF-HOT | 76.5s | 7.36 GB |
| PromoTech RF-TETRA | 58.4s | 7.36 GB |
| PromoterLCNN | 2.62s | 7.36 GB |
| MLDSPP XGBoost | 1.73s | 7.36 GB |

---

## 4. AUC POR CONSERVACIÓN DE IGRs

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

**Conclusión**: Todas las tools bajan su AUC en TSS intragénicos. PromoTech HOT es la más robusta (sin caída).

---

## 5. ALINEAMIENTO IGRs D39V ↔ TIGR4 (MMseqs2)

### Cross-strain
- **1,323 pares** conservados D39V→TIGR4 (971/1670 IGRs, 58.1%)
- **2,240 pares** reverso TIGR4→D39V
- **823 intersección** (reciprocal best by direction)
- Identidad media: **96.2%**, 525 pares (39.7%) al 100%
- **81.9%** sinténicos por arquitectura génica
- **643 pares** con genes flanqueantes ortólogos (SPV_* ↔ SP_RS*)
- **209 pares** 100% idénticos con TSS en ambas cepas

### Intra-strain
- D39V: 1,670 IGRs → 1,580 clusters (98.4% singletons)
- TIGR4: 1,784 IGRs → 1,584 clusters (97.9% singletons)
- 31 IGRs repetitivas (BOX/RUP/IS) = 28.9% de pares cross-strain

### nucmer
- 1,025 bloques colineales, 86% en orden genómico
- 52.1% de pares MMseqs2 validados por nucmer

---

## 6. TSS — POSICIÓN Y FACTIBILIDAD

### D39V (989 TSS curados de 1003 brutos)
| Categoría | N | % |
|-----------|---|---|
| En IGR (usables) | 804 | 81.4% |
| └─ Ventana 100% en IGR | 307 | 31.1% |
| └─ <50% CDS overlap | 490 | 49.6% |
| En CDS | 184 | 18.6% |
| └─ Deep internal (>50bp) | 95 | 9.6% |
| **TSS usables para IGR** | **891** | **90.2%** |
| **TSS perdidos** | **97** | **9.8%** |

### TIGR4 (738 TSS)
| Categoría | N | % |
|-----------|---|---|
| En IGR (usables) | 558 | 75.6% |
| En CDS | 180 | 24.4% |
| **TSS usables** | **627** | **85.0%** |
| **TSS perdidos** | **111** | **15.0%** |

### Sigma factors D39V
| Sigma | Total | En IGRs conservadas | % |
|-------|-------|--------------------|----|
| SigA | 397 | 271 | 68.3% |
| SigX | 21 | 11 | 52.4% |
| None | 570 | — | — |

---

## 6b. IGR BENCHMARK — EXPERIMENTAL (no consolidado)

> **Estado: experimental.** Extensión en paralelo al benchmark canónico; mismos
> runners y CLI, solo cambia el dataset. Resultados preliminares sujetos a cambio.

| Dataset | Pos/Neg | Contenido |
|---|---|---|
| D39V IGR | 723 / 723 | Promotores en IGRs refinadas vs fondo intergénico |
| TIGR4 IGR subset_1 | 553 / 553 | high-conf primary |
| TIGR4 IGR subset_2 | 578 / 578 | high-conf all |
| TIGR4 IGR subset_3 | 971 / 971 | all primary |
| TIGR4 IGR subset_4 | 1009 / 1009 | all comprehensive |

- **Linaje D39V**: GFF 1003 TSS (Victor + Axel) → 989 curados (proximidad <25bp) → 723 en IGRs refinadas.
- **Clusters IGR cross-strain (2 cepas)**: 2,247 clusters MMseqs2 (1,074 pares 1:1, 1,124 singletons, 49 multi-hit).
- **Ejecución**: pura configuración del CLI (`--pos/--neg`), sin código dedicado — ver `docs/RUNNING.md`.
- **Datasets versionados**: `data/benchmark_igr/` + split MLDSPP 723 (seed 42).

---

## 7. PENDIENTE (para cerrar el proyecto)

| # | Tarea | Esfuerzo |
|---|-------|----------|
| 1 | Asignar sigma putativo a 570 TSS "None" de D39V | 30 min |
| 2 | AUC per sigma factor (usando scores cacheados) | 15 min |
| 3 | Ejecutar MEME/FIMO en TIGR4 localmente | 5 min |
| 4 | iPro-MP en TIGR4 (requiere GPU/Codon) | Slurm |
| 5 | Documentar limitación TIGR4 sigma | 5 min |

---

## 8. HERRAMIENTAS Y SUS ESTADOS

| # | Tool | D39V scores | TIGR4 scores | Runner | Tipo |
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

*MLDSPP 75% = data leakage (75% S. pneumoniae en training). Valor real: MLDSPP 0%.
