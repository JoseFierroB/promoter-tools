# MEME Suite Pipeline — promoter-tools

## Pipeline Overview

```
                    ┌─────────────┐
                    │  Positives   │  988 promotores D39V (81bp, -60/+20)
                    │  Negatives   │  1000 CDS (81bp, sin TATAAT, GC ~41%)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌──────────┐
         │ STREME  │ │  MEME   │ │ XSTREME  │
         │ (local) │ │(classic)│ │  (web)   │
         └────┬────┘ └────┬────┘ └────┬─────┘
              │            │            │
              ▼            ▼            ▼
         streme.txt   meme.txt    xstreme.txt
         (PWM motifs) (PWM motifs) (combined)
              │            │
              └─────┬──────┘
                    ▼
              ┌─────────┐
              │  FIMO   │  Escanea secuencias contra los PWM
              └────┬────┘
                   │
                   ▼
              fimo.tsv
              (por secuencia: max -log10(p-value))
                   │
                   ▼
              ┌──────────┐
              │  TOMTOM  │  Compara motivos contra DBs conocidas
              └────┬─────┘
                   │
                   ▼
              tomtom.tsv
              (matches: rpoD, MalR, PurR, CcpA...)
                   │
                   ▼
              ┌──────────────┐
              │  AUC / Stats  │
              └──────────────┘
```

## Step by Step

### STEP 1: STREME — Discriminative Motif Discovery
```bash
streme -oc streme_out -dna -minw 10 -maxw 20 \
       -p positives_81bp.fasta \
       -n negatives_81bp.fasta
```
**Input:** 2 FASTA files (pos + neg)
**Output:** `streme_out/streme.txt` — PWM de motivos enriquecidos en positivos vs negativos
**Qué hace:** Encuentra motivos SOBRE-REPRESENTADOS en positivos respecto a negativos
**Tiempo:** ~4s

### STEP 2: FIMO — Motif Scanning
```bash
fimo --text --skip-matched-sequence streme_out/streme.txt combined.fasta
```
**Input:** PWM (streme.txt) + FASTA a escanear
**Output:** TSV con `sequence_name, start, stop, p-value, q-value, matched_sequence`
**Qué hace:** Escanea cada secuencia contra cada PWM, devuelve hits con p<1e-4 (default)
**Tiempo:** ~0.5s

### STEP 3: TOMTOM — Motif Annotation
```bash
tomtom -no-ssc -text -min-overlap 4 streme.txt collectf.meme
```
**Input:** PWM descubierto + base de datos de motivos conocidos (.meme)
**Output:** TSV con `Query_ID, Target_ID, p-value, q-value, Orientation`
**Qué hace:** Compara cada motivo contra DB de motivos anotados (CollecTF, PRODORIC)
**Tiempo:** ~2s

### STEP 4: Scoring & AUC
```python
# Por cada secuencia: max(-log10(p-value)) de todos los motivos que la matchean
# Secuencias sin hits → score = 0
# AUC = roc_auc_score(y_true, scores)
```
**Input:** FIMO TSV + ground truth labels
**Output:** AUC, F1, MCC, Sens, Spec, matriz de confusión
**Tiempo:** <0.1s

## 2-Fold Cross-Validation (Anti-Leakage)

```
Fold 1: STREME(pos[:494], neg[:500]) → FIMO(pos[494:], neg[500:])
Fold 2: STREME(pos[494:], neg[500:]) → FIMO(pos[:494], neg[:500])
Combinar scores → AUC
```
Cada secuencia se evalúa con motivos descubiertos de secuencias que NUNCA vio.

## Bases de Datos para TOMTOM (tools/meme/motif_databases/)

| Base | Motivos | Procariotas | Fuente |
|------|---------|-------------|--------|
| `PROKARYOTE/collectf.meme` | 84 | Sí | CollecTF (curada) |
| `PROKARYOTE/prodoric.meme` | 201 | Sí | PRODORIC |
| `PROKARYOTE/regtransbase.meme` | 99 | Sí | RegTransBase |
| `PROKARYOTE/fan2020.meme` | ~50 | Sí | Regulones especie-específicos |
| `ECOLI/dpinteract.meme` | 68 | E. coli | DPInteract |
| `ECOLI/SwissRegulon_e_coli.meme` | ~100 | E. coli | SwissRegulon |

## Resultados Clave (D39V, 988 pos + 1000 neg)

| Motivo | E-value | Sitios | Tomtom match |
|--------|---------|--------|-------------|
| `YTATTATAYCAYAWWWW` (17bp) | 3.2×10⁻³³ | 843/988 (85%) | **rpoD (σ⁷⁰)** — PRODORIC |
| `AAAAAAKMAWA` (11bp) | 1.6×10⁻⁹ | 632/988 | UP element / unión inespecífica |
| `AAAACGCTTRCA` (12bp) | 1.7×10⁻¹ | 36/988 | **MalR_S. pneumoniae** — Fan2020 |
| Otros motivos (3) | >1.0 | <80 | Shine-Dalgarno, poly-T |

**AUC 2-fold CV honesto: 0.854**

## ¿Es Viable MEME → FIMO → TOMTOM?

**Sí.** Es el pipeline estándar de la suite. Nosotros ya lo implementamos completo:

1. ✓ STREME descubre motivos (discriminativo, pos vs neg)
2. ✓ FIMO escanea y puntúa (max -log10 p-value)
3. ✓ TOMTOM anota contra CollecTF + PRODORIC (74 matches significativos)
4. ✓ AUC honesto vía 2-fold CV

**Lo que XSTREME web añade:** MEME classic (modelo EM, sin negativos) + SEA (enriquecimiento) + logos automáticos. No es necesario para el benchmark.

## Expansión Propuesta

A partir de esta base, se puede construir una narrativa:

1. **Descubrimiento:** STREME encuentra el extended -10 como motivo dominante
2. **Validación:** TOMTOM confirma que es rpoD (σ⁷⁰) contra PRODORIC
3. **Anotación funcional:** MalR_S. pneumoniae, PurR, CcpA aparecen como reguladores
4. **Expresión:** Correlación score MEME vs Cappable-seq (ρ=0.21, p<0.0001)
5. **Genome-wide:** 2,717 promotores predichos, 1,200 novel
6. **Cross-strain:** MLDSPP generaliza (ΔG es universal), MEME es strain-específico
