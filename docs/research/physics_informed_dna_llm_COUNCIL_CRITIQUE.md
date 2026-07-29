# Council Critique — Physics-Informed DNA LLM Proposal
**Date:** 2026-07-27
**Reviewers:** Math Modeling Expert, Molecular Biology Expert, Bioinformatics Expert

## Consensus: Promising idea, broken execution plan

All three experts agree the core concept is novel and worth pursuing, but the proposal
has critical design flaws that must be fixed before any implementation.

---

## CRITICAL ISSUES

### 1. Data Leakage — GUARANTEED
**Bioinformatics Expert**
Three-phase training (external → mixed → spn-only) followed by standard 5-fold CV means
the model sees test sequences during Phase 2 and 3. Only fix: nested 5-fold CV where
Phases 1-3 are re-run per fold. This triples training cost.

### 2. Baseline Beats the Projection 
**Molecular Biology Expert**
The project's existing MLDSPP already achieves AUC 0.97 (corrected, real CDS negatives).
The proposal projects AUC 0.94-0.96 — this is a REGRESSION. Why use a 470 MB
transformer when 79 dinucleotide features + XGBoost already does better?

### 3. Human Pre-training → Bacterial Domain
**Molecular Biology Expert**
DNABERT-2 was pre-trained on GRCh38 (human genome). Bacterial genome architecture is
fundamentally different: 85% coding, compact intergenic regions, no CpG islands, no
repetitive elements. Risk severely underestimated (rated "High" not "Medium").

### 4. Sample Size: Phase 3 Mathematically Indefensible
**Math Modeling Expert**
1,988 spn sequences for ~295K LoRA params = 148× more params than examples.
Catastrophic overfitting guaranteed. Drop Phase 3 or reduce to r=2 LoRA with
dropout 0.3-0.5.

---

## MUST FIX

| # | Issue | Expert | Fix |
|---|-------|--------|-----|
| 1 | Standard CV leaks test data into training | Bioinformatics | Nested 5-fold CV |
| 2 | AUC projection 0.04-0.06 too high | Math | Revise to 0.91 ± 0.02 |
| 3 | Mean pooling dilutes position signal | Math | Attention pooling (865 extra params) |
| 4 | Ablation can't answer research question | Math | 7 factorial variants (not 4) |
| 5 | DNAshape R dependency is integration trap | Bioinformatics | Pure Python pentamer lookup (15 lines) |
| 6 | Zero-shot vs CV comparison unfair | Bioinformatics | Report BOTH regimes, separate columns |
| 7 | GENA-LM/NT pre-trained on bacteria | MolBio | Consider instead of DNABERT-2 |
| 8 | Reproducibility unspecified | Bioinformatics | Fix seeds, splits, hyperparams, DNAshape version |

---

## COULD BE PROBLEMATIC

| # | Issue | Expert |
|---|-------|--------|
| 1 | Shape features likely redundant with k-mers for consensus promoters | MolBio |
| 2 | False positive hypothesis not supported by biology | MolBio |
| 3 | Bacterial supercoiling invalidates pentamer table predictions | MolBio |
| 4 | Missing biology (supercoiling, TF cooperativity, stringent response) | MolBio |
| 5 | "30 min CPU fine-tuning" is likely 2-4 hours | Bioinformatics |

---

## VERDICT

**Worth pursuing but needs redesign.** The strongest contribution might be a NEGATIVE result:
showing that a 470 MB transformer FAILS to beat 79 ΔG features + XGBoost in
cross-species promoter prediction. This would be an important cautionary finding
for the field, especially relevant given the hype around genomic LLMs.

**Recommended next step:** Run a minimal experiment — DNABERT-2 (zero-shot, no fine-tuning)
vs MLDSPP (ΔG + XGBoost) on S. pneumoniae. If MLDSPP wins, the proposal needs
fundamental reframing. If DNABERT-2 wins, proceed with physics features.
