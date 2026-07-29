# Physics-Informed DNA Language Model for Bacterial Promoter Prediction

**STATUS:** Research Proposal — Parallel Project  
**Created:** 2026-07-27  
**Location:** `docs/research/physics_informed_dna_llm.md`

## 1. Motivation

Current DNA language models (DNABERT, DNABERT-2, Nucleotide Transformer, HyenaDNA, GENA-LM) are purely sequence-based — they learn k-mer patterns through self-supervised pre-training on genomes. None incorporate physical properties of DNA (thermodynamics, structural geometry) into their embeddings.

For bacterial promoter prediction, physics matters:
- The **extended -10 box** (TATAAT) has characteristic minor groove width (MGW ~3-4Å, narrower than background)
- **DNA melting** requires AT-rich regions with low SantaLucia ΔG
- **False positives** from AT-rich CDS regions can be distinguished by structural features (shape differs from true promoters)

The central hypothesis: **combining DNABERT-2 sequence embeddings with per-position thermodynamic (ΔG) and structural (DNA shape) features improves promoter prediction accuracy over sequence-only models.**

## 2. Precedents & Gap Analysis

| Paper | Year | Approach | Gap |
|-------|------|----------|-----|
| iEnhancer-CADS (Ren & Zheng) | 2025 | DNABERT + DNA shape → enhancer prediction | Eukaryotic enhancers only; no bacterial promoters; no thermodynamic features |
| Jung (Sci Reports) | 2026 | DNA LLM → core promoter activity | Sequence-only; no physics |
| Zhou et al. (NAR) | 2025 | Transformer → E. coli promoter design | Sequence-only; no structural features |
| Suárez-Villagrán (CSBJ) | 2025 | Physics-informed ML → DNA mutations | Mutations, not promoters; no LLM |
| MLDSPP | 2020 | SantaLucia ΔG + XGBoost | No LLM; XGBoost only |
| PMID search: "physics informed DNA language model promoter" | — | **0 results** | — |

**Novelty claim:** First model to combine (1) DNA language model embeddings, (2) nearest-neighbor thermodynamic stability, and (3) DNA structural parameters for bacterial promoter prediction.

## 3. Architecture

```
Input: 81bp DNA sequence (−60 to +20, TSS at position 60)
│
├── Pipeline A: Sequence embeddings
│   └── DNABERT-2 (zhihan1996/DNABERT-2-117M)
│       ├── Tokenizer: byte-level BPE (1 token per nucleotide)
│       ├── 81 tokens → [81, 768] hidden states
│       └── Frozen backbone + LoRA adapters (r=8, alpha=16)
│           └── Trainable params: ~295K
│
├── Pipeline B: Thermodynamic features
│   └── SantaLucia nearest-neighbor model
│       ├── 79 dinucleotides → 79 ΔG values (kcal/mol)
│       └── Linear projection → [79, 32] → pad → [81, 32]
│
├── Pipeline C: Structural features
│   └── DNAshape (Zhou et al., NAR 2013)
│       ├── 4 features per position:
│       │   ├── Minor Groove Width (MGW)
│       │   ├── Propeller Twist (ProT)
│       │   ├── Roll
│       │   └── Helix Twist (HelT)
│       └── [81, 4] → Linear projection → [81, 64]
│
└── Fusion & Classification
    └── Concatenate: [81, 768+32+64] = [81, 864]
        └── Mean pooling → [864]
            └── Dropout(0.1) → Linear(864, 256) → ReLU → Linear(256, 1)
                └── Sigmoid → promoter probability
```

## 4. Data Requirements

| Source | Positives | Negatives | Total |
|--------|-----------|-----------|-------|
| S. pneumoniae D39V (benchmark) | 988 | 1,000 | 1,988 |
| External species (MLDSPP, 12 spp) | 4,800 | 4,800* | 9,600 |
| **Total (with augmentation)** | ~5,800 | ~5,800 | ~11,600 |

*Negatives generated via dinucleotide-shuffle preserving composition.

**Sample size adequacy:**
- LoRA fine-tuning: 500-5,000 examples sufficient for binary classification
- Jung (2026) succeeded with ~5K sequences on DNABERT base
- Our ~11,600 is comfortable for LoRA (r=8, ~295K trainable params)
- Data augmentation: ±10bp window shifts → 3-5× effective sample size

## 5. Training Strategy

| Phase | Data | Epochs | Learning Rate |
|-------|------|--------|---------------|
| Phase 1: LoRA warmup | 12 external species only | 5 | 1e-3 |
| Phase 2: Mixed fine-tune | External + 50% S. pneumoniae | 5 | 5e-4 |
| Phase 3: Species-specific | S. pneumoniae only | 3 | 1e-4 |

**Evaluation:**
- Primary: 5-fold stratified CV on S. pneumoniae (AUC, F1, MCC)
- External: Zero-shot on TIGR4 strain (validation set, never used for training)
- Ablation: Compare full model vs. DNABERT-2-only vs. ΔG-only vs. shape-only

## 6. Expected Improvements

| Component | Expected contribution | Biological rationale |
|-----------|---------------------|---------------------|
| DNABERT-2 alone | AUC ~0.90 | Sequence patterns (k-mers, motifs) |
| + SantaLucia ΔG | +0.02-0.04 AUC | Thermodynamic stability at -10 box |
| + DNA shape | +0.02-0.04 AUC | MGW narrowing at TATAAT, ProT in spacer |
| Full model (projected) | AUC ~0.94-0.96 | Complementary physics features |

**Key test:** MEME false positives (12 CDS sequences with high scores) should be correctly classified as negatives by the physics-informed model because their DNA shape differs from true promoters despite sequence similarity.

## 7. Computational Requirements

| Resource | Local | Codon (HPC) |
|----------|-------|-------------|
| DNABERT-2 model | ~470 MB | Same |
| Fine-tuning time | ~30 min (CPU) | ~5 min (A100 GPU) |
| Inference (1988 seqs) | ~2 min | ~10 sec |
| DNAshape computation | ~1 sec | Same |
| Total GPU memory | ~4 GB | Same |

## 8. Comparison with Existing Tools in Our Benchmark

| Tool | AUC | Type | Physics? |
|------|-----|------|----------|
| iPro-MP sp12 | 0.962 | DNABERT-6 + classifier | No |
| PromoterLCNN | 0.953 | CNN (pre-trained) | No |
| PromoTech RF-HOT | 0.931 | RF on k-mers | No |
| MEME (2-fold CV) | 0.854 | Motif discovery | No |
| MLDSPP XGBoost | 0.863 | SantaLucia ΔG + XGBoost | **Partial** (ΔG only) |
| **Proposed model** | **0.94-0.96 (projected)** | **DNABERT-2 + ΔG + shape** | **Yes** |

## 9. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DNABERT-2 pre-trained on human, not bacteria | Medium | LoRA adapts domain; test with bacterial-only pre-training if needed |
| Overfitting with 1,988 spn sequences | Medium | External 4,800 seqs + window-shift augmentation |
| DNAshape features add noise, not signal | Low | Ablation study to measure per-feature contribution |
| ΔG features redundant with DNABERT attention | Low | Attention captures k-mer context, not thermodynamic physics |
| Computational cost on CPU | High (local) | Target GPU (Codon A100), CPU only for prototyping |

## 10. Implementation Plan

| Week | Deliverable |
|------|-------------|
| Week 1 | Install DNAshape, prototype feature extraction, verify on 5 promoters |
| Week 2 | Set up LoRA fine-tuning pipeline, train baseline (DNABERT-2 only) |
| Week 3 | Integrate ΔG + shape features, train full model |
| Week 4 | Ablation study, TIGR4 validation, paper draft |

## 11. Key References

1. Ren & Zheng (2025). iEnhancer-CADS: Cross-Modal Attention for Enhancer Identification by Integrating DNABERT and DNA Shape Features. *IEEE SMC*.
2. Jung (2026). Decoding promoter activity from DNA sequence using pre-trained language models. *Scientific Reports*.
3. Zhou et al. (2013). DNAshape: a method for the high-throughput prediction of DNA structural features on a genomic scale. *Nucleic Acids Research*.
4. SantaLucia (1998). A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics. *PNAS*.
5. Zhou et al. (2025). Deep learning guided programmable design of E. coli core promoters. *Nucleic Acids Research*.
6. Suárez-Villagrán et al. (2025). Beyond sequence: A physics-informed ML framework for predicting DNA mutations. *CSBJ*.
