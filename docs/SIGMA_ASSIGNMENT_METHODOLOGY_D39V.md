# Methodological Report: Experimental Identification and Sigma Factor Assignment in *Streptococcus pneumoniae* D39V

**Date:** August 28, 2026
**Area:** Computational Genomics, Systems Biology and Promoter Modeling
**Model Organism:** *Streptococcus pneumoniae* strain D39V (GenBank: `CP027540.1`, $2{,}046{,}572\text{ bp}$)
**Reference Files:** [`data/reference/D39V.gff3`](../data/reference/D39V.gff3), `data/reference/D39V_TSS.gff3` (historical, not versioned)

---

## 1. Executive Summary and Transcriptome Balance

The baseline transcriptome of *Streptococcus pneumoniae* D39V comprises **$988\text{ Transcription Start Sites (TSSs)}$** primary sites mapped experimentally at single-nucleotide resolution ($1\text{ bp}$). The functional assignment of RNA polymerase holoenzyme and transcription factors is summarized in the following consolidated quantitative balance:

$$\begin{array}{|l|c|c|l|}
\hline
\textbf{Functional Category} & \textbf{Count} & \textbf{Percentage} & \textbf{Molecular and Structural Definition} \\
\hline
\mathbf{Complete\text{ Bipartite }RpoD\text{ (}}\sigma^A\mathbf{)} & \mathbf{382} & 38.7\% & \text{-35 box } (TTGACA) + 15\text{--}19\text{ bp spacer} + \text{-10 box } (TATAAT) \\
\mathbf{\text{-10 only / Extended }-10\text{ (Putative }\sigma^A\text{)}} & \mathbf{449} & 45.4\% & \text{Lack a } -35\text{ box; dependent on Extended } -10 \; (TRTGNT) \text{ or } -10 \text{ box} \\
\mathbf{ComE\text{ (TF dependent on }}\text{E}\sigma^A\mathbf{)} & \mathbf{13} & 1.3\% & \text{ComE response regulator recruiting the } \text{E}\sigma^A \text{ holoenzyme} \\
\mathbf{Additional Basal Promoters} & \mathbf{2} & 0.2\% & \text{Vegetative promoters with basal constitutive expression} \\
\hline
\textbf{TOTAL WITH }\mathbf{\sigma^A}\textbf{ MACHINERY} & \mathbf{846} & \mathbf{85.6\%} & \text{All promoters recognized by RNA polymerase } \text{E}\sigma^A \\
\hline
\mathbf{ComX\text{ (}}\sigma^X\text{ Competence)} & \mathbf{19\text{ to }21} & 2.1\% & \text{Alternative competence sigma factor (combox/CIN-box motif } TACGAATA) \\
\textbf{Pure Orphans / TF-dependent} & \mathbf{121\text{ to }123} & 12.3\% & \text{No detectable basal box; activated by CodY, CcpA, CiaR or riboswitches} \\
\hline
\textbf{TOTAL D39V TSSs} & \mathbf{988} & \mathbf{100.0\%} & \text{Complete transcriptional census under standard condition} \\
\hline
\end{array}$$

> [!NOTE]
> **Consolidation of the 397 benchmark-confirmed $\sigma^A$ promoters:**
> The **$397\text{ confirmed }\sigma^A\text{ promoters}$** of reference correspond to the **$382\text{ complete bipartite RpoD}$** plus the **$13\text{ ComE sites}$** (which recruit $\text{E}\sigma^A$) and $2\text{ basal promoters}$ ($382 + 13 + 2 = 397$).

---

## 2. Experimental TSS Identification: dRNA-seq with TEX Digestion

Physical mapping of the $+1$ nucleotide was performed by the Jan-Willem Veening laboratory (*Slager et al., 2018; Aprianto et al., 2018*) using **differential dRNA-seq**:

```
S. pneumoniae D39V Total RNA
        │
   ┌────┴───────────────────────────────┐
   ▼ TEX- Fraction                      ▼ TEX+ Fraction (Enzymatic Digestion)
Sequencing of all RNA                TEX degrades RNAs with 5'-monophosphate (processed/degraded RNAs).
(primary and processed transcripts)  Primary transcripts (5'-triphosphate / 5'-PPP) are TEX-RESISTANT.
   │                                    │
   └────────────────┬───────────────────┘
                    ▼
     Alignment to the D39V Genome
                    │
                    ▼
 Coverage Peaks Enriched in TEX+ / TEX- ──► Exact TSS +1 location (1 bp)
```

1. **Enzymatic Treatment:** *Terminator 5′-Phosphate-Dependent Exonuclease* (**TEX**) selectively degrades degraded or processed RNA fragments carrying a $5'\text{-monophosphate}$ ($5'\text{-P}$) end.
2. **Resistance of Native Transcripts:** Newly synthesized primary transcripts retain the $5'\text{-triphosphate}$ ($5'\text{-PPP}$) end and survive digestion.
3. **$1\text{ bp}$ Mapping:** Peaks massively enriched in $\text{TEX}+$ relative to $\text{TEX}-$ define the transcription start ($+1$) at single-nucleotide precision.

---

## 3. Reference Literature and External Data Sources

The Veening group did not invent the binding sequences nor extract them circularly from their own genome; they used the following bibliographic sources and external databases:

1. **RpoD ($\sigma^A$) — Citation 93 (192 in full version):**
   *Shimada, T., Yamazaki, Y., Tanaka, K. and Ishihama, A. (2014). The whole set of constitutive promoters recognized by RNA polymerase RpoD holoenzyme of Escherichia coli. **PLoS ONE**, 9(3), e90447.*
   - **Contribution:** Experimental in vitro mapping by **Genomic SELEX (gSELEX)** of **$669\text{ constitutive promoters}$** (tabulated in Shimada Supplementary Table S2, from which $550\text{ complete pairs}$ $-35/-10$ were extracted).
2. **Pneumococcal Extended $-10$ — Citation 96 (206):**
   *de Jong, B., et al. (2011). Regulatory networks and extended -10 elements in Streptococcus pneumoniae. **Applied and Environmental Microbiology**.*
   - **Contribution:** Definition of the extended motif **`TRTGNT`** (`TG` dinucleotide at $-15/-14$).
3. **ComX ($\sigma^X$) — Citation 94 (72):**
   *Campbell, E.A., et al. (1998) & Peterson, S.N., et al. (2004).*
   - **Contribution:** Consensus sequence of the combox/CIN-box (**`TACGAATA`**).
4. **ComE — Citation 95 (151):**
   *Martin, B., et al. & Ween, O., et al.*
   - **Contribution:** Binding site of the ComE response regulator dimer (**`TCAGTTGAG`**).
5. **Transcription Factors (CodY, CcpA, CiaR, Rex) — Citation 97 (162):**
   *Novichkov, P.S., et al. (2013). RegPrecise 3.0: a database of curated regulatory interactions in bacteria. **Nucleic Acids Research**.*
   - **Contribution:** Collection of TF binding sites adopted directly in D39V.

---

## 4. Mathematical Construction of the $\sigma^A$ PWM Matrices

The computational procedure to model $\sigma^A$ was executed as follows:

### 4.1. Background Frequency Model (Background Distribution)
The nucleotide composition of the **$500\text{ bp}$ upstream regions of all D39V TSSs** was extracted:
$$f_A = 0.368 \quad|\quad f_T = 0.332 \quad|\quad f_C = 0.124 \quad|\quad f_G = 0.175$$

### 4.2. Position Probability Matrix (PPM) Generation
From the Shimada et al. $-35$ and $-10$ box sequences, normalized probability was computed with background pseudo-counting:
$$P_{b, i} = \frac{C_{b, i} + \alpha \cdot f_b}{N + \alpha}$$

### 4.3. Assembly of the 5 Composite Bipartite Models
The $\sigma^A$ subunit binds DNA simultaneously via its 4.2 domain (at $-35$) and 2.4 domain (at $-10$). To capture the torsional flexibility of the double helix, **5 independent composite matrices of width $W = 6 + S + 6$** were built ($S \in \{15, 16, 17, 18, 19\}\text{ bp}$):

```
┌─────────────────────────┬───────────────────────────────────┬─────────────────────────┐
│     -35 box (6 bp)      │      Neutral Spacer (S bp)        │     -10 box (6 bp)      │
│  -35 box PPM (Shimada)  │  Columns fixed at fb frequencies  │  -10 box PPM (Shimada)  │
└─────────────────────────┴───────────────────────────────────┴─────────────────────────┘
```

- `RPOD_COMPOSITE_SP15` (width = $27\text{ bp}$)
- `RPOD_COMPOSITE_SP16` (width = $28\text{ bp}$)
- `RPOD_COMPOSITE_SP17` (width = $29\text{ bp}$, the optimal geometric conformation)
- `RPOD_COMPOSITE_SP18` (width = $30\text{ bp}$)
- `RPOD_COMPOSITE_SP19` (width = $31\text{ bp}$)

### 4.4. Log-Odds Score Computation in FIMO
$$\text{Score}(S) = \sum_{k=1}^{W} \log_2 \left( \frac{P_{s_k, k}}{f_{s_k}} \right)$$
At neutral spacer positions, $\log_2(f_{s_k} / f_{s_k}) = \log_2(1) = 0$, ensuring the spacer **does not penalize sequence but rigidly enforces the exact molecular distance of $S$ nucleotides**.

---

## 5. FIMO Scanning Protocol and Spacing Filters

```
[988 D39V TSSs]
       │
       ├───────────────────────────────────┬───────────────────────────────────┐
       ▼ 40 bp scan (p < 0.001)            ▼ 20 bp scan (p < 0.001)            ▼ 40 bp scan (p < 1e-5)
[Composite Bipartite RpoD]           [Extended -10 / Pribnow]            [ComX CIN-box]
Filter: 3 bp <= Spacing <= 8 bp      Filter: 3 bp <= Spacing <= 8 bp     Filter: Spacing < 6 bp
       │                                   │                                   │
       ▼                                   ▼                                   ▼
 382 Bipartite Sites                 449 Sites without -35               19 ComX Sites
```

$$\begin{array}{|l|c|c|c|l|}
\hline
\textbf{Regulatory Element} & \textbf{Analyzed Window} & \textbf{FIMO P-value} & \textbf{Spacing Filter to }+1 & \textbf{Result} \\
\hline
\mathbf{Composite\text{ Bipartite }RpoD} & -40\text{ to }-1 & p < 0.001 & \mathbf{3\text{ to }8\text{ bp}} & \mathbf{382\text{ complete sites}} \\
\mathbf{Extended }-10\text{ / Isolated Pribnow} & -20\text{ to }-1 & p < 0.001 & \mathbf{3\text{ to }8\text{ bp}} & \mathbf{449\text{ promoters without }}-35 \\
\mathbf{ComX\text{ (}}\sigma^X\mathbf{)} & -40\text{ to }-1 & p < 0.00001 & < \mathbf{6\text{ bp}} & \mathbf{19\text{ competence promoters}} \\
\mathbf{ComE\text{ (TF)}} & -100\text{ to }-1 & p < 0.00001 & \text{Free} & \mathbf{13\text{ regulatory sites}} \\
\hline
\end{array}$$

---

## 6. Reconciliation of the 4 Putative $\sigma^A$ Approaches

During benchmark development, 4 bioinformatic strategies were evaluated to classify the $570\text{ TSSs}$ with no bipartite RpoD assignment in the primary table:

$$\begin{array}{|c|l|l|c|c|}
\hline
\textbf{Count} & \textbf{Methodology Evaluated} & \textbf{Algorithmic Criterion} & \textbf{Orphans} & \textbf{ROC-AUC} \\
\hline
\mathbf{561} & \textbf{PRODORIC General} & \text{External general bacterial matrix } (p < 0.001) & 9 & 0.9497 \\
\mathbf{451} & \textbf{Veening Regex} & \text{Window } [-18, -4] \text{ with } TATAAT \le 1\text{ mismatch} & 119 & 0.9968 \\
\mathbf{447} & \textbf{Reconstructed Slager} & \textbf{D39V de novo PWM + FIMO } (p < 0.001) + 3\text{--}8\text{ bp} & \mathbf{122} & \mathbf{0.9969} \\
\mathbf{376\text{--}390} & \textbf{External Shimada Bases} & \text{Independent gSELEX matrix (550 pairs) + FIMO} & 180 & 0.9892 \\
\hline
\end{array}$$

### Why did the 449 putatives lack the $-35$ box?
1. **Compensation by Domain 3.0:** In low-GC Gram-positive bacteria, the `TG` dinucleotide of the **Extended $-10$ (`TRTGNT`)** anchors $\text{E}\sigma^A$ polymerase strongly enough to make the $-35$ box contact unnecessary.
2. **TF Recruitment:** Many promoters are activated by transcription factors (**CodY, CcpA, CiaR, ComE**), where protein-protein interactions compensate for the missing $-35$ contact.
3. **GFF3 Annotation Criterion:** Slager et al. restricted the formal tag `Predicted RpoD recognition site` exclusively to the **$382\text{ complete bipartite promoters}$** to maintain maximum specificity in the genomic annotation file.

---

## 7. Key File Mapping in the Repository

- 📂 **Shimada Composite MEME Matrix:**
  [`output/fimo_shimada_composite/shimada_composite_motifs.meme`](../output/fimo_shimada_composite/shimada_composite_motifs.meme) (generated, gitignored)
- 📂 **Shimada 669 Constitutive Promoters Table:**
  `_experimentos_analysis/sigma_assignment_rebuild/data/shimada_2014_669_constitutive_promoters.tsv` (local, outside the repo)
- 📂 **Reconstructed Slager Pipeline Script:**
  `_proyectos_tools/promoter-tools-extra/backup/experiments/reconstruct_slager_pipeline.py` (local, outside the repo)
- 📂 **D39V Positives Metadata and Classification:**
  [`data/benchmark/d39v/positives_81bp_metadata.tsv`](../data/benchmark/d39v/positives_81bp_metadata.tsv)
