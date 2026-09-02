# Memoria Metodológica: Identificación Experimental y Asignación de Factores Sigma en *Streptococcus pneumoniae* D39V

**Fecha de Elaboración:** 28 de Agosto de 2026  
**Área:** Genómica Computacional, Biología de Sistemas y Modelamiento de Promotores  
**Organismo Modelo:** *Streptococcus pneumoniae* cepa D39V (GenBank: `CP027540.1`, $2.046.572\text{ pb}$)  
**Archivos de Referencia:** [`data/reference/D39V.gff3`](../data/reference/D39V.gff3), `data/reference/D39V_TSS.gff3` (histórico, no versionado)

---

## 1. Resumen Ejecutivo y Balance del Transcriptoma

El transcriptoma base de *Streptococcus pneumoniae* D39V comprende **$988\text{ Sitios de Inicio de la Transcripción (TSSs)}$** primarios mapeados experimentalmente a resolución de nucleótido único ($1\text{ pb}$). La asignación funcional de la ARN polimerasa holoenzima y los factores de transcripción se resume en el siguiente balance cuantitativo consolidado:

$$\begin{array}{|l|c|c|l|}
\hline
\textbf{Categoría Funcional} & \textbf{Cantidad} & \textbf{Porcentaje} & \textbf{Definición Molecular y Estructural} \\
\hline
\mathbf{RpoD\text{ Bipartitos Completos (}}\sigma^A\mathbf{)} & \mathbf{382} & 38,7\% & \text{Caja } -35 \; (TTGACA) + \text{espaciador } 15\text{--}19\text{ pb} + \text{Caja } -10 \; (TATAAT) \\
\mathbf{\text{Solo }}-10\text{ / Extended }-10\text{ (}\sigma^A\text{ Putativos)} & \mathbf{449} & 45,4\% & \text{Carecen de caja } -35\text{; dependientes de Extended } -10 \; (TRTGNT) \text{ o caja } -10 \\
\mathbf{ComE\text{ (TF dependiente de }}\text{E}\sigma^A\mathbf{)} & \mathbf{13} & 1,3\% & \text{Regulador de respuesta ComE que recluta a la holoenzima } \text{E}\sigma^A \\
\mathbf{Promotores Basales Adicionales} & \mathbf{2} & 0,2\% & \text{Promotores vegetativos de expresión constitutiva basal} \\
\hline
\textbf{TOTAL CON MAQUINARIA }\mathbf{\sigma^A} & \mathbf{846} & \mathbf{85,6\%} & \text{Totalidad de promotores reconocidos por la polimerasa } \text{E}\sigma^A \\
\hline
\mathbf{ComX\text{ (}}\sigma^X\text{ Competencia)} & \mathbf{19\text{ a }21} & 2,1\% & \text{Factor sigma alternativo de competencia (motivo combox/CIN-box } TACGAATA) \\
\textbf{Huérfanos Puros / TF-dependientes} & \mathbf{121\text{ a }123} & 12,3\% & \text{Sin caja basal detectable; activados por CodY, CcpA, CiaR o riboswitches} \\
\hline
\textbf{TOTAL TSSs D39V} & \mathbf{988} & \mathbf{100,0\%} & \text{Censo transcripcional completo en condición estándar} \\
\hline
\end{array}$$

> [!NOTE]
> **Consolidación de los 397 $\sigma^A$ Confirmados del Benchmark:**  
> Los **$397\text{ promotores }\sigma^A\text{ Confirmados}$** de referencia corresponden a los **$382\text{ RpoD bipartitos completados}$** más los **$13\text{ sitios ComE}$** (que reclutan a $\text{E}\sigma^A$) y $2\text{ promotores basales}$ ($382 + 13 + 2 = 397$).

---

## 2. Identificación Experimental de TSSs: dRNA-seq con Digestión TEX

El mapeo físico del nucleótido $+1$ fue realizado por el laboratorio de Jan-Willem Veening (*Slager et al., 2018; Aprianto et al., 2018*) mediante **dRNA-seq diferencial**:

```
ARN Total de S. pneumoniae D39V
        │
   ┌────┴───────────────────────────────┐
   ▼ Fracción TEX-                      ▼ Fracción TEX+ (Digestión Enzimática)
Secuenciación de todo el ARN         TEX degrada ARNs con 5'-monofosfato (ARNs procesados/degradados).
(transcriptos primarios y procesados) Transcriptos primarios (5'-trifosfato / 5'-PPP) son RESISTENTES a TEX.
   │                                    │
   └────────────────┬───────────────────┘
                    ▼
     Alineamiento al Genoma D39V
                    │
                    ▼
 Picos de Cobertura Enriquecidos en TEX+ / TEX- ──► Ubicación exacta del TSS +1 (1 pb)
```

1. **Tratamiento Enzimático:** La enzima *Terminator 5′-Phosphate-Dependent Exonuclease* (**TEX**) degrada selectivamente los fragmentos de ARN degradados o procesados que poseen un extremo $5'\text{-monofosfato}$ ($5'\text{-P}$).
2. **Resistencia de Transcriptos Nativos:** Los transcriptos primarios recién sintetizados conservan el extremo $5'\text{-trifosfato}$ ($5'\text{-PPP}$) y sobreviven a la digestión.
3. **Mapeo a $1\text{ pb}$:** Los picos masivamente enriquecidos en $\text{TEX}+$ respecto a $\text{TEX}-$ definen con precisión de nucleótido único el inicio de la transcripción ($+1$).

---

## 3. Literatura de Referencia y Fuentes de Datos Externas

El grupo de Veening no inventó las secuencias de unión ni las extrajo de forma circular de su propio genoma, sino que utilizó las siguientes fuentes bibliográficas y bases de datos externas:

1. **RpoD ($\sigma^A$) — Cita 93 (192 en versión completa):**  
   *Shimada, T., Yamazaki, Y., Tanaka, K. and Ishihama, A. (2014). The whole set of constitutive promoters recognized by RNA polymerase RpoD holoenzyme of Escherichia coli. **PLoS ONE**, 9(3), e90447.*  
   - **Aporte:** Mapeo experimental in vitro por **Genomic SELEX (gSELEX)** de **$669\text{ promotores constitutivos}$** (tabulados en la Tabla Suplementaria S2 de Shimada, de los cuales $550\text{ pares completos}$ $-35/-10$ fueron extraídos).
2. **Extended $-10$ de Neumococo — Cita 96 (206):**  
   *de Jong, B., et al. (2011). Regulatory networks and extended -10 elements in Streptococcus pneumoniae. **Applied and Environmental Microbiology**.*  
   - **Aporte:** Definición del motivo extendido **`TRTGNT`** (dinucleótido `TG` en $-15/-14$).
3. **ComX ($\sigma^X$) — Cita 94 (72):**  
   *Campbell, E.A., et al. (1998) & Peterson, S.N., et al. (2004).*  
   - **Aporte:** Secuencia consenso del combox/CIN-box (**`TACGAATA`**).
4. **ComE — Cita 95 (151):**  
   *Martin, B., et al. & Ween, O., et al.*  
   - **Aporte:** Sitio de unión del dímero del regulador de respuesta ComE (**`TCAGTTGAG`**).
5. **Factores de Transcripción (CodY, CcpA, CiaR, Rex) — Cita 97 (162):**  
   *Novichkov, P.S., et al. (2013). RegPrecise 3.0: a database of curated regulatory interactions in bacteria. **Nucleic Acids Research**.*  
   - **Aporte:** Colección de sitios de unión de TFs adoptados directamente en D39V.

---

## 4. Construcción Matemática de las Matrices PWM de $\sigma^A$

El procedimiento computacional para modelar $\sigma^A$ se ejecutó de la siguiente forma:

### 4.1. Modelo de Frecuencias de Fondo (Background Distribution)
Se extrajo la composición nucleotídica de las regiones de **$500\text{ pb}$ aguas arriba de todos los TSSs** de D39V:
$$f_A = 0,368 \quad|\quad f_T = 0,332 \quad|\quad f_C = 0,124 \quad|\quad f_G = 0,175$$

### 4.2. Generación de Matrices de Probabilidad de Posición (PPM)
A partir de las secuencias de las cajas $-35$ y $-10$ de Shimada et al., se calculó la probabilidad normalizada con pseudoconteo de fondo:
$$P_{b, i} = \frac{C_{b, i} + \alpha \cdot f_b}{N + \alpha}$$

### 4.3. Ensamblaje de los 5 Modelos Bipartitos Compuestos
La subunidad $\sigma^A$ une el ADN simultáneamente mediante su dominio 4.2 (en $-35$) y dominio 2.4 (en $-10$). Para capturar la flexibilidad torsional de la doble hélice, se construyeron **5 matrices compuestas independientes de ancho $W = 6 + S + 6$** ($S \in \{15, 16, 17, 18, 19\}\text{ pb}$):

```
┌─────────────────────────┬───────────────────────────────────┬─────────────────────────┐
│     Caja -35 (6 pb)     │       Espaciador Neutro (S pb)     │     Caja -10 (6 pb)     │
│   PPM Caja -35 (Shimada) │  Columnas fijas en frecuencias fb  │  PPM Caja -10 (Shimada) │
└─────────────────────────┴───────────────────────────────────┴─────────────────────────┘
```

- `RPOD_COMPOSITE_SP15` (ancho = $27\text{ pb}$)
- `RPOD_COMPOSITE_SP16` (ancho = $28\text{ pb}$)
- `RPOD_COMPOSITE_SP17` (ancho = $29\text{ pb}$, la conformación geométrica óptima)
- `RPOD_COMPOSITE_SP18` (ancho = $30\text{ pb}$)
- `RPOD_COMPOSITE_SP19` (ancho = $31\text{ pb}$)

### 4.4. Cálculo del Log-Odds Score en FIMO
$$\text{Score}(S) = \sum_{k=1}^{W} \log_2 \left( \frac{P_{s_k, k}}{f_{s_k}} \right)$$
En las posiciones del espaciador neutro, $\log_2(f_{s_k} / f_{s_k}) = \log_2(1) = 0$, asegurando que el espaciador **no penaliza la secuencia pero impone rígidamente la distancia molecular exacta de $S$ nucleótidos**.

---

## 5. Protocolo de Escaneo con FIMO y Filtros de Espaciado

```
[988 TSSs D39V] 
       │
       ├───────────────────────────────────┬───────────────────────────────────┐
       ▼ Escaneo 40 pb (p < 0.001)         ▼ Escaneo 20 pb (p < 0.001)         ▼ Escaneo 40 pb (p < 1e-5)
[RpoD Bipartito Compuesto]           [Extended -10 / Pribnow]            [ComX CIN-box]
Filtro: 3 pb <= Espaciado <= 8 pb    Filtro: 3 pb <= Espaciado <= 8 pb   Filtro: Espaciado < 6 pb
       │                                   │                                   │
       ▼                                   ▼                                   ▼
 382 Sitios Bipartitos               449 Sitios sin -35                   19 Sitios ComX
```

$$\begin{array}{|l|c|c|c|l|}
\hline
\textbf{Elemento Regulador} & \textbf{Ventana Analizada} & \textbf{P-valor FIMO} & \textbf{Filtro Espacial al }+1 & \textbf{Resultado} \\
\hline
\mathbf{RpoD\text{ Bipartito Compuesto}} & -40\text{ a }-1 & p < 0,001 & \mathbf{3\text{ a }8\text{ pb}} & \mathbf{382\text{ sitios completos}} \\
\mathbf{Extended }-10\text{ / Pribnow Aislado} & -20\text{ a }-1 & p < 0,001 & \mathbf{3\text{ a }8\text{ pb}} & \mathbf{449\text{ promotores sin }}-35 \\
\mathbf{ComX\text{ (}}\sigma^X\mathbf{)} & -40\text{ a }-1 & p < 0,00001 & < \mathbf{6\text{ pb}} & \mathbf{19\text{ promotores de competencia}} \\
\mathbf{ComE\text{ (TF)}} & -100\text{ a }-1 & p < 0,00001 & \text{Libre} & \mathbf{13\text{ sitios reguladores}} \\
\hline
\end{array}$$

---

## 6. Reconciliación de las 4 Aproximaciones de $\sigma^A$ Putativos

Durante el desarrollo del benchmark, se evaluaron 4 estrategias bioinformáticas para clasificar los $570\text{ TSSs}$ que no tenían asignación de RpoD bipartito en la tabla primaria:

$$\begin{array}{|c|l|l|c|c|}
\hline
\textbf{Cantidad} & \textbf{Metodología Evaluada} & \textbf{Criterio Algorítmico} & \textbf{Huérfanos} & \textbf{ROC-AUC} \\
\hline
\mathbf{561} & \textbf{PRODORIC General} & \text{Matriz externa general bacteriana } (p < 0,001) & 9 & 0,9497 \\
\mathbf{451} & \textbf{Veening Regex} & \text{Ventana } [-18, -4] \text{ con } TATAAT \le 1\text{ mismatch} & 119 & 0,9968 \\
\mathbf{447} & \textbf{Slager Reconstruido} & \textbf{PWM de novo D39V + FIMO } (p < 0,001) + 3\text{--}8\text{ pb} & \mathbf{122} & \mathbf{0,9969} \\
\mathbf{376\text{--}390} & \textbf{Bases Externas Shimada} & \text{Matriz gSELEX independiente (550 pares) + FIMO} & 180 & 0,9892 \\
\hline
\end{array}$$

### ¿Por qué los 449 putativos carecían de la caja $-35$?
1. **Compensación por Dominio 3.0:** En bacterias Gram-positivas de bajo GC, el dinucleótido `TG` del **Extended $-10$ (`TRTGNT`)** ancla con suficiente fuerza a la polimerasa $\text{E}\sigma^A$, haciendo innecesario el contacto con la caja $-35$.
2. **Reclutamiento por TFs:** Muchos promotores son activados por factores de transcripción (**CodY, CcpA, CiaR, ComE**), donde las interacciones proteína-proteína compensan la falta del contacto $-35$.
3. **Criterio de Anotación GFF3:** Slager et al. restringieron el tag formal `Predicted RpoD recognition site` exclusivamente a los **$382\text{ promotores bipartitos completados}$** para mantener la máxima especificidad en el archivo de anotación genómica.

---

## 7. Mapeo de Archivos Clave en el Repositorio

- 📂 **Matriz MEME Compuesta de Shimada:**  
  [`output/fimo_shimada_composite/shimada_composite_motifs.meme`](../output/fimo_shimada_composite/shimada_composite_motifs.meme) (generado, gitignored)
- 📂 **Tabla de 669 Promotores Constitutivos de Shimada:**  
  `_experimentos_analysis/sigma_assignment_rebuild/data/shimada_2014_669_constitutive_promoters.tsv` (local, fuera del repo)
- 📂 **Script del Pipeline de Slager Reconstruido:**  
  `_proyectos_tools/promoter-tools-extra/backup/experiments/reconstruct_slager_pipeline.py` (local, fuera del repo)
- 📂 **Metadatos y Clasificación de Positivos D39V:**  
  [`data/benchmark/d39v/positives_81bp_metadata.tsv`](../data/benchmark/d39v/positives_81bp_metadata.tsv)
