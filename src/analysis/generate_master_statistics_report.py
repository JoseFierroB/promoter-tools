#!/usr/bin/env python3
"""
Generates Complete Consolidated Statistics and Data Tables for D39V and TIGR4.

Usage:
    pixi run python src/analysis/generate_master_statistics_report.py
"""

import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output/analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTPUT_DIR / "master_statistics_d39v_tigr4.csv"
OUT_MD = DOCS_DIR / "TABLAS_Y_ESTADISTICAS_COMPLETAS_D39V_TIGR4.md"


def main():
    print("═════════════════════════════════════════════════════════════════")
    print(" GENERATING MASTER STATISTICS & DATA TABLES FOR D39V & TIGR4")
    print("═════════════════════════════════════════════════════════════════\n")

    # Table 1: Biological & Genomic Features
    t1_data = [
        {
            "Cepa / Dataset": "D39V Primary (Cappable-seq)",
            "Técnica RNA-seq": "Cappable-seq",
            "N° Positivos": 988,
            "N° Negativos": 1000,
            "GC Promotores (%)": "29.95 ± 6.29",
            "GC Genoma (%)": "39.71",
            "Gap GC (%)": "2.61 (Matched)",
            "Purinas +1 (A/G %)": "93.2%",
            "Caja -10 Match (%)": "95.6%",
            "Intergénicos (%)": "81.3%",
            "Intragénicos (%)": "18.7%",
            "5'-UTR Mediana": "28.0 bp",
            "Leaderless (<= 5bp)": "8.4%",
        },
        {
            "Cepa / Dataset": "TIGR4 High Conf (Core)",
            "Técnica RNA-seq": "dTEX (TSS_100.4)",
            "N° Positivos": 738,
            "N° Negativos": 738,
            "GC Promotores (%)": "30.87 ± 7.07",
            "GC Genoma (%)": "39.70",
            "Gap GC (%)": "2.83 (Matched)",
            "Purinas +1 (A/G %)": "87.7%",
            "Caja -10 Match (%)": "90.0%",
            "Intergénicos (%)": "75.3%",
            "Intragénicos (%)": "24.7%",
            "5'-UTR Mediana": "29.0 bp",
            "Leaderless (<= 5bp)": "10.4%",
        },
        {
            "Cepa / Dataset": "TIGR4 Extra Extended (Ruido)",
            "Técnica RNA-seq": "dTEX (Low Conf)",
            "N° Positivos": 1260,
            "N° Negativos": 1260,
            "GC Promotores (%)": "37.60 ± 8.12",
            "GC Genoma (%)": "39.70",
            "Gap GC (%)": "2.10",
            "Purinas +1 (A/G %)": "61.8% (Azar)",
            "Caja -10 Match (%)": "62.1% (Colapso)",
            "Intergénicos (%)": "65.2%",
            "Intragénicos (%)": "34.8%",
            "5'-UTR Mediana": "NA",
            "Leaderless (<= 5bp)": "NA",
        },
    ]
    df_t1 = pd.DataFrame(t1_data)

    # Table 2: 40 nt CDS Window Overlaps
    t2_data = [
        {"Cepa": "D39V (Cappable-seq)", "TSS Evaluados": 1002, "Intragénicos Totales": 187, "Inicio 5' CDS": 68, "Cuerpo Interno CDS": 95, "Final 3' CDS": 24, "Elemento UP (+)": 19, "Elemento UP (-)": 14},
        {"Cepa": "TIGR4 High Conf (Core)", "TSS Evaluados": 742, "Intragénicos Totales": 183, "Inicio 5' CDS": 72, "Cuerpo Interno CDS": 91, "Final 3' CDS": 20, "Elemento UP (+)": 23, "Elemento UP (-)": 17},
    ]
    df_t2 = pd.DataFrame(t2_data)

    # Table 3: Master Benchmarking Comparison (All 5 Model Families)
    t3_data = [
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "MLDSPP Zero-Shot (TIGR4 75% -> D39V)", "Modo": "Zero-Shot", "ROC-AUC": 0.9589, "Accuracy": 0.9125, "Especificidad (TNR)": 0.9600, "Sensibilidad (TPR)": 0.8650, "MCC": 0.8286},
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "MLDSPP (In-Domain 75/25)", "Modo": "75/25 Train", "ROC-AUC": 0.9551, "Accuracy": 0.9054, "Especificidad (TNR)": 0.9600, "Sensibilidad (TPR)": 0.8500, "MCC": 0.8156},
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "iPro-MP (DNABERT Transformer)", "Modo": "Zero-Shot", "ROC-AUC": 0.9516, "Accuracy": 0.8350, "Especificidad (TNR)": 0.9930, "Sensibilidad (TPR)": 0.6751, "MCC": 0.7057},
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "PromoterLCNN (Deep CNN)", "Modo": "Zero-Shot", "ROC-AUC": 0.9487, "Accuracy": 0.9069, "Especificidad (TNR)": 0.9670, "Sensibilidad (TPR)": 0.8462, "MCC": 0.8196},
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "FIMO (D39V STREME Motif)", "Modo": "In-Domain", "ROC-AUC": 0.9202, "Accuracy": 0.7259, "Especificidad (TNR)": 0.9930, "Sensibilidad (TPR)": 0.4550, "MCC": 0.5329},
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "PromoTech RF-HOT Nativo", "Modo": "Zero-Shot", "ROC-AUC": 0.9145, "Accuracy": 0.8682, "Especificidad (TNR)": 0.9380, "Sensibilidad (TPR)": 0.7980, "MCC": 0.7434},
        {"Cepa / Dataset": "D39V (GC-Matched)", "Modelo Evaluado": "FIMO Canónico (SigA Motif)", "Modo": "Zero-Shot", "ROC-AUC": 0.7756, "Accuracy": 0.7042, "Especificidad (TNR)": 0.9640, "Sensibilidad (TPR)": 0.4413, "MCC": 0.4762},

        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "FIMO Zero-Shot (D39V Motif -> T4)", "Modo": "Zero-Shot", "ROC-AUC": 0.9922, "Accuracy": 0.7974, "Especificidad (TNR)": 0.9932, "Sensibilidad (TPR)": 0.6016, "MCC": 0.6465},
        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "FIMO Canónico (SigA Motif)", "Modo": "Zero-Shot", "ROC-AUC": 0.9929, "Accuracy": 0.7134, "Especificidad (TNR)": 0.9986, "Sensibilidad (TPR)": 0.4282, "MCC": 0.5197},
        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "MLDSPP Zero-Shot (D39V 75% -> T4)", "Modo": "Zero-Shot", "ROC-AUC": 0.9224, "Accuracy": 0.8909, "Especificidad (TNR)": 0.9864, "Sensibilidad (TPR)": 0.7954, "MCC": 0.7965},
        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "MLDSPP (In-Domain 75/25)", "Modo": "75/25 Train", "ROC-AUC": 0.9140, "Accuracy": 0.8595, "Especificidad (TNR)": 0.9459, "Sensibilidad (TPR)": 0.7730, "MCC": 0.7299},
        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "iPro-MP (DNABERT Transformer)", "Modo": "Zero-Shot", "ROC-AUC": 0.8996, "Accuracy": 0.8022, "Especificidad (TNR)": 0.9919, "Sensibilidad (TPR)": 0.6125, "MCC": 0.6532},
        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "PromoTech RF-HOT Nativo", "Modo": "Zero-Shot", "ROC-AUC": 0.8673, "Accuracy": 0.8286, "Especificidad (TNR)": 0.9160, "Sensibilidad (TPR)": 0.7412, "MCC": 0.6675},
        {"Cepa / Dataset": "TIGR4 High Conf (Core)", "Modelo Evaluado": "PromoterLCNN (Deep CNN)", "Modo": "Zero-Shot", "ROC-AUC": 0.8731, "Accuracy": 0.8638, "Especificidad (TNR)": 0.9390, "Sensibilidad (TPR)": 0.7886, "MCC": 0.7360},

        {"Cepa / Dataset": "TIGR4 Extended (2,000)", "Modelo Evaluado": "FIMO Zero-Shot (D39V Motif -> T4)", "Modo": "Zero-Shot", "ROC-AUC": 0.9828, "Accuracy": 0.7428, "Especificidad (TNR)": 0.9890, "Sensibilidad (TPR)": 0.4965, "MCC": 0.5578},
        {"Cepa / Dataset": "TIGR4 Extended (2,000)", "Modelo Evaluado": "MLDSPP Zero-Shot (D39V 75% -> T4)", "Modo": "Zero-Shot", "ROC-AUC": 0.7352, "Accuracy": 0.6923, "Especificidad (TNR)": 0.9295, "Sensibilidad (TPR)": 0.4550, "MCC": 0.4368},
        {"Cepa / Dataset": "TIGR4 Extended (2,000)", "Modelo Evaluado": "iPro-MP (DNABERT Transformer)", "Modo": "Zero-Shot", "ROC-AUC": 0.7258, "Accuracy": 0.6885, "Especificidad (TNR)": 0.9855, "Sensibilidad (TPR)": 0.3915, "MCC": 0.4686},
        {"Cepa / Dataset": "TIGR4 Extended (2,000)", "Modelo Evaluado": "PromoTech RF-HOT Nativo", "Modo": "Zero-Shot", "ROC-AUC": 0.6820, "Accuracy": 0.6683, "Especificidad (TNR)": 0.9230, "Sensibilidad (TPR)": 0.4135, "MCC": 0.3911},
        {"Cepa / Dataset": "TIGR4 Extended (2,000)", "Modelo Evaluado": "PromoterLCNN (Deep CNN)", "Modo": "Zero-Shot", "ROC-AUC": 0.6621, "Accuracy": 0.6770, "Especificidad (TNR)": 0.9510, "Sensibilidad (TPR)": 0.4030, "MCC": 0.4232},
    ]
    df_t3 = pd.DataFrame(t3_data)

    df_t3.to_csv(OUT_CSV, index=False)
    print(f"[SUCCESS] Master statistics CSV saved ➔ {OUT_CSV}\n")

    # Generate Markdown File
    md_content = f"""# Tablas de Datos y Estadísticas Completas — D39V y TIGR4

**Proyecto:** promoter-tools  
**Organismo:** *Streptococcus pneumoniae* (D39V y TIGR4)  
**Última Actualización:** 3 de Agosto, 2026

---

## 📌 1. Características Biológicas, Transcriptómicas y Genómicas Comparativas

| Característica / Métrica | D39V Primary (Cappable-seq) | TIGR4 High Conf (Core) ⭐ | TIGR4 Extra Extended (Ruido 1,260) |
| :--- | :---: | :---: | :---: |
| **Técnica RNA-seq** | Cappable-seq (Slager 2018) | dTEX (`TSS_100.4`, Aprianto 2018) | dTEX (`Low Confidence`) |
| **N° Secuencias Positivas / Negativas** | **988 / 1,000** | **738 / 738** | **1,260 / 1,260** |
| **GC Promotores ($\text{{Mean}} \pm \text{{SD}}$)** | **$29.95\% \pm 6.29\%$** | **$30.87\% \pm 7.07\%$** | **$37.60\% \pm 8.12\%$** |
| **GC Controles Negativos (CDS)** | **$32.56\% \pm 2.77\%$** | **$33.70\% \pm 3.17\%$** | **$37.55\% \pm 3.66\%$** |
| **Gap de GC ($\Delta$)** | **`2.61%`** *(GC-Matched)* | **`2.83%`** *(GC-Matched)* | **`2.10%`** |
| **Purinas en $+1$ (A/G %)** | **`93.2%`** ⭐ | **`87.7%`** ⭐ | **`61.8%`** *(Azar)* |
| **Caja $-10$ $TATAAT$ Match (%)** | **`95.6%`** ⭐ | **`90.0%`** ⭐ | **`62.1%`** *(Colapso)* |
| **Promotores Intergénicos (%)** | **`81.3%`** | **`75.3%`** | **`65.2%`** |
| **Promotores Intragénicos (%)** | **`18.7%`** | **`24.7%`** | **`34.8%`** |
| **$5'$-UTR Mediana Length** | **$28.0\text{{ pb}}$** | **$29.0\text{{ pb}}$** | NA |
| **Leaderless mRNAs ($\le 5\text{{ pb}}$)** | **$8.4\%$** | **$10.4\%$** | NA |

---

## 🧬 2. Sobreposición en CDS y Ventana de 40 nt (Directivas Victor)

| Cepa / Dataset Evaluado | Total TSSs | Intragénicos Totales | Inicio $5'$ del CDS | Cuerpo Interno CDS | Final $3'$ del CDS | Elemento UP ($+$) | Elemento UP ($-$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D39V Cappable-seq** | 1,002 | **187 ($18.7\%$)** | 68 | 95 | 24 | 19 | 14 |
| **TIGR4 High Conf (Core)** | 742 | **183 ($24.7\%$)** | 72 | 91 | 20 | 23 | 17 |

---

## 📊 3. Tabla Maestra de Benchmarking Unificado (5 Familias de Modelos $\times$ Datasets GC-Matched)

| Cepa / Dataset Evaluado | Modelo Evaluado | Modo de Evaluación | ROC-AUC | Accuracy | Especificidad (TNR) | Sensibilidad (TPR) | Coef. Matthews (MCC) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **D39V (GC-Matched)** 🏆 | **`MLDSPP Zero-Shot (TIGR4 75% ➔ D39V)`** | Zero-Shot | **`0.9589`** ⭐ | **`91.25%`** | **`96.00%`** | **`86.50%`** | **`0.8286`** |
| **D39V (GC-Matched)** | **`MLDSPP (75/25 In-Domain)`** | In-Domain | **`0.9551`** | **`90.54%`** | **`96.00%`** | **`85.00%`** | **`0.8156`** |
| **D39V (GC-Matched)** | **`iPro-MP (DNABERT Transformer)`** | Zero-Shot | **`0.9516`** | $83.50\%$ | **`99.30%`** | $67.51\%$ | `0.7057` |
| **D39V (GC-Matched)** | **`PromoterLCNN (Deep CNN)`** | Zero-Shot | **`0.9487`** | **`90.69%`** | $96.70\%$ | $84.62\%$ | **`0.8196`** |
| **D39V (GC-Matched)** | **`FIMO (D39V STREME Motif)`** | In-Domain | **`0.9202`** | $72.59\%$ | **`99.30%`** | $45.50\%$ | `0.5329` |
| **D39V (GC-Matched)** | **`PromoTech RF-HOT Nativo`** | Zero-Shot | **`0.9145`** | $86.82\%$ | $93.80\%$ | $79.80\%$ | `0.7434` |
| **D39V (GC-Matched)** | **`FIMO Canónico (SigA Motif)`** | Zero-Shot | `0.7756` | $70.42\%$ | $96.40\%$ | $44.13\%$ | `0.4762` |
| **TIGR4 High Conf (Core)** 🏆 | **`FIMO Zero-Shot (D39V Motif ➔ T4)`** | Zero-Shot | **`0.9922`** ⭐ | $79.74\%$ | **`99.32%`** | $60.16\%$ | `0.6465` |
| **TIGR4 High Conf (Core)** | **`FIMO Canónico (SigA Motif)`** | Zero-Shot | **`0.9929`** ⭐ | $71.34\%$ | **`99.86%`** | $42.82\%$ | `0.5197` |
| **TIGR4 High Conf (Core)** | **`MLDSPP Zero-Shot (D39V 75% ➔ T4)`** | Zero-Shot | **`0.9224`** 📈 | **`89.09%`** | **`98.64%`** | **`79.54%`** | **`0.7965`** |
| **TIGR4 High Conf (Core)** | **`MLDSPP (75/25 In-Domain)`** | In-Domain | `0.9140` | $85.95\%$ | $94.59\%$ | $77.30\%$ | `0.7299` |
| **TIGR4 High Conf (Core)** | **`iPro-MP (DNABERT Transformer)`** | Zero-Shot | `0.8996` | $80.22\%$ | $99.19\%$ | $61.25\%$ | `0.6532` |
| **TIGR4 High Conf (Core)** | **`PromoterLCNN (Deep CNN)`** | Zero-Shot | `0.8731` | $86.38\%$ | $93.90\%$ | $78.86\%$ | `0.7360` |
| **TIGR4 High Conf (Core)** | **`PromoTech RF-HOT Nativo`** | Zero-Shot | `0.8673` | $82.86\%$ | $91.60\%$ | $74.12\%$ | `0.6675` |
| **TIGR4 Extended (2,000)** 🏆 | **`FIMO Zero-Shot (D39V Motif ➔ T4)`** | Zero-Shot | **`0.9828`** ⭐ | $74.28\%$ | **`98.90%`** | $49.65\%$ | `0.5578` |
| **TIGR4 Extended (2,000)** | **`MLDSPP Zero-Shot (D39V 75% ➔ TIGR4)`** | Zero-Shot | **`0.7352`** | $69.23\%$ | $92.95\%$ | $45.50\%$ | `0.4368` |
| **TIGR4 Extended (2,000)** | **`iPro-MP (DNABERT Transformer)`** | Zero-Shot | `0.7258` | $68.85\%$ | $98.55\%$ | $39.15\%$ | `0.4686` |
| **TIGR4 Extended (2,000)** | **`PromoTech RF-HOT Nativo`** | Zero-Shot | `0.6820` | $66.83\%$ | $92.30\%$ | $41.35\%$ | `0.3911` |
| **TIGR4 Extended (2,000)** | **`PromoterLCNN (Deep CNN)`** | Zero-Shot | `0.6621` | $67.70\%$ | $95.10\%$ | $40.30\%$ | `0.4232` |

---

## 🎨 4. Enlaces Directos a Gráficos de Curvas ROC Individuales (300 DPI)

* 🔵 **D39V Cappable-seq Benchmark (GC-Matched)**: [roc_d39v_gc_matched.png](file:///home/fierro/Desktop/promoter-tools/output/plots/individual_benchmarks/roc_d39v_gc_matched.png)
* 🟢 **TIGR4 High Confidence Primary (Core)**: [roc_tigr4_high_conf.png](file:///home/fierro/Desktop/promoter-tools/output/plots/individual_benchmarks/roc_tigr4_high_conf.png)
* 🟠 **TIGR4 Extended Primary (2,000 Secuencias)**: [roc_tigr4_extended.png](file:///home/fierro/Desktop/promoter-tools/output/plots/individual_benchmarks/roc_tigr4_extended.png)
"""

    with open(OUT_MD, "w") as f:
        f.write(md_content)
    print(f"[SUCCESS] Master statistics Markdown document written ➔ {OUT_MD}\n")


if __name__ == "__main__":
    main()
