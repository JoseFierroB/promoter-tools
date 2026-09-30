#!/usr/bin/env python3
"""Update the TFM P1 plan with the final benchmark-focused scope.

The script preserves the existing Word layout and tables, while replacing the
draft content with the PEC1-oriented version.  It creates a recoverable copy
of the preceding document beside the updated file.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "TFM_P1_Plan_de_trabajo_Borrador.docx"
BACKUP = ROOT / "TFM_P1_Plan_de_trabajo_Borrador_pre_revision.docx"
TEMP = ROOT / "TFM_P1_Plan_de_trabajo_Borrador.tmp.docx"


ITALIC_TERMS = ("Streptococcus pneumoniae", "S. pneumoniae", "Escherichia coli")


def write_paragraph(paragraph, text: str) -> None:
    """Replace paragraph text while preserving its paragraph style."""
    paragraph.clear()
    cursor = 0
    while cursor < len(text):
        match = None
        for term in ITALIC_TERMS:
            pos = text.find(term, cursor)
            if pos != -1 and (match is None or pos < match[0]):
                match = (pos, term)
        if match is None:
            paragraph.add_run(text[cursor:])
            break
        pos, term = match
        if pos > cursor:
            paragraph.add_run(text[cursor:pos])
        run = paragraph.add_run(term)
        run.italic = True
        cursor = pos + len(term)


def write_cell(cell, text: str) -> None:
    """Replace one table cell without changing its table-level formatting."""
    first = cell.paragraphs[0]
    write_paragraph(first, text)
    for para in cell.paragraphs[1:]:
        para.clear()
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table(table, rows: list[list[str]]) -> None:
    if len(rows) != len(table.rows):
        raise ValueError(f"Expected {len(table.rows)} rows, received {len(rows)}")
    for row, values in zip(table.rows, rows):
        if len(values) != len(row.cells):
            raise ValueError("Unexpected table column count")
        for cell, value in zip(row.cells, values):
            write_cell(cell, value)


def set_table_widths(table, widths: list[float]) -> None:
    """Set deliberate column widths so dense planning tables remain readable."""
    if len(widths) != len(table.columns):
        raise ValueError("Unexpected table column count")
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.first_child_found_in("w:tblLayout")
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)


def repeat_header_row(table) -> None:
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    if tr_pr.find(qn("w:tblHeader")) is not None:
        return
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def delete_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)
    paragraph._p = paragraph._element = None


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    if not BACKUP.exists():
        shutil.copy2(INPUT, BACKUP)

    # Rebuild from the original draft on every run so the update is idempotent.
    doc = Document(BACKUP)
    p = doc.paragraphs

    write_paragraph(p[0], "Plan de trabajo del TFM")
    write_paragraph(
        p[1],
        "Benchmark reproducible de predictores de promotores en Streptococcus pneumoniae",
    )
    write_paragraph(p[2], "Borrador revisado para la actividad P1")
    write_paragraph(
        p[5],
        "Este documento presenta el alcance, los objetivos, el enfoque metodológico y la planificación del TFM. Las fechas concretas se ajustarán al calendario oficial de la asignatura.",
    )

    # 0. Title and keywords
    set_table(
        doc.tables[1],
        [
            ["Elemento", "Propuesta"],
            [
                "Título",
                "Benchmark reproducible de predictores de promotores en Streptococcus pneumoniae",
            ],
            [
                "Palabras clave",
                "promotores bacterianos; Streptococcus pneumoniae; predicción de promotores; sitios de inicio de transcripción; benchmarking; reproducibilidad; recursos computacionales",
            ],
        ],
    )
    write_cell(doc.tables[0].cell(6, 1), "Borrador revisado P1")

    # 1. Context and justification
    context = [
        "Este Trabajo de Fin de Máster se enmarca en el ámbito de la bioinformática y la genómica bacteriana, concretamente en el análisis computacional de regiones promotoras. Los promotores bacterianos condicionan el inicio de la transcripción y, por tanto, influyen en la expresión génica, la adaptación celular y la virulencia. Su identificación facilita la anotación funcional de genomas. Sin embargo, el número de genomas bacterianos secuenciados supera ampliamente la capacidad de caracterizar experimentalmente sus promotores.",
        "Para abordar esta limitación se han desarrollado predictores basados en motivos, reglas, modelos estadísticos, aprendizaje automático, aprendizaje profundo y modelos de lenguaje genómico. Estas herramientas se han entrenado con especies, tipos de promotor, conjuntos negativos y criterios de evaluación diferentes. Sus resultados no siempre son comparables de manera directa, por lo que son necesarios benchmarks con conjuntos de referencia bien definidos, métricas homogéneas y condiciones de ejecución documentadas (Cassiano y Silva Rocha, 2020; Weber et al., 2019).",
        "Este trabajo se centra en Streptococcus pneumoniae, una bacteria Gram positiva y anaerobia facultativa de importancia clínica, capaz de colonizar la nasofaringe y de causar otitis, sinusitis, neumonía, bacteriemia y meningitis (CDC, 2026). Además de su relevancia clínica, S. pneumoniae presenta una elevada diversidad poblacional. Variantes en sus regiones promotoras pueden modificar la expresión génica, la encapsulación, la adhesión y la virulencia entre cepas (Wen et al., 2016; Barton et al., 2025).",
        "En particular, S. pneumoniae posee un factor sigma principal, SigA, y un factor sigma alternativo, SigX o ComX, implicado en el estado de competencia y la transformación genética. Esto permite estudiar predictores en grupos de promotores con funciones reguladoras diferentes. El TFM desarrollará un benchmark reproducible de herramientas de predicción de promotores en S. pneumoniae, comparando precisión, comportamiento entre cepas y factores sigma, requisitos computacionales y facilidad de ejecución. El resultado servirá de base para futuros estudios de variación promotora a gran escala.",
    ]
    for paragraph, text in zip(p[9:12], context[:3]):
        write_paragraph(paragraph, text)
    fourth_context_paragraph = p[12].insert_paragraph_before(style="Normal")
    write_paragraph(fourth_context_paragraph, context[3])

    # 2. General description
    description = [
        "El TFM consistirá en diseñar, ejecutar y analizar un benchmark reproducible de nueve herramientas de predicción de promotores bacterianos. El benchmark base utilizará un conjunto canónico de alta confianza de S. pneumoniae D39V y TIGR4, compuesto por 3.465 secuencias: 1.727 positivas asociadas a sitios de inicio de transcripción y 1.738 controles negativos procedentes de regiones codificantes. Cada conjunto se documentará mediante metadatos y manifiestos que permitan identificar su origen, composición y versión.",
        "Las herramientas se integrarán en un flujo común de ejecución local. Cada corrida generará tres grupos de resultados: inferencia, recursos y tablas. La inferencia contendrá predicciones y curvas ROC y de precisión-recall; los recursos recogerán tiempo de ejecución, RAM, VRAM y tamaño de los modelos; y las tablas almacenarán métricas, matrices de confusión y resúmenes comparativos. Se compararán regímenes con uno y dieciséis núcleos de CPU, con y sin GPU cuando la herramienta lo permita, manteniendo el mismo conjunto de entrada.",
        "El producto final será un protocolo de benchmark, datasets trazables, un flujo de ejecución y análisis reutilizable, resultados comparativos en PNG y PDF, tablas TSV y una memoria que interprete los hallazgos desde una perspectiva bioinformática y computacional.",
    ]
    for paragraph, text in zip(p[13:16], description):
        write_paragraph(paragraph, text)

    # 3. Sustainability, ethics and diversity
    write_paragraph(
        p[18],
        "El trabajo medirá los recursos necesarios para ejecutar cada herramienta. Esta información permitirá identificar alternativas que ofrezcan una relación razonable entre desempeño y coste computacional, evitando repetir corridas o utilizar configuraciones de alta demanda cuando no aporten una mejora relevante. La reutilización de datasets, entornos documentados, manifiestos y resultados versionados reducirá cálculos innecesarios y favorecerá un uso eficiente de la infraestructura disponible.",
    )
    write_paragraph(
        p[20],
        "El estudio se basa en datos genómicos públicos y no contempla intervención clínica, experimentación con personas ni manipulación de muestras biológicas. Las herramientas se evaluarán como métodos de apoyo a la investigación y no como sistemas de diagnóstico o de decisión clínica. El código, los parámetros, las versiones y las limitaciones del benchmark se documentarán para permitir la revisión crítica de los resultados.",
    )
    write_paragraph(
        p[22],
        "La comparación incluirá configuraciones de cómputo moderadas y de mayor capacidad para no asumir que todas las personas usuarias tienen acceso a GPU o a servidores especializados. La presentación separará desempeño predictivo y demanda de recursos, facilitando decisiones informadas en laboratorios y entornos académicos con capacidades computacionales diferentes.",
    )

    # 4. Objectives
    write_paragraph(
        p[25],
        "Desarrollar un benchmark reproducible para evaluar herramientas de predicción de promotores en Streptococcus pneumoniae, integrando desempeño predictivo, análisis por cepa y factor sigma, consumo de recursos y escalabilidad.",
    )
    write_paragraph(p[26], "")
    write_paragraph(p[27], "")
    set_table(
        doc.tables[2],
        [
            ["OG", "Código", "Objetivo específico"],
            ["OG1", "OE1.1", "Definir el conjunto canónico, los criterios de inclusión y la trazabilidad de las secuencias positivas y negativas."],
            ["OG1", "OE1.2", "Documentar datasets y manifiestos que permitan verificar su composición, origen y versión."],
            ["OG1", "OE1.3", "Integrar nueve herramientas bajo una interfaz de línea de comandos y condiciones de entrada comparables."],
            ["OG1", "OE1.4", "Ejecutar las herramientas en regímenes de CPU y GPU definidos."],
            ["OG1", "OE1.5", "Calcular métricas de clasificación, curvas ROC y de precisión-recall, y matrices de confusión por herramienta."],
            ["OG1", "OE1.6", "Analizar resultados por cepa y por grupos de promotores asociados a SigA, ComX y otras categorías."],
            ["OG1", "OE1.7", "Medir tiempo de ejecución, RAM, VRAM y tamaño de modelo en regímenes equivalentes."],
            ["OG1", "OE1.8", "Generar gráficos individuales en PNG y PDF, tablas TSV y comparaciones organizadas por corrida."],
            ["OG1", "OE1.9", "Comparar el desempeño y los recursos entre herramientas y configuraciones de hardware."],
            ["OG1", "OE1.10", "Documentar el flujo y elaborar la memoria final y la presentación de resultados."],
        ],
    )
    set_table_widths(doc.tables[2], [0.45, 0.7, 5.35])
    repeat_header_row(doc.tables[2])

    # 5. Approach and method
    set_table(
        doc.tables[3],
        [
            ["Estrategia", "Aportación", "Decisión"],
            ["Desarrollar un predictor nuevo", "Podría aportar un modelo propio, pero ampliaría el alcance hacia entrenamiento, ajuste y validación de un método adicional.", "No priorizada"],
            ["Revisión bibliográfica", "Permitirá situar las herramientas y los datos disponibles, pero no sustituye una evaluación experimental propia.", "Complementaria"],
            ["Benchmark empírico reproducible", "Permitirá comparar herramientas existentes con datos, métricas, versiones y condiciones de cómputo homogéneas.", "Estrategia elegida"],
        ],
    )
    write_paragraph(
        p[32],
        "Se elige un benchmark empírico reproducible porque responde directamente a la pregunta de qué herramientas son adecuadas para predecir promotores en S. pneumoniae y bajo qué restricciones computacionales. La revisión bibliográfica acompañará el diseño de los datasets y la interpretación de los resultados. El desarrollo de un predictor nuevo queda fuera del alcance del TFM para concentrar el trabajo en una comparación sólida, trazable y evaluable.",
    )
    methodology = [
        "Revisión de herramientas disponibles, publicaciones de referencia y características de sus modelos, entradas y salidas.",
        "Definición y documentación del dataset canónico a partir de información de TSS de D39V y TIGR4 y controles negativos procedentes de regiones codificantes.",
        "Estandarización de entradas y configuración de nueve herramientas mediante una interfaz de línea de comandos común.",
        "Ejecución de corridas por régimen de cómputo y registro estructurado de predicciones, recursos y metadatos de cada corrida.",
        "Cálculo de ROC-AUC, PR-AUC, sensibilidad, especificidad, precisión, F1, MCC y matrices de confusión, incluyendo análisis por cepa y categoría sigma cuando sea posible.",
        "Medición de tiempo de pared, RAM, VRAM y tamaño de modelo, junto con análisis de escalabilidad entre regímenes equivalentes.",
        "Generación de gráficos individuales en PNG y PDF, tablas TSV y comparaciones por hardware, preservando los manifiestos de entrada y configuración.",
        "Interpretación integrada de desempeño, coste computacional, escalabilidad y aplicabilidad práctica de cada herramienta.",
    ]
    for paragraph, text in zip(p[34:42], methodology):
        write_paragraph(paragraph, text)

    # 6. Planning
    write_paragraph(
        p[43],
        "La planificación se expresa en semanas de trabajo y deberá alinearse con las fechas de entrega, tutorías e hitos establecidos en el Plan Docente. Las tareas se solapan de forma controlada para documentar el benchmark y redactar la memoria desde las primeras fases del proyecto.",
    )
    set_table(
        doc.tables[4],
        [
            ["Código", "Tarea", "Objetivos", "Duración", "Resultado"],
            ["T1", "Delimitar el benchmark y revisar bibliografía", "OG1", "Semanas 1 a 2", "Marco teórico y protocolo inicial"],
            ["T2", "Definir dataset canónico y criterios de trazabilidad", "OE1.1, OE1.2", "Semanas 2 a 4", "Especificación de datos y manifiestos"],
            ["T3", "Integrar herramientas y estandarizar la CLI", "OE1.3", "Semanas 3 a 6", "Flujo de ejecución reproducible"],
            ["T4", "Ejecutar prueba piloto y validar métricas y recursos", "OE1.4, OE1.6", "Semanas 5 a 7", "Protocolo validado"],
            ["T5", "Realizar benchmark por régimen de cómputo", "OE1.4, OE1.6", "Semanas 7 a 10", "Corridas y tablas de recursos"],
            ["T6", "Analizar cepas y categorías sigma", "OE1.5", "Semanas 9 a 11", "Resultados estratificados"],
            ["T7", "Generar comparativas, figuras y tablas finales", "OE1.7", "Semanas 11 a 13", "Resultados consolidados"],
            ["T8", "Redactar memoria, revisar y preparar presentación", "OE1.8, OE1.9", "Semanas 1 a 16", "Memoria y presentación"],
        ],
    )
    set_table(
        doc.tables[5],
        [
            ["Tarea", "S1-2", "S3-4", "S5-6", "S7-8", "S9-10", "S11-12", "S13-14", "S15-16"],
            ["T1 Alcance", "●", "", "", "", "", "", "", ""],
            ["T2 Datos", "", "●", "", "", "", "", "", ""],
            ["T3 Integración", "", "●", "●", "", "", "", "", ""],
            ["T4 Piloto", "", "", "●", "●", "", "", "", ""],
            ["T5 Benchmark", "", "", "", "●", "●", "", "", ""],
            ["T6 Estratificación", "", "", "", "", "●", "●", "", ""],
            ["T7 Análisis", "", "", "", "", "", "●", "●", ""],
            ["T8 Memoria", "●", "●", "●", "●", "●", "●", "●", "●"],
        ],
    )
    set_table(
        doc.tables[6],
        [
            ["Hito", "Criterio de consecución", "Fecha propuesta"],
            ["H1", "Alcance, pregunta de trabajo y bibliografía inicial validados", "Final de la semana 2"],
            ["H2", "Dataset canónico y manifiestos de procedencia documentados", "Final de la semana 4"],
            ["H3", "Flujo integrado y prueba piloto completados", "Final de la semana 7"],
            ["H4", "Corridas principales y mediciones de recursos finalizadas", "Final de la semana 10"],
            ["H5", "Análisis por cepa y categoría sigma, tablas y gráficos cerrados", "Final de la semana 13"],
            ["H6", "Memoria revisada y presentación preparada", "Final de la semana 16"],
        ],
    )
    set_table(
        doc.tables[7],
        [
            ["Riesgo", "Impacto potencial", "Medida preventiva"],
            ["Disponibilidad y calidad de datos", "Las etiquetas de TSS y los controles negativos pueden requerir armonización antes de su uso.", "Definir criterios de inclusión, validación y trazabilidad; conservar versiones y manifiestos."],
            ["Comparabilidad de herramientas", "Las herramientas pueden requerir entradas, dependencias y preprocesamientos distintos.", "Estandarizar entradas, documentar versiones y realizar una prueba piloto temprana."],
            ["Recursos de cómputo", "Las ejecuciones con GPU o datasets grandes pueden requerir más tiempo del previsto.", "Priorizar un benchmark base y mantener configuraciones de contingencia en CPU."],
            ["Alcance del TFM", "La diversidad de herramientas y análisis puede ampliar excesivamente el trabajo.", "Mantener nueve herramientas, el dataset canónico y extensiones claramente delimitadas."],
            ["Tiempo disponible", "La ejecución, el análisis y la redacción pueden competir por el tiempo final.", "Redactar de forma incremental y usar hitos de revisión quincenales."],
            ["Interpretación de resultados", "Las métricas pueden variar entre herramientas por sus supuestos y dominios de entrenamiento.", "Separar desempeño, coste computacional y aplicabilidad; evitar recomendaciones fuera del dominio evaluado."],
        ],
    )

    # 7. Expected results
    expected = [
        "Un protocolo de benchmark reproducible para herramientas de predicción de promotores bacterianos en S. pneumoniae.",
        "Un dataset canónico de D39V y TIGR4 con 3.465 secuencias, metadatos y manifiestos de procedencia.",
        "Un flujo de línea de comandos que ejecute nueve herramientas y organice cada corrida en directorios de inferencia, recursos y tablas.",
        "Predicciones, curvas ROC y de precisión-recall, matrices de confusión y tablas de métricas por herramienta.",
        "Resultados estratificados por cepa y categoría sigma cuando la evidencia disponible permita esta separación.",
        "Mediciones comparables de tiempo, RAM, VRAM, tamaño de modelo y escalabilidad para configuraciones de uno y dieciséis núcleos de CPU y GPU.",
        "Gráficos comparativos individuales en PNG y PDF, junto con tablas TSV de resultados consolidados.",
        "La memoria final, el código documentado y una presentación virtual de los resultados.",
    ]
    for paragraph, text in zip(p[51:58], expected[:7]):
        write_paragraph(paragraph, text)
    write_paragraph(p[58], "8  Bibliografía inicial")
    final_expected = p[58].insert_paragraph_before(style="List Bullet")
    write_paragraph(final_expected, expected[7])

    # 8. Bibliography: no URL text in the document.
    bibliography = [
        "Barton, T. E., Green, A. E., Mellor, K. C., et al. (2025). Naturally acquired promoter variation influences Streptococcus pneumoniae infection outcomes. Cell Host & Microbe, 33(9), 1473-1483.e6. doi: 10.1016/j.chom.2025.08.005",
        "Cassiano, M. H. A., y Silva Rocha, R. (2020). Benchmarking bacterial promoter prediction tools: Potentialities and limitations. mSystems, 5(4), e00439-20. doi: 10.1128/mSystems.00439-20",
        "Centers for Disease Control and Prevention. (2026). Clinical overview of pneumococcal disease.",
        "Chevez Guardado, R., y Peña Castillo, L. (2022). Critical assessment of computational tools for prokaryotic and eukaryotic promoter prediction. Nucleic Acids Research, 50(5), 2576-2596. doi: 10.1093/nar/gkac133",
        "Sandve, G. K., Nekrutenko, A., Taylor, J., y Hovig, E. (2013). Ten simple rules for reproducible computational research. PLoS Computational Biology, 9(10), e1003285. doi: 10.1371/journal.pcbi.1003285",
        "Slager, J., Aprianto, R., y Veening, J.-W. (2018). Deep genome annotation of the opportunistic human pathogen Streptococcus pneumoniae D39. Genome Biology, 19, 112. doi: 10.1186/s13059-018-1501-5",
        "Warrier, I., et al. (2018). The transcriptional landscape of Streptococcus pneumoniae TIGR4 reveals a complex operon architecture and abundant riboregulation critical for growth and virulence. PLoS Pathogens, 14(12), e1007461. doi: 10.1371/journal.ppat.1007461",
        "Weber, L. M., Saelens, W., Cannoodt, R., Soneson, C., et al. (2019). Essential guidelines for computational method benchmarking. Genome Biology, 20, 125. doi: 10.1186/s13059-019-1738-8",
        "Wen, Z., Liu, Y., Qu, F., y Zhang, J.-R. (2016). Allelic variation of the capsule promoter diversifies encapsulation and virulence in Streptococcus pneumoniae. Scientific Reports, 6, 30176. doi: 10.1038/srep30176",
    ]
    fresh_paragraphs = doc.paragraphs
    bibliography_heading = next(
        paragraph
        for paragraph in fresh_paragraphs
        if "Bibliografía inicial" in paragraph.text
    )
    note = next(
        paragraph
        for paragraph in fresh_paragraphs
        if paragraph.text.startswith("Nota para la versión final:")
    )
    start = fresh_paragraphs.index(bibliography_heading) + 1
    end = fresh_paragraphs.index(note)
    for paragraph in reversed(fresh_paragraphs[start:end]):
        delete_paragraph(paragraph)
    for text in bibliography:
        inserted = note.insert_paragraph_before(style="Normal")
        write_paragraph(inserted, text)
    write_paragraph(
        note,
        "Nota para la versión final: completar los datos personales y alinear las semanas, hitos y entregas con el Plan Docente y la plantilla oficial de la asignatura.",
    )

    # Update document metadata and header while preserving the existing page number fields.
    doc.core_properties.title = "Plan de trabajo del TFM"
    doc.core_properties.subject = "Benchmark reproducible de promotores en Streptococcus pneumoniae"
    doc.core_properties.comments = "Borrador revisado para la actividad P1"
    for section in doc.sections:
        for header_paragraph in section.header.paragraphs:
            if header_paragraph.text.strip():
                write_paragraph(header_paragraph, "Plan de trabajo del TFM")

    doc.save(TEMP)
    os.replace(TEMP, INPUT)
    print(f"Updated: {INPUT}")
    print(f"Backup:  {BACKUP}")


if __name__ == "__main__":
    main()
