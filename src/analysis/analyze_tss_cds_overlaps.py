#!/usr/bin/env python3
"""
TSS-CDS Overlap Analysis — D39V vs TIGR4.
Computes intragenic TSS counts, strand-specific overlaps, promoter window overlaps,
regulatory element proximity, and sigma factor breakdown per strain.
Outputs: output/tables/tss_cds_overlap_analysis.tsv
"""
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from BCBio import GFF


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_FILE = ROOT / "output" / "tables" / "tss_cds_overlap_analysis.tsv"


def load_d39v_cds():
    """Load CDS coordinates from D39V sequence.gff3 sub-features."""
    cds_list = []
    with open(ROOT / "data/reference/sequence.gff3") as f:
        for rec in GFF.parse(f):
            for feat in rec.features:
                for sf in getattr(feat, "sub_features", []):
                    if sf.type == "CDS":
                        cds_list.append({
                            "start": int(sf.location.start),
                            "end": int(sf.location.end),
                            "strand": "+" if sf.location.strand == 1 else "-",
                            "gene": feat.id,
                        })
    return cds_list


def load_d39v_binding_sites():
    """Load protein binding sites from D39V GFF3."""
    sites = []
    with open(ROOT / "data/reference/sequence.gff3") as f:
        for rec in GFF.parse(f):
            for feat in rec.features:
                if feat.type == "protein_binding_site":
                    bound = feat.qualifiers.get("bound_moiety", [""])[0]
                    sites.append({
                        "start": int(feat.location.start),
                        "end": int(feat.location.end),
                        "strand": "+" if feat.location.strand == 1 else "-",
                        "bound_moiety": bound,
                    })
    return sites


def load_tigr4_genes():
    """Load gene boundaries from all sheets of TIGR4 S1_TSS.xlsx."""
    xlsx = ROOT / "data/tigr4/S1_TSS.xlsx"
    all_genes = {}
    for sheet in ["High Confidence (TSS_100.4)", "Low Confidence (TSS_2.1)",
                  "Secondary TSS, High confidence", "Secondary TSS, Low confidence"]:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        c_locus = "Locus_tag" if "Locus_tag" in df.columns else "Locus"
        c_start = "Locus_start" if "Locus_start" in df.columns else "Start"
        c_end = "Locus_end" if "Locus_end" in df.columns else "End"
        for _, row in df.iterrows():
            if pd.notna(row.get(c_locus)) and pd.notna(row.get(c_start)):
                all_genes[row[c_locus]] = (int(row[c_start]), int(row[c_end]), row.get("Strand", "+"))
    return all_genes


def load_tigr4_tss():
    """Load TSS from all Excel sheets."""
    xlsx = ROOT / "data/tigr4/S1_TSS.xlsx"
    tss_all = []
    for sheet in ["High Confidence (TSS_100.4)", "Low Confidence (TSS_2.1)",
                  "Secondary TSS, High confidence", "Secondary TSS, Low confidence"]:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        c_pos = "TSS_position" if "TSS_position" in df.columns else None
        c_strand = "Strand" if "Strand" in df.columns else None
        if c_pos:
            for _, row in df.iterrows():
                if pd.notna(row.get(c_pos)):
                    tss_all.append({
                        "sheet": sheet,
                        "pos": int(row[c_pos]),
                        "strand": row.get(c_strand, "+"),
                        "coding": row.get("Within_coding_vs_intergenic", ""),
                    })
    return tss_all


def analyze_d39v():
    rows = []
    cds = load_d39v_cds()
    binding = load_d39v_binding_sites()
    tss = pd.read_csv(ROOT / "data/benchmark/d39v/positives_81bp_metadata.tsv", sep="\t")

    n_tss = len(tss)
    n_cds = len(cds)
    n_binding = len(binding)

    rows.append(("D39V", "Total TSS", str(n_tss)))
    rows.append(("D39V", "Total CDS (GFF3)", str(n_cds)))

    # TSS inside/outside CDS
    inside = 0; same_s = 0; opp_s = 0
    in_5p = 0; in_3p = 0; in_mid = 0
    prom_overlap = 0; near_5p = 0; near_3p = 0

    for _, t in tss.iterrows():
        pos = t["TSS_Position_0based"]
        strand = t["Strand"]
        prom_start = pos - 60
        prom_end = pos + 20

        in_cds = False
        for c in cds:
            if c["start"] <= pos < c["end"]:
                in_cds = True
                inside += 1
                if strand == c["strand"]: same_s += 1
                else: opp_s += 1
                rel = pos - c["start"]; c_len = c["end"] - c["start"]
                if rel < 40: in_5p += 1
                elif rel > c_len - 40: in_3p += 1
                else: in_mid += 1
                break
            if prom_start < c["end"] and prom_end > c["start"]:
                prom_overlap += 1
                break
        if not in_cds:
            for c in cds:
                if 0 < c["start"] - pos <= 40:
                    near_5p += 1; break
                if 0 < pos - c["end"] <= 40:
                    near_3p += 1; break

    rows.append(("D39V", "TSS dentro de CDS", f"{inside} ({inside/n_tss*100:.1f}%)"))
    rows.append(("D39V", "  - Misma hebra que CDS", str(same_s)))
    rows.append(("D39V", "  - Hebra opuesta", str(opp_s)))
    rows.append(("D39V", "  - En primeros 40nt (5')", str(in_5p)))
    rows.append(("D39V", "  - En últimos 40nt (3')", str(in_3p)))
    rows.append(("D39V", "  - En cuerpo medio", str(in_mid)))
    rows.append(("D39V", "TSS fuera de CDS", f"{n_tss - inside} ({ (n_tss-inside)/n_tss*100:.1f}%)"))
    rows.append(("D39V", "Ventana promotora (81bp) solapa CDS", f"{prom_overlap} ({prom_overlap/n_tss*100:.1f}%)"))
    rows.append(("D39V", "TSS ≤40nt upstream de 5' CDS", f"{near_5p} ({near_5p/n_tss*100:.1f}%)"))
    rows.append(("D39V", "TSS ≤40nt downstream de 3' CDS", f"{near_3p} ({near_3p/n_tss*100:.1f}%)"))

    # Binding sites
    bs_in_cds = sum(1 for b in binding if any(c["start"] <= b["start"] < c["end"] for c in cds))
    bs_near_5p = sum(1 for b in binding if any(0 < c["start"] - b["end"] <= 40 for c in cds))
    bs_near_3p = sum(1 for b in binding if any(0 < b["start"] - c["end"] <= 40 for c in cds))
    bs_near_tss = sum(1 for b in binding if any(abs(b["start"] - t["TSS_Position_0based"]) <= 40 for _, t in tss.iterrows()))

    rows.append(("D39V", "Sitios de unión proteica totales", str(n_binding)))
    rows.append(("D39V", "  - Dentro de CDS", f"{bs_in_cds} ({bs_in_cds/n_binding*100:.1f}%)"))
    rows.append(("D39V", "  - Cerca 5' CDS (≤40nt)", str(bs_near_5p)))
    rows.append(("D39V", "  - Cerca 3' CDS (≤40nt)", str(bs_near_3p)))
    rows.append(("D39V", "  - Cerca de algún TSS (≤40nt)", f"{bs_near_tss} ({bs_near_tss/n_binding*100:.1f}%)"))

    # Sigma factors
    siga = (tss["Sigma_Factor"] == "SigA").sum()
    sigx = (tss["Sigma_Factor"] == "SigX").sum()
    none_s = n_tss - siga - sigx
    rows.append(("D39V", "SigA / SigX / None", f"{siga} / {sigx} / {none_s}"))

    siga_inside = sum(1 for _, t in tss.iterrows() if t["Sigma_Factor"] == "SigA" and any(c["start"] <= t["TSS_Position_0based"] < c["end"] for c in cds))
    sigx_inside = sum(1 for _, t in tss.iterrows() if t["Sigma_Factor"] == "SigX" and any(c["start"] <= t["TSS_Position_0based"] < c["end"] for c in cds))
    rows.append(("D39V", "SigA dentro de CDS", f"{siga_inside} ({siga_inside/siga*100:.1f}%)" if siga else "n/a"))
    rows.append(("D39V", "SigX dentro de CDS", f"{sigx_inside} ({sigx_inside/sigx*100:.1f}%)" if sigx else "n/a"))

    return rows, n_tss


def analyze_tigr4():
    rows = []
    genes = load_tigr4_genes()
    all_tss = load_tigr4_tss()
    t4_meta = pd.read_csv(ROOT / "data/tigr4/positives_high_81bp_metadata.tsv", sep="\t")
    n_tss = len(t4_meta)

    rows.append(("TIGR4", "Total TSS (high conf)", str(n_tss)))
    rows.append(("TIGR4", "Total genes (Excel, all sheets)", str(len(genes))))

    gene_list = [(locus, gs, ge, st) for locus, (gs, ge, st) in genes.items()]

    inside = 0; same_s = 0; opp_s = 0
    in_5p = 0; in_3p = 0; in_mid = 0
    near_5p = 0; near_3p = 0
    prom_overlap = 0

    for _, t in t4_meta.iterrows():
        pos = t["TSS_Position"]
        strand = t["Strand"]
        prom_start = pos - 60
        prom_end = pos + 20
        in_gene = False

        for _, gs, ge, gs_strand in gene_list:
            if gs <= pos <= ge:
                in_gene = True; inside += 1
                if strand == gs_strand: same_s += 1
                else: opp_s += 1
                rel = pos - gs; g_len = ge - gs
                if rel < 40: in_5p += 1
                elif rel > g_len - 40: in_3p += 1
                else: in_mid += 1
                break
            if prom_start < ge and prom_end > gs:
                prom_overlap += 1; break
        if not in_gene:
            for _, gs, ge, _ in gene_list:
                if 0 < gs - pos <= 40: near_5p += 1; break
                if 0 < pos - ge <= 40: near_3p += 1; break

    rows.append(("TIGR4", "TSS dentro de gen", f"{inside} ({inside/n_tss*100:.1f}%)"))
    rows.append(("TIGR4", "  - Misma hebra que gen", str(same_s)))
    rows.append(("TIGR4", "  - Hebra opuesta", str(opp_s)))
    rows.append(("TIGR4", "  - En primeros 40nt (5')", str(in_5p)))
    rows.append(("TIGR4", "  - En últimos 40nt (3')", str(in_3p)))
    rows.append(("TIGR4", "  - En cuerpo medio", str(in_mid)))
    rows.append(("TIGR4", "TSS fuera de gen", f"{n_tss - inside} ({(n_tss-inside)/n_tss*100:.1f}%)"))
    rows.append(("TIGR4", "  - Cerca 5' gen (≤40nt)", str(near_5p)))
    rows.append(("TIGR4", "  - Cerca 3' gen (≤40nt)", str(near_3p)))
    rows.append(("TIGR4", "Ventana promotora (81bp) solapa gen", f"{prom_overlap} ({prom_overlap/n_tss*100:.1f}%)"))

    # dTEX labels
    inter_coding = t4_meta["Location_Type"].value_counts()
    inter = inter_coding.get("tss_intergenic", 0)
    coding = inter_coding.get("tss_within_coding", 0)
    rows.append(("TIGR4", "Label dTEX: intergenic / coding", f"{inter} / {coding}"))

    intra = t4_meta["Is_Intragenic"].value_counts()
    rows.append(("TIGR4", "Is_Intragenic: False / True", f"{intra.get(False, 0)} / {intra.get(True, 0)}"))

    cds_pos = t4_meta["CDS_Position_Type"].value_counts()
    rows.append(("TIGR4", "CDS_Position_Type: intergenic / 5' / 3' / body",
                 f"{cds_pos.get('intergenic',0)} / {cds_pos.get('5_prime_start',0)} / {cds_pos.get('3_prime_end',0)} / {cds_pos.get('internal_body',0)}"))

    up = (t4_meta["UP_Element_Overlap"] == True).sum()
    rows.append(("TIGR4", "UP_Element_Overlap = True", f"{up} ({up/n_tss*100:.1f}%)"))

    # Secondary TSS
    sec_high = pd.read_excel(ROOT / "data/tigr4/S1_TSS.xlsx", sheet_name="Secondary TSS, High confidence")
    sec_low = pd.read_excel(ROOT / "data/tigr4/S1_TSS.xlsx", sheet_name="Secondary TSS, Low confidence")
    sec_pos = list(pd.to_numeric(sec_high["Secondary_TSS"], errors="coerce").dropna())
    sec_pos += list(pd.to_numeric(sec_low["Secondary_TSS"], errors="coerce").dropna())

    sec_in = sum(1 for p in sec_pos if any(gs <= p <= ge for _, gs, ge, _ in gene_list))
    rows.append(("TIGR4", "TSS secundarios totales", f"{len(sec_pos)} (48 high + 74 low)"))
    rows.append(("TIGR4", "  - Dentro de gen", f"{sec_in} ({sec_in/len(sec_pos)*100:.1f}%)"))
    rows.append(("TIGR4", "  - Fuera de gen", f"{len(sec_pos)-sec_in} ({(len(sec_pos)-sec_in)/len(sec_pos)*100:.1f}%)"))

    # All sheets summary
    all_inside = sum(1 for t in all_tss if any(gs <= t["pos"] <= ge for _, gs, ge, _ in gene_list))
    all_coding = sum(1 for t in all_tss if t["coding"] == "tss_within_coding")
    all_inter = sum(1 for t in all_tss if t["coding"] == "tss_intergenic")
    rows.append(("TIGR4", "Total TSS (all sheets)", str(len(all_tss))))
    rows.append(("TIGR4", "TSS all sheets: intergenic / coding", f"{all_inter} / {all_coding}"))
    rows.append(("TIGR4", "TSS all sheets dentro de gen", f"{all_inside} ({all_inside/len(all_tss)*100:.1f}%)"))

    return rows, n_tss


def main():
    parser = argparse.ArgumentParser(description="TSS-CDS overlap analysis for D39V and TIGR4")
    parser.add_argument("-o", "--output", default=str(OUT_FILE), help="Output TSV path")
    args = parser.parse_args()

    rows = []
    d39v_rows, _ = analyze_d39v()
    t4_rows, _ = analyze_tigr4()
    rows.extend(d39v_rows)
    rows.extend(t4_rows)

    df = pd.DataFrame(rows, columns=["Cepa", "Métrica", "Valor"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    print(f"Saved: {out}")
    print(f"  {len(rows)} metrics ({len(d39v_rows)} D39V + {len(t4_rows)} TIGR4)")


if __name__ == "__main__":
    main()
