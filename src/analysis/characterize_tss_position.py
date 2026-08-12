#!/usr/bin/env python3
"""
Characterize TSS positions relative to IGRs and CDS.

For each TSS, computes:
  - Whether the 81bp window falls in IGR, CDS, or a mix
  - Classification: pure_intergenic, edge, intragenic
  - How many TSS are "lost" to IGR-based analysis

Output: output/tables/tss_position_classification.tsv
"""

import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "output" / "tables" / "tss_position_classification.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)

D39V_TSS = ROOT / "data" / "benchmark" / "d39v" / "positives_81bp_metadata.tsv"
TIGR4_TSS = ROOT / "data" / "tigr4" / "positives_high_81bp_metadata.tsv"
D39V_GFF = ROOT / "data" / "reference" / "D39V.gff3"
TIGR4_GFF = ROOT / "data" / "reference" / "NC_003028.gff3"
D39V_IGR = ROOT / "output" / "intergenic" / "d39v" / "D39V_igrs.tsv"
TIGR4_IGR = ROOT / "output" / "intergenic" / "tigr4" / "TIGR4_igrs.tsv"


def load_cds(gff_path, chrom_tag):
    """Load CDS coordinates from GFF3 → list of (start, end, strand, gene)."""
    cds_list = []
    with open(gff_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            if parts[2] != "CDS":
                continue
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]
            # Extract locus_tag or gene ID
            gene = ""
            for a in attrs.split(";"):
                if "locus_tag=" in a:
                    gene = a.split("=")[1].strip()
                    break
            if not gene:
                for a in attrs.split(";"):
                    if "Parent=" in a:
                        gene = a.split("=")[1].strip()
                        break
            cds_list.append({"chrom": chrom, "start": start, "end": end,
                             "strand": strand, "gene": gene})
    return cds_list


def load_igrs(igr_path):
    """Load IGRs → list of (chrom, start, end, igr_id, length, orientation)."""
    igrs = []
    with open(igr_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            igrs.append({
                "chrom": row["chrom"].strip("\r"),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "igr_id": row["igr_id"].strip("\r"),
                "length": int(row["length"]),
                "orient": row["orientation_type"].strip("\r"),
            })
    return igrs


def window_overlap_cds(win_start, win_end, cds_list):
    """Calculate what fraction of the 81bp window overlaps CDS."""
    total_overlap = 0
    window_len = win_end - win_start + 1
    cds_genes = []
    for cds in cds_list:
        o_start = max(win_start, cds["start"])
        o_end = min(win_end, cds["end"])
        if o_start <= o_end:
            total_overlap += (o_end - o_start + 1)
            cds_genes.append(cds["gene"])
    return min(total_overlap, window_len) / window_len, cds_genes


def find_containing_igr(pos, igrs):
    """Find the IGR containing a genomic position."""
    for igr in igrs:
        if igr["start"] <= pos <= igr["end"]:
            return igr
    return None


def process_strain(tss_path, cds_list, igr_list, strain, pos_col, pos_0based):
    """Classify all TSS for one strain."""
    results = []
    with open(tss_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tss_id = row.get("Sequence_ID", row.get("Locus_Tag", "")).strip("\r")
            pos = int(row[pos_col].strip("\r"))
            if pos_0based:
                pos += 1
            strand = row.get("Strand", ".").strip("\r")
            sigma = row.get("Sigma_Factor", row.get("Location_Type", "")).strip("\r") or "?"
            confidence = row.get("Confidence_Level",
                                 row.get("Confidence_Sheet", "")).strip("\r")

            # 81bp window centered on TSS
            half = 40
            win_start = pos - half
            win_end = pos + half

            # Overlap with CDS
            cds_pct, cds_genes = window_overlap_cds(win_start, win_end, cds_list)

            # Find containing IGR
            igr = find_containing_igr(pos, igr_list)
            igr_id = igr["igr_id"] if igr else ""
            igr_len = igr["length"] if igr else 0
            igr_orient = igr["orient"] if igr else ""

            # Classification: based on whether the TSS POINT is in IGR or CDS
            tss_in_cds = False
            nearest_cds_gene = ""
            for cds in cds_list:
                if cds["start"] <= pos <= cds["end"]:
                    tss_in_cds = True
                    nearest_cds_gene = cds["gene"]
                    break

            if tss_in_cds:
                # Check if near CDS start (<50bp)
                near_start = False
                for cds in cds_list:
                    if cds["start"] <= pos <= cds["end"]:
                        if cds["strand"] == "+" and (pos - cds["start"]) < 50:
                            near_start = True
                        elif cds["strand"] == "-" and (cds["end"] - pos) < 50:
                            near_start = True
                if near_start:
                    classification = "CDS_near_start"
                    tss_lost = False  # possible 5' UTR extension
                else:
                    classification = "CDS_deep"
                    tss_lost = True

            elif igr:
                # TSS is in an IGR
                if igr_len < 20:
                    classification = "IGR_short"
                    tss_lost = True
                elif cds_pct == 0:
                    classification = "IGR_pure"
                    tss_lost = False
                elif cds_pct < 0.5:
                    classification = "IGR_minor_CDS_overlap"
                    tss_lost = False
                else:
                    classification = "IGR_major_CDS_overlap"
                    tss_lost = False
            else:
                classification = "UNKNOWN"
                tss_lost = True

            # In short IGR?
            in_short_igr = (igr_len > 0 and igr_len < 20)

            results.append({
                "tss_id": tss_id,
                "strain": strain,
                "pos": pos,
                "strand": strand,
                "sigma": sigma,
                "confidence": confidence,
                "cds_overlap_pct": round(cds_pct * 100, 1),
                "cds_genes": ",".join(cds_genes[:3]),
                "classification": classification,
                "igr_id": igr_id,
                "igr_len": igr_len,
                "igr_orient": igr_orient,
                "in_short_igr": in_short_igr,
                "tss_lost": tss_lost,
            })

    return results


def print_summary(results, strain):
    n = len(results)
    igr_pure = sum(1 for r in results if r["classification"] == "IGR_pure")
    igr_minor = sum(1 for r in results if r["classification"] == "IGR_minor_CDS_overlap")
    igr_major = sum(1 for r in results if r["classification"] == "IGR_major_CDS_overlap")
    igr_short = sum(1 for r in results if r["classification"] == "IGR_short")
    cds_near = sum(1 for r in results if r["classification"] == "CDS_near_start")
    cds_deep = sum(1 for r in results if r["classification"] == "CDS_deep")
    unknown = sum(1 for r in results if r["classification"] == "UNKNOWN")
    lost = sum(1 for r in results if r["tss_lost"])
    kept = n - lost

    in_igr = igr_pure + igr_minor + igr_major + igr_short
    in_cds = cds_near + cds_deep

    print(f"\n{'='*55}")
    print(f"  {strain} — TSS Position vs IGR/CDS")
    print(f"{'='*55}")
    print(f"  Total TSS:                     {n}")
    print(f"  ─── IN IGR ───")
    print(f"  IGR, pure (window 0%% CDS):     {igr_pure} ({igr_pure/n*100:.1f}%)")
    print(f"  IGR, <50%% CDS overlap:         {igr_minor} ({igr_minor/n*100:.1f}%)")
    print(f"  IGR, ≥50%% CDS overlap:         {igr_major} ({igr_major/n*100:.1f}%)")
    print(f"  IGR, <20bp (too short):        {igr_short} ({igr_short/n*100:.1f}%)")
    print(f"    Total in IGR:                {in_igr} ({in_igr/n*100:.1f}%)")
    print(f"  ─── IN CDS ───")
    print(f"  CDS, near start (<50bp):       {cds_near} ({cds_near/n*100:.1f}%)")
    print(f"  CDS, deep internal:            {cds_deep} ({cds_deep/n*100:.1f}%)")
    print(f"    Total in CDS:                {in_cds} ({in_cds/n*100:.1f}%)")
    if unknown:
        print(f"  Unknown:                       {unknown}")
    print(f"  ─────────────────────────")
    print(f"  TSS USABLE for IGR analysis:   {kept} ({kept/n*100:.1f}%)")
    print(f"  TSS LOST (CDS_deep + IGR_short + unknown): {lost} ({lost/n*100:.1f}%)")


def main():
    print("Loading CDS annotations...")
    d39v_cds = load_cds(D39V_GFF, "D39V")
    tigr4_cds = load_cds(TIGR4_GFF, "NC_003028")
    print(f"  D39V CDS: {len(d39v_cds)}  |  TIGR4 CDS: {len(tigr4_cds)}")

    print("Loading IGRs...")
    d39v_igrs = load_igrs(D39V_IGR)
    tigr4_igrs = load_igrs(TIGR4_IGR)
    print(f"  D39V IGRs: {len(d39v_igrs)}  |  TIGR4 IGRs: {len(tigr4_igrs)}")

    print("Processing D39V TSS...")
    d39v_results = process_strain(D39V_TSS, d39v_cds, d39v_igrs,
                                  "D39V", "TSS_Position_0based", True)
    print("Processing TIGR4 TSS...")
    tigr4_results = process_strain(TIGR4_TSS, tigr4_cds, tigr4_igrs,
                                   "TIGR4", "TSS_Position", False)

    # Print summaries
    print_summary(d39v_results, "D39V")
    print_summary(tigr4_results, "TIGR4")

    # Combined
    all_results = d39v_results + tigr4_results
    print(f"\n{'='*55}")
    print(f"  COMBINED (D39V + TIGR4)")
    print(f"{'='*55}")
    print(f"  Total: {len(all_results)}")
    for strain in ["D39V", "TIGR4"]:
        subset = [r for r in all_results if r["strain"] == strain]
        kept = [r for r in subset if not r["tss_lost"]]
        print(f"  {strain}: {len(kept)}/{len(subset)} kept ({len(kept)/len(subset)*100:.1f}%)")

    # Write output
    fieldnames = list(d39v_results[0].keys())
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Output: {OUT}")


if __name__ == "__main__":
    main()
