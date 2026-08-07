#!/usr/bin/env python3
"""
Validate which conserved IGRs (D39V ↔ TIGR4) contain known TSS.

Uses BEDTools intersect for robust TSS-to-IGR overlap detection,
replacing the slower Python-loop approach.

Output: analysis_igr/outputs/tables/conserved_igrs_tss_validation.tsv
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_igr_metadata(path: Path) -> dict:
    """Load IGR TSV → dict[igr_id] = {chrom, start_1, end_1, len, orient, flank_info}."""
    igrs = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            igrs[row["igr_id"].strip("\r")] = {
                "chrom": row["chrom"].strip("\r"),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "length": int(row["length"]),
                "orientation": row["orientation_type"].strip("\r"),
                "left_gene": f"{row['left_cds'].strip(chr(13))}({row['left_strand'].strip(chr(13))})",
                "right_gene": f"{row['right_cds'].strip(chr(13))}({row['right_strand'].strip(chr(13))})",
            }
    return igrs


def make_tss_bed(meta_path: Path, pos_col: str, pos_is_0based: bool) -> str:
    """Generate a BED file from TSS metadata → path to temp BED file.
    BED columns: chrom start end name strand sigma
    """
    tmppath = None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as tmp:
        tmppath = tmp.name
        with open(meta_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                pos = int(row[pos_col].strip("\r"))
                if pos_is_0based:
                    pos += 1
                chrom = row.get("Chromosome", "").strip("\r")
                strand = row.get("Strand", ".").strip("\r")
                name = row.get("Sequence_ID", row.get("Locus_Tag", ".")).strip("\r")
                sigma = row.get("Sigma_Factor", "").strip("\r") or "?"
                gene = row.get("Downstream_Gene", row.get("Locus_Tag", "")).strip("\r")

                if not chrom:
                    continue

                # BED: 0-based start, 1-based end. TSS is a point → start=pos-1, end=pos
                bed_start = pos - 1
                bed_end = pos
                tmp.write(f"{chrom}\t{bed_start}\t{bed_end}\t{name}\t{strand}\t{sigma}\t{gene}\n")
    return tmppath


def make_igr_bed(igr_path: Path) -> str:
    """Generate a BED file from IGR TSV → path to temp BED file."""
    tmppath = None
    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as tmp:
        tmppath = tmp.name
        with open(igr_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                chrom = row["chrom"].strip("\r")
                start = int(row["start"])
                end = int(row["end"])
                igr_id = row["igr_id"].strip("\r")
                orient = row["orientation_type"].strip("\r")
                left = f"{row['left_cds'].strip(chr(13))}({row['left_strand'].strip(chr(13))})"
                right = f"{row['right_cds'].strip(chr(13))}({row['right_strand'].strip(chr(13))})"
                # BED: 0-based start, 1-based end
                tmp.write(f"{chrom}\t{start - 1}\t{end}\t{igr_id}\t.\t{orient}\t{left}\t{right}\n")
    return tmppath


def run_bedtools_intersect(tss_bed: str, igr_bed: str) -> dict:
    """Run bedtools intersect → dict[igr_id] = list of TSS entries."""
    result = defaultdict(list)
    try:
        proc = subprocess.run(
            ["bedtools", "intersect", "-a", tss_bed, "-b", igr_bed, "-wa", "-wb"],
            capture_output=True, text=True, check=True,
        )
        for line in proc.stdout.strip().split("\n"):
            if not line.strip():
                continue
            fields = line.strip().split("\t")
            # TSS cols: 0=chrom, 1=start, 2=end, 3=name, 4=strand, 5=sigma, 6=gene
            # IGR cols: 7=chrom, 8=start, 9=end, 10=igr_id, 11=., 12=orient, 13=left, 14=right
            igr_id = fields[10] if len(fields) > 10 else ""
            if igr_id:
                result[igr_id].append({
                    "name": fields[3],
                    "strand": fields[4],
                    "sigma": fields[5],
                    "gene": fields[6],
                })
    except FileNotFoundError:
        print("WARNING: bedtools not found, using fallback (empty results)")
    except subprocess.CalledProcessError as e:
        print(f"WARNING: bedtools error: {e}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate TSS presence in conserved IGRs")
    parser.add_argument("--m8", default=str(ROOT / "output/intergenic/mmseqs2/cross/D39V_vs_TIGR4.m8"))
    parser.add_argument("--d39v-igr", default=str(ROOT / "output/intergenic/d39v/D39V_igrs.tsv"))
    parser.add_argument("--tigr4-igr", default=str(ROOT / "output/intergenic/tigr4/TIGR4_igrs.tsv"))
    parser.add_argument("--d39v-tss", default=str(ROOT / "data/benchmark/positives_81bp_metadata.tsv"))
    parser.add_argument("--tigr4-tss", default=str(ROOT / "data/tigr4/positives_high_81bp_metadata.tsv"))
    parser.add_argument("--out", default=str(OUT_DIR / "conserved_igrs_tss_validation.tsv"))
    args = parser.parse_args()

    print("Loading IGR metadata...")
    d39v_igrs = load_igr_metadata(Path(args.d39v_igr))
    tigr4_igrs = load_igr_metadata(Path(args.tigr4_igr))

    print("Generating BED files...")
    d39v_tss_bed = make_tss_bed(Path(args.d39v_tss), "TSS_Position_0based", pos_is_0based=True)
    tigr4_tss_bed = make_tss_bed(Path(args.tigr4_tss), "TSS_Position", pos_is_0based=False)
    d39v_igr_bed = make_igr_bed(Path(args.d39v_igr))
    tigr4_igr_bed = make_igr_bed(Path(args.tigr4_igr))

    print("Running BEDTools intersect...")
    d39v_tss_in_igr = run_bedtools_intersect(d39v_tss_bed, d39v_igr_bed)
    tigr4_tss_in_igr = run_bedtools_intersect(tigr4_tss_bed, tigr4_igr_bed)

    print(f"  D39V IGRs with TSS: {len(d39v_tss_in_igr)}")
    print(f"  TIGR4 IGRs with TSS: {len(tigr4_tss_in_igr)}")

    # Cleanup temp files
    for f in [d39v_tss_bed, tigr4_tss_bed, d39v_igr_bed, tigr4_igr_bed]:
        Path(f).unlink(missing_ok=True)

    print("Cross-referencing conserved pairs with TSS...")
    rows = []
    stats = {"both": 0, "d39v_only": 0, "tigr4_only": 0, "neither": 0}

    with open(args.m8, newline="") as f:
        for line in f:
            parts = line.strip().split("\t")
            q_id, t_id = parts[0], parts[1]
            fident = float(parts[2])
            alnlen = int(parts[3])
            bitscore = float(parts[11])

            q_igr = d39v_igrs.get(q_id)
            t_igr = tigr4_igrs.get(t_id)
            if not q_igr or not t_igr:
                continue

            q_tss = d39v_tss_in_igr.get(q_id, [])
            t_tss = tigr4_tss_in_igr.get(t_id, [])

            q_has = len(q_tss) > 0
            t_has = len(t_tss) > 0

            if q_has and t_has:
                cat = "both"
            elif q_has:
                cat = "d39v_only"
            elif t_has:
                cat = "tigr4_only"
            else:
                cat = "neither"
            stats[cat] += 1

            q_sigmas = ",".join(sorted(set(t["sigma"] for t in q_tss))) if q_tss else ""
            t_sigmas = ",".join(sorted(set(t["sigma"] for t in t_tss))) if t_tss else ""
            q_genes = ",".join(t["gene"] for t in q_tss) if q_tss else ""
            t_genes = ",".join(t["gene"] for t in t_tss) if t_tss else ""

            rows.append({
                "query_d39v": q_id,
                "target_tigr4": t_id,
                "fident": f"{fident:.3f}",
                "alnlen": alnlen,
                "bitscore": bitscore,
                "has_tss_d39v": q_has,
                "has_tss_tigr4": t_has,
                "category": cat,
                "tss_count_d39v": len(q_tss),
                "tss_count_tigr4": len(t_tss),
                "sigmas_d39v": q_sigmas,
                "sigmas_tigr4": t_sigmas,
                "genes_d39v": q_genes,
                "genes_tigr4": t_genes,
                "igr_orient_d39v": q_igr["orientation"],
                "igr_orient_tigr4": t_igr["orientation"],
                "flank_d39v": f"{q_igr['left_gene']}–{q_igr['right_gene']}",
                "flank_tigr4": f"{t_igr['left_gene']}–{t_igr['right_gene']}",
            })

    print("Writing output...")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    total = sum(stats.values())
    print(f"\n=== Results ({total} conserved pairs) ===")
    print(f"  TSS in both strains:  {stats['both']:>5} ({stats['both']/total*100:.1f}%)")
    print(f"  TSS only in D39V:     {stats['d39v_only']:>5} ({stats['d39v_only']/total*100:.1f}%)")
    print(f"  TSS only in TIGR4:    {stats['tigr4_only']:>5} ({stats['tigr4_only']/total*100:.1f}%)")
    print(f"  No TSS in either:     {stats['neither']:>5} ({stats['neither']/total*100:.1f}%)")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
