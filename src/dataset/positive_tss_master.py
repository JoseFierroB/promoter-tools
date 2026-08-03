#!/usr/bin/env python3
"""
Universal Positive TSS Sequence Extractor (Master Multi-Species Pipeline).

Extracts promoter sequence windows centered on TSS coordinates from either Excel (.xlsx)
or GFF3 (.gff3) annotation files against genomic FASTA references.

Pipeline Steps:
  1. Parse CLI arguments & load genome FASTA.
  2. Read TSS coordinates from Excel or GFF3 (auto-detected).
  3. Extract 81-mer promoter windows (-60 to +20 bp) with reverse complement handling.
  4. Resolve same-strand steric hindrance conflicts (< 25 bp).
  5. Calculate GC bias statistics (Z-score, Cohen's d) & biological validation metrics.
  6. Export standardized FASTA & TSV metadata files.

Usage:
    pixi run python src/dataset/positive_tss_master.py \
      --annotation output/tigr4_data/S1_TSS.xlsx \
      --fasta data/reference/NC_003028.fasta \
      --tier high_conf_primary \
      -o output/tigr4_data/positives_tigr4_master_81bp
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


# ════════════════════════════════════════════════════════════════
# 1. Command-Line Interface
# ════════════════════════════════════════════════════════════════

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal Positive TSS Sequence Extractor for bacterial genomes."
    )

    parser.add_argument(
        "-a",
        "--annotation",
        required=True,
        help="Path to TSS annotation file (.xlsx/.xls or .gff/.gff3).",
    )
    parser.add_argument(
        "-f",
        "--fasta",
        required=True,
        help="Path to genome FASTA file.",
    )
    parser.add_argument(
        "--gff-cds",
        default=None,
        help="(Optional) Path to structural CDS GFF3 file for 5'-UTR mapping.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output/positive_dataset_master",
        help="Output prefix (generates .fasta and .tsv).",
    )
    parser.add_argument(
        "--tier",
        choices=["high_conf_primary", "extended_primary", "all_tss"],
        default="high_conf_primary",
        help="Sheet selection tier for Excel files (default: 'high_conf_primary').",
    )
    parser.add_argument(
        "-u",
        "--upstream",
        type=int,
        default=60,
        help="Upstream bp from TSS (default: 60).",
    )
    parser.add_argument(
        "-d",
        "--downstream",
        type=int,
        default=20,
        help="Downstream bp from TSS (default: 20).",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=int,
        default=25,
        help="Same-strand steric hindrance distance threshold in bp (default: 25).",
    )

    return parser.parse_args()


# ════════════════════════════════════════════════════════════════
# 2. Genome Loading
# ════════════════════════════════════════════════════════════════

def load_genome(fasta_path: Path) -> Tuple[str, Seq, int]:
    """Loads and validates the primary genomic sequence from a FASTA file."""
    if not fasta_path.exists():
        print(f"[ERROR] FASTA file not found at {fasta_path}", file=sys.stderr)
        sys.exit(1)

    genome_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    if not genome_dict:
        print(f"[ERROR] Could not parse FASTA from {fasta_path}", file=sys.stderr)
        sys.exit(1)

    chrom_id = list(genome_dict.keys())[0]
    seq = genome_dict[chrom_id].seq
    print(f"[INFO] Loaded genome '{chrom_id}' ({len(seq):,} bp) from {fasta_path.name}")
    return chrom_id, seq, len(seq)


# ════════════════════════════════════════════════════════════════
# 3. Structural CDS Map Reader (Optional for GFF3 5'-UTR Mapping)
# ════════════════════════════════════════════════════════════════

def load_cds_map(gff_cds_path: Path) -> Dict[str, List[Tuple[int, int, str, str]]]:
    """Reads CDS coordinates from a structural GFF3 file to compute 5'-UTR distances."""
    cds_dict = {}
    if not gff_cds_path or not Path(gff_cds_path).exists():
        return cds_dict

    print(f"[INFO] Loading structural CDS annotations from {Path(gff_cds_path).name}...")
    with open(gff_cds_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and parts[2] == "CDS":
                chrom = parts[0]
                start, end, strand = int(parts[3]), int(parts[4]), parts[6]
                attr = dict(item.split("=") for item in parts[8].split(";") if "=" in item)
                locus = attr.get("locus_tag", attr.get("Name", "unknown_cds"))
                cds_dict.setdefault(chrom, []).append((start, end, strand, locus))

    return cds_dict


# ════════════════════════════════════════════════════════════════
# 4. Universal Annotation Readers (.xlsx & .gff3)
# ════════════════════════════════════════════════════════════════

def load_annotations(annotation_path: Path, tier: str, cds_map: Dict = None) -> Tuple[List[Dict], str]:
    """Auto-detects file format (.xlsx vs .gff3) and loads raw TSS records."""
    ext = annotation_path.suffix.lower()

    if ext in [".xlsx", ".xls"]:
        records = _load_excel(annotation_path, tier)
        fmt = "Excel"
    elif ext in [".gff", ".gff3", ".gtf"]:
        records = _load_gff(annotation_path, cds_map)
        fmt = "GFF3"
    else:
        print(f"[ERROR] Unsupported annotation file extension: {ext}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Parsed {len(records):,} candidate TSS entries from {fmt} file {annotation_path.name}")
    return records, fmt


def _load_excel(xlsx_path: Path, tier: str) -> List[Dict]:
    """Parses TSS entries from Excel sheets according to the selected tier."""
    tier_sheets = {
        "high_conf_primary": ["High Confidence (TSS_100.4)"],
        "extended_primary": ["High Confidence (TSS_100.4)", "Low Confidence (TSS_2.1)"],
        "all_tss": [
            "High Confidence (TSS_100.4)",
            "Secondary TSS, High confidence",
            "Low Confidence (TSS_2.1)",
            "Secondary TSS, Low confidence",
        ],
    }.get(tier, ["High Confidence (TSS_100.4)"])

    xl = pd.ExcelFile(xlsx_path)
    dfs = [pd.read_excel(xlsx_path, sheet_name=s).assign(_Source_Sheet=s) for s in tier_sheets if s in xl.sheet_names]

    if not dfs:
        dfs = [pd.read_excel(xlsx_path, sheet_name=0).assign(_Source_Sheet=xl.sheet_names[0])]

    df = pd.concat(dfs, ignore_index=True)
    records = []

    for idx, row in df.iterrows():
        locus = str(row.get("Locus_tag", row.get("Locus", f"TSS_{idx}"))).strip()
        strand = str(row.get("Strand", "+")).strip()
        tss_col = "TSS_position" if "TSS_position" in row else ("Primary_TSS" if "Primary_TSS" in row else "TSS")

        if tss_col not in row or pd.isna(row[tss_col]):
            continue

        records.append({
            "Locus_Tag": locus,
            "TSS_Position": int(row[tss_col]),
            "Strand": strand,
            "UTR5_Length": row.get("5'-UTR_length", row.get("Primary_5'-UTR_length", "NA")),
            "Location_Type": row.get("Within_coding_vs_intergenic", "intergenic"),
            "Confidence_Sheet": row.get("_Source_Sheet", "Excel"),
            "Score": row.get("Processed_coverage", row.get("Primary_Processed_cov", "curated")),
        })

    return records


def _load_gff(gff_path: Path, cds_map: Dict = None) -> List[Dict]:
    """Parses TSS features from GFF3 lines."""
    records = []
    with open(gff_path) as f:
        for idx, line in enumerate(f):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 9 and any(k in parts[2].lower() for k in ["tss", "transcription_start_site", "feature"]):
                strand = parts[6]
                tss_pos = int(parts[3])
                attr = dict(item.split("=") for item in parts[8].split(";") if "=" in item)
                locus = attr.get("ID", attr.get("Name", f"GFF_TSS_{idx}"))

                # 5'-UTR distance calculation if CDS map is present
                utr_len = "NA"
                if cds_map:
                    chrom_key = list(cds_map.keys())[0] if len(cds_map) == 1 else parts[0]
                    matches = [(c[0] - tss_pos if strand == "+" else tss_pos - c[1], c[3])
                               for c in cds_map.get(chrom_key, []) if c[2] == strand and -20 <= (c[0] - tss_pos if strand == "+" else tss_pos - c[1]) <= 1000]
                    if matches:
                        matches.sort(key=lambda x: x[0])
                        utr_len, locus = matches[0]

                records.append({
                    "Locus_Tag": locus,
                    "TSS_Position": tss_pos,
                    "Strand": strand,
                    "UTR5_Length": utr_len,
                    "Location_Type": "GFF3",
                    "Confidence_Sheet": "GFF3",
                    "Score": parts[5] if parts[5] != "." else "curated",
                })

    return records


# ════════════════════════════════════════════════════════════════
# 5. Promoter Window Extraction
# ════════════════════════════════════════════════════════════════

def extract_promoter_windows(
    raw_records: List[Dict], genome_seq: Seq, chrom_id: str, upstream: int, downstream: int
) -> Tuple[List[Dict], Dict[str, int]]:
    """Extracts sequence windows centered on TSS coordinates with reverse complement handling."""
    extracted = []
    seq_len = len(genome_seq)
    window_size = upstream + 1 + downstream
    stats = {"total_evaluated": len(raw_records), "skipped_boundary": 0, "skipped_invalid_n": 0}
    seen = set()

    for r in raw_records:
        locus, strand, tss_pos = r["Locus_Tag"], r["Strand"], r["TSS_Position"]
        pos_0 = tss_pos - 1

        if (tss_pos, strand) in seen:
            continue
        seen.add((tss_pos, strand))

        if strand == "+":
            start_idx, end_idx = pos_0 - upstream, pos_0 + downstream + 1
            if start_idx < 0 or end_idx > seq_len:
                stats["skipped_boundary"] += 1
                continue
            kmer = str(genome_seq[start_idx:end_idx]).upper()
        elif strand == "-":
            start_idx, end_idx = pos_0 - downstream, pos_0 + upstream + 1
            if start_idx < 0 or end_idx > seq_len:
                stats["skipped_boundary"] += 1
                continue
            kmer = str(genome_seq[start_idx:end_idx].reverse_complement()).upper()
        else:
            continue

        if len(kmer) != window_size or "N" in kmer:
            stats["skipped_invalid_n"] += 1
            continue

        gc_content = ((kmer.count("G") + kmer.count("C")) / window_size) * 100.0

        extracted.append({
            "Sequence_ID": f"TSS_{locus}_{chrom_id}_{tss_pos}_{strand}",
            "Locus_Tag": locus,
            "Chromosome": chrom_id,
            "TSS_Position": tss_pos,
            "Strand": strand,
            "Upstream_bp": upstream,
            "Downstream_bp": downstream,
            "Window_Size": window_size,
            "Sequence": kmer,
            "GC_Content": round(gc_content, 2),
            "UTR5_Length": r.get("UTR5_Length", "NA"),
            "Location_Type": r.get("Location_Type", "intergenic"),
            "Confidence_Sheet": r.get("Confidence_Sheet", "Unknown"),
            "Score": r.get("Score", "curated"),
        })

    return extracted, stats


# ════════════════════════════════════════════════════════════════
# 6. Steric Hindrance Conflict Resolution
# ════════════════════════════════════════════════════════════════

def resolve_steric_conflicts(records: List[Dict], threshold: int) -> Tuple[List[Dict], int]:
    """Filters same-strand TSS entries that fall closer than the threshold distance."""
    if threshold <= 0 or not records:
        return records, 0

    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in records:
        groups.setdefault((r["Chromosome"], r["Strand"]), []).append(r)

    resolved, discarded_count = [], 0

    for item_list in groups.values():
        item_list.sort(key=lambda x: x["TSS_Position"])
        cluster = []

        for item in item_list:
            if not cluster or (item["TSS_Position"] - cluster[-1]["TSS_Position"] < threshold):
                cluster.append(item)
            else:
                best = max(cluster, key=lambda x: float(x.get("Score", 0)) if str(x.get("Score", 0)).replace(".", "", 1).isdigit() else 0.0)
                resolved.append(best)
                discarded_count += len(cluster) - 1
                cluster = [item]

        if cluster:
            best = max(cluster, key=lambda x: float(x.get("Score", 0)) if str(x.get("Score", 0)).replace(".", "", 1).isdigit() else 0.0)
            resolved.append(best)
            discarded_count += len(cluster) - 1

    return resolved, discarded_count


# ════════════════════════════════════════════════════════════════
# 7. GC Bias & Biological Validation Metrics
# ════════════════════════════════════════════════════════════════

def compute_statistics(records: List[Dict], genome_seq: Seq, upstream: int = 60) -> Dict:
    """Calculates GC content stats (Z-score, Cohen's d) and biological validation metrics."""
    if not records:
        return {}

    sampled_gc = [r["GC_Content"] for r in records]
    mean_gc = float(np.mean(sampled_gc))
    stdev_gc = float(np.std(sampled_gc, ddof=1)) if len(sampled_gc) > 1 else 0.0

    gen_str = str(genome_seq).upper()
    gen_gc_mean = ((gen_str.count("G") + gen_str.count("C")) / len(gen_str)) * 100.0

    n = len(sampled_gc)
    z_score = (mean_gc - gen_gc_mean) / (stdev_gc / math.sqrt(n)) if n > 0 and stdev_gc > 0 else 0.0
    cohen_d = (mean_gc - gen_gc_mean) / stdev_gc if stdev_gc > 0 else 0.0

    # +1 Purine Preference
    plus1 = [r["Sequence"][upstream] for r in records if len(r["Sequence"]) > upstream]
    n_plus1 = len(plus1)
    purines_pct = ((plus1.count("A") + plus1.count("G")) / n_plus1 * 100.0) if n_plus1 > 0 else 0.0

    # -10 Box Match Percentage
    pribnow_matches = sum(1 for r in records if len(r["Sequence"]) == upstream + 21 and
                          ("TATAAT" in r["Sequence"][40:57] or re.search(r"TA[ATGC]{2,3}[AT]", r["Sequence"][40:57])))
    pribnow_pct = (pribnow_matches / n * 100.0) if n > 0 else 0.0

    # Canonical 5'-UTR Spacing
    utrs = [float(r["UTR5_Length"]) for r in records if str(r["UTR5_Length"]).replace("-", "", 1).isdigit()]
    canonical_utr_pct = (sum(1 for u in utrs if 15 <= u <= 45) / len(utrs) * 100.0) if utrs else 0.0

    return {
        "n_samples": n,
        "mean_gc": mean_gc,
        "stdev_gc": stdev_gc,
        "genome_gc_mean": gen_gc_mean,
        "z_score": z_score,
        "cohen_d": cohen_d,
        "plus1_purines_pct": purines_pct,
        "plus1_a_pct": (plus1.count("A") / n_plus1 * 100.0) if n_plus1 > 0 else 0.0,
        "plus1_g_pct": (plus1.count("G") / n_plus1 * 100.0) if n_plus1 > 0 else 0.0,
        "pribnow_pct": pribnow_pct,
        "canonical_utr_pct": canonical_utr_pct,
        "n_utrs": len(utrs),
    }


# ════════════════════════════════════════════════════════════════
# 8. Exporters (.fasta & .tsv)
# ════════════════════════════════════════════════════════════════

def write_files(records: List[Dict], out_prefix: Path) -> Tuple[Path, Path]:
    """Writes extracted promoter records to FASTA and metadata TSV files."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fasta_out = out_prefix.with_suffix(".fasta")
    tsv_out = out_prefix.with_suffix(".tsv")

    with open(fasta_out, "w") as f:
        for r in records:
            f.write(f">{r['Sequence_ID']}\n{r['Sequence']}\n")

    df_meta = pd.DataFrame(records).drop(columns=["Sequence"], errors="ignore")
    df_meta.to_csv(tsv_out, sep="\t", index=False)
    return fasta_out, tsv_out


# ════════════════════════════════════════════════════════════════
# 9. Executive Console Summary
# ════════════════════════════════════════════════════════════════

def report_summary(
    records: List[Dict], stats: Dict, excl_stats: Dict, conflict_count: int,
    fmt: str, fasta_out: Path, tsv_out: Path, upstream: int, downstream: int
):
    n_pos = sum(1 for r in records if r["Strand"] == "+")
    n_neg = sum(1 for r in records if r["Strand"] == "-")
    total_eval = excl_stats.get("total_evaluated", len(records))
    n_ext = stats.get("n_samples", len(records))
    kmer_len = upstream + 1 + downstream

    print("\n" + "═" * 65)
    print(" UNIVERSAL POSITIVE TSS EXTRACTION SUMMARY")
    print("═" * 65)
    print(f"Annotation Format:          {fmt}")
    print(f"Total TSS Candidates Evaluated: {total_eval:,}")
    print(f"Total {kmer_len}-mer Sequences Extracted: {n_ext:,} (Efficiency: {(n_ext / total_eval * 100.0) if total_eval > 0 else 0:.1f}%)")
    print(f"K-mer Window Size (bp):     {kmer_len} bp (-{upstream} to +{downstream} relative to TSS)")
    print(f"Total Nucleotides Sampled:   {n_ext * kmer_len:,} bp")
    print("─" * 65)
    print("EXTRACTION EXCLUSIONS & RESOLUTIONS:")
    print(f" • Steric Conflicts Discarded (<25 bp): {conflict_count:,}")
    print(f" • Skipped Boundary Coordinates:         {excl_stats.get('skipped_boundary', 0):,}")
    print(f" • Skipped Invalid 'N' Bases:            {excl_stats.get('skipped_invalid_n', 0):,}")
    print("─" * 65)
    print("DATASET COMPOSITION & BIAS:")
    print(f" • Strand Distribution:     + strand: {n_pos:,} | - strand: {n_neg:,}")
    print(f" • Sampled GC Content:       {stats.get('mean_gc', 0):.2f}% ± {stats.get('stdev_gc', 0):.2f}%")
    print(f" • Genome Background GC:     {stats.get('genome_gc_mean', 0):.2f}%")
    print(f" • GC Bias Significance:     Z-score = {stats.get('z_score', 0):+.2f} | Cohen's d = {stats.get('cohen_d', 0):+.2f}")
    print("─" * 65)
    print("BIOLOGICAL VALIDATION METRICS:")
    print(f" • +1 Initiator Purines (A+G): {stats.get('plus1_purines_pct', 0):.1f}% (A: {stats.get('plus1_a_pct', 0):.1f}%, G: {stats.get('plus1_g_pct', 0):.1f}%)")
    print(f" • -10 Box Variant Match:       {stats.get('pribnow_pct', 0):.1f}%")
    if stats.get("n_utrs", 0) > 0:
        print(f" • Canonical 5'-UTR (15-45bp): {stats.get('canonical_utr_pct', 0):.1f}% of evaluated UTRs")
    print("═" * 65)
    print(f"[SUCCESS] FASTA dataset ➔ {fasta_out}")
    print(f"[SUCCESS] Metadata TSV ➔ {tsv_out}\n")


# ════════════════════════════════════════════════════════════════
# 10. Main Pipeline Orchestrator
# ════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()

    chrom_id, genome_seq, _ = load_genome(Path(args.fasta))
    cds_map = load_cds_map(Path(args.gff_cds)) if args.gff_cds else None
    raw_anno, fmt = load_annotations(Path(args.annotation), args.tier, cds_map)

    raw_records, excl_stats = extract_promoter_windows(raw_anno, genome_seq, chrom_id, args.upstream, args.downstream)
    resolved_records, conflict_count = resolve_steric_conflicts(raw_records, args.conflict_threshold)

    stats = compute_statistics(resolved_records, genome_seq, args.upstream)
    fasta_out, tsv_out = write_files(resolved_records, Path(args.output))

    report_summary(resolved_records, stats, excl_stats, conflict_count, fmt, fasta_out, tsv_out, args.upstream, args.downstream)


if __name__ == "__main__":
    main()
