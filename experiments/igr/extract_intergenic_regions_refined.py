#!/usr/bin/env python3
"""
Refined Intergenic Region (IGR) Extraction & Structural RNA Masking Tool.

Enhancements over baseline:
  1. Non-coding RNA & Structural Feature Masking: Excludes CDS, tRNA, rRNA, ncRNA,
     scRNA, sRNA, pseudogenes, riboswitches, tmRNA, RNase P, and SRP RNA from IGR space.
  2. Circular Chromosome Wrap-Around: Merges the terminal spacer (after last feature)
     and initial spacer (before first feature) across the replication origin (coordinate L -> 1).

Usage:
  python experiments/igr/extract_intergenic_regions_refined.py \
    --fasta data/reference/D39V.fna \
    --gff data/reference/D39V.gff3 \
    --out-dir output/intergenic_refined/d39v \
    --mask-types CDS,tRNA,rRNA,ncRNA,pseudogene,riboswitch,tmRNA,RNase_P_RNA,SRP_RNA \
    --circular
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


DEFAULT_MASK_TYPES = [
    "CDS",
    "TRNA",
    "RRNA",
    "NCRNA",
    "SCRNA",
    "SRNA",
    "PSEUDOGENE",
    "RIBOSWITCH",
    "TMRNA",
    "RNASE_P_RNA",
    "SRP_RNA"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract refined IGRs with ncRNA masking and circular origin wrap-around."
    )
    parser.add_argument(
        "--fasta", required=True, type=Path, help="Path to reference genome FASTA."
    )
    parser.add_argument(
        "--gff", required=True, type=Path, help="Path to structural annotation GFF3."
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for refined datasets.",
    )
    parser.add_argument(
        "--mask-types",
        default=",".join(DEFAULT_MASK_TYPES),
        type=str,
        help="Comma-separated feature types to mask (default: CDS,tRNA,rRNA,ncRNA,pseudogene,etc.)",
    )
    parser.add_argument(
        "--no-mask-ncrna",
        action="store_true",
        help="Disable ncRNA masking (CDS-only mode, for backward compatibility tests).",
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        default=True,
        help="Enable circular chromosome origin wrap-around (default: True).",
    )
    parser.add_argument(
        "--no-circular",
        dest="circular",
        action="store_false",
        help="Disable circular wrap-around (linear mode).",
    )
    parser.add_argument(
        "--min-len",
        default=10,
        type=int,
        help="Minimum intergenic length in bp (default: 10).",
    )
    return parser.parse_args()


def parse_gff_features(gff_path, chrom_id, mask_types):
    """
    Parses GFF3 to extract all designated gene and structural RNA features for a specific chromosome,
    returning a sorted list of dicts: {start, end, strand, locus, feature_type, product}.
    """
    mask_types_set = {t.strip().upper() for t in mask_types}
    features = []

    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            seqid, source, feature_type, start_str, end_str, score, strand, phase, attrs = parts
            
            # Match seqid
            if seqid != chrom_id and seqid not in chrom_id and chrom_id not in seqid:
                continue

            f_type_upper = feature_type.strip().upper()
            if f_type_upper in mask_types_set:
                start_1 = int(start_str)
                end_1 = int(end_str)
                
                # Parse attributes
                attr_dict = {}
                for kv in attrs.split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        attr_dict[k] = v
                
                locus = attr_dict.get("locus_tag", attr_dict.get("ID", attr_dict.get("Name", f"{feature_type}_{start_1}")))
                product = attr_dict.get("product", attr_dict.get("gene", feature_type))
                
                features.append({
                    "start": start_1,
                    "end": end_1,
                    "strand": strand if strand in ["+", "-"] else ".",
                    "locus": locus,
                    "feature_type": feature_type,
                    "product": product
                })

    # Sort features by start coordinate
    features.sort(key=lambda x: (x["start"], x["end"]))
    return features


def determine_orientation_type(strand1, strand2):
    """
    Determines intergenic orientation type from flanking feature strands:
      - DIVERGENT (<- ->): left feature is '-', right feature is '+'
      - CONVERGENT (-> <-): left feature is '+', right feature is '-'
      - TANDEM (-> -> or <- <-): both features have same strand
    """
    if strand1 == "-" and strand2 == "+":
        return "DIVERGENT"
    elif strand1 == "+" and strand2 == "-":
        return "CONVERGENT"
    elif strand1 in ["+", "-"] and strand1 == strand2:
        return f"TANDEM({strand1}{strand2})"
    elif strand1 == "." or strand2 == ".":
        return "UNSTRANDED"
    else:
        return "UNKNOWN"


def mask_genome_sequence(genome_seq, features):
    """
    Replaces all annotated feature nucleotide positions in the full genome with 'N'.
    Returns the masked sequence string.
    """
    seq_chars = list(str(genome_seq))
    genome_len = len(seq_chars)

    for feat in features:
        start_0 = max(0, feat["start"] - 1)
        end_0 = min(genome_len, feat["end"])
        for i in range(start_0, end_0):
            seq_chars[i] = "N"

    return "".join(seq_chars)


def extract_refined_igrs(genome_record, features, min_len=10, circular=True):
    """
    Extracts refined IGRs between consecutive non-overlapping features,
    optionally stitching the origin-spanning spacer for circular chromosomes.
    """
    genome_seq = str(genome_record.seq)
    chrom_id = genome_record.id
    genome_len = len(genome_seq)

    igr_records = []
    igr_table_rows = []

    if not features:
        return igr_records, igr_table_rows

    # Merge overlapping feature intervals to ensure proper intergenic boundary definition
    merged_intervals = []
    for feat in features:
        if not merged_intervals:
            merged_intervals.append({
                "start": feat["start"],
                "end": feat["end"],
                "left_feat": feat,
                "right_feat": feat,
                "sub_features": [feat]
            })
        else:
            prev = merged_intervals[-1]
            if feat["start"] <= prev["end"] + 1:  # Overlapping or directly abutting
                prev["end"] = max(prev["end"], feat["end"])
                prev["right_feat"] = feat
                prev["sub_features"].append(feat)
            else:
                merged_intervals.append({
                    "start": feat["start"],
                    "end": feat["end"],
                    "left_feat": feat,
                    "right_feat": feat,
                    "sub_features": [feat]
                })

    igr_idx = 1
    first_feat = merged_intervals[0]
    last_feat = merged_intervals[-1]

    # If linear (non-circular), handle initial region
    if not circular:
        if first_feat["start"] > 1:
            igr_start_1 = 1
            igr_end_1 = first_feat["start"] - 1
            igr_len = igr_end_1 - igr_start_1 + 1
            if igr_len >= min_len:
                igr_seq = genome_seq[0:igr_end_1]
                igr_id = f"IGR_{chrom_id}_{igr_idx:04d}"
                record = SeqRecord(
                    Seq(igr_seq),
                    id=igr_id,
                    description=f"chrom={chrom_id} pos={igr_start_1}-{igr_end_1} len={igr_len} type=UNKNOWN left=CHROM_START(.) right={first_feat['left_feat']['locus']}({first_feat['left_feat']['strand']})"
                )
                igr_records.append(record)
                igr_table_rows.append({
                    "igr_id": igr_id,
                    "chrom": chrom_id,
                    "start": str(igr_start_1),
                    "end": str(igr_end_1),
                    "length": igr_len,
                    "orientation_type": "UNKNOWN",
                    "left_feature": "CHROM_START",
                    "left_type": "BOUNDARY",
                    "left_strand": ".",
                    "right_feature": first_feat["left_feat"]["locus"],
                    "right_type": first_feat["left_feat"]["feature_type"],
                    "right_strand": first_feat["left_feat"]["strand"],
                    "is_circular_origin_wrap": False,
                    "sequence": igr_seq
                })
                igr_idx += 1

    # Internal IGRs between consecutive feature clusters
    for i in range(len(merged_intervals) - 1):
        prev_interval = merged_intervals[i]
        curr_interval = merged_intervals[i + 1]

        igr_start_1 = prev_interval["end"] + 1
        igr_end_1 = curr_interval["start"] - 1
        igr_len = igr_end_1 - igr_start_1 + 1

        if igr_len >= min_len:
            igr_seq = genome_seq[igr_start_1 - 1 : igr_end_1]
            flank_left = prev_interval["right_feat"]
            flank_right = curr_interval["left_feat"]

            orient_type = determine_orientation_type(flank_left["strand"], flank_right["strand"])
            igr_id = f"IGR_{chrom_id}_{igr_idx:04d}"

            desc = (
                f"chrom={chrom_id} pos={igr_start_1}-{igr_end_1} len={igr_len} type={orient_type} "
                f"left={flank_left['locus']}[{flank_left['feature_type']}]({flank_left['strand']}) "
                f"right={flank_right['locus']}[{flank_right['feature_type']}]({flank_right['strand']})"
            )

            record = SeqRecord(Seq(igr_seq), id=igr_id, description=desc)
            igr_records.append(record)
            igr_table_rows.append({
                "igr_id": igr_id,
                "chrom": chrom_id,
                "start": str(igr_start_1),
                "end": str(igr_end_1),
                "length": igr_len,
                "orientation_type": orient_type,
                "left_feature": flank_left["locus"],
                "left_type": flank_left["feature_type"],
                "left_strand": flank_left["strand"],
                "right_feature": flank_right["locus"],
                "right_type": flank_right["feature_type"],
                "right_strand": flank_right["strand"],
                "is_circular_origin_wrap": False,
                "sequence": igr_seq
            })
            igr_idx += 1

    # Last interval / Circular wrap-around
    if circular:
        # Stitch: [last_feat.end + 1 ... L] + [1 ... first_feat.start - 1]
        tail_len = max(0, genome_len - last_feat["end"])
        head_len = max(0, first_feat["start"] - 1)
        total_origin_len = tail_len + head_len

        if total_origin_len >= min_len:
            tail_seq = genome_seq[last_feat["end"] : genome_len] if tail_len > 0 else ""
            head_seq = genome_seq[0 : first_feat["start"] - 1] if head_len > 0 else ""
            origin_seq = tail_seq + head_seq

            flank_left = last_feat["right_feat"]
            flank_right = first_feat["left_feat"]
            orient_type = determine_orientation_type(flank_left["strand"], flank_right["strand"])
            igr_id = f"IGR_{chrom_id}_{igr_idx:04d}_ORIGIN"

            desc = (
                f"chrom={chrom_id} pos={last_feat['end']+1}..{genome_len}^1..{first_feat['start']-1} "
                f"len={total_origin_len} type={orient_type} "
                f"left={flank_left['locus']}[{flank_left['feature_type']}]({flank_left['strand']}) "
                f"right={flank_right['locus']}[{flank_right['feature_type']}]({flank_right['strand']}) [CIRCULAR_ORIGIN_WRAP]"
            )

            record = SeqRecord(Seq(origin_seq), id=igr_id, description=desc)
            igr_records.append(record)
            igr_table_rows.append({
                "igr_id": igr_id,
                "chrom": chrom_id,
                "start": f"{last_feat['end']+1}..{genome_len}^1..{first_feat['start']-1}",
                "end": f"{first_feat['start']-1}",
                "length": total_origin_len,
                "orientation_type": orient_type,
                "left_feature": flank_left["locus"],
                "left_type": flank_left["feature_type"],
                "left_strand": flank_left["strand"],
                "right_feature": flank_right["locus"],
                "right_type": flank_right["feature_type"],
                "right_strand": flank_right["strand"],
                "is_circular_origin_wrap": True,
                "sequence": origin_seq
            })
    else:
        # Linear mode: save tail separately
        if last_feat["end"] < genome_len:
            igr_start_1 = last_feat["end"] + 1
            igr_end_1 = genome_len
            igr_len = igr_end_1 - igr_start_1 + 1
            if igr_len >= min_len:
                igr_seq = genome_seq[igr_start_1 - 1 : igr_end_1]
                igr_id = f"IGR_{chrom_id}_{igr_idx:04d}"
                flank_left = last_feat["right_feat"]
                record = SeqRecord(
                    Seq(igr_seq),
                    id=igr_id,
                    description=f"chrom={chrom_id} pos={igr_start_1}-{igr_end_1} len={igr_len} type=UNKNOWN left={flank_left['locus']}({flank_left['strand']}) right=CHROM_END(.)"
                )
                igr_records.append(record)
                igr_table_rows.append({
                    "igr_id": igr_id,
                    "chrom": chrom_id,
                    "start": str(igr_start_1),
                    "end": str(igr_end_1),
                    "length": igr_len,
                    "orientation_type": "UNKNOWN",
                    "left_feature": flank_left["locus"],
                    "left_type": flank_left["feature_type"],
                    "left_strand": flank_left["strand"],
                    "right_feature": "CHROM_END",
                    "right_type": "BOUNDARY",
                    "right_strand": ".",
                    "is_circular_origin_wrap": False,
                    "sequence": igr_seq
                })

    return igr_records, igr_table_rows


def main():
    args = parse_args()

    if not args.fasta.exists():
        sys.exit(f"Error: FASTA file not found: {args.fasta}")
    if not args.gff.exists():
        sys.exit(f"Error: GFF3 file not found: {args.gff}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mask_types = ["CDS"] if args.no_mask_ncrna else args.mask_types.split(",")
    genome_name = args.fasta.stem

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        sys.exit(f"Error: No FASTA records found in {args.fasta}")

    print(f"=== Extracting Refined IGRs: {genome_name} ===")
    print(f"Features to mask: {mask_types}")
    print(f"Circular origin wrap-around: {args.circular}")

    all_masked_records = []
    all_igr_records = []
    all_igr_rows = []

    for rec in records:
        features = parse_gff_features(args.gff, rec.id, mask_types)
        print(f" -> Contig: {rec.id} ({len(rec.seq):,} bp) | Filtered features: {len(features):,}")

        # Mask genome
        masked_str = mask_genome_sequence(rec.seq, features)
        masked_rec = SeqRecord(
            Seq(masked_str),
            id=f"{rec.id}_masked_refined",
            description=f"{rec.description} (CDS+ncRNA masked with N)"
        )
        all_masked_records.append(masked_rec)

        # Extract IGRs
        igr_records, igr_rows = extract_refined_igrs(
            rec, features, min_len=args.min_len, circular=args.circular
        )
        all_igr_records.extend(igr_records)
        all_igr_rows.extend(igr_rows)

    # Save outputs
    masked_path = args.out_dir / f"{genome_name}_genome_masked_refined.fasta"
    igrs_fasta_path = args.out_dir / f"{genome_name}_igrs_refined.fasta"
    igrs_tsv_path = args.out_dir / f"{genome_name}_igrs_refined.tsv"

    SeqIO.write(all_masked_records, masked_path, "fasta")
    SeqIO.write(all_igr_records, igrs_fasta_path, "fasta")

    df_igrs = pd.DataFrame(all_igr_rows)
    df_igrs.to_csv(igrs_tsv_path, sep="\t", index=False)

    print(f"\n[Refined Extraction Complete]")
    print(f"Masked genome: {masked_path}")
    print(f"Refined IGR FASTA ({len(all_igr_records)} IGRs): {igrs_fasta_path}")
    print(f"Refined IGR TSV metadata: {igrs_tsv_path}")
    print(f"Total IGR length: {df_igrs['length'].sum():,} bp")
    print(f"Mean IGR length: {df_igrs['length'].mean():.3f} bp")
    print("\nOrientation Breakdown:")
    print(df_igrs["orientation_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
