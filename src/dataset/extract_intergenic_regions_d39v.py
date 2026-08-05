#!/usr/bin/env python3
"""
Intergenic Region (IGR) Extraction & CDS Masking Tool.

Provides two complementary approaches for intergenic sequence analysis:
  Approach 1: Hard-Masking of CDS regions in the reference genome (replacing CDS with 'N's).
  Approach 2: Extraction of discrete IGRs into a multi-FASTA file + metadata TSV table,
              annotating orientation types (DIVERGENT <- ->, CONVERGENT -> <-, TANDEM -> -> / <- <-).

Usage:
  python src/dataset/extract_intergenic_regions.py \\
    --fasta data/reference/D39V.fna \\
    --gff data/reference/D39V.gff3 \\
    --out-dir output/intergenic
"""

import argparse
import sys
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract intergenic regions (IGRs) and generate CDS-masked genome FASTA."
    )
    parser.add_argument(
        "--fasta", required=True, type=Path, help="Path to reference genome FASTA."
    )
    parser.add_argument(
        "--gff", required=True, type=Path, help="Path to structural annotation GFF3."
    )
    parser.add_argument(
        "--out-dir",
        default=Path("output/intergenic"),
        type=Path,
        help="Output directory (default: output/intergenic).",
    )
    parser.add_argument(
        "--min-len",
        default=10,
        type=int,
        help="Minimum intergenic length in bp (default: 10).",
    )
    return parser.parse_args()


def parse_gff_cds(gff_path, chrom_id):
    """
    Parses GFF3 to extract all CDS features for a specific chromosome,
    returning a sorted list of tuples: (start_1based, end_1based, strand, locus_tag/ID).
    """
    cds_list = []
    with open(gff_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            seqid, source, feature_type, start_str, end_str, score, strand, phase, attrs = parts
            
            # Match seqid (allow matching even if chrom_id matches parts of seqid)
            if seqid != chrom_id and seqid not in chrom_id and chrom_id not in seqid:
                continue

            if feature_type.upper() == "CDS":
                start_1 = int(start_str)
                end_1 = int(end_str)
                
                # Extract locus_tag or ID from attributes
                attr_dict = {}
                for kv in attrs.split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        attr_dict[k] = v
                
                locus = attr_dict.get("locus_tag", attr_dict.get("ID", attr_dict.get("Name", "unknown")))
                cds_list.append((start_1, end_1, strand, locus))

    # Sort CDS features by start coordinate
    cds_list.sort(key=lambda x: x[0])
    return cds_list


def determine_orientation_type(strand1, strand2):
    """
    Determines intergenic orientation type from flanking CDS strands:
      - DIVERGENT (<- ->): left CDS is '-', right CDS is '+'
      - CONVERGENT (-> <-): left CDS is '+', right CDS is '-'
      - TANDEM (-> -> or <- <-): both CDS have same strand
    """
    if strand1 == "-" and strand2 == "+":
        return "DIVERGENT"
    elif strand1 == "+" and strand2 == "-":
        return "CONVERGENT"
    elif strand1 == strand2:
        return f"TANDEM({strand1}{strand2})"
    else:
        return "UNKNOWN"


def approach_1_mask_genome(genome_seq, cds_list):
    """
    Approach 1: Replace all CDS nucleotide positions in the full genome with 'N'.
    Returns the masked sequence as a string.
    """
    seq_chars = list(str(genome_seq))
    genome_len = len(seq_chars)

    for start_1, end_1, strand, locus in cds_list:
        # Convert 1-based inclusive to 0-based slice
        start_0 = max(0, start_1 - 1)
        end_0 = min(genome_len, end_1)
        for i in range(start_0, end_0):
            seq_chars[i] = "N"

    return "".join(seq_chars)


def approach_2_extract_igrs(genome_record, cds_list, min_len=10):
    """
    Approach 2: Extract discrete intergenic regions (IGRs) between consecutive non-overlapping CDSs.
    Returns:
      - igr_records: List of Bio.SeqRecord objects for multi-FASTA output
      - igr_table_rows: List of metadata dictionaries for TSV output
    """
    genome_seq = str(genome_record.seq)
    chrom_id = genome_record.id
    genome_len = len(genome_seq)

    igr_records = []
    igr_table_rows = []

    if not cds_list:
        return igr_records, igr_table_rows

    # Merge overlapping CDS ranges for boundary finding, but keep flanking gene info
    prev_end = 0
    prev_strand = None
    prev_locus = None

    igr_idx = 1

    for i in range(len(cds_list)):
        curr_start, curr_end, curr_strand, curr_locus = cds_list[i]

        if i == 0:
            # Region from chromosome start to first CDS
            igr_start_1 = 1
            igr_end_1 = curr_start - 1
            flank_left_locus = "CHROM_START"
            flank_left_strand = "."
            flank_right_locus = curr_locus
            flank_right_strand = curr_strand
        else:
            prev_s, prev_e, prev_st, prev_loc = cds_list[i - 1]
            igr_start_1 = prev_e + 1
            igr_end_1 = curr_start - 1
            flank_left_locus = prev_loc
            flank_left_strand = prev_st
            flank_right_locus = curr_locus
            flank_right_strand = curr_strand

        igr_len = igr_end_1 - igr_start_1 + 1

        if igr_len >= min_len:
            # Extract 0-based slice
            igr_start_0 = igr_start_1 - 1
            igr_end_0 = igr_end_1
            igr_seq = genome_seq[igr_start_0:igr_end_0]

            orient_type = determine_orientation_type(flank_left_strand, flank_right_strand)
            igr_id = f"IGR_{chrom_id}_{igr_idx:04d}"

            header_desc = (
                f"chrom={chrom_id} pos={igr_start_1}-{igr_end_1} len={igr_len} "
                f"type={orient_type} left={flank_left_locus}({flank_left_strand}) right={flank_right_locus}({flank_right_strand})"
            )

            record = SeqRecord(
                Seq(igr_seq),
                id=igr_id,
                description=header_desc
            )
            igr_records.append(record)

            igr_table_rows.append({
                "igr_id": igr_id,
                "chrom": chrom_id,
                "start": igr_start_1,
                "end": igr_end_1,
                "length": igr_len,
                "orientation_type": orient_type,
                "left_cds": flank_left_locus,
                "left_strand": flank_left_strand,
                "right_cds": flank_right_locus,
                "right_strand": flank_right_strand,
                "sequence": igr_seq
            })

            igr_idx += 1

    # Final region after last CDS
    last_start, last_end, last_strand, last_locus = cds_list[-1]
    if last_end < genome_len:
        igr_start_1 = last_end + 1
        igr_end_1 = genome_len
        igr_len = igr_end_1 - igr_start_1 + 1

        if igr_len >= min_len:
            igr_seq = genome_seq[igr_start_1 - 1 : igr_end_1]
            igr_id = f"IGR_{chrom_id}_{igr_idx:04d}"
            orient_type = determine_orientation_type(last_strand, ".")
            
            record = SeqRecord(
                Seq(igr_seq),
                id=igr_id,
                description=f"chrom={chrom_id} pos={igr_start_1}-{igr_end_1} len={igr_len} type={orient_type} left={last_locus}({last_strand}) right=CHROM_END(.)"
            )
            igr_records.append(record)
            igr_table_rows.append({
                "igr_id": igr_id,
                "chrom": chrom_id,
                "start": igr_start_1,
                "end": igr_end_1,
                "length": igr_len,
                "orientation_type": orient_type,
                "left_cds": last_locus,
                "left_strand": last_strand,
                "right_cds": "CHROM_END",
                "right_strand": ".",
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

    records = list(SeqIO.parse(args.fasta, "fasta"))
    if not records:
        sys.exit(f"Error: No FASTA records found in {args.fasta}")

    genome_name = args.fasta.stem

    all_masked_records = []
    all_igr_records = []
    all_igr_rows = []

    print(f"Processing genome: {genome_name} ({len(records)} contig/chromosome records)")

    for rec in records:
        print(f" -> Parsing CDS for sequence: {rec.id} (length={len(rec.seq)} bp)")
        cds_list = parse_gff_cds(args.gff, rec.id)
        print(f"    Found {len(cds_list)} CDS features.")

        # --- Approach 1: Mask Genome ---
        masked_str = approach_1_mask_genome(rec.seq, cds_list)
        masked_rec = SeqRecord(
            Seq(masked_str),
            id=f"{rec.id}_masked",
            description=f"{rec.description} (CDS hard-masked with N)"
        )
        all_masked_records.append(masked_rec)

        # --- Approach 2: Extract IGRs ---
        igr_records, igr_rows = approach_2_extract_igrs(rec, cds_list, min_len=args.min_len)
        all_igr_records.extend(igr_records)
        all_igr_rows.extend(igr_rows)

    # Output Approach 1
    masked_fasta_path = args.out_dir / f"{genome_name}_genome_cds_masked.fasta"
    SeqIO.write(all_masked_records, masked_fasta_path, "fasta")
    print(f"\n[Approach 1 Complete] Masked genome FASTA saved to: {masked_fasta_path}")

    # Output Approach 2
    igrs_fasta_path = args.out_dir / f"{genome_name}_igrs.fasta"
    SeqIO.write(all_igr_records, igrs_fasta_path, "fasta")

    import pandas as pd
    df_igrs = pd.DataFrame(all_igr_rows)
    igrs_tsv_path = args.out_dir / f"{genome_name}_igrs.tsv"
    df_igrs.to_csv(igrs_tsv_path, sep="\t", index=False)

    print(f"[Approach 2 Complete] Multi-FASTA with {len(all_igr_records)} IGRs saved to: {igrs_fasta_path}")
    print(f"                      Metadata table saved to: {igrs_tsv_path}")

    # Summary statistics
    if not df_igrs.empty:
        print("\n--- Summary Statistics of Extracted IGRs ---")
        print(f"Total IGRs extracted: {len(df_igrs)}")
        print(f"Mean IGR length: {df_igrs['length'].mean():.1f} bp (min={df_igrs['length'].min()}, max={df_igrs['length'].max()})")
        print("\nOrientation Breakdown:")
        print(df_igrs["orientation_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
