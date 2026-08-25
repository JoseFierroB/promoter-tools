#!/usr/bin/env python3
"""
Positive TSS Sequence Extractor for Promoter Prediction Models.
Parses GFF3 and FASTA files to extract promoter windows centered on TSS.
Includes biological strand corrections, proximity conflict indexing,
advanced GC bias statistics, and transcription factor cross-validation.
"""

import argparse
import sys
import os
import csv
import re
import statistics
from typing import Dict, List, Tuple
from Bio import SeqIO
from BCBio import GFF

# Hardcoded chromosome mappings (e.g. NCBI RefSeq accession to friendly/custom name)

# Mapping of regulator binding site names to their respective sigma factor class
SIGMA_FACTOR_MAPPING = {
    "SigA": "SigA",
    "RpoD": "SigA",
    "SigX": "SigX",
    "ComX": "SigX",
    "ComE": "SigA"
}

# Friendly name mapping for GFF bound_moiety values
REGULATOR_NAMES = {
    "RpoD": "SigA",
    "ComX": "SigX",  # Map ComX to SigX (alternative sigma factor)
    "ComE": "ComE",
    "CiaR": "CiaR",
    "PurR": "PurR",
    "CcpA": "CcpA",
    "CodY": "CodY",
    "ArgR": "ArgR",
    "BlpR": "BlpR",
    "ParB": "ParB",
    "DnaA": "DnaA"
}

def parse_arguments() -> argparse.Namespace:
    """Sets up and parses the command-line arguments provided by the user."""
    parser = argparse.ArgumentParser(description="Extract positive promoter windows centered on TSS.")
    
    # Input/Output paths
    parser.add_argument("--gff", required=True, help="Path to the TSS GFF file.")
    parser.add_argument("--fasta", required=True, help="Path to the genome FASTA file.")
    parser.add_argument("--gff-cds", required=False, default=None, help="(Optional) Path to GFF3 structural annotation containing TSS with corrected orientation.")
    parser.add_argument("-o", "--output", default="positive_dataset", help="Output prefix (generates .fasta and .tsv).")
    
    # Filtering and extraction parameters
    parser.add_argument("--exclude-low-conf", action="store_true", help="If set, filters out sequences marked as 'lower_confidence'.")
    parser.add_argument("-u", "--upstream", type=int, default=60, help="Upstream bp (default: 60).")
    parser.add_argument("-d", "--downstream", type=int, default=20, help="Downstream bp (default: 20).")
    parser.add_argument("--conflict-threshold", type=int, default=25, help="Distance threshold (bp) to flag steric hindrance same-strand conflicts (default: 25).")
    
    return parser.parse_args()


def load_cds_and_regulators(gff_cds_path: str) -> Tuple[Dict[str, List[Tuple[int, int, str]]], Dict[str, List[Tuple[int, int, str, str, str]]]]:
# CDS incorporate the start, end, strand, and locus_tag; regulators incorporate start, end, strand, regulator_name, and feature_id   
    """Loads CDS coordinates and protein binding site coordinates from the GFF3 structural annotation."""
    cds_data = {}
    regulator_data = {}
    if not gff_cds_path or not os.path.exists(gff_cds_path):
        return cds_data, regulator_data
    
    print(f"[INFO] Loading CDS and regulators from GFF {gff_cds_path}...", file=sys.stderr)
    try:
        with open(gff_cds_path) as f:
            for rec in GFF.parse(f):
                chrom = rec.id
                if chrom not in cds_data:
                    cds_data[chrom] = []
                if chrom not in regulator_data:
                    regulator_data[chrom] = []
                    
                # Recursive traversal
                def parse_feat(f_obj):
                    if f_obj.type == 'CDS':
                        start = int(f_obj.location.start)
                        end = int(f_obj.location.end)
                        strand = '+' if f_obj.location.strand == 1 else '-'
                        locus = f_obj.qualifiers.get('locus_tag', f_obj.qualifiers.get('Name', ['unknown_cds']))[0]
                        cds_data[chrom].append((start, end, strand, locus))
                    elif f_obj.type == 'protein_binding_site':
                        start = int(f_obj.location.start)
                        end = int(f_obj.location.end)
                        strand = '+' if f_obj.location.strand == 1 else '-'
                        bound_moiety = f_obj.qualifiers.get('bound_moiety', [''])[0].strip()
                        reg_name = REGULATOR_NAMES.get(bound_moiety, bound_moiety)
                        if reg_name:
                            regulator_data[chrom].append((start, end, strand, reg_name, f_obj.id))
                            
                    sub_f = getattr(f_obj, 'sub_features', None) or getattr(f_obj, 'features', [])
                    for sf in sub_f:
                        parse_feat(sf)
                        
                for feat in rec.features:
                    parse_feat(feat)
    except Exception as e:
        print(f"[WARNING] Could not parse CDS GFF file: {e}", file=sys.stderr)
        
    return cds_data, regulator_data

def find_associated_regulator(chrom: str, pos: int, strand: str, regulator_data: Dict, chrom_map: Dict, max_dist=50) -> Tuple[str, str]:
    """Finds the closest upstream protein binding site on the same strand within max_dist bp."""
    resolved_reg_chrom = chrom
    if resolved_reg_chrom not in regulator_data or not regulator_data[resolved_reg_chrom]:
        return "None", "N/A"
        
    candidates = regulator_data[resolved_reg_chrom]
    valid = []
    for start, end, s_r, reg_name, feat_id in candidates:
        if s_r == strand:
            dist = (pos - end) if strand == '+' else (start - pos)
            if 0 <= dist <= max_dist:
                valid.append((dist, reg_name))
                
    if valid:
        best_dist, best_reg = min(valid, key=lambda x: x[0])
        return best_reg, str(best_dist)
    return "None", "N/A"

def find_downstream_cds_distance(chrom: str, pos: int, strand: str, cds_data: Dict, chrom_map: Dict) -> Tuple[str, str]:
    """Finds the closest downstream CDS on the same strand and calculates the distance (5' UTR length)."""
    resolved_cds_chrom = chrom
    if resolved_cds_chrom not in cds_data or not cds_data[resolved_cds_chrom]:
        return "None", "N/A"
        
    candidates = cds_data[resolved_cds_chrom]
    valid = []
    for start, end, s_g, locus in candidates:
        if s_g == strand:
            dist = (start - pos) if strand == '+' else (pos - end)
            if dist >= -20:
                valid.append((dist, locus))
                
    if valid:
        best_dist, best_locus = min(valid, key=lambda x: x[0])
        return best_locus, str(best_dist)
    return "None", "N/A"

def find_promoter_boxes(seq_str: str, sigma_factor: str, tss_idx: int) -> Tuple[str, str, str, str]:
    """Finds putative -10 and -35 boxes in the sequence window (TSS at index tss_idx).
    Returns a tuple of (minus_10_seq, minus_10_dist, minus_35_seq, minus_35_dist).
    If no match passes the similarity threshold, returns 'None' or 'N/A'.
    """
    if len(seq_str) < tss_idx + 1:
        return "N/A", "N/A", "N/A", "N/A"
        
    m10_cons = "TACGAT" if sigma_factor == "SigX" else "TATAAT"
    
    # Clean inner function to scan a coordinate window for a specific motif consensus
    def scan(consensus: str, start: int, end: int) -> Tuple[str, str]:
        best_seq, best_dist, min_mis = "None", "N/A", 99
        start_bounded = max(0, start)
        end_bounded = min(len(seq_str) - len(consensus), end)
        for i in range(start_bounded, end_bounded + 1):
            slice_seq = seq_str[i:i+len(consensus)]
            if len(slice_seq) != len(consensus):
                continue
            mis = sum(a != b for a, b in zip(slice_seq, consensus))
            if mis < min_mis:
                min_mis, best_seq, best_dist = mis, slice_seq, str(tss_idx - i)
        return (best_seq, best_dist) if min_mis <= 3 else ("None", "N/A")

    m10_seq, m10_dist = scan(m10_cons, tss_idx - 20, tss_idx - 6)
    m35_seq, m35_dist = scan("TTGACA", tss_idx - 42, tss_idx - 28)
    return m10_seq, m10_dist, m35_seq, m35_dist

def determine_sigma_factor(chrom: str, pos: int, strand: str, regulator_data: Dict, chrom_map: Dict, max_dist=100) -> Tuple[str, Tuple]:
    """Classifies TSS to exactly one sigma factor (either SigA or SigX).
    Looks for RpoD (SigA) or ComX (SigX) binding sites upstream on the same strand.
    If both exist, resolves conflict by selecting the one closer to the TSS.
    Returns (sigma_name, assoc_site_tuple) where assoc_site_tuple is (chrom, start, end, strand, reg_name).
    """
    resolved_reg_chrom = chrom
    if resolved_reg_chrom not in regulator_data or not regulator_data[resolved_reg_chrom]:
        return "None", None
        
    candidates = regulator_data[resolved_reg_chrom]
    sigma_candidates = []
    
    for start, end, s_r, reg_name, feat_id in candidates:
        if s_r != strand:
            continue
        if reg_name not in SIGMA_FACTOR_MAPPING:
            continue
            
        sigma_name = SIGMA_FACTOR_MAPPING[reg_name]
            
        if strand == '+':
            dist = pos - end
        else:
            dist = start - pos
            
        if 0 <= dist <= max_dist:
            sigma_candidates.append((sigma_name, dist, (resolved_reg_chrom, start, end, s_r, reg_name)))
            
    if not sigma_candidates:
        return "None", None
        
    # Select the closest sigma binding site
    sigma_candidates.sort(key=lambda x: x[1])
    return sigma_candidates[0][0], sigma_candidates[0][2]

def run_proximity_analysis(extracted_positives: List[Dict], window_size: int, conflict_threshold: int) -> List[Dict]:
    """Analyzes same-strand TSS proximity conflicts and labels redundancy status.
    Resolves conflicts using a hierarchy: technique priority (manual/curation > Cappable-seq > RNA-seq),
    followed by upstream position (outer boundary), and finally score.
    """
    # Sort by chrom, strand, sigma_factor, and position to resolve same-strand conflicts within each sigma factor group
    extracted_positives.sort(key=lambda x: (x["chrom"], x["strand"], x["sigma_factor"], x["pos"]))
    
    # 1. Identify clusters of conflicts (< 25 bp) within the same sigma factor group
    i = 0
    while i < len(extracted_positives):
        cluster = [extracted_positives[i]]
        strand = extracted_positives[i]["strand"]
        chrom = extracted_positives[i]["chrom"]
        sigma_factor = extracted_positives[i]["sigma_factor"]
        pos = extracted_positives[i]["pos"]
        
        j = i + 1
        while j < len(extracted_positives):
            next_tss = extracted_positives[j]
            if (next_tss["chrom"] == chrom and 
                next_tss["strand"] == strand and 
                next_tss["sigma_factor"] == sigma_factor and 
                (next_tss["pos"] - pos) < conflict_threshold):
                cluster.append(next_tss)
                pos = next_tss["pos"]
                j += 1
            else:
                break
                
        # Priority-based selection key
        def get_technique_priority(source_str: str) -> int:
            source_lower = source_str.lower() if source_str else ""
            if "manual" in source_lower or "curation" in source_lower or "literature" in source_lower or "experimental" in source_lower:
                return 3
            elif "cappable" in source_lower:
                return 2
            elif "rna-seq" in source_lower or "rnaseq" in source_lower or "rna seq" in source_lower:
                return 1
            return 0

        def representative_key(tss):
            tech_pri = get_technique_priority(tss.get("source", ""))
            # On positive strand (+), more UP (upstream) means lower coordinate (minimum).
            # We want to select the one that maximizes this key, so we use -pos.
            # On negative strand (-), more UP (upstream) means higher coordinate (maximum).
            # We want to select the one that maximizes this key, so we use pos.
            pos_key = -tss["pos"] if tss["strand"] == '+' else tss["pos"]
            # Curated/literature TSSs are mapped to 'curated' in headers/files, but we treat them as 0
            # for internal numerical comparison. This is safe because 'tech_pri' is the first element
            # in the tuple: a manually curated TSS (tech_pri=3) always outranks an RNA-seq TSS (tech_pri=1)
            # regardless of their score.
            score_key = 0 if tss["score"] == 'curated' else (int(tss["score"]) if str(tss["score"]).isdigit() else 0)
            return (tech_pri, pos_key, score_key)

        best_tss = max(cluster, key=representative_key)
        for tss in cluster:
            tss["is_representative"] = (tss == best_tss)
            
        if len(cluster) > 1:
            cluster_positions = [x['pos'] for x in cluster]
            discarded_positions = [x['pos'] for x in cluster if x != best_tss]
            print(f"[INFO] TSS same-strand conflict on {chrom} ({strand}) for {sigma_factor} at positions {cluster_positions}. "
                  f"Keeping representative: {best_tss['pos']} (score: {best_tss['score']}, source: {best_tss.get('source', 'unknown')}), "
                  f"discarding: {discarded_positions}.", file=sys.stderr)
            
        i = j
        
    # 2. Compute overlap statuses and percentages
    for k in range(len(extracted_positives)):
        tss = extracted_positives[k]
        pos = tss["pos"]
        chrom = tss["chrom"]
        strand = tss["strand"]
        
        neighbors = []
        if k > 0:
            prev_tss = extracted_positives[k - 1]
            if prev_tss["chrom"] == chrom and prev_tss["strand"] == strand:
                neighbors.append(abs(pos - prev_tss["pos"]))
        if k < len(extracted_positives) - 1:
            next_tss = extracted_positives[k + 1]
            if next_tss["chrom"] == chrom and next_tss["strand"] == strand:
                neighbors.append(abs(next_tss["pos"] - pos))
                
        min_distance = min(neighbors) if neighbors else float('inf')
        
        if min_distance < window_size:
            tss["overlap_status"] = "Redundant_Overlap"
            tss["overlap_pct"] = round(max(0.0, 100.0 * (window_size - min_distance) / window_size), 2)
        elif min_distance <= 150:
            tss["overlap_status"] = "Cluster_Isoform"
            tss["overlap_pct"] = 0.0
        else:
            tss["overlap_status"] = "Isolated"
            tss["overlap_pct"] = 0.0
            
    return extracted_positives

def extract_positives() -> None:
    """Main function to parse GFF, validate coordinates, and extract sequences."""
    args = parse_arguments()

    print(f"[INFO] Loading genome from {args.fasta}...", file=sys.stderr)
    genome = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))
    if not genome:
        sys.exit("[ERROR] FASTA file is empty or invalid.")

    # Calculate global background GC fraction
    total_genome_bases = 0
    gc_genome_bases = 0
    for rec in genome.values():
        seq_upper = str(rec.seq).upper()
        total_genome_bases += len(seq_upper)
        gc_genome_bases += seq_upper.count('G') + seq_upper.count('C')
    p_genome = gc_genome_bases / total_genome_bases if total_genome_bases > 0 else 0.0

    # Calculate total window size dynamically
    window_size = args.upstream + args.downstream + 1

    # Use default hardcoded chromosome mapping

    # Load structural annotations (CDS & Regulators)
    cds_data, regulator_data = load_cds_and_regulators(args.gff_cds)
    
    # Track all unique sigma binding site locations from structural annotation
    all_sigma_binding_sites = set()
    if regulator_data:
        for chrom_id, regs in regulator_data.items():
            for r_start, r_end, r_strand, r_name, r_feat_id in regs:
                if r_name in SIGMA_FACTOR_MAPPING:
                    all_sigma_binding_sites.add((chrom_id, r_start, r_end, r_strand, r_name))
    
    skipped = {"bounds": 0, "n_bases": 0, "low_conf": 0, "strand_corrected": 0}
    valid_bases = {'A', 'C', 'G', 'T'}
    unique_positives = {}

    print(f"[INFO] Scanning TSS coordinates in {args.gff}...", file=sys.stderr)
    try:
        parsed_records = []
        with open(args.gff) as f:
            for rec in GFF.parse(f):
                parsed_records.append(rec)
    except Exception as e:
        sys.exit(f"[ERROR] Could not parse TSS GFF file: {e}")

    for rec in parsed_records:
        chrom = rec.id
        resolved_genome_chrom = chrom
        if resolved_genome_chrom not in genome:
            print(f"[WARNING] Chromosome '{chrom}' (resolved as '{resolved_genome_chrom}') not found in genome FASTA.", file=sys.stderr)
            continue
            
        genome_seq = genome[resolved_genome_chrom].seq

        for feat in get_all_tss(rec.features):
            name = feat.qualifiers.get('Name', [''])[0].lower()
            is_low_conf = 'lower_confidence' in name
            
            if is_low_conf and args.exclude_low_conf:
                skipped["low_conf"] += 1
                continue
            
            appendix = '-LOWCONF' if is_low_conf else ''
            if feat.location.strand == 1:
                original_strand = '+'
            elif feat.location.strand == -1:
                original_strand = '-'
            else:
                print(f"[WARNING] TSS '{feat.id}' has no strand ({feat.location.strand}), skipping.", file=sys.stderr)
                continue
            tss_id = feat.id
            pos = int(feat.location.start)

            # Strictly respect the experimental strand from the TSS GFF file
            strand = original_strand

            # Coordinate boundaries slicing
            if strand == '+':
                s = pos
                start, end = s - args.upstream, s + args.downstream + 1
            else:
                s = int(feat.location.end) - 1
                start, end = s - args.downstream, s + args.upstream + 1

            if start < 0 or end > len(genome_seq):
                skipped["bounds"] += 1
                continue

            subseq = genome_seq[start:end]
            if strand == '-':
                subseq = subseq.reverse_complement()
            
            seq_str = str(subseq).upper()
            
            if not set(seq_str).issubset(valid_bases):
                skipped["n_bases"] += 1
                continue

            clean_id = re.sub(r'^(transcription_start_site|tss)[\._]?', '', tss_id, flags=re.IGNORECASE)
            if 'score' in feat.qualifiers:
                score_raw = feat.qualifiers['score'][0]
                score = 'curated' if score_raw == '.' else score_raw
            else:
                score = 'curated'

            # Cross-reference closest upstream regulator within the k-mer window size
            assoc_reg, reg_dist = find_associated_regulator(chrom, s, strand, regulator_data, chrom_map, max_dist=window_size)

            # Determine the source/technique of the TSS
            source = ""
            if hasattr(feat, 'source') and feat.source:
                source = feat.source
            elif 'source' in feat.qualifiers:
                source = feat.qualifiers['source'][0]

            # Determine the exact sigma factor association within the k-mer window size
            sigma_factor, assoc_site = determine_sigma_factor(chrom, s, strand, regulator_data, chrom_map, max_dist=window_size)

            # Find closest downstream CDS start (5' UTR length) if CDS data is loaded
            downstream_cds, cds_dist = "None", "N/A"
            if cds_data:
                downstream_cds, cds_dist = find_downstream_cds_distance(chrom, s, strand, cds_data, chrom_map)

            coord_key = (resolved_genome_chrom, s, strand)
            if coord_key in unique_positives:
                continue

            unique_positives[coord_key] = {
                "clean_id": clean_id,
                "appendix": appendix,
                "chrom": chrom,
                "pos": s,
                "strand": strand,
                "tss_id": tss_id,
                "is_low_conf": is_low_conf,
                "score": score,
                "source": source,
                "associated_reg": assoc_reg,
                "regulator_dist": reg_dist,
                "sigma_factor": sigma_factor,
                "assoc_site": assoc_site,
                "downstream_cds": downstream_cds,
                "cds_dist": cds_dist,
                "seq_str": seq_str
            }

    # Proximity cluster analysis
    raw_extracted_list = list(unique_positives.values())
    analyzed_list = run_proximity_analysis(raw_extracted_list, window_size, args.conflict_threshold)

    # Sort final list back to natural genomic coordinate order
    analyzed_list.sort(key=lambda x: (x["chrom"], x["pos"]))

    # Output file paths
    fasta_out = f"{args.output}.fasta"
    tsv_out = f"{args.output}_metadata.tsv"
    sig_a_fasta = f"{args.output}_SigA.fasta"
    sig_a_tsv = f"{args.output}_SigA_metadata.tsv"
    sig_x_fasta = f"{args.output}_SigX.fasta"
    sig_x_tsv = f"{args.output}_SigX_metadata.tsv"

    # Unified TSV header columns list
    headers = [
        "Sequence_ID", "Chromosome", "TSS_Position_0based", "Strand", 
        "TSS_ID", "Confidence_Level", "Score", "Associated_Regulator", "Regulator_Distance_bp",
        "Sigma_Factor", "Downstream_Gene", "5UTR_Length_bp", 
        "Minus_10_Seq", "Minus_10_Dist_bp", "Minus_35_Seq", "Minus_35_Dist_bp",
        "GC_Content(%)", "Overlap_Status", "Overlap_Percentage", "Is_Cluster_Representative"
    ]

    extracted = 0
    sig_a_count = 0
    sig_x_count = 0
    gc_values = []
    purines_at_plus1 = 0
    
    # Write all datasets in a single, unified pass
    with open(fasta_out, 'w') as f_out, open(tsv_out, 'w', newline='') as tsvfile, \
         open(sig_a_fasta, 'w') as f_a, open(sig_a_tsv, 'w', newline='') as tsv_a, \
         open(sig_x_fasta, 'w') as f_x, open(sig_x_tsv, 'w', newline='') as tsv_x:
         
        writer = csv.writer(tsvfile, delimiter='\t')
        writer_a = csv.writer(tsv_a, delimiter='\t')
        writer_x = csv.writer(tsv_x, delimiter='\t')
        
        # Write headers to all files
        writer.writerow(headers)
        writer_a.writerow(headers)
        writer_x.writerow(headers)
        
        associated_sigma_sites = set()
        for meta in analyzed_list:
            if not meta.get("is_representative", True):
                continue
                
            if meta.get("assoc_site"):
                associated_sigma_sites.add(meta["assoc_site"])
                
            reg_header = meta['sigma_factor'] if meta['sigma_factor'] != "None" else meta['associated_reg']
            header = f"TSS_{meta['clean_id']}_{meta['chrom']}_{meta['pos']+1}_{meta['strand']}_{meta['score']}_{reg_header}{meta['appendix']}"
            seq = meta["seq_str"]
            
            # GC content & theoretical base calculations (performed once)
            gc_content = calculate_gc(seq)
            gc_values.append(gc_content)
            
            if len(seq) > args.upstream:
                plus1_base = seq[args.upstream]
                if plus1_base in ('A', 'G'):
                    purines_at_plus1 += 1
            
            # Scan sequence for putative -10 and -35 promoter boxes
            m10_seq, m10_dist, m35_seq, m35_dist = find_promoter_boxes(seq, meta["sigma_factor"], args.upstream)
            
            row = [
                header, meta["chrom"], meta["pos"], meta["strand"], 
                meta["clean_id"], "Low" if meta["is_low_conf"] else "High", 
                meta["score"], meta["associated_reg"], meta["regulator_dist"],
                meta["sigma_factor"], meta["downstream_cds"], meta["cds_dist"],
                m10_seq, m10_dist, m35_seq, m35_dist,
                round(gc_content, 2), meta["overlap_status"],
                meta["overlap_pct"], meta["is_representative"]
            ]
            
            # 1. Write to general dataset
            f_out.write(f">{header}\n{seq}\n")
            writer.writerow(row)
            extracted += 1
            
            # 2. Write to SigA dataset if applicable
            if meta["sigma_factor"] == "SigA":
                f_a.write(f">{header}\n{seq}\n")
                writer_a.writerow(row)
                sig_a_count += 1
                
            # 3. Write to SigX dataset if applicable
            elif meta["sigma_factor"] == "SigX":
                f_x.write(f">{header}\n{seq}\n")
                writer_x.writerow(row)
                sig_x_count += 1

    print(f"[INFO] Sigma factor subset generated -> {sig_a_fasta} ({sig_a_count} sequences)", file=sys.stderr)
    print(f"[INFO] Sigma factor subset generated -> {sig_x_fasta} ({sig_x_count} sequences)", file=sys.stderr)

    # Calculate background and promoter GC content metrics
    mean_gc = statistics.mean(gc_values) if gc_values else 0.0
    stdev_gc = statistics.stdev(gc_values) if len(gc_values) > 1 else 0.0
    expected_gc = p_genome * 100

    # Calculate proximity cluster variables first
    representatives = sum(1 for x in analyzed_list if x["is_representative"])
    overlap_count = sum(1 for x in analyzed_list if x["overlap_status"] == "Redundant_Overlap")
    isoform_count = sum(1 for x in analyzed_list if x["overlap_status"] == "Cluster_Isoform")
    isolated_count = sum(1 for x in analyzed_list if x["overlap_status"] == "Isolated")
    
    # Calculate conflict breakdown by sigma factor
    discarded_sig_a = sum(1 for x in analyzed_list if not x["is_representative"] and x["sigma_factor"] == "SigA")
    discarded_sig_x = sum(1 for x in analyzed_list if not x["is_representative"] and x["sigma_factor"] == "SigX")
    discarded_none = sum(1 for x in analyzed_list if not x["is_representative"] and x["sigma_factor"] == "None")

    # Analyze unmapped binding sites distances to closest downstream TSS
    unmapped_sites = all_sigma_binding_sites - associated_sigma_sites
    for u in unmapped_sites:
        print(f"[UNMAPPED_SIGMA_SITE] {u}", file=sys.stderr)
    unmapped_by_dist = {
        "close_exceeded": 0,    # distance <= 150 bp (exceeded window)
        "distant": 0,           # 150 < distance <= 500 bp
        "very_distant": 0,      # distance > 500 bp
        "no_tss": 0             # no TSS downstream on same strand
    }
    
    for u_chrom, u_start, u_end, u_strand, u_name in unmapped_sites:
        min_dist = float('inf')
        for tss in analyzed_list:
            if not tss["is_representative"]:
                continue
            res_tss_chrom = tss["chrom"]
            if res_tss_chrom != u_chrom or tss["strand"] != u_strand:
                continue
            
            if u_strand == '+':
                if tss["pos"] >= u_end:
                    min_dist = min(min_dist, tss["pos"] - u_end)
            else:
                if tss["pos"] <= u_start:
                    min_dist = min(min_dist, u_start - tss["pos"])
                    
        if min_dist == float('inf'):
            unmapped_by_dist["no_tss"] += 1
        elif min_dist <= 150:
            unmapped_by_dist["close_exceeded"] += 1
        elif min_dist <= 500:
            unmapped_by_dist["distant"] += 1
        else:
            unmapped_by_dist["very_distant"] += 1

    print("\n" + "="*50, file=sys.stderr)
    print("POSITIVE DATASET EXTRACTOR REPORT:", file=sys.stderr)
    print(f"Total TSS candidates in GFF:          {len(analyzed_list)}", file=sys.stderr)
    print(f"Total sequences successfully saved:   {extracted}", file=sys.stderr)
    print(f"  - TSS associated with SigA:         {sig_a_count}", file=sys.stderr)
    print(f"  - TSS associated with SigX:         {sig_x_count}", file=sys.stderr)
    print(f"  - TSS NOT associated with anyone:   {extracted - sig_a_count - sig_x_count}", file=sys.stderr)
    print(f"Total proximity conflicts (filtered): {len(analyzed_list) - representatives}", file=sys.stderr)
    print(f"  - Discarded from SigA group:        {discarded_sig_a}", file=sys.stderr)
    print(f"  - Discarded from SigX group:        {discarded_sig_x}", file=sys.stderr)
    print(f"  - Discarded from None group:        {discarded_none}", file=sys.stderr)
    print(f"Total Sigma binding sites in GFF:     {len(all_sigma_binding_sites)}", file=sys.stderr)
    print(f"Sigma binding sites NOT mapped to TSS:{len(unmapped_sites)}", file=sys.stderr)
    print(f"  - Exceeded window limit (81-150bp): {unmapped_by_dist['close_exceeded']}", file=sys.stderr)
    print(f"  - Distant from TSS (150-500bp):     {unmapped_by_dist['distant']}", file=sys.stderr)
    print(f"  - Very distant from TSS (>500bp):   {unmapped_by_dist['very_distant']}", file=sys.stderr)
    print(f"  - Silent (no TSS downstream):       {unmapped_by_dist['no_tss']}", file=sys.stderr)
    print(f"Strand adjustments applied (corrected):{skipped['strand_corrected']}", file=sys.stderr)
    
    print("\nPROXIMITY CONFLCT ANALYSIS:", file=sys.stderr)
    
    print(f"Redundant Overlaps (< {window_size} bp):       {overlap_count}", file=sys.stderr)
    print(f"Cluster Promoter Isoforms (81-150 bp): {isoform_count}", file=sys.stderr)
    print(f"Isolated Promoters (> 150 bp):        {isolated_count}", file=sys.stderr)
    print(f"Steric Hindrance Representatives:     {representatives} (Satellite peaks filtered: {len(analyzed_list) - representatives})", file=sys.stderr)

    print("\nBIOLOGICAL & STATISTICAL QC:", file=sys.stderr)
    print(f"Global GC Content (Promoters):        {mean_gc:.2f}% (Mean: {mean_gc:.2f}% ± {stdev_gc:.2f}%)", file=sys.stderr)
    print(f"Background Genome GC Content:         {expected_gc:.2f}%", file=sys.stderr)
    
    if extracted > 0:
        purine_pct = (purines_at_plus1 / extracted) * 100
        print(f"TSS +1 Purine Preference (A/G):       {purine_pct:.2f}%", file=sys.stderr)
        if purine_pct < 50:
            print("[WARNING] Low purine density at +1 site. Please check coordinate orientations.", file=sys.stderr)

    print("\nLOST CANDIDATES:", file=sys.stderr)
    print(f"Due to genome boundary limits:        {skipped['bounds']}", file=sys.stderr)
    print(f"Due to invalid 'N' characters:        {skipped['n_bases']}", file=sys.stderr)
    print(f"Due to excluded low-confidence flags: {skipped['low_conf']}", file=sys.stderr)
    print("="*50, file=sys.stderr)
    
    print(f"[SUCCESS] Dataset generated -> {fasta_out}", file=sys.stderr)
    print(f"[SUCCESS] Metadata generated -> {tsv_out}", file=sys.stderr)

if __name__ == '__main__':
    extract_positives()