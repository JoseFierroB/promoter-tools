#!/usr/bin/env python3
"""
Split benchmark AUCs by IGR conservation status.

Uses cached predictions + conservation tags to compute:
  AUC_global | AUC_conserved | AUC_nonconserved | AUC_intragenic

For tools with sequence-order predictions (meme, fimo, mldspp, lcnn, promotech),
predictions are matched 1:1 with FASTA input order. For iPro-MP, the combined CSV
has Sequence/Prediction/Probability columns.

Output: output/tables/auc_by_conservation.tsv
"""

import csv
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "output" / "tables" / "auc_by_conservation.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_fasta_ids(fasta_path):
    """Return list of sequence IDs in FASTA order."""
    ids = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def load_simple_scores(csv_path):
    """Load predictions from single-column CSV (PRED header)."""
    scores = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if row:
                scores.append(float(row[0]))
    return scores


def load_ipromp_scores(csv_path, n_pos=988, n_neg=1000):
    """Load iPro-MP combined predictions (Sequence,Prediction,Probability).
    First n_pos rows are positives, next n_neg are negatives (combined FASTA order)."""
    pos_scores = []
    neg_scores = []
    with open(csv_path) as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) >= 3:
                score = float(parts[2])
            elif len(parts) >= 2:
                score = float(parts[1])
            else:
                score = float(parts[0])

            if i < n_pos:
                pos_scores.append(score)
            else:
                neg_scores.append(score)

    return pos_scores[:n_pos], neg_scores[:n_neg]


def load_igr_index(igr_path):
    """Load IGRs → list of {chrom, start, end, igr_id}."""
    igrs = []
    with open(igr_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            igrs.append({
                "chrom": row["chrom"].strip("\r"),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "igr_id": row["igr_id"].strip("\r"),
            })
    return igrs


def find_containing_igr(chrom, pos, igrs):
    """Find IGR containing a genomic position (1-based)."""
    for igr in igrs:
        if igr["chrom"] == chrom and igr["start"] <= pos <= igr["end"]:
            return igr["igr_id"]
    return ""


def load_conserved_set(m8_path, strain="d39v"):
    """Load IGR IDs that have a match in the other strain.
    D39V: conserved = D39V queries in column 1
    TIGR4: conserved = TIGR4 targets in column 2
    """
    conserved = set()
    col = 0 if strain == "d39v" else 1
    with open(m8_path) as f:
        for line in f:
            conserved.add(line.strip().split("\t")[col])
    return conserved


def build_tags_on_the_fly(tss_metadata_path, igr_path, m8_path, strain):
    """Compute conserved_igr tags directly from TSS positions + IGRs + MMseqs2."""
    igrs = load_igr_index(igr_path)
    conserved = load_conserved_set(m8_path, strain)

    tags = {}
    with open(tss_metadata_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row1 in reader:
            row = {k.strip("\r"): v.strip("\r") for k, v in row1.items()}
            sid = row.get("Sequence_ID", row.get("Locus_Tag", ""))
            if strain == "d39v":
                pos = int(row.get("TSS_Position_0based", 0)) + 1
                chrom = row.get("Chromosome", "D39V")
            else:
                pos = int(row.get("TSS_Position", 0))
                chrom = row.get("Chromosome", "NC_003028.3")

            igr_id = find_containing_igr(chrom, pos, igrs)
            if igr_id and igr_id in conserved:
                tags[sid] = "yes"
            elif igr_id:
                tags[sid] = "no"
            else:
                tags[sid] = "intragenic"
    return tags


def load_tags(tagged_tsv):
    """Load conservation tags → dict[Sequence_ID] = conserved_igr."""
    tags = {}
    with open(tagged_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row.get("Sequence_ID", "").strip("\r")
            cons = row.get("conserved_igr", "").strip("\r")
            if sid:
                tags[sid] = cons
    return tags


def compute_group_auc(pos_scores, neg_scores, pos_tags, group_name):
    """Compute AUC for a subset of positive sequences matching a given conservation group."""
    # Filter positive scores to only those in the group
    group_pos = []
    for sid, score in zip(pos_tags.keys(), pos_scores):
        if pos_tags.get(sid, "") == group_name:
            group_pos.append(score)
    if len(group_pos) == 0 or len(neg_scores) == 0:
        return None, len(group_pos)
    # Use all negatives
    y_true = [1] * len(group_pos) + [0] * len(neg_scores)
    y_score = group_pos + list(neg_scores)
    return roc_auc_score(y_true, y_score), len(group_pos)


def process_tool(name, pos_csv, neg_csv, pos_fasta, neg_fasta, pos_tags, tool_type="simple"):
    """Process one tool and return list of dicts with AUC per group."""
    results = []

    if tool_type == "ipromp":
        # Load FASTA to get count
        pos_ids = load_fasta_ids(pos_fasta)
        neg_ids = load_fasta_ids(neg_fasta)
        pos_scores, neg_scores = load_ipromp_scores(pos_csv, len(pos_ids), len(neg_ids))
    else:
        pos_scores = load_simple_scores(pos_csv)
        neg_scores = load_simple_scores(neg_csv)

    pos_ids = load_fasta_ids(pos_fasta)
    # Build ordered tag list matching pos_scores
    ordered_tags = [pos_tags.get(sid, "intragenic") for sid in pos_ids]

    # Global AUC
    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_score = pos_scores + list(neg_scores)
    auc_global = roc_auc_score(y_true, y_score)

    # Per-group AUC
    auc_cons, n_cons = compute_group_auc(pos_scores, neg_scores,
                                          dict(zip(pos_ids, ordered_tags)), "yes")
    auc_ncons, n_ncons = compute_group_auc(pos_scores, neg_scores,
                                            dict(zip(pos_ids, ordered_tags)), "no")
    auc_intra, n_intra = compute_group_auc(pos_scores, neg_scores,
                                            dict(zip(pos_ids, ordered_tags)), "intragenic")

    return {
        "tool": name,
        "AUC_global": round(auc_global, 4),
        "AUC_conserved": round(auc_cons, 4) if auc_cons else None,
        "AUC_nonconserved": round(auc_ncons, 4) if auc_ncons else None,
        "AUC_intragenic": round(auc_intra, 4) if auc_intra else None,
        "n_conserved": n_cons,
        "n_nonconserved": n_ncons,
        "n_intragenic": n_intra,
        "n_total_pos": len(pos_scores),
        "n_negatives": len(neg_scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strain", choices=["d39v", "tigr4"], default="d39v")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    strain = args.strain

    if strain == "d39v":
        PRED_DIR = ROOT / "output" / "predictions"
        POS_FASTA = ROOT / "data" / "benchmark" / "positives_81bp.fasta"
        NEG_FASTA = ROOT / "data" / "benchmark" / "negatives_81bp.fasta"
        TSS_META = ROOT / "data" / "benchmark" / "positives_81bp_metadata.tsv"
        IGR_TSV = ROOT / "output" / "intergenic" / "d39v" / "D39V_igrs.tsv"
        M8_PATH = ROOT / "output" / "intergenic" / "mmseqs2" / "cross" / "D39V_vs_TIGR4.m8"
    else:
        PRED_DIR = ROOT / "output" / "tigr4" / "predictions"
        POS_FASTA = ROOT / "data" / "tigr4" / "positives_high_81bp.fasta"
        NEG_FASTA = ROOT / "data" / "tigr4" / "negatives_high_81bp.fasta"
        TSS_META = ROOT / "data" / "tigr4" / "positives_high_81bp_metadata.tsv"
        IGR_TSV = ROOT / "output" / "intergenic" / "tigr4" / "TIGR4_igrs.tsv"
        M8_PATH = ROOT / "output" / "intergenic" / "mmseqs2" / "cross" / "D39V_vs_TIGR4.m8"

    print(f"Building conservation tags on-the-fly for {strain}...")
    pos_tags = build_tags_on_the_fly(TSS_META, IGR_TSV, M8_PATH, strain)
    print(f"  {len(pos_tags)} TSS tagged")

    # Tool definitions: (name, pos_csv, neg_csv, tool_type)
    tools = [
        ("meme",             PRED_DIR / "meme_pos.csv",       PRED_DIR / "meme_neg.csv",       "simple"),
        ("fimo_db",          PRED_DIR / "fimo_db_pos.csv",    PRED_DIR / "fimo_db_neg.csv",    "simple"),
        ("fimo_prok",        PRED_DIR / "fimo_prok_pos.csv",  PRED_DIR / "fimo_prok_neg.csv",  "simple"),
        ("mldspp",           PRED_DIR / "mldspp_pos.csv",     PRED_DIR / "mldspp_neg.csv",     "simple"),
        ("lcnn",             PRED_DIR / "lcnn/lcnn_pos.csv",  PRED_DIR / "lcnn/lcnn_neg.csv",  "simple"),
        ("promotech_hot",    PRED_DIR / "promotech/workdir/hot_pg_pos/sequences_predictions.csv",
                             PRED_DIR / "promotech/workdir/hot_pg_neg/sequences_predictions.csv", "simple"),
        ("promotech_tetra",  PRED_DIR / "promotech/workdir/tetra_pg_pos/sequences_predictions.csv",
                             PRED_DIR / "promotech/workdir/tetra_pg_neg/sequences_predictions.csv", "simple"),
        ("ipromp_sp12",      PRED_DIR / "ipromp/ipromp_12_predictions.csv", None,              "ipromp"),
    ]

    all_results = []
    for name, pos_p, neg_p, ttype in tools:
        if not pos_p.exists():
            print(f"  SKIP {name}: {pos_p} not found")
            continue
        if ttype != "ipromp" and not neg_p.exists():
            print(f"  SKIP {name}: {neg_p} not found")
            continue
        print(f"  Processing {name}...")
        try:
            r = process_tool(name, pos_p, neg_p, POS_FASTA, NEG_FASTA, pos_tags, ttype)
            r["strain"] = strain
            all_results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")

    # Write output
    if all_results:
        fieldnames = ["tool", "strain", "AUC_global", "AUC_conserved",
                      "AUC_nonconserved", "AUC_intragenic",
                      "n_conserved", "n_nonconserved", "n_intragenic",
                      "n_total_pos", "n_negatives"]
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                               lineterminator="\n")
            w.writeheader()
            w.writerows(all_results)

        # Print summary
        print(f"\n{'='*80}")
        print(f"  AUC BY CONSERVATION — {strain.upper()}")
        print(f"{'='*80}")
        print(f"{'Tool':<20} {'Global':>8} {'Conserved':>10} {'Non-Cons':>10} {'Intragenic':>11}")
        print("-" * 65)
        for r in all_results:
            auc_g = f"{r['AUC_global']:.4f}" if r["AUC_global"] else "N/A"
            auc_c = f"{r['AUC_conserved']:.4f}" if r["AUC_conserved"] else "N/A"
            auc_n = f"{r['AUC_nonconserved']:.4f}" if r["AUC_nonconserved"] else "N/A"
            auc_i = f"{r['AUC_intragenic']:.4f}" if r["AUC_intragenic"] else "N/A"
            print(f"{r['tool']:<20} {auc_g:>8} {auc_c:>10} {auc_n:>10} {auc_i:>11}")
        print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
