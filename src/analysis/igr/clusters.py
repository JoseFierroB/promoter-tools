#!/usr/bin/env python3
"""
IGR Clustering Analysis — D39V vs TIGR4
=========================================
Self-contained pipeline for intergenic region conservation analysis
using MMseqs2 easy-cluster on combined D39V+TIGR4 IGRs.

Modes:
  cluster   Run MMseqs2 easy-cluster on combined FASTA
  analyze   Classify clusters + compute validation metrics (default)
  sensitive Sensitivity analysis across identity/coverage thresholds
  shuffled  Dinucleotide-shuffled control (FDR estimation)
  tss       Cross-reference TSS positions with cluster types
  align     Generate pairwise alignments + full stats for all pairs
  export    Write annotated TSV of all 1+1 ortholog pairs

Examples:
  python src/analysis/igr/clusters.py cluster --id 0.95 --cov 0.70
  python src/analysis/igr/clusters.py analyze --cluster-file clusters.tsv
  python src/analysis/igr/clusters.py sensitive
  python src/analysis/igr/clusters.py shuffled
  python src/analysis/igr/clusters.py tss
  python src/analysis/igr/clusters.py export --out pairs_annotated.tsv

Output directory: output/tables/igr/
"""

import argparse
import csv
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict, Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA = ROOT / "data"
OUT_IGR = ROOT / "output" / "intergenic"
OUT_TABLES = ROOT / "output" / "tables" / "igr"
OUT_DIR = OUT_TABLES

# ── Paths ──
D39V_IGR_TSV = OUT_IGR / "d39v" / "D39V_igrs.tsv"
TIGR4_IGR_TSV = OUT_IGR / "tigr4" / "TIGR4_igrs.tsv"
D39V_IGR_FASTA = OUT_IGR / "d39v" / "D39V_igrs.fasta"
TIGR4_IGR_FASTA = OUT_IGR / "tigr4" / "TIGR4_igrs.fasta"
COMBINED_FASTA = OUT_IGR / "combined" / "All_IGRs.fasta"
M8_FWD = OUT_IGR / "mmseqs2" / "cross" / "D39V_vs_TIGR4.m8"
M8_REV = OUT_IGR / "mmseqs2" / "cross" / "TIGR4_vs_D39V.m8"
NUCMER_COORDS = OUT_IGR / "mummer" / "d39v_vs_tigr4.coords"
D39V_TSS = DATA / "benchmark" / "positives_81bp_metadata.tsv"
TIGR4_TSS = DATA / "tigr4" / "positives_high_81bp_metadata.tsv"
GENE_MAP_FILE = Path("/tmp/gene_orthology/gene_map.tsv")

CLUSTER_DIR = OUT_IGR / "mmseqs2" / "combined_clusters"


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_clusters(path):
    """Load MMseqs2 cluster TSV → dict[rep] = [member, ...]."""
    cl = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cl[parts[0]].append(parts[1])
    return dict(cl)


def load_igr_metadata():
    """Load IGR TSVs → (d39v_dict, tigr4_dict)."""
    d = {}; t = {}
    with open(D39V_IGR_TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            d[row["igr_id"].strip()] = row
    with open(TIGR4_IGR_TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t[row["igr_id"].strip()] = row
    return d, t


def load_gene_map():
    """Load SPV_* ↔ SP_RS* orthology from GFF3 annotations."""
    if not GENE_MAP_FILE.exists():
        _build_gene_map()
    gm = {}
    with open(GENE_MAP_FILE) as f:
        for line in f:
            d39v, sp, tigr4 = line.strip().split("\t")
            gm[d39v] = tigr4
    return gm


def _build_gene_map():
    """Regenerate gene orthology map from GFF3 files."""
    GENE_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    tigr4_map = {}
    with open(DATA / "reference" / "NC_003028.gff3") as f:
        for line in f:
            if "locus_tag=" in line and "old_locus_tag=" in line:
                import re
                lt = re.search(r"locus_tag=([^;]+)", line)
                old = re.search(r"old_locus_tag=([^;]+)", line)
                if lt and old:
                    tigr4_map[old.group(1)] = lt.group(1)
    with open(DATA / "reference" / "D39V.gff3") as f_in, open(GENE_MAP_FILE, "w") as f_out:
        for line in f_in:
            if "locus_tag=" in line and "Corresponds to SP_" in line:
                lt = re.search(r"locus_tag=([^;]+)", line)
                sp = re.search(r"Corresponds to (SP_\d+)", line)
                if lt and sp and sp.group(1) in tigr4_map:
                    f_out.write(f"{lt.group(1)}\t{sp.group(1)}\t{tigr4_map[sp.group(1)]}\n")


def load_tss_positions():
    """Load TSS positions → (d39v_list, tigr4_list). Each: (id, pos_1based, extra...)."""
    d = []; t = []
    with open(D39V_TSS, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            d.append((row["Sequence_ID"].strip(),
                      int(row["TSS_Position_0based"].strip()) + 1,
                      row.get("Sigma_Factor", "").strip() or "None"))
    with open(TIGR4_TSS, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t.append((row["Sequence_ID"].strip(), int(row["TSS_Position"].strip())))
    return d, t


def load_nucmer_blocks():
    """Parse nucmer show-coords → [(d39v_start, d39v_end, tigr4_start, tigr4_end), ...]."""
    blocks = []
    with open(NUCMER_COORDS) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("/") or s.startswith("NUCMER") or \
               s.startswith("=") or s.startswith("[") or not s[0].isdigit():
                continue
            parts = s.split("|")
            if len(parts) < 4: continue
            try:
                dc = parts[0].strip().split(); tc = parts[1].strip().split()
                blocks.append((int(dc[0]), int(dc[1]), int(tc[0]), int(tc[1])))
            except (ValueError, IndexError): continue
    return blocks


# ═══════════════════════════════════════════════════════════════════
#  CLUSTER CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def classify_clusters(clusters):
    """
    Classify each cluster and return structured results.
    
    Returns:
        summary: dict with counts per category
        pairs_1_1: list of (d39v_id, tigr4_id) for 1+1 orthologs
        multi: list of multi-hit clusters with details
    """
    stats = Counter()
    pairs_1_1 = []
    multi_hit = []
    
    for rep, members in clusters.items():
        d_m = [m for m in members if "D39V" in m]
        t_m = [m for m in members if "NC_003" in m]
        sz = len(members)
        
        if sz == 1:
            stats["singleton"] += 1
        elif len(d_m) == 1 and len(t_m) == 1 and sz == 2:
            stats["ortholog_1+1"] += 1
            pairs_1_1.append((d_m[0], t_m[0]))
        elif len(d_m) == 1 and len(t_m) > 1:
            stats["multi_1+N"] += 1
            multi_hit.append(("1+N", d_m[0], t_m))
        elif len(d_m) > 1 and len(t_m) == 1:
            stats["multi_N+1"] += 1
            multi_hit.append(("N+1", d_m, t_m[0]))
        elif len(d_m) > 1 and len(t_m) > 1:
            stats["multi_N+N"] += 1
            multi_hit.append(("N+N", d_m, t_m))
        elif len(d_m) > 1:
            stats["d39v_only_multi"] += 1
        elif len(t_m) > 1:
            stats["tigr4_only_multi"] += 1
    
    return dict(stats), pairs_1_1, multi_hit


# ═══════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_orthologs(pairs_1_1, d_igrs, t_igrs, gene_map, d_tss, t_tss, nucmer_blocks):
    """Compute all validation metrics for 1+1 ortholog pairs."""
    total = len(pairs_1_1)
    same_orient = 0; both_gene = 0; one_gene = 0
    same_len = 0; len_ratios = []
    tss_both = 0; tss_d = 0; tss_t = 0; nucmer_ok = 0
    
    for d_id, t_id in pairs_1_1:
        di = d_igrs.get(d_id, {}); ti = t_igrs.get(t_id, {})
        if not di or not ti: continue
        
        # Orientation
        do = di.get("orientation_type", "").replace("(++)","").replace("(--)","")
        to = ti.get("orientation_type", "").replace("(++)","").replace("(--)","")
        if do == to: same_orient += 1
        
        # Gene orthology
        dl = di.get("left_cds", "").strip(); dr = di.get("right_cds", "").strip()
        tl = ti.get("left_cds", "").strip(); tr = ti.get("right_cds", "").strip()
        lok = gene_map.get(dl, "") == tl; rok = gene_map.get(dr, "") == tr
        if lok and rok: both_gene += 1
        elif lok or rok: one_gene += 1
        
        # Length ratio
        dlen = int(di.get("length", 1)); tlen = int(ti.get("length", 1))
        if dlen > 0 and tlen > 0:
            r = dlen / tlen
            len_ratios.append(r)
            if 0.9 < r < 1.1: same_len += 1
        
        # TSS
        ds = sum(1 for _, p, *_ in d_tss if int(di["start"]) <= p <= int(di["end"]))
        ts = sum(1 for _, p in t_tss   if int(ti["start"]) <= p <= int(ti["end"]))
        if ds and ts: tss_both += 1
        elif ds: tss_d += 1
        elif ts: tss_t += 1
        
        # Nucmer
        if nucmer_blocks:
            d_s, d_e = int(di["start"]), int(di["end"])
            t_s, t_e = int(ti["start"]), int(ti["end"])
            for bds, bde, bts, bte in nucmer_blocks:
                if max(d_s, bds) <= min(d_e, bde) and max(t_s, bts) <= min(t_e, bte):
                    nucmer_ok += 1; break
    
    nucmer_pct = nucmer_ok / total * 100 if nucmer_blocks else None
    
    return {
        "total_1+1": total,
        "same_orientation": same_orient, "pct_same_orient": same_orient / total * 100,
        "both_genes_orthologs": both_gene, "pct_both_genes": both_gene / total * 100,
        "any_gene_ortholog": both_gene + one_gene, "pct_any_gene": (both_gene + one_gene) / total * 100,
        "same_length": same_len, "pct_same_len": same_len / total * 100,
        "mean_len_ratio": statistics.mean(len_ratios) if len_ratios else 0,
        "median_len_ratio": statistics.median(len_ratios) if len_ratios else 0,
        "tss_both": tss_both, "tss_d39v_only": tss_d, "tss_tigr4_only": tss_t,
        "nucmer_overlap": nucmer_ok, "pct_nucmer": nucmer_pct,
    }


def compute_tss_inventory(clusters, d_igrs, t_igrs, d_tss, t_tss):
    """Cross-reference all TSS with cluster types. Returns per-strain dicts."""
    def find_igr(pos, igrs):
        for iid, data in igrs.items():
            if int(data["start"]) <= pos <= int(data["end"]):
                return iid
        return None
    
    def get_ctype(iid, clusters):
        if not iid: return "in_CDS"
        for rep, members in clusters.items():
            if iid in members:
                d_m = sum(1 for m in members if "D39V" in m)
                t_m = len(members) - d_m
                if len(members) == 1: return "singleton"
                if d_m == 1 and t_m == 1 and len(members) == 2: return "ortholog_1+1"
                if d_m > 0 and t_m > 0: return "multi_hit"
                return "cepa_especifico_multi"
        return "singleton"
    
    results = {}
    for strain, tss_list, igrs in [
        ("D39V", d_tss, d_igrs), ("TIGR4", t_tss, t_igrs)]:
        cats = Counter()
        for _, pos, *_ in tss_list:
            iid = find_igr(pos, igrs)
            cats[get_ctype(iid, clusters)] += 1
        results[strain] = dict(cats)
    return results


# ═══════════════════════════════════════════════════════════════════
#  MMSEQS2 EXECUTION
# ═══════════════════════════════════════════════════════════════════

def run_easy_cluster(id_thresh, cov_thresh, output_prefix, fasta=None):
    """Run MMseqs2 easy-cluster. Returns path to cluster.tsv."""
    fasta_path = fasta or COMBINED_FASTA
    out_prefix = CLUSTER_DIR / output_prefix
    tmp = Path(f"/tmp/mmseqs_igr_{output_prefix}")
    
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_path), str(out_prefix), str(tmp),
        "--min-seq-id", str(id_thresh), "-c", str(cov_thresh),
        "--cov-mode", "2", "-k", "5", "--mask", "0", "--cluster-mode", "1",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    cluster_file = CLUSTER_DIR / f"{output_prefix}_cluster.tsv"
    return cluster_file if cluster_file.exists() else None


# ═══════════════════════════════════════════════════════════════════
#  SHUFFLED CONTROL
# ═══════════════════════════════════════════════════════════════════

def dinuc_shuffle(seq):
    """Altschul-Erickson dinucleotide-preserving shuffle."""
    s = list(seq.upper()); n = len(s)
    if n < 3: random.shuffle(s); return "".join(s)
    edges = [(s[i], s[i+1]) for i in range(n-1)]
    used = [False] * len(edges)
    result = [s[0]]; current = s[0]
    for _ in range(1, n):
        candidates = [(i, e) for i, e in enumerate(edges) if not used[i] and e[0] == current]
        if candidates:
            i, (_, b) = random.choice(candidates)
            used[i] = True; result.append(b); current = b
        else:
            avail = [(i, e) for i, e in enumerate(edges) if not used[i]]
            if avail:
                i, (_, b) = random.choice(avail)
                used[i] = True; result.append(b); current = b
            else:
                result.append(random.choice("ACGT"))
    return "".join(result)


def create_shuffled_fasta(input_fasta, output_fasta, seed=42):
    """Create dinucleotide-shuffled version of a FASTA file."""
    random.seed(seed)
    with open(input_fasta) as fin, open(output_fasta, "w") as fout:
        header = ""; seq = ""; count = 0
        for line in fin:
            if line.startswith(">"):
                if header and seq:
                    shuf = dinuc_shuffle(seq)
                    fout.write(f"{header}\n")
                    for i in range(0, len(shuf), 80):
                        fout.write(shuf[i:i+80] + "\n")
                    count += 1
                header = line.strip(); seq = ""
            else:
                seq += line.strip()
        if header and seq:
            shuf = dinuc_shuffle(seq)
            fout.write(f"{header}\n")
            for i in range(0, len(shuf), 80):
                fout.write(shuf[i:i+80] + "\n")
            count += 1
    return count


# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_pairs(pairs_1_1, valid, d_igrs, t_igrs, gene_map, output_path):
    """Export annotated 1+1 pairs to TSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "d39v_id", "tigr4_id",
        "d39v_chrom", "d39v_start", "d39v_end", "d39v_length", "d39v_orient",
        "d39v_left_gene", "d39v_right_gene",
        "tigr4_chrom", "tigr4_start", "tigr4_end", "tigr4_length", "tigr4_orient",
        "tigr4_left_gene", "tigr4_right_gene",
        "same_orientation", "both_genes_orthologs", "len_ratio",
    ]
    out = OUT_DIR / output_path
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for d_id, t_id in pairs_1_1:
            di = d_igrs.get(d_id, {}); ti = t_igrs.get(t_id, {})
            if not di or not ti: continue
            do = di.get("orientation_type","").replace("(++)","").replace("(--)","")
            to = ti.get("orientation_type","").replace("(++)","").replace("(--)","")
            dl = di.get("left_cds","").strip(); dr = di.get("right_cds","").strip()
            tl = ti.get("left_cds","").strip(); tr = ti.get("right_cds","").strip()
            lok = gene_map.get(dl,"") == tl; rok = gene_map.get(dr,"") == tr
            dlen = int(di.get("length",1)); tlen = int(ti.get("length",1))
            ratio = dlen / tlen if tlen > 0 else 0
            w.writerow({
                "d39v_id": d_id, "tigr4_id": t_id,
                "d39v_chrom": di.get("chrom",""), "d39v_start": di.get("start",""),
                "d39v_end": di.get("end",""), "d39v_length": di.get("length",""),
                "d39v_orient": di.get("orientation_type",""),
                "d39v_left_gene": f"{dl}({di.get('left_strand','')})",
                "d39v_right_gene": f"{dr}({di.get('right_strand','')})",
                "tigr4_chrom": ti.get("chrom",""), "tigr4_start": ti.get("start",""),
                "tigr4_end": ti.get("end",""), "tigr4_length": ti.get("length",""),
                "tigr4_orient": ti.get("orientation_type",""),
                "tigr4_left_gene": f"{tl}({ti.get('left_strand','')})",
                "tigr4_right_gene": f"{tr}({ti.get('right_strand','')})",
                "same_orientation": do == to,
                "both_genes_orthologs": lok and rok,
                "len_ratio": f"{ratio:.2f}",
            })
    return out


# ═══════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════

def print_summary(stats, valid):
    """Print clustering summary table."""
    print(f"\n{'='*65}")
    print(f"  CLUSTER CLASSIFICATION")
    print(f"{'='*65}")
    total_c = sum(stats.values())
    print(f"  Total clusters:            {total_c}")
    print(f"  Orthologs 1+1:             {stats.get('ortholog_1+1',0)}")
    print(f"  Multi 1+N:                 {stats.get('multi_1+N',0)}")
    print(f"  Multi N+1:                 {stats.get('multi_N+1',0)}")
    print(f"  Multi N+N:                 {stats.get('multi_N+N',0)}")
    print(f"  D39V-only multi:           {stats.get('d39v_only_multi',0)}")
    print(f"  TIGR4-only multi:          {stats.get('tigr4_only_multi',0)}")
    print(f"  Singletons:                {stats.get('singleton',0)} ({stats.get('singleton',0)/total_c*100:.0f}%)")
    
    if valid:
        print(f"\n{'='*65}")
        print(f"  1+1 ORTHOLOG VALIDATION")
        print(f"{'='*65}")
        print(f"  Total pairs:               {valid['total_1+1']}")
        print(f"  Same orientation:          {valid['same_orientation']} ({valid['pct_same_orient']:.1f}%)")
        print(f"  Both genes orthologs:      {valid['both_genes_orthologs']} ({valid['pct_both_genes']:.1f}%)")
        print(f"  >=1 gene ortholog:         {valid['any_gene_ortholog']} ({valid['pct_any_gene']:.1f}%)")
        print(f"  Same length (0.9-1.1):     {valid['same_length']} ({valid['pct_same_len']:.0f}%)")
        print(f"  TSS in both strains:       {valid['tss_both']} ({valid['tss_both']/valid['total_1+1']*100:.1f}%)")
        if valid['pct_nucmer'] is not None:
            print(f"  Nucmer overlap:            {valid['nucmer_overlap']} ({valid['pct_nucmer']:.1f}%)")


def print_sensitivity(results):
    """Print sensitivity analysis table."""
    print(f"\n{'='*80}")
    print(f"  SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    print(f"  {'id':>5} {'cov':>5} {'clusters':>10} {'singletons':>12} {'1+1':>6} {'both':>6} {'D-only':>7} {'T-only':>7} {'multi':>6} {'max':>5}")
    print(f"  {'-'*70}")
    for idv, cov, nc, sing, d1t1, both, d_only, t_only, multi, maxsz in results:
        pct = sing/nc*100
        print(f"  {idv:>5} {cov:>5} {nc:>10} {sing:>5} ({pct:4.0f}%) {d1t1:>6} {both:>6} {d_only:>7} {t_only:>7} {multi:>6} {maxsz:>5}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def make_label(id_val, cov_val):
    """Create file label from id/cov: 0.95, 0.70 -> id0_95_cov0_70"""
    return f"id{id_val:.2f}".replace('.','_') + f"_cov{cov_val:.2f}".replace('.','_')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    p = argparse.ArgumentParser(description="IGR Clustering Analysis — D39V vs TIGR4",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("mode", choices=["cluster", "analyze", "sensitive", "shuffled", "tss", "align", "export"],
                   nargs="?", default="analyze")
    p.add_argument("--cluster-file", default=None, help="Path to cluster.tsv")
    p.add_argument("--id", type=float, default=0.95, help="Min identity (default: 0.95)")
    p.add_argument("--cov", type=float, default=0.70, help="Min coverage (default: 0.70)")
    p.add_argument("--out", default="orthologs_1+1.tsv", help="Output filename for export")
    args = p.parse_args()
    
    # ── cluster mode ──
    if args.mode == "cluster":
        COMBINED_FASTA.parent.mkdir(parents=True, exist_ok=True)
        if not COMBINED_FASTA.exists():
            print("Merging FASTA...")
            with open(COMBINED_FASTA, "w") as fout:
                for fa in [D39V_IGR_FASTA, TIGR4_IGR_FASTA]:
                    with open(fa) as fin:
                        fout.write(fin.read())
        label = make_label(args.id, args.cov)
        out_f = run_easy_cluster(args.id, args.cov, f"All_{label}")
        print(f"Done → {out_f}")
        return
    
    # ── load clustering ──
    if args.cluster_file:
        cf = Path(args.cluster_file)
    else:
        label = make_label(args.id, args.cov)
        cf = CLUSTER_DIR / f"All_{label}_cluster.tsv"
    
    if not cf.exists():
        print(f"ERROR: {cf} not found. Run 'cluster' mode first.")
        sys.exit(1)
    
    clusters = load_clusters(cf)
    stats, pairs_1_1, multi = classify_clusters(clusters)
    
    # ── analyze mode (default) ──
    if args.mode == "analyze":
        d_igrs, t_igrs = load_igr_metadata()
        gene_map = load_gene_map()
        d_tss, t_tss = load_tss_positions()
        nucmer = load_nucmer_blocks() if NUCMER_COORDS.exists() else []
        valid = validate_orthologs(pairs_1_1, d_igrs, t_igrs, gene_map, d_tss, t_tss, nucmer)
        print_summary(stats, valid)
        
        # Also run TSS inventory
        inv = compute_tss_inventory(clusters, d_igrs, t_igrs, d_tss, t_tss)
        print(f"\n{'='*65}")
        print(f"  TSS INVENTORY")
        print(f"{'='*65}")
        for strain, cats in inv.items():
            total = sum(cats.values())
            print(f"\n  {strain} — {total} TSS")
            for ctype in ["ortholog_1+1", "multi_hit", "singleton", "cepa_especifico_multi", "in_CDS"]:
                n = cats.get(ctype, 0)
                if n > 0:
                    print(f"    {ctype:<25} {n:>4} ({n/total*100:5.1f}%)")
        
        # Export
        out = export_pairs(pairs_1_1, valid, d_igrs, t_igrs, gene_map, "orthologs_1+1.tsv")
        print(f"\n  Exported {len(pairs_1_1)} pairs → {out}")
        return
    
    # ── sensitive mode ──
    if args.mode == "sensitive":
        COMBINED_FASTA.parent.mkdir(parents=True, exist_ok=True)
        if not COMBINED_FASTA.exists():
            with open(COMBINED_FASTA, "w") as fout:
                for fa in [D39V_IGR_FASTA, TIGR4_IGR_FASTA]:
                    with open(fa) as fin: fout.write(fin.read())
        
        results = []
        for idv in [0.70, 0.95]:
            for cov in [0.70, 0.80, 0.90, 0.95]:
                label = make_label(idv, cov)
                cf = CLUSTER_DIR / f"All_{label}_cluster.tsv"
                if not cf.exists():
                    run_easy_cluster(idv, cov, f"All_{label}")
                    cf = CLUSTER_DIR / f"All_{label}_cluster.tsv"
                if cf.exists():
                    cl = load_clusters(cf)
                    s, _, _ = classify_clusters(cl)
                    nc = sum(s.values()); sing = s.get("singleton", 0)
                    d1t1 = s.get("ortholog_1+1", 0)
                    both = d1t1 + s.get("multi_1+N", 0) + s.get("multi_N+1", 0) + s.get("multi_N+N", 0)
                    d_only = s.get("d39v_only_multi", 0)
                    t_only = s.get("tigr4_only_multi", 0)
                    multi_count = s.get("multi_1+N",0)+s.get("multi_N+1",0)+s.get("multi_N+N",0)
                    maxsz = max(len(m) for m in cl.values())
                    results.append((f"{idv:.0%}", f"{cov:.0%}", nc, sing, d1t1, both, d_only, t_only, multi_count, maxsz))
        print_sensitivity(results)
        return
    
    # ── shuffled mode ──
    if args.mode == "shuffled":
        shuf_fa = OUT_IGR / "combined" / "All_IGRs_shuffled.fasta"
        if not shuf_fa.exists():
            print("Creating shuffled FASTA...")
            create_shuffled_fasta(COMBINED_FASTA, shuf_fa)
        label = "shuffled"
        cf_shuf = CLUSTER_DIR / f"All_{label}_cluster.tsv"
        if not cf_shuf.exists():
            run_easy_cluster(0.95, 0.70, f"All_{label}", fasta=shuf_fa)
            cf_shuf = CLUSTER_DIR / f"All_{label}_cluster.tsv"
        cl_shuf = load_clusters(cf_shuf)
        s_shuf, p_shuf, _ = classify_clusters(cl_shuf)
        fdr = s_shuf.get("ortholog_1+1", 0) / stats.get("ortholog_1+1", 1) * 100
        print(f"\n  SHUFFLED CONTROL (FDR)")
        print(f"  Real 1+1:   {stats.get('ortholog_1+1', 0)}")
        print(f"  Shuffled:   {s_shuf.get('ortholog_1+1', 0)}")
        print(f"  FDR:        {fdr:.1f}%")
        return
    
    # ── tss mode ──
    if args.mode == "tss":
        d_igrs, t_igrs = load_igr_metadata()
        d_tss, t_tss = load_tss_positions()
        inv = compute_tss_inventory(clusters, d_igrs, t_igrs, d_tss, t_tss)
        for strain, cats in inv.items():
            total = sum(cats.values())
            print(f"\n  {strain} — {total} TSS")
            for ctype in ["ortholog_1+1", "multi_hit", "singleton", "cepa_especifico_multi", "in_CDS"]:
                n = cats.get(ctype, 0)
                if n > 0:
                    print(f"    {ctype:<25} {n:>4} ({n/total*100:5.1f}%)")
        return
    
    # ── align mode (MMseqs2-native) ──
    if args.mode == "align":
        d_igrs, t_igrs = load_igr_metadata()
        d_seqs = {k: v["sequence"].strip() for k, v in d_igrs.items()}
        t_seqs = {k: v["sequence"].strip() for k, v in t_igrs.items()}
        
        # Write 1+1 pairs as query+target FASTA for MMseqs2
        q_fa = Path("/tmp/igr_query.fasta"); t_fa = Path("/tmp/igr_target.fasta")
        with open(q_fa, "w") as fq, open(t_fa, "w") as ft:
            for d_id, t_id in pairs_1_1:
                d_seq = d_seqs.get(d_id, ""); t_seq = t_seqs.get(t_id, "")
                if d_seq and t_seq:
                    fq.write(f">{d_id}\n{d_seq}\n"); ft.write(f">{t_id}\n{t_seq}\n")
        
        # Run MMseqs2 search for exact pairwise alignments
        aln_out = Path("/tmp/igr_mmseqs_alignment")
        mmseqs_bin = ROOT / ".pixi" / "envs" / "default" / "bin" / "mmseqs"
        subprocess.run([
            str(mmseqs_bin), "easy-search", str(q_fa), str(t_fa), str(aln_out), "/tmp/igr_mmseqs_tmp",
            "--search-type", "3", "--min-seq-id", "0.0", "-c", "0.0", "--cov-mode", "0",
            "-k", "5", "--mask", "0", "-s", "7.5",
        ], check=True, capture_output=True)
        
        # Parse MMseqs2 M8 output
        cols = ["query", "target", "pident", "alnlen", "mismatch", "gapopen",
                "qstart", "qend", "tstart", "tend", "evalue", "bits"]
        all_results = []
        with open(aln_out) as f:
            for line in f:
                parts = line.strip().split("\t")
                row = dict(zip(cols, parts))
                row["pident"] = float(row["pident"])
                row["alnlen"] = int(row["alnlen"])
                row["bits"] = float(row["bits"])
                
                # Get best hit per query
                d_id = row["query"]; t_id = row["target"]
                di = d_igrs.get(d_id, {}); ti = t_igrs.get(t_id, {})
                all_results.append({
                    "cluster_type": "ortholog_1+1", "d39v_id": d_id, "tigr4_id": t_id,
                    "d39v_len": len(d_seqs.get(d_id, "")), "tigr4_len": len(t_seqs.get(t_id, "")),
                    "aln_len": row["alnlen"], "identity_pct": round(row["pident"]*100, 1),
                    "mismatches": int(row["mismatch"]), "gaps": int(row["gapopen"]),
                    "qstart": int(row["qstart"]), "qend": int(row["qend"]),
                    "tstart": int(row["tstart"]), "tend": int(row["tend"]),
                    "evalue": row["evalue"], "bitscore": f"{row['bits']:.0f}",
                    "d39v_chrom": di.get("chrom",""), "d39v_start": di.get("start",""),
                    "d39v_end": di.get("end",""), "d39v_orient": di.get("orientation_type",""),
                    "tigr4_chrom": ti.get("chrom",""), "tigr4_start": ti.get("start",""),
                    "tigr4_end": ti.get("end",""), "tigr4_orient": ti.get("orientation_type",""),
                })
        
        # Export
        fields = list(all_results[0].keys())
        out = OUT_DIR / "all_alignments.tsv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
            w.writeheader(); w.writerows(all_results)
        
        # Summary
        ids = [r["identity_pct"] for r in all_results]
        print(f"\n  === MMseqs2 Alignment Summary ===")
        print(f"  Pairs aligned:     {len(all_results)}")
        print(f"  Mean identity:     {statistics.mean(ids):.1f}%")
        print(f"  Median:            {statistics.median(ids):.1f}%")
        for lo, hi, label in [(100,100,"100%"), (95,99,"95-99%"), (90,94,"90-94%"),
                                (80,89,"80-89%"), (70,79,"70-79%"), (0,69,"<70%")]:
            n = sum(1 for i in ids if lo <= i <= hi)
            if n > 0: print(f"    {label:<10} {n:>5} ({n/len(ids)*100:4.1f}%)")
        print(f"\n  Exported → {out}")
        
        # Cleanup
        for f in [q_fa, t_fa, aln_out]:
            f.unlink(missing_ok=True)
        return
        d_igrs, t_igrs = load_igr_metadata()
        gene_map = load_gene_map()
        out = export_pairs(pairs_1_1, {}, d_igrs, t_igrs, gene_map, args.out)
        print(f"Exported {len(pairs_1_1)} pairs → {out}")
        return


if __name__ == "__main__":
    main()
