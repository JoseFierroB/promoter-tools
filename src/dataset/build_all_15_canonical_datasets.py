#!/usr/bin/env python3
"""
================================================================================
DATASET BUILDER: All 15 Canonical Benchmark Combinations + 200k Scaling DB
================================================================================
Author: José Fierro Bustos & Víctor Rodríguez Bouza
Repository: promoter-tools
================================================================================
"""

import sys
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

OUT_DIR = ROOT_DIR / "data/benchmark/canonical_15_datasets"
SCALE_DIR = ROOT_DIR / "data/benchmark/scale_db"

for d in [OUT_DIR, SCALE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def write_fasta(records, path):
    SeqIO.write(records, path, "fasta")
    print(f"  [SAVED] {path.name:40s}: {len(records)} sequences")

def main():
    print("=" * 80)
    print("BUILDING ALL 15 CANONICAL BENCHMARK DATASET COMBINATIONS")
    print("=" * 80)

    # Base Files
    d39v_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/positives_81bp.fasta", "fasta"))
    d39v_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark/d39v/negatives_81bp.fasta", "fasta"))
    
    tigr4_pos_high = list(SeqIO.parse(ROOT_DIR / "data/tigr4/positives_high_81bp.fasta", "fasta"))
    tigr4_neg_high = list(SeqIO.parse(ROOT_DIR / "data/tigr4/negatives_high_81bp.fasta", "fasta"))
    
    tigr4_pos_all = list(SeqIO.parse(ROOT_DIR / "data/tigr4/positives_all_81bp.fasta", "fasta")) if (ROOT_DIR / "data/tigr4/positives_all_81bp.fasta").exists() else list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/positives_81bp.fasta", "fasta"))
    tigr4_neg_all = list(SeqIO.parse(ROOT_DIR / "data/tigr4/negatives_all_81bp.fasta", "fasta")) if (ROOT_DIR / "data/tigr4/negatives_all_81bp.fasta").exists() else list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/negatives_81bp.fasta", "fasta"))

    d39v_igr_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/d39v/positives_81bp_igr.fasta", "fasta"))
    d39v_igr_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/d39v/negatives_81bp_igr.fasta", "fasta"))

    tigr4_igr_high_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_1_high_conf_primary/positives_81bp.fasta", "fasta"))
    tigr4_igr_high_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_1_high_conf_primary/negatives_81bp.fasta", "fasta"))

    tigr4_igr_all_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/positives_81bp.fasta", "fasta"))
    tigr4_igr_all_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark_igr/tigr4/subset_4_all_comprehensive/negatives_81bp.fasta", "fasta"))

    # D39V CDS (TSSs not in IGRs)
    igr_pos_ids = {r.id for r in d39v_igr_pos}
    d39v_cds_pos = [r for r in d39v_pos if r.id not in igr_pos_ids]
    d39v_cds_neg = d39v_neg[:len(d39v_cds_pos)]

    # TIGR4 CDS High and All
    tigr4_cds_high_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_high_confidence/positives_81bp.fasta", "fasta"))
    tigr4_cds_high_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_high_confidence/negatives_81bp.fasta", "fasta"))

    tigr4_cds_all_pos = list(SeqIO.parse(ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_all/positives_81bp.fasta", "fasta"))
    tigr4_cds_all_neg = list(SeqIO.parse(ROOT_DIR / "data/benchmark/specialized_niches/cds_internal_all/negatives_81bp.fasta", "fasta"))

    # 1. d39v
    d1_dir = OUT_DIR / "1_d39v"
    d1_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_pos, d1_dir / "positives_81bp.fasta")
    write_fasta(d39v_neg, d1_dir / "negatives_81bp.fasta")

    # 2. tigr4 high
    d2_dir = OUT_DIR / "2_tigr4_high"
    d2_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tigr4_pos_high, d2_dir / "positives_81bp.fasta")
    write_fasta(tigr4_neg_high, d2_dir / "negatives_81bp.fasta")

    # 3. tigr4 all
    d3_dir = OUT_DIR / "3_tigr4_all"
    d3_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tigr4_pos_all, d3_dir / "positives_81bp.fasta")
    write_fasta(tigr4_neg_all, d3_dir / "negatives_81bp.fasta")

    # 4. d39v - tigr4 high
    d4_dir = OUT_DIR / "4_d39v_tigr4_high"
    d4_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_pos + tigr4_pos_high, d4_dir / "positives_81bp.fasta")
    write_fasta(d39v_neg + tigr4_neg_high, d4_dir / "negatives_81bp.fasta")

    # 5. d39v - tigr4 all
    d5_dir = OUT_DIR / "5_d39v_tigr4_all"
    d5_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_pos + tigr4_pos_all, d5_dir / "positives_81bp.fasta")
    write_fasta(d39v_neg + tigr4_neg_all, d5_dir / "negatives_81bp.fasta")

    # 6. d39v IGR
    d6_dir = OUT_DIR / "6_d39v_igr"
    d6_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_igr_pos, d6_dir / "positives_81bp.fasta")
    write_fasta(d39v_igr_neg, d6_dir / "negatives_81bp.fasta")

    # 7. tigr4 igr high
    d7_dir = OUT_DIR / "7_tigr4_igr_high"
    d7_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tigr4_igr_high_pos, d7_dir / "positives_81bp.fasta")
    write_fasta(tigr4_igr_high_neg, d7_dir / "negatives_81bp.fasta")

    # 8. tigr4 igr all
    d8_dir = OUT_DIR / "8_tigr4_igr_all"
    d8_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tigr4_igr_all_pos, d8_dir / "positives_81bp.fasta")
    write_fasta(tigr4_igr_all_neg, d8_dir / "negatives_81bp.fasta")

    # 9. d39v igr - tigr4 igr high
    d9_dir = OUT_DIR / "9_d39v_igr_tigr4_igr_high"
    d9_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_igr_pos + tigr4_igr_high_pos, d9_dir / "positives_81bp.fasta")
    write_fasta(d39v_igr_neg + tigr4_igr_high_neg, d9_dir / "negatives_81bp.fasta")

    # 10. d39v igr - tigr4 igr all
    d10_dir = OUT_DIR / "10_d39v_igr_tigr4_igr_all"
    d10_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_igr_pos + tigr4_igr_all_pos, d10_dir / "positives_81bp.fasta")
    write_fasta(d39v_igr_neg + tigr4_igr_all_neg, d10_dir / "negatives_81bp.fasta")

    # 11. d39v cds
    d11_dir = OUT_DIR / "11_d39v_cds"
    d11_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_cds_pos, d11_dir / "positives_81bp.fasta")
    write_fasta(d39v_cds_neg, d11_dir / "negatives_81bp.fasta")

    # 12. tigr4 cds high
    d12_dir = OUT_DIR / "12_tigr4_cds_high"
    d12_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tigr4_cds_high_pos, d12_dir / "positives_81bp.fasta")
    write_fasta(tigr4_cds_high_neg, d12_dir / "negatives_81bp.fasta")

    # 13. tigr4 cds all
    d13_dir = OUT_DIR / "13_tigr4_cds_all"
    d13_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(tigr4_cds_all_pos, d13_dir / "positives_81bp.fasta")
    write_fasta(tigr4_cds_all_neg, d13_dir / "negatives_81bp.fasta")

    # 14. d39v cds - tigr4 cds high
    d14_dir = OUT_DIR / "14_d39v_cds_tigr4_cds_high"
    d14_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_cds_pos + tigr4_cds_high_pos, d14_dir / "positives_81bp.fasta")
    write_fasta(d39v_cds_neg + tigr4_cds_high_neg, d14_dir / "negatives_81bp.fasta")

    # 15. d39v cds - tigr4 cds all
    d15_dir = OUT_DIR / "15_d39v_cds_tigr4_cds_all"
    d15_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(d39v_cds_pos + tigr4_cds_all_pos, d15_dir / "positives_81bp.fasta")
    write_fasta(d39v_cds_neg + tigr4_cds_all_neg, d15_dir / "negatives_81bp.fasta")

    # 16. Scale DB up to 200,000 sequences
    print("\n" + "=" * 80)
    print("BUILDING MASSIVE SCALING DATASETS (UP TO 200,000 SEQUENCES)")
    print("=" * 80)
    
    all_pos_pool = d39v_pos + tigr4_pos_all
    all_neg_pool = d39v_neg + tigr4_neg_all
    
    for n_target in [10000, 30000, 60000, 100000, 200000]:
        n_half = n_target // 2
        pos_reps = (all_pos_pool * (n_half // len(all_pos_pool) + 1))[:n_half]
        neg_reps = (all_neg_pool * (n_half // len(all_neg_pool) + 1))[:n_half]
        
        pos_final = [SeqRecord(r.seq, id=f"POS_SCALE_{i}", description="") for i, r in enumerate(pos_reps)]
        neg_final = [SeqRecord(r.seq, id=f"NEG_SCALE_{i}", description="") for i, r in enumerate(neg_reps)]
        
        scale_sub = SCALE_DIR / f"scale_{n_target//1000}k"
        scale_sub.mkdir(parents=True, exist_ok=True)
        write_fasta(pos_final, scale_sub / "positives_81bp.fasta")
        write_fasta(neg_final, scale_sub / "negatives_81bp.fasta")

    print("\n[DONE] All 15 canonical datasets and 200k scaling datasets built successfully!")

if __name__ == "__main__":
    main()
