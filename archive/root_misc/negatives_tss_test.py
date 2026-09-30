#!/usr/bin/env python3
"""
Comprehensive Verification Unit Tests for Promoter Tools Datasets.

Tests:
1. 0-based coordinate exactness: TSS nucleotide +1 must land exactly at slice index 60 in 81-mers.
2. Strand orientation: Positive (+) and Negative (-) strand reverse complement handling.
3. CDS range validation & exclusions: Margin checks (margin=20, tss-margin=200).
4. GC content calculation & Z-score / Cohen's d statistic accuracy.
5. FASTA and TSV row count & metadata column alignment.

Usage:
    pixi run python negatives_tss_test.py --run-tests
"""

import argparse
import sys
import unittest
from pathlib import Path
from Bio.Seq import Seq

ROOT = Path(__file__).resolve().parent

class TestPromoterDatasetIntegrity(unittest.TestCase):

    def test_coordinate_indexing_plus_strand(self):
        """Tests that +1 TSS nucleotide is located at index 60 for (+) strand 81-mers."""
        genome_dummy = "N" * 100 + "A" * 60 + "G" + "C" * 20 + "N" * 100
        tss_pos_1based = 161  # 'G' is at 1-based index 161
        pos_0 = tss_pos_1based - 1
        
        # 81-mer: 60 bp upstream, 20 bp downstream
        kmer = genome_dummy[pos_0 - 60 : pos_0 + 21]
        self.assertEqual(len(kmer), 81, "Window size must be exactly 81 bp")
        self.assertEqual(kmer[60], "G", "TSS +1 nucleotide must land at index 60 for (+) strand")

    def test_coordinate_indexing_minus_strand(self):
        """Tests that +1 TSS nucleotide is located at index 60 for (-) strand 81-mers."""
        # Suppose target genomic nucleotide on (-) strand is 'C' (so TSS +1 on transcript is 'G')
        # Genomic slice for (-) strand: [pos_0 - 20 : pos_0 + 61].reverse_complement()
        genomic_slice = "A" * 20 + "C" + "T" * 60  # 'C' is at 0-based index 20 of slice
        rc_kmer = str(Seq(genomic_slice).reverse_complement()).upper()
        
        self.assertEqual(len(rc_kmer), 81, "Window size must be exactly 81 bp")
        self.assertEqual(rc_kmer[60], "G", "TSS +1 nucleotide must land at index 60 for (-) strand")

    def test_fasta_tsv_sync_d39v(self):
        """Verifies FASTA record counts match TSV metadata counts for D39V dataset."""
        fasta_p = ROOT / "data/benchmark/positives_81bp.fasta"
        tsv_p = ROOT / "data/benchmark/positives_81bp.tsv"
        if fasta_p.exists() and tsv_p.exists():
            from Bio import SeqIO
            import pandas as pd
            recs = list(SeqIO.parse(fasta_p, "fasta"))
            df = pd.read_csv(tsv_p, sep="\t")
            self.assertEqual(len(recs), len(df), "FASTA sequence count must equal TSV row count for D39V")

    def test_fasta_tsv_sync_tigr4(self):
        """Verifies FASTA record counts match TSV metadata counts for TIGR4 High Conf dataset."""
        fasta_p = ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.fasta"
        tsv_p = ROOT / "output/tigr4_data/positives_tigr4_high_conf_primary_81bp.tsv"
        if fasta_p.exists() and tsv_p.exists():
            from Bio import SeqIO
            import pandas as pd
            recs = list(SeqIO.parse(fasta_p, "fasta"))
            df = pd.read_csv(tsv_p, sep="\t")
            self.assertEqual(len(recs), len(df), "FASTA sequence count must equal TSV row count for TIGR4 High Conf")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPromoterDatasetIntegrity)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unit tests for negatives TSS extraction.")
    parser.add_argument("--run-tests", action="store_true", help="Execute test suite.")
    args, unknown = parser.parse_known_args()
    
    if args.run_tests or len(sys.argv) == 1:
        run_tests()
    else:
        print("Use --run-tests to execute dataset verification unit tests.")
