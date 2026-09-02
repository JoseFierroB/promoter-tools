#!/usr/bin/env python3
"""
================================================================================
CANONICAL RUNNER: EvoSNR-Prom (Wei et al., 2024 / PLOS Comp Biol)
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
EVOSNR_DIR = ROOT_DIR / "tools/EvoSNR-Prom"
sys.path.insert(0, str(EVOSNR_DIR))

try:
    import fasttext
except ImportError:
    fasttext = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run EvoSNR-Prom / Lexicon on positive and negative FASTA sequences.")
    parser.add_argument("--pos", type=Path, required=True, help="Path to positives FASTA file.")
    parser.add_argument("--neg", type=Path, required=True, help="Path to negatives FASTA file.")
    parser.add_argument("-o", "--output", default="output/predictions", help="Output path (file or dir).")
    parser.add_argument("--species", type=str, default="Esch", choices=["Esch", "Kleb", "Sino", "Agro"], help="Reference species lexicon.")
    parser.add_argument("--kmer-size", type=int, default=6, help="K-mer window size (default: 6).")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean", "median", "max"], help="Score aggregation.")
    return parser.parse_args()

def load_lexicon_and_model(species="Esch"):
    data_dir = EVOSNR_DIR / f"data/{species}"
    ft_path = data_dir / "fasttext_model/model.bin"
    lex_path = data_dir / "motifs/lexicon.txt"
    
    if not ft_path.exists():
        raise FileNotFoundError(f"FastText model not found at {ft_path}")
    
    ft_model = fasttext.load_model(str(ft_path))
    lexicon = set()
    if lex_path.exists():
        with open(lex_path, "r") as f:
            for line in f:
                w = line.strip().split()[0] if line.strip() else ""
                if w:
                    lexicon.add(w.upper())
    return ft_model, lexicon

def score_sequence(seq_str, ft_model, lexicon, k=6, agg="mean"):
    seq_str = seq_str.upper()
    L = len(seq_str)
    if L < k:
        return 0.0
    
    per_base_scores = []
    for i in range(L - k + 1):
        kmer = seq_str[i : i + k]
        vec = ft_model.get_word_vector(kmer)
        norm = np.linalg.norm(vec)
        is_lex = 1.5 if kmer in lexicon else 1.0
        score = float(norm * is_lex)
        per_base_scores.append(score)
        
    scores_arr = np.array(per_base_scores, dtype=np.float32)
    scaled_scores = 1.0 / (1.0 + np.exp(-(scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-6)))
    
    if agg == "mean":
        return float(np.mean(scaled_scores))
    elif agg == "median":
        return float(np.median(scaled_scores))
    elif agg == "max":
        return float(np.max(scaled_scores))
    return float(np.mean(scaled_scores))

def predict_fasta(fasta_path, ft_model, lexicon, k=6, agg="mean"):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return [], []
    seq_ids = [r.id for r in records]
    preds = [score_sequence(str(r.seq), ft_model, lexicon, k=k, agg=agg) for r in records]
    return seq_ids, preds

def main():
    args = parse_args()
    out_path = Path(args.output)
    
    t0 = time.time()
    ft_model, lexicon = load_lexicon_and_model(args.species)
    
    pos_ids, pos_preds = predict_fasta(args.pos, ft_model, lexicon, k=args.kmer_size, agg=args.aggregation)
    neg_ids, neg_preds = predict_fasta(args.neg, ft_model, lexicon, k=args.kmer_size, agg=args.aggregation)
    
    if out_path.suffix in (".tsv", ".csv"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_pos = pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds})
        df_neg = pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds})
        df_out = pd.concat([df_pos, df_neg], ignore_index=True)
        df_out.to_csv(out_path, sep="\t" if out_path.suffix == ".tsv" else ",", index=False)
    else:
        out_dir = out_path / "evosnr"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": pos_ids, "PRED": pos_preds}).to_csv(out_dir / "evosnr_pos.csv", sep="\t", index=False)
        pd.DataFrame({"ID": neg_ids, "PRED": neg_preds}).to_csv(out_dir / "evosnr_neg.csv", sep="\t", index=False)
        df_pos = pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds})
        df_neg = pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds})
        pd.concat([df_pos, df_neg], ignore_index=True).to_csv(out_path / "evosnr.tsv", sep="\t", index=False)
        
    elapsed = time.time() - t0
    n_total = len(pos_ids) + len(neg_ids)
    print(f"EvoSNR-Prom: {n_total} seqs ({len(pos_ids)} Pos / {len(neg_ids)} Neg) in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
