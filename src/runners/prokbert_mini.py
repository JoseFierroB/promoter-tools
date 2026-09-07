#!/usr/bin/env python3
"""
================================================================================
CANONICAL RUNNER: ProkBERT-mini-promoter (NeuralBioInfo / Ligeti et al., 2023)
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "tools/prokbert/src"))

from transformers import AutoModelForSequenceClassification
from prokbert.prokbert_tokenizer import ProkBERTTokenizer

MODEL_NAME = "neuralbioinfo/prokbert-mini-promoter"

def parse_args():
    parser = argparse.ArgumentParser(description="Run ProkBERT-mini-promoter on positive and negative FASTA sequences.")
    parser.add_argument("--pos", type=Path, default=None, help="Path to positives FASTA file (optional).")
    parser.add_argument("--neg", type=Path, default=None, help="Path to negatives FASTA file (optional).")
    parser.add_argument("-o", "--output", default="output/predictions", help="Output path (file or dir).")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    return parser.parse_args()

def predict_fasta(model, tokenizer, fasta_path, device, batch_size=64):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return [], []
    seq_ids = [r.id for r in records]
    seq_texts = [str(r.seq).upper() for r in records]

    preds = []
    for i in range(0, len(seq_texts), batch_size):
        batch_seqs = seq_texts[i : i + batch_size]
        encoded = tokenizer.batch_encode_plus(batch_seqs, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            promoter_probs = probs[:, 1].cpu().numpy()
            preds.extend(promoter_probs)
    return seq_ids, preds

def main():
    args = parse_args()
    if args.pos is None and args.neg is None:
        raise SystemExit("error: at least one of --pos / --neg is required")
    out_path = Path(args.output)

    t0 = time.time()
    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    pos_ids, pos_preds = predict_fasta(model, tokenizer, args.pos, args.device, args.batch_size) if args.pos else ([], [])
    neg_ids, neg_preds = predict_fasta(model, tokenizer, args.neg, args.device, args.batch_size) if args.neg else ([], [])
    
    if out_path.suffix in (".tsv", ".csv"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        if pos_ids:
            frames.append(pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds}))
        if neg_ids:
            frames.append(pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds}))
        df_out = pd.concat(frames, ignore_index=True)
        df_out.to_csv(out_path, sep="\t" if out_path.suffix == ".tsv" else ",", index=False)
    else:
        out_dir = out_path / "prokbert"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        if pos_ids:
            pd.DataFrame({"ID": pos_ids, "PRED": pos_preds}).to_csv(out_dir / "prokbert_pos.csv", sep="\t", index=False)
            frames.append(pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds}))
        if neg_ids:
            pd.DataFrame({"ID": neg_ids, "PRED": neg_preds}).to_csv(out_dir / "prokbert_neg.csv", sep="\t", index=False)
            frames.append(pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds}))
        pd.concat(frames, ignore_index=True).to_csv(out_path / "prokbert.tsv", sep="\t", index=False)
        
    elapsed = time.time() - t0
    n_total = len(pos_ids) + len(neg_ids)
    print(f"ProkBERT: {n_total} seqs ({len(pos_ids)} Pos / {len(neg_ids)} Neg) in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
