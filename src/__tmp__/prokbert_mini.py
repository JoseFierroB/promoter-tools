#!/usr/bin/env python3
"""
================================================================================
BENCHMARK RUNNER: ProkBERT-mini-promoter (NeuralBioInfo / Ligeti et al., 2023)
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

_TOK_CACHE = {}


def _worker_tokenizer():
    """Per-process tokenizer for prefetch workers (cf. ipromp_sp12 pattern)."""
    if "tok" not in _TOK_CACHE:
        _TOK_CACHE["tok"] = ProkBERTTokenizer(
            tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence')
    return _TOK_CACHE["tok"]


def _tokenize_chunk(chunk):
    """Same batch_encode_plus call as the serial path, executed ahead."""
    encoded = _worker_tokenizer().batch_encode_plus(chunk, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProkBERT-mini-promoter on positive and negative FASTA sequences.")
    parser.add_argument("--pos", type=Path, default=None, help="Path to positives FASTA file (optional).")
    parser.add_argument("--neg", type=Path, default=None, help="Path to negatives FASTA file (optional).")
    parser.add_argument("-o", "--output", default="output/predictions", help="Output path (file or dir).")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64).")
    parser.add_argument("--prefetch-workers", type=int, default=0,
                        help="Tokenize batches ahead in N worker processes (0 = serial, default).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu).")
    return parser.parse_args()

def predict_fasta(model, tokenizer, fasta_path, device, batch_size=64, prefetch_workers=0):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return [], []
    seq_ids = [r.id for r in records]
    seq_texts = [str(r.seq).upper() for r in records]

    preds = []
    use_cuda = str(device).startswith("cuda")
    if prefetch_workers and prefetch_workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [seq_texts[i:i + batch_size]
                  for i in range(0, len(seq_texts), batch_size)]
        with ProcessPoolExecutor(max_workers=prefetch_workers) as ex:
            encoded_iter = ex.map(_tokenize_chunk, chunks)
            for input_ids, attention_mask in encoded_iter:
                if use_cuda:
                    input_ids = input_ids.pin_memory().to(device, non_blocking=True)
                    attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
                else:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    promoter_probs = probs[:, 1].cpu().numpy()
                    preds.extend(promoter_probs)
        return seq_ids, preds
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

    pos_ids, pos_preds = predict_fasta(model, tokenizer, args.pos, args.device, args.batch_size, args.prefetch_workers) if args.pos else ([], [])
    neg_ids, neg_preds = predict_fasta(model, tokenizer, args.neg, args.device, args.batch_size, args.prefetch_workers) if args.neg else ([], [])
    
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
    print(f"ProkBERT: {n_total} seqs ({len(pos_ids)} Pos / {len(neg_ids)} Neg) in {elapsed:.2f}s [batch={args.batch_size} prefetch={args.prefetch_workers}]")

if __name__ == "__main__":
    main()
