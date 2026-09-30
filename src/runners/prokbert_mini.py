#!/usr/bin/env python3
"""
================================================================================
BENCHMARK RUNNER: ProkBERT-mini-promoter (NeuralBioInfo / Ligeti et al., 2023)
================================================================================
"""

import argparse
import faulthandler
import os
import signal
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

# Single-process inner tokenization: ProkBERT's batch_encode_plus spawns
# its own multiprocessing.Pool(cpu_cores_for_tokenization) per call
# (sequtils.batch_tokenize_segments_with_ids). The library default is all
# cores, which fork-bombs from our prefetch workers and exhausts fds at
# scale (EMFILE ~450 chunks of 128). Parallelism comes from OUR outer
# pool; the inner one stays serial.
TOK_COMP_PARAMS = {'cpu_cores_for_tokenization': 1}


def _worker_tokenizer():
    """Per-process tokenizer for prefetch workers (cf. ipromp_sp12 pattern)."""
    if "tok" not in _TOK_CACHE:
        _TOK_CACHE["tok"] = ProkBERTTokenizer(
            tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence',
            comp_params=dict(TOK_COMP_PARAMS))
    return _TOK_CACHE["tok"]


def _tokenize_chunk(chunk):
    """Same batch_encode_plus call as the serial path, executed ahead.

    Returns NUMPY arrays, not torch tensors: torch's multiprocessing
    pickling shares tensor storage via file descriptors (fd-passing),
    which exhausts fds at scale (EMFILE ~450 chunks of 128 with the
    default 1024 limit). Numpy pickles inline. forward_batches()
    converts back with zero-copy torch.as_tensor.
    """
    encoded = _worker_tokenizer().batch_encode_plus(chunk, return_tensors="pt")
    return (encoded["input_ids"].numpy(), encoded["attention_mask"].numpy())


def parse_args():
    parser = argparse.ArgumentParser(description="Run ProkBERT-mini-promoter on positive and negative FASTA sequences.")
    parser.add_argument("--pos", type=Path, default=None, help="Path to positives FASTA file (optional).")
    parser.add_argument("--neg", type=Path, default=None, help="Path to negatives FASTA file (optional).")
    parser.add_argument("-o", "--output", default="output/predictions", help="Output path (file or dir).")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64).")
    parser.add_argument("--prefetch-workers", type=int, default=0,
                        help="Tokenize batches ahead in N worker processes (0 = serial, default).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu; default: cuda if available).")
    return parser.parse_args()

def tokenize_texts(seq_texts, batch_size, prefetch_workers=0):
    """Tokenize all chunks up front, in order. Must run BEFORE any forward
    pass: forking worker processes after CPU-threaded MKL work started can
    crash them (BrokenProcessPool). Multiprocess tokenization only when
    prefetch_workers > 0, else same serial calls as the reference path."""
    chunks = [seq_texts[i:i + batch_size]
              for i in range(0, len(seq_texts), batch_size)]
    print(f"[tokenize] {len(seq_texts)} seqs in {len(chunks)} chunks "
          f"(workers={prefetch_workers})", flush=True)
    if prefetch_workers and prefetch_workers > 0:
        from concurrent.futures import ProcessPoolExecutor
        # Bounded waves (64 chunks): a single 782-future map() deadlocks
        # the pool at scale (feeder blocks on a full call queue once a
        # worker dies); waves keep the queue shallow so failures surface
        # as BrokenProcessPool instead of hanging shutdown.
        out = []
        with ProcessPoolExecutor(max_workers=prefetch_workers) as ex:
            for w in range(0, len(chunks), 64):
                out.extend(list(ex.map(_tokenize_chunk, chunks[w:w + 64])))
        print(f"[tokenize] done ({len(out)} batches)", flush=True)
        return out
    return [_tokenize_chunk(c) for c in chunks]


def forward_batches(model, device, encoded_list, pin_memory):
    """Forward pass over pre-tokenized (input_ids, attention_mask) batches,
    preserving order."""
    use_cuda = str(device).startswith("cuda")
    if os.environ.get("PROMOTER_TOOLS_PROKBERT_PIN", "1") == "0":
        pin_memory = False
    preds = []
    total = len(encoded_list)
    with torch.no_grad():
        for i, (input_ids, attention_mask) in enumerate(encoded_list):
            # Zero-copy back to tensor (same process; see _tokenize_chunk).
            input_ids = torch.as_tensor(input_ids)
            attention_mask = torch.as_tensor(attention_mask)
            if use_cuda and pin_memory:
                input_ids = input_ids.pin_memory().to(device, non_blocking=True)
                attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
            else:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            promoter_probs = probs[:, 1].cpu().numpy()
            preds.extend(promoter_probs)
            if (i + 1) % 200 == 0 or (i + 1) == total:
                print(f"[forward] {i + 1}/{total} batches", flush=True)
    return preds


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
    # Hang diagnosis: `kill -USR1 <pid>` dumps a traceback without killing.
    try:
        faulthandler.register(signal.SIGUSR1)
    except Exception:
        pass
    if args.pos is None and args.neg is None:
        raise SystemExit("error: at least one of --pos / --neg is required")
    out_path = Path(args.output)

    t0 = time.time()

    if args.prefetch_workers and args.prefetch_workers > 0:
        # Tokenize EVERYTHING up front, BEFORE the model touches CUDA:
        # forking worker processes with a live CUDA context is undefined
        # behaviour (hangs at scale), same family as fork-after-MKL.
        # Order preserved throughout; pools close before any forward pass.
        pos_recs = list(SeqIO.parse(args.pos, "fasta")) if args.pos else []
        neg_recs = list(SeqIO.parse(args.neg, "fasta")) if args.neg else []
        pos_ids = [r.id for r in pos_recs]
        neg_ids = [r.id for r in neg_recs]
        pos_texts = [str(r.seq).upper() for r in pos_recs]
        neg_texts = [str(r.seq).upper() for r in neg_recs]
        # NOTE: tokenize pos/neg separately: batches must not straddle the
        # pos/neg boundary. Both pools still close before any forward pass.
        t_tok = time.time()
        pos_encoded = tokenize_texts(pos_texts, args.batch_size,
                                     args.prefetch_workers)
        neg_encoded = tokenize_texts(neg_texts, args.batch_size,
                                     args.prefetch_workers)
        tok_s = time.time() - t_tok

    tokenizer = ProkBERTTokenizer(tokenization_params={'kmer': 6, 'shift': 1}, operation_space='sequence',
                                    comp_params=dict(TOK_COMP_PARAMS))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.to(args.device)
    model.eval()

    if args.prefetch_workers and args.prefetch_workers > 0:
        t_fwd = time.time()
        pos_preds = forward_batches(model, args.device, pos_encoded,
                                    pin_memory=True)
        neg_preds = forward_batches(model, args.device, neg_encoded,
                                    pin_memory=True)
        # Pure compute (LCNN/iProMP convention): tokenize + forward,
        # model loading excluded.
        elapsed = tok_s + (time.time() - t_fwd)
    else:
        pos_ids, pos_preds = predict_fasta(model, tokenizer, args.pos, args.device, args.batch_size) if args.pos else ([], [])
        neg_ids, neg_preds = predict_fasta(model, tokenizer, args.neg, args.device, args.batch_size) if args.neg else ([], [])
        elapsed = time.time() - t0
    
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

    # elapsed already set per branch (pure compute; serial includes parse).
    n_total = len(pos_ids) + len(neg_ids)
    print(f"ProkBERT: {n_total} seqs ({len(pos_ids)} Pos / {len(neg_ids)} Neg) in {elapsed:.2f}s [batch={args.batch_size} prefetch={args.prefetch_workers}]")

if __name__ == "__main__":
    main()
