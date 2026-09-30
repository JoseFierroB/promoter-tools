#!/usr/bin/env python3
"""ProkBERT runner A/B experiment (fp32, no numeric changes).

Modes:
  baseline   - exact replica of src/runners/prokbert_mini.py predict_fasta:
               batch 64, serial tokenize -> forward, blocking transfers.
               Sanity gate: output must byte-match the canonical run TSV.
  prefetch64 - same 64-seq chunks, same batch_encode_plus call per chunk,
               tokenized ahead in worker processes (consumed in order),
               pin_memory + non_blocking transfers. Bit-identical by design.
  batch256   - same as prefetch64 with 256-seq chunks. Gate: byte-diff; if
               it fails, mode is discarded (stays out of the runner).
  prefetch128  - same as prefetch64 with 128-seq chunks (common batch with
               iPro-MP). Gate: tolerance envelope + identical AUC/OP.

All modes: same checkpoint, same tokenizer params (kmer=6, shift=1),
fp32, torch.no_grad, softmax[:, 1] as score, same TSV schema/column order.

Usage (WS, 5090):
  CUDA_VISIBLE_DEVICES=1 pixi run python experiments/prokbert_runner_ab/ab_run.py \
      --mode baseline --input-dir data/benchmark/scale_db/scale_10k \
      -o output/experiments/prokbert_ab/baseline_10k.tsv
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools" / "prokbert" / "src"))

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

MODEL_NAME = "neuralbioinfo/prokbert-mini-promoter"

_TOK_CACHE = {}


def _tokenizer():
    if "tok" not in _TOK_CACHE:
        from prokbert.prokbert_tokenizer import ProkBERTTokenizer
        _TOK_CACHE["tok"] = ProkBERTTokenizer(
            tokenization_params={"kmer": 6, "shift": 1},
            operation_space="sequence")
    return _TOK_CACHE["tok"]


def tokenize_chunk(chunk):
    """Same call as the runner's predict_fasta, executed in a worker."""
    t0 = time.perf_counter()
    encoded = _tokenizer().batch_encode_plus(chunk, return_tensors="pt")
    dt = time.perf_counter() - t0
    return encoded["input_ids"], encoded["attention_mask"], dt


class GpuSampler:
    def __init__(self, smi_index):
        self.idx = smi_index
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-i", str(self.idx),
                     "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                self.samples.append(float(out.stdout.strip()))
            except Exception:
                pass
            self._stop.wait(0.3)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["baseline", "prefetch64", "prefetch128", "batch256"])
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = Path(args.input_dir)
    pos_fasta, neg_fasta = d / "positives_81bp.fasta", d / "negatives_81bp.fasta"
    seq_ids, labels, seq_texts = [], [], []
    for fasta, lab in ((pos_fasta, 1), (neg_fasta, 0)):
        for r in SeqIO.parse(fasta, "fasta"):
            seq_ids.append(r.id)
            labels.append(lab)
            seq_texts.append(str(r.seq).upper())

    batch_size = {"baseline": 64, "prefetch64": 64, "prefetch128": 128,
                  "batch256": 256}[args.mode]
    chunks = [seq_texts[i:i + batch_size]
              for i in range(0, len(seq_texts), batch_size)]

    # ---- load phase (same scope as the runner's internal timer) ----
    t_load0 = time.perf_counter()
    tok = _tokenizer()
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    t_load = time.perf_counter() - t_load0

    # ---- predict phase ----
    smi_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    preds = []
    t_tok_sum = 0.0
    t0 = time.perf_counter()
    with GpuSampler(smi_idx) as sampler:
        if args.mode == "baseline":
            for chunk in chunks:
                encoded = tok.batch_encode_plus(chunk, return_tensors="pt")
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded["attention_mask"].to(args.device)
                with torch.no_grad():
                    probs = torch.softmax(
                        model(input_ids=input_ids,
                              attention_mask=attention_mask).logits, dim=-1)
                preds.extend(probs[:, 1].cpu().numpy())
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for input_ids, attention_mask, dt in ex.map(tokenize_chunk, chunks):
                    t_tok_sum += dt
                    input_ids = input_ids.pin_memory().to(args.device,
                                                          non_blocking=True)
                    attention_mask = attention_mask.pin_memory().to(
                        args.device, non_blocking=True)
                    with torch.no_grad():
                        probs = torch.softmax(
                            model(input_ids=input_ids,
                                  attention_mask=attention_mask).logits, dim=-1)
                    preds.extend(probs[:, 1].cpu().numpy())
        torch.cuda.synchronize() if args.device == "cuda" else None
    t_predict = time.perf_counter() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ID": seq_ids, "LABEL": labels, "PRED": preds})
    df.to_csv(out, sep="\t", index=False)

    u = sampler.samples
    n = len(seq_texts)
    print(f"[{args.mode}] n={n} batch={batch_size} workers="
          f"{args.workers if args.mode != 'baseline' else 0}")
    print(f"  load={t_load:.1f}s predict={t_predict:.1f}s "
          f"total={t_load + t_predict:.1f}s")
    if args.mode != "baseline":
        print(f"  tokenize(sum over chunks)={t_tok_sum:.1f}s "
              f"(overlapped, not additive)")
    print(f"  throughput={n / t_predict:.1f} seq/s (predict phase)")
    print(f"  gpu_util mean={np.mean(u):.1f}% max={np.max(u):.0f}% "
          f"({len(u)} samples)" if u else "  gpu_util: n/a")
    print(f"  output={out}")


if __name__ == "__main__":
    main()
