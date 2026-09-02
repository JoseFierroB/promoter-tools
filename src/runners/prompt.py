#!/usr/bin/env python3
"""
================================================================================
CANONICAL RUNNER: Prompt (Du et al., 2024 / Interdisciplinary Sciences)
================================================================================
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = ROOT_DIR / "tools/Prompt"

class MLP_(nn.Module):                    
    def __init__(self, num_classes=2):
        super(MLP_, self).__init__()      
        self.layer1 = nn.Linear(81, 16)
        self.Tanh = nn.Tanh()
        self.layer3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.Tanh(x)
        x = self.layer3(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description="Run Prompt MLP on positive and negative FASTA sequences.")
    parser.add_argument("--pos", type=Path, required=True, help="Path to positives FASTA file.")
    parser.add_argument("--neg", type=Path, required=True, help="Path to negatives FASTA file.")
    parser.add_argument("-o", "--output", default="output/predictions", help="Output path (file or dir).")
    parser.add_argument("--model-path", type=Path, default=PROMPT_DIR / "mlp/ckpt/168/model_500.pth", help="Model checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    return parser.parse_args()

DATA_DICT = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'M': 5, 'Y': 6, 'W': 7}

def predict_fasta(model, fasta_path, batch_size=128):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        return [], []
    seq_ids = [r.id for r in records]
    seq_texts = [str(r.seq).upper()[:81] for r in records]
    
    padded = []
    for s in seq_texts:
        if len(s) < 81:
            s = s + 'A' * (81 - len(s))
        padded.append([DATA_DICT.get(ch, 0) for ch in s])
        
    mat = np.array(padded, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(mat), batch_size):
            batch_t = torch.tensor(mat[i : i + batch_size], dtype=torch.float32)
            out = model(batch_t)
            probs = torch.softmax(out, dim=1)[:, 1].numpy()
            preds.extend(probs)
    return seq_ids, preds

def main():
    args = parse_args()
    out_path = Path(args.output)
    
    t0 = time.time()
    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.eval()
    
    pos_ids, pos_preds = predict_fasta(model, args.pos, args.batch_size)
    neg_ids, neg_preds = predict_fasta(model, args.neg, args.batch_size)
    
    if out_path.suffix in (".tsv", ".csv"):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_pos = pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds})
        df_neg = pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds})
        df_out = pd.concat([df_pos, df_neg], ignore_index=True)
        df_out.to_csv(out_path, sep="\t" if out_path.suffix == ".tsv" else ",", index=False)
    else:
        out_dir = out_path / "prompt"
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": pos_ids, "PRED": pos_preds}).to_csv(out_dir / "prompt_pos.csv", sep="\t", index=False)
        pd.DataFrame({"ID": neg_ids, "PRED": neg_preds}).to_csv(out_dir / "prompt_neg.csv", sep="\t", index=False)
        df_pos = pd.DataFrame({"ID": pos_ids, "LABEL": 1, "PRED": pos_preds})
        df_neg = pd.DataFrame({"ID": neg_ids, "LABEL": 0, "PRED": neg_preds})
        pd.concat([df_pos, df_neg], ignore_index=True).to_csv(out_path / "prompt.tsv", sep="\t", index=False)
        
    elapsed = time.time() - t0
    n_total = len(pos_ids) + len(neg_ids)
    print(f"Prompt: {n_total} seqs ({len(pos_ids)} Pos / {len(neg_ids)} Neg) in {elapsed:.3f}s")

if __name__ == "__main__":
    main()
