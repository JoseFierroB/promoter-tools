#!/usr/bin/env python3
"""iPro-MP sp12 (H. pylori) runner — DNABERT-6 inference."""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools"

_TOK_CACHE = {}


def _tokenize_seq(s: str, dnabert_dir: str) -> dict:
    """Tokenize one sequence into k-mers -> BERT input. Runs in a worker process."""
    if dnabert_dir not in _TOK_CACHE:
        from transformers import BertTokenizer
        _TOK_CACHE[dnabert_dir] = BertTokenizer.from_pretrained(dnabert_dir)
    tok = _TOK_CACHE[dnabert_dir]
    kmers = [s[i:i+6] for i in range(len(s) - 5)]
    inp = tok(kmers, is_split_into_words=True, padding="max_length",
              max_length=128, return_tensors="np", truncation=True)
    return {k: inp[k][0] for k in ["input_ids", "attention_mask"] if k in inp}


def main():
    p = argparse.ArgumentParser(description="iPro-MP sp12 (H. pylori)")
    p.add_argument("--pos", required=True, help="Positive test FASTA")
    p.add_argument("--neg", required=True, help="Negative test FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    p.add_argument("-m", "--model-dir", default="tools/iPro-MP/07-final",
                    help="Model directory")
    p.add_argument("-d", "--dnabert-dir", default="tools/iPro-MP/DNABERT-6",
                    help="DNABERT-6 directory")
    args = p.parse_args()

    out_base = Path(args.output)
    out_base.mkdir(parents=True, exist_ok=True)

    combined_path = out_base / "bench_combined.fasta"
    with open(combined_path, "w") as f_out:
        for fasta in [args.pos, args.neg]:
            with open(fasta) as f_in:
                for line in f_in:
                    f_out.write(line)

    sys.path.insert(0, str(TOOLS_DIR / "iPro-MP"))
    from importlib.util import spec_from_file_location
    spec = spec_from_file_location("ip", str(TOOLS_DIR / "iPro-MP/iPro-MP_predict.py"))
    ip = spec.loader.load_module()

    import torch

    seqs = []
    with open(combined_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seqs.append(line.upper())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for fold in range(1, 6):
        model = ip.DNABERTPromoterClassifier(dnabert_dir=args.dnabert_dir)
        state_dict = torch.load(f"{args.model_dir}/12_fold_{fold}.pth", map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        models.append(model)

    t0 = time.perf_counter()
    batch_size = 128
    n_workers = int(os.environ.get("OMP_NUM_THREADS", "1") or 1)
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            inputs = list(ex.map(partial(_tokenize_seq, dnabert_dir=args.dnabert_dir), seqs))
    else:
        inputs = [_tokenize_seq(s, args.dnabert_dir) for s in seqs]

    probs = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start:start + batch_size]
        batch = {k: torch.stack([torch.from_numpy(d[k]) for d in chunk]).to(device)
                 for k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            fold_probs = torch.stack([torch.softmax(m(**batch), dim=1)[:, 1] for m in models])
            probs.extend(fold_probs.mean(dim=0).tolist())
    elapsed = time.perf_counter() - t0

    out_ipromp = out_base / "ipromp"
    out_ipromp.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame({"PRED": probs}).to_csv(
        out_ipromp / "ipromp_12_predictions.csv", sep="\t", index=False)

    combined_path.unlink(missing_ok=True)
    print(f"iPro-MP: {len(seqs)} seqs in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
