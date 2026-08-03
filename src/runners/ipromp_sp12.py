#!/usr/bin/env python3
"""iPro-MP sp12 (H. pylori) runner — DNABERT-6 inference."""
import argparse
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "tools"


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
    from transformers import BertTokenizer

    seqs = []
    with open(combined_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(">"):
                seqs.append(line.upper())

    tok = BertTokenizer.from_pretrained(args.dnabert_dir)
    model = ip.DNABERTPromoterClassifier(dnabert_dir=args.dnabert_dir)
    state_dict = torch.load(f"{args.model_dir}/12_fold_1.pth", map_location="cpu")
    state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    t0 = time.perf_counter()
    probs = []
    for s in seqs:
        kmers = [s[i:i+6] for i in range(len(s) - 5)]
        inp = tok(kmers, is_split_into_words=True, padding="max_length",
                   max_length=128, return_tensors="pt", truncation=True)
        inp = {k: inp[k] for k in ["input_ids", "attention_mask"] if k in inp}
        with torch.no_grad():
            output = model(**inp)
            prob = torch.softmax(output, dim=1)[0, 1].item()
            probs.append(prob)
    elapsed = time.perf_counter() - t0

    import pandas as pd
    pd.DataFrame({"PRED": probs}).to_csv(
        out_base / "ipromp_sp12_predictions.csv", sep="\t", index=False)

    combined_path.unlink(missing_ok=True)
    print(f"iPro-MP: {len(seqs)} seqs in {elapsed:.3f}s")


if __name__ == "__main__":
    main()
