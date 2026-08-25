#!/usr/bin/env python3
"""PromoterLCNN runner — one-hot encode + TF predict (batched)."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

MODEL_DIR = "tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5"
DEFAULT_BATCH = 10000
_TBL = np.zeros(256, dtype=np.uint8)
_TBL[[ord(c) for c in "ATCG"]] = [0, 1, 2, 3]  # A,T,C,G -> rows of eye(4)
_EYE = np.eye(4, dtype=np.float32)


def onehot(seqs):
    out = np.zeros((len(seqs), 81, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        out[i] = _EYE[_TBL[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]]
    return out


def main():
    p = argparse.ArgumentParser(description="PromoterLCNN prediction")
    p.add_argument("--pos", required=True, help="Positive test FASTA")
    p.add_argument("--neg", required=True, help="Negative test FASTA")
    p.add_argument("-o", "--output", default="output/predictions", help="Output dir")
    p.add_argument("-m", "--model", default=MODEL_DIR, help="Model directory")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Inference batch size (0 = all sequences in one batch)")
    args = p.parse_args()

    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = None  # resolved after counting sequences
    env_batch = os.environ.get("PROMOTER_TOOLS_LCNN_BATCH", "")
    if env_batch not in ("", None):
        batch_size = int(env_batch)
        if batch_size == 0:
            batch_size = None

    pos = list(SeqIO.parse(args.pos, "fasta"))
    neg = list(SeqIO.parse(args.neg, "fasta"))
    seqs = [str(r.seq).upper() for r in pos + neg]

    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    probs = np.empty(len(seqs), dtype=np.float32)
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.model)
        signature = meta_graph_def.signature_def["serving_default"]
        in_tensor_name = signature.inputs[list(signature.inputs.keys())[0]].name
        out_tensor_name = signature.outputs[list(signature.outputs.keys())[0]].name

        in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
        out_tensor = sess.graph.get_tensor_by_name(out_tensor_name)

        t0 = time.perf_counter()
        eff_batch = batch_size or len(seqs)
        for start in range(0, len(seqs), eff_batch):
            X = onehot(seqs[start:start + eff_batch])
            preds = sess.run(out_tensor, feed_dict={in_tensor: X})
            probs[start:start + eff_batch] = (
                preds[:, 1] if preds.ndim == 2 and preds.shape[1] >= 2 else preds.ravel())

    elapsed = time.perf_counter() - t0

    out_dir = Path(args.output) / "lcnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PRED": probs[:len(pos)]}).to_csv(
        out_dir / "lcnn_pos.csv", sep="\t", index=False)
    pd.DataFrame({"PRED": probs[len(pos):]}).to_csv(
        out_dir / "lcnn_neg.csv", sep="\t", index=False)

    print(f"LCNN: {len(seqs)} seqs in {elapsed:.3f}s (batch={batch_size or len(seqs)})")


if __name__ == "__main__":
    main()
