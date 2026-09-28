#!/usr/bin/env python3
"""Operating points (Youden) for one CLI run: global + sigma cuts.

For every tool, finds the ROC threshold maximizing Youden's J
(sensitivity + specificity - 1) and reports the confusion matrix,
F1, MCC and ACC at that threshold. Loaders are the same as
`analyze_run.py`; sigma ID-mapping is reused from `sigma_stratify.py`
so cut AUCs match `sigma_factor_benchmark_metrics.tsv`.

Writes into the run dir:
  3_tables/operating_point_metrics.tsv         (global, one row per tool)
  3_tables/operating_point_metrics_sigma.tsv   (SigA/ComX/None, needs --metadata)

Usage:
    pixi run python src/analysis/operating_points.py \
      --run-dir output/260917_runs/4_d39v_tigr4_high_sigma/16cpu-gpu \
      --name 4_d39v_tigr4_high_sigma \
      --metadata data/benchmark/datasets/2_d39v_tigr4_high_sigma/positives_81bp_metadata.tsv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (roc_curve, auc, confusion_matrix, f1_score,
                             matthews_corrcoef, accuracy_score)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from analyze_run import load_all, TOOL_ORDER  # noqa: E402


def operating_point(y_true, y_score):
    """Youden threshold + confusion metrics. Constant scores -> BROKEN (None)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(set(y_score.tolist())) < 2:
        return None
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    i = int(np.argmax(j))
    pred = (y_score >= thr[i]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"auc": round(float(auc(fpr, tpr)), 4),
            "threshold": float(thr[i]),  # full precision: rounding moves borderline calls
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
            "mcc": round(float(matthews_corrcoef(y_true, pred)), 4),
            "acc": round(float(accuracy_score(y_true, pred)), 4)}


def main():
    ap = argparse.ArgumentParser(description="Youden operating points for one run.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metadata", default=None,
                    help="positives metadata TSV (enables SigA/ComX/None cuts)")
    ap.add_argument("--neg", default=None,
                    help="negatives FASTA (with --metadata: enables d39v/tigr4 cuts; "
                         "neg IDs carry _D39V_/_TIGR4_ tags, scores align by FASTA order)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_root = run_dir / "1_inference" / "predictions"
    if not pred_root.exists():  # legacy fallback
        pred_root = run_dir / "predictions"

    scored = load_all(pred_root, args.name)
    if not scored:
        raise SystemExit(f"ERROR: no predictions in {pred_root}")

    tables = run_dir / "3_tables"
    tables.mkdir(parents=True, exist_ok=True)

    # Global cut.
    rows, broken = [], []
    order_idx = {t: i for i, t in enumerate(TOOL_ORDER)}
    for key in sorted(scored, key=lambda k: order_idx.get(scored[k][0], 999)):
        label, y_true, y_score = scored[key]
        op = operating_point(y_true, y_score)
        if op is None:
            broken.append(label)
            continue
        rows.append({"dataset": args.name, "tool": label, "cut": "global",
                     "n_pos": int(np.asarray(y_true).sum()),
                     "n_neg": int(len(y_true) - np.asarray(y_true).sum()), **op})
    pd.DataFrame(rows).to_csv(tables / "operating_point_metrics.tsv",
                              sep="\t", index=False)
    print(f"  [op] global: {len(rows)} rows -> 3_tables/operating_point_metrics.tsv")

    # Sigma cuts (ID-grounded, same mapping as sigma_stratify).
    if args.metadata:
        from sigma_stratify import load_classes, per_id_scores, CLASS_ORDER
        pos_ids, classes = load_classes(Path(args.metadata))
        tool_maps = per_id_scores(pred_root, args.name, pos_ids)
        srows = []
        for label, (pmap, nscores) in tool_maps.items():
            for sig in CLASS_ORDER:
                sub = [p for p in pos_ids if classes.get(p) == sig and p in pmap]
                if not sub:
                    continue
                yt = np.array([1] * len(sub) + [0] * len(nscores))
                ys = np.array([pmap[p] for p in sub] + list(map(float, nscores)))
                op = operating_point(yt, ys)
                if op is None:
                    broken.append(f"{label}/{sig}")
                    continue
                srows.append({"dataset": args.name, "tool": label, "cut": sig,
                              "n_pos": len(sub), "n_neg": len(nscores), **op})
        pd.DataFrame(srows).to_csv(tables / "operating_point_metrics_sigma.tsv",
                                   sep="\t", index=False)
        print(f"  [op] sigma: {len(srows)} rows -> 3_tables/operating_point_metrics_sigma.tsv")

    # Strain cuts (pos: metadata Chromosome; neg: FASTA ID tags).
    if args.metadata and args.neg:
        meta = pd.read_csv(Path(args.metadata), sep="\t")
        if "Chromosome" not in meta.columns:
            raise SystemExit(f"ERROR: {args.metadata} needs a Chromosome column")
        strain_of = {str(r["Sequence_ID"]): ("d39v" if str(r["Chromosome"]) == "D39V"
                                             else "tigr4")
                     for _, r in meta.iterrows()}
        neg_ids = [r.id for r in SeqIO.parse(args.neg, "fasta")]

        def nstrain(nid):
            if "_D39V_" in nid or "D39V" in nid.split("_"):
                return "d39v"
            if "_TIGR4_" in nid or "TIGR4" in nid.split("_"):
                return "tigr4"
            raise SystemExit(f"ERROR: neg ID without strain tag: {nid}")

        pos_ids = meta["Sequence_ID"].astype(str).tolist()
        tool_maps = per_id_scores(pred_root, args.name, pos_ids)
        trows = []
        for label, (pmap, nscores) in tool_maps.items():
            if len(nscores) != len(neg_ids):
                print(f"  [op] skip strain for {label}: "
                      f"{len(nscores)} scores vs {len(neg_ids)} neg IDs")
                continue
            n_by_strain = {"d39v": [float(s) for i, s in zip(neg_ids, nscores)
                                    if nstrain(i) == "d39v"],
                           "tigr4": [float(s) for i, s in zip(neg_ids, nscores)
                                     if nstrain(i) == "tigr4"]}
            for st in ("d39v", "tigr4"):
                sub = [p for p in pos_ids if strain_of.get(p) == st and p in pmap]
                if not sub or not n_by_strain[st]:
                    continue
                yt = np.array([1] * len(sub) + [0] * len(n_by_strain[st]))
                ys = np.array([pmap[p] for p in sub] + n_by_strain[st])
                op = operating_point(yt, ys)
                if op is None:
                    broken.append(f"{label}/{st}")
                    continue
                trows.append({"dataset": args.name, "tool": label, "cut": st,
                              "n_pos": len(sub), "n_neg": len(n_by_strain[st]), **op})
        pd.DataFrame(trows).to_csv(tables / "operating_point_metrics_strain.tsv",
                                   sep="\t", index=False)
        print(f"  [op] strain: {len(trows)} rows -> 3_tables/operating_point_metrics_strain.tsv")

    if broken:
        print(f"  [op] BROKEN (constant scores, no threshold): {broken}")
        raise SystemExit(1)
    print("  [op] OK: no broken runners")


if __name__ == "__main__":
    main()
