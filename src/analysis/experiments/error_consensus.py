#!/usr/bin/env python3
"""Error analysis + consensus across the 7 tools, per dataset (canonical + GC-matched)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.pred_io import load_preds

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(os.environ.get("PROMOTER_DATA_DIR", "/home/fierro/Desktop"))
OUT = ROOT / "output/gc_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TOOLS = [("lcnn","LCNN"),("ipromp","iPro-MP"),("mldspp","MLDSPP"),("mldspp_75","MLDSPP75"),
         ("promotech","PromoTech"),("fimo","FIMO"),("meme","MEME")]
SETS = [("d39v","cds"),("d39v","gc30"),("d39v","gc33"),("tigr4","cds"),("tigr4","gc31")]
NPOS = {"d39v": 988, "tigr4": 738}

def youden_threshold(y, s):
    fpr, tpr, th = roc_curve(y, s)
    return th[np.argmax(tpr - fpr)]

# metadata
meta = pd.read_csv(ROOT/"data/benchmark/d39v/positives_81bp_metadata.tsv", sep="\t")
from src.analysis.experiments._conservation import build_conservation_classes
meta = build_conservation_classes(Path(__file__).resolve().parents[3] / "data/benchmark/d39v/positives_81bp_metadata.tsv")
meta["GC"] = pd.to_numeric(meta["GC_Content(%)"], errors="coerce")

def phi(a, b):
    a = a.astype(int); b = b.astype(int)
    n11 = ((a == 1) & (b == 1)).sum(); n00 = ((a == 0) & (b == 0)).sum()
    n10 = ((a == 1) & (b == 0)).sum(); n01 = ((a == 0) & (b == 1)).sum()
    den = np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return (n11*n00 - n10*n01)/den if den > 0 else 0.0

summary = []
for sname, sset in SETS:
    root = BASE/f"{sname}_gc/{sset}/predictions"
    npos = NPOS[sname]
    scores = {}
    for key, lab in TOOLS:
        try:
            pos, neg = load_preds(root, key, npos)
        except Exception:
            continue
        scores[lab] = np.r_[pos, neg]
    n = npos + (len(scores[list(scores)[0]]) - npos)
    y = np.r_[np.ones(npos), np.zeros(n - npos)]

    # binary predictions at Youden
    preds = {}
    for lab, s in scores.items():
        th = youden_threshold(y, s)
        preds[lab] = (s >= th).astype(int)

    # error correlation
    errs = {lab: (y != preds[lab]) for lab in preds}
    labs = list(errs)
    phi_mat = np.zeros((len(labs), len(labs)))
    for i in range(len(labs)):
        for j in range(len(labs)):
            phi_mat[i, j] = phi(errs[labs[i]], errs[labs[j]])

    phi_rows = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            phi_rows.append({"tool_a": labs[i], "tool_b": labs[j], "phi": round(phi_mat[i, j], 4)})
    pd.DataFrame(phi_rows).to_csv(OUT/f"error_phi_{sname}_{sset}.tsv", sep="\t", index=False)

    # hard cases
    fn_all = np.zeros(npos, bool)
    fp_all = np.zeros(n - npos, bool)
    for lab in labs:
        fn_all |= (y[:npos] != preds[lab][:npos])  # union of FN (missed by any)
    fn_intersection = np.ones(npos, bool)
    for lab in labs:
        fn_intersection &= (y[:npos] != preds[lab][:npos])
    fp_intersection = np.ones(n - npos, bool)
    for lab in labs:
        fp_intersection &= (y[npos:] != preds[lab][npos:])

    if sname == "d39v" and sset == "cds":
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        im = ax.imshow(phi_mat, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=8)
        for i in range(len(labs)):
            for j in range(len(labs)):
                ax.text(j, i, f"{phi_mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(phi_mat[i, j]) > 0.5 else "black")
        ax.set_title("Error correlation (phi) — D39V cds")
        plt.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(OUT/"error_phi_heatmap_d39v_cds.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # consensus
    rank_means = np.mean([pd.Series(s).rank(pct=True).values for s in scores.values()], axis=0)
    votes = np.mean([preds[lab] for lab in labs], axis=0)
    iso_scores = []
    for lab, s in scores.items():
        iso = IsotonicRegression(out_of_bounds="clip").fit(s, y)
        iso_scores.append(iso.predict(s))
    iso_mean = np.mean(iso_scores, axis=0)

    aucs = {lab: roc_auc_score(y, s) for lab, s in scores.items()}
    aucs["CONS_rank"] = roc_auc_score(y, rank_means)
    aucs["CONS_vote"] = roc_auc_score(y, votes)
    aucs["CONS_iso"] = roc_auc_score(y, iso_mean)

    best = max((v, k) for k, v in aucs.items() if not k.startswith("CONS"))
    lift = {k: v - best[0] for k, v in aucs.items() if k.startswith("CONS")}

    summary.append({"strain": sname, "set": sset, **{k: round(v, 4) for k, v in aucs.items()},
                    **{f"{k}_lift": round(v, 4) for k, v in lift.items()},
                    "n_hard_fn": int(fn_intersection.sum()), "n_hard_fp": int(fp_intersection.sum())})

    # hard-case profile (D39V only)
    if sname == "d39v":
        prof = meta.iloc[:npos].copy()
        prof["hard"] = fn_intersection
        print(f"\n=== {sname} {sset}: positivos que fallan TODAS las tools: {fn_intersection.sum()} ===")
        if fn_intersection.sum() > 0:
            h = prof[prof["hard"]]
            print(f"  sigma: {h['Sigma_Factor'].fillna('None').value_counts().to_dict()}")
            print(f"  clase: {h['class_cons'].value_counts().to_dict()}")
            print(f"  GC hard: {h['GC'].mean():.1f}% vs resto {prof[~prof['hard']]['GC'].mean():.1f}%")
        print(f"  negativos FP de todas: {fp_intersection.sum()}")

df = pd.DataFrame(summary)
df.to_csv(OUT/"consensus_analysis.tsv", sep="\t", index=False)
print("\n=== CONSENSO (AUC) por set ===")
cols = ["strain","set"] + [k for k in df.columns if k not in ("strain","set")]
print(df[cols].to_string(index=False))
print("\nGuardado:", OUT/"consensus_analysis.tsv")
