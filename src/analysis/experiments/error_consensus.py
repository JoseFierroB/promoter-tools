#!/usr/bin/env python3
"""Error analysis + consensus across the 7 tools, per dataset (canonical + GC-matched)."""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from pathlib import Path
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("/home/fierro/Desktop")
OUT = ROOT / "output/gc_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TOOLS = [("lcnn","LCNN"),("ipromp","iPro-MP"),("mldspp","MLDSPP"),("mldspp_75","MLDSPP75"),
         ("promotech","PromoTech"),("fimo","FIMO"),("meme","MEME")]
SETS = [("d39v","cds"),("d39v","gc30"),("d39v","gc33"),("tigr4","cds"),("tigr4","gc31")]
NPOS = {"d39v": 988, "tigr4": 738}

def load_preds(root, key, npos):
    pats = {"lcnn": ("lcnn/lcnn_pos.csv","lcnn/lcnn_neg.csv"),
            "ipromp": ("ipromp/ipromp_12_predictions.csv",),
            "mldspp": ("mldspp_pos.csv","mldspp_neg.csv"),
            "mldspp_75": ("mldspp_75spn_pos.csv","mldspp_75spn_neg.csv"),
            "promotech": ("promotech/workdir/hot_pg_pos/sequences_predictions.csv","promotech/workdir/hot_pg_neg/sequences_predictions.csv"),
            "fimo": ("fimo_prok_pos.csv","fimo_prok_neg.csv"),
            "meme": ("meme_pos.csv","meme_neg.csv")}
    p = Path(root)
    if key == "ipromp":
        df = pd.read_csv(p/pats[key][0], sep="\t")
        col = "PRED" if "PRED" in df.columns else "Probability"
        return df[col].values[:npos], df[col].values[npos:]
    a, b = pats[key]
    return pd.read_csv(p/a, sep="\t")["PRED"].values, pd.read_csv(p/b, sep="\t")["PRED"].values

def youden_threshold(y, s):
    fpr, tpr, th = roc_curve(y, s)
    return th[np.argmax(tpr - fpr)]

# metadata
meta = pd.read_csv(ROOT/"data/benchmark/d39v/positives_81bp_metadata.tsv", sep="\t")
cls_tab = pd.read_csv(ROOT/"output/tables/tss_position_classification.tsv", sep="\t")
cls_tab = cls_tab[cls_tab["strain"]=="D39V"].set_index("tss_id")
val = pd.read_csv(ROOT/"output/tables/conserved_igrs_tss_validation.tsv", sep="\t")
hit_igrs = set(val["query_d39v"].astype(str))
classes = []
for _, r in meta.iterrows():
    c = cls_tab.loc[r["Sequence_ID"], "classification"]
    if c in ("CDS_deep", "CDS_near_start"):
        classes.append("intragenic")
    elif c.startswith("IGR"):
        gid = cls_tab.loc[r["Sequence_ID"], "igr_id"]
        classes.append("conserved" if str(gid) in hit_igrs else "nonconserved")
    else:
        classes.append("other")
meta["class_cons"] = classes
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
