#!/usr/bin/env python3
"""GC experiment full analysis: calibration, conservation, confusion, DeLong+Holm, duplicates, sigma."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from src.utils.pred_io import load_preds


def load_preds_gc(root, key):
    npos = NPOS.get(Path(root).parent.parent.name.split("_")[0], None)
    return load_preds(root, key, npos)
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/gc_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TOOLS = [("lcnn","LCNN"),("ipromp","iPro-MP"),("mldspp","MLDSPP"),("mldspp_75","MLDSPP75"),
         ("promotech","PromoTech"),("fimo","FIMO"),("meme","MEME")]
SETS = [("d39v","cds"),("d39v","gc30"),("d39v","gc33"),("tigr4","cds"),("tigr4","gc31")]
NPOS = {"d39v": 988, "tigr4": 738}


def get_scores(root, key, npos):
    pos, neg = load_preds_gc(root, key)
    return np.r_[pos, neg[:npos-len(pos)] if len(pos) < npos else pos], np.r_[np.ones(npos), np.zeros(len(pos)+len(neg)-npos)]

# ── Brier + reliability (isotonic-calibrated) ──
def brier(y, s):
    return float(np.mean((s - y)**2))

def isotonic_cal(y, s, n_bins=10):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(s, y)
    sc = iso.predict(s)
    return float(np.mean((sc - y)**2))

def reliability(s, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    rows = []
    for i in range(n_bins):
        m = (s > bins[i]) & (s <= bins[i+1])
        if m.sum() == 0:
            continue
        rows.append((round((bins[i]+bins[i+1])/2, 3), round(float(s[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())))
    return rows

# ── classes: conservation mapping rebuild ──
igr = pd.read_csv(ROOT/"output/intergenic/d39v/D39V_igrs.tsv", sep="\t")
igr = igr.sort_values("start")
val = pd.read_csv(ROOT/"output/tables/conserved_igrs_tss_validation.tsv", sep="\t")
hit_igrs = set(val["query_d39v"].astype(str)) | set(val["target_tigr4"].astype(str))
# CDS intervals
import re
cds_int = []
for line in open(ROOT/"data/reference/D39V.gff3"):
    if line.startswith("#"): continue
    parts = line.split("\t")
    if len(parts) < 9: continue
    if "CDS" in parts[2] or "gene" == parts[2]:
        cds_int.append((int(parts[3])-1, int(parts[4]), parts[6]))
meta = pd.read_csv(ROOT/"data/benchmark/d39v/positives_81bp_metadata.tsv", sep="\t")

def in_cds(pos):
    for s, e, st in cds_int:
        if s <= pos < e:
            return True
    return False

cls_tab = pd.read_csv(ROOT/"output/tables/tss_position_classification.tsv", sep="\t")
cls_tab = cls_tab[cls_tab["strain"]=="D39V"].set_index("tss_id")
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
print("=== verificación de clases ===")
print(meta["class_cons"].value_counts().to_string())
print("original: conserved=647 nonconserved=157 intragenic=184")

# ── A2: conservation-stratified AUC por set ──
print("\n=== A2: AUC por clase de conservación (d39v) ===")
rows = []
for sname, sset in SETS:
    if sname != "d39v": continue
    root = Path("/home/fierro/Desktop")/f"d39v_gc/{sset}/predictions"
    for key, lab in TOOLS:
        try:
            pos, neg = load_preds_gc(root, key)
        except Exception:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        s = np.r_[pos, neg]
        for cls in ["conserved", "nonconserved", "intragenic"]:
            idx = meta.index[meta["class_cons"] == cls]
            mask_pos = np.zeros(len(pos), bool); mask_pos[idx] = True
            yc = np.r_[np.ones(mask_pos.sum()), np.zeros(len(neg))]
            sc = np.r_[pos[mask_pos], neg]
            a = roc_auc_score(yc, sc)
            rows.append((lab, sset, cls, round(a, 4)))
df = pd.DataFrame(rows, columns=["tool","set","class","AUC"])
print(df.pivot_table(index="tool", columns=["set","class"], values="AUC").round(4).to_string())

# ── A1: Brier ──
print("\n=== A1: Brier (raw / isotonic-calibrated) ===")
rows = []
for sname, sset in SETS:
    root = Path("/home/fierro/Desktop")/f"{sname}_gc/{sset}/predictions"
    npos = NPOS[sname]
    for key, lab in TOOLS:
        try:
            pos, neg = load_preds_gc(root, key)
        except Exception:
            continue
        s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        rows.append((lab, sname, sset, round(brier(y, s), 4), round(isotonic_cal(y, s), 4)))
df = pd.DataFrame(rows, columns=["tool","strain","set","Brier_raw","Brier_iso"])
df.to_csv(OUT/"calibration_brier.tsv", sep="\t", index=False)
print(df.pivot_table(index="tool", columns=["strain","set"], values="Brier_raw").round(4).to_string())

# ── A3: confusion (Youden) por set ──
print("\n=== A3: Confusion Youden (canónico vs gc30 d39v) ===")
for sname, sset in [("d39v","cds"),("d39v","gc30"),("tigr4","cds"),("tigr4","gc31")]:
    root = Path("/home/fierro/Desktop")/f"{sname}_gc/{sset}/predictions"
    print(f"\n-- {sname} {sset} --")
    print(f"{'tool':<11}{'AUC':>7}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}{'Sens':>7}{'Spec':>7}{'F1':>7}")
    for key, lab in TOOLS:
        try:
            pos, neg = load_preds_gc(root, key)
        except Exception:
            continue
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]
        fpr, tpr, th = roc_curve(y, s)
        j = np.argmax(tpr - fpr)
        pred = (s >= th[j]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        sens = tp/(tp+fn); spec = tn/(tn+fp); prec = tp/(tp+fp); f1 = 2*prec*sens/(prec+sens)
        print(f"{lab:<11}{roc_auc_score(y,s):>7.4f}{tp:>5}{fn:>5}{fp:>5}{tn:>5}{sens:>7.3f}{spec:>7.3f}{f1:>7.3f}")

# ── A4: DeLong + Holm por set ──
def delong_var(y, s):
    pos_s = s[y==1]; neg_s = s[y==0]
    v10 = np.zeros(len(s))
    for i, ss in enumerate(s):
        if y[i] == 1:
            v10[i] = np.mean(ss > neg_s) + 0.5*np.mean(ss == neg_s)
        else:
            v10[i] = np.mean(ss < pos_s) + 0.5*np.mean(ss == pos_s)
    return v10

print("\n=== A4: DeLong pareado (d39v cds vs gc30), Holm corregido ===")
rows = []
root_c = Path("/home/fierro/Desktop/d39v_gc/cds/predictions")
root_g = Path("/home/fierro/Desktop/d39v_gc/gc30/predictions")
data = {}
for key, lab in TOOLS:
    try:
        pc, nc = load_preds_gc(root_c, key); pg, ng = load_preds_gc(root_g, key)
    except Exception:
        continue
    data[lab] = (pc, nc, pg, ng)
labs = list(data)
pvals = []
for i in range(len(labs)):
    for j in range(i+1, len(labs)):
        la, lb = labs[i], labs[j]
        pa, na, pag, nag = data[la]; pb, nb, pbg, nbg = data[lb]
        n = len(pa)+len(na); y = np.r_[np.ones(len(pa)), np.zeros(len(na))]
        dA = roc_auc_score(y, np.r_[pag, nag]) - roc_auc_score(y, np.r_[pa, na])
        dB = roc_auc_score(y, np.r_[pbg, nbg]) - roc_auc_score(y, np.r_[pb, nb])
        vA = delong_var(y, np.r_[pa, na]); vAg = delong_var(y, np.r_[pag, nag])
        vB = delong_var(y, np.r_[pb, nb]); vBg = delong_var(y, np.r_[pbg, nbg])
        n_pos = len(pa); n_neg = len(na)
        var_A = np.var(vA[y==1], ddof=1)/n_pos + np.var(vA[y==0], ddof=1)/n_neg
        var_B = np.var(vB[y==1], ddof=1)/n_pos + np.var(vB[y==0], ddof=1)/n_neg
        var_gA = np.var(vAg[y==1], ddof=1)/n_pos + np.var(vAg[y==0], ddof=1)/n_neg
        var_gB = np.var(vBg[y==1], ddof=1)/n_pos + np.var(vBg[y==0], ddof=1)/n_neg
        covA = np.cov(vAg[y==1], vA[y==1])[0,1]/n_pos + np.cov(vAg[y==0], vA[y==0])[0,1]/n_neg
        covB = np.cov(vBg[y==1], vB[y==1])[0,1]/n_pos + np.cov(vBg[y==0], vB[y==0])[0,1]/n_neg
        var_d = var_gA + var_A + var_gB + var_B - 2*covA - 2*covB
        z = (dA - dB)/np.sqrt(var_d) if var_d > 0 else 0
        p = 2*(1-norm.cdf(abs(z))) if var_d > 0 else 1.0
        pvals.append((la, lb, p))
# Holm
m = len(pvals)
pvals_sorted = sorted(pvals, key=lambda x: x[2])
adj = {}
for k, (la, lb, p) in enumerate(pvals_sorted):
    adj[(la, lb)] = min(1.0, p * (m - k))
for la, lb, p in pvals_sorted:
    print(f"  Δ(ΔAUC) {la} vs {lb}: diff={round((data[la][2][0]>0 and True) or 0, 3)}, p_holm={adj[(la,lb)]:.4f}" if False else f"  {la} vs {lb}: p_raw={p:.5f}  p_holm={adj[(la,lb)]:.4f}")

# ── A5: duplicados sensibilidad ──
print("\n=== A5: duplicados — AUC 972 únicas vs 988 (d39v cds) ===")
pos_seqs = [str(r.seq) for r in __import__("Bio.SeqIO", fromlist=["SeqIO"]).parse(ROOT/"data/benchmark/d39v/positives_81bp.fasta", "fasta")]
keep = np.ones(len(pos_seqs), bool)
seen = set()
for i, s in enumerate(pos_seqs):
    if s in seen: keep[i] = False
    seen.add(s)
root = Path("/home/fierro/Desktop/d39v_gc/cds/predictions")
for key, lab in TOOLS:
    try:
        pos, neg = load_preds_gc(root, key)
    except Exception:
        continue
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]
    a_all = roc_auc_score(y, s)
    a_uniq = roc_auc_score(np.r_[np.ones(keep.sum()), np.zeros(len(neg))], np.r_[pos[keep], neg])
    print(f"  {lab:<11} AUC 988={a_all:.4f}  AUC 972 únicas={a_uniq:.4f}  Δ={a_uniq-a_all:+.4f}")

print("\n=== A6: AUC por Sigma_Factor (d39v cds y gc30) ===")
sig = meta["Sigma_Factor"].fillna("None").values
for sset in ["cds", "gc30"]:
    root = Path("/home/fierro/Desktop")/f"d39v_gc/{sset}/predictions"
    print(f"-- {sset} --")
    for key, lab in TOOLS:
        try:
            pos, neg = load_preds_gc(root, key)
        except Exception:
            continue
        s = np.r_[pos, neg]
        out = []
        for cls in sorted(set(sig)):
            idx = sig == cls
            if idx.sum() < 10:
                continue
            yc = np.r_[np.ones(idx.sum()), np.zeros(len(neg))]
            sc = np.r_[pos[idx], neg]
            out.append(f"{cls}({idx.sum()})={roc_auc_score(yc, sc):.4f}")
        print(f"  {lab:<11} " + "  ".join(out))

print("\nGuardado:", OUT)
