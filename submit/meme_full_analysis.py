#!/usr/bin/env python3
"""
MEME Complete Analysis — equivalent to local machine analysis.
Runs: STREME→FIMO, MEME→FIMO→TOMTOM, generates all plots.
Usage: BENCH_OUT=/nfs/research/jlees/fierro/resultados pixi run --manifest-path tools/meme/pixi.toml python submit/meme_full_analysis.py
"""
import subprocess, tempfile, shutil, csv, math, re, random, time, json, os
from pathlib import Path
import numpy as np
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
POS = ROOT / "data/benchmark/positives_81bp.fasta"
NEG = ROOT / "data/benchmark/negatives_81bp.fasta"
OUT_DIR = Path(os.environ.get("BENCH_OUT", str(ROOT / "output" / "meme_analysis")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
pos = list(SeqIO.parse(POS, "fasta"))
neg = list(SeqIO.parse(NEG, "fasta"))
print(f"D39V: {len(pos)} pos + {len(neg)} neg = {len(pos)+len(neg)} total\n")

# ═══════════════════════════════════════════════════════════════
# 1. STREME → FIMO (2-fold CV, discriminative)
# ═══════════════════════════════════════════════════════════════
print("═══ 1. STREME → FIMO (2-fold CV) ═══")
random.shuffle(pos); random.shuffle(neg)
mp, mn = len(pos)//2, len(neg)//2
scores_streme = {}

for fold in range(2):
    tp = pos[:mp] if fold==0 else pos[mp:]
    tn = neg[:mn] if fold==0 else neg[mn:]
    tp_test = pos[mp:] if fold==0 else pos[:mp]
    tn_test = neg[mn:] if fold==0 else neg[:mn]
    
    td = Path(tempfile.mkdtemp(prefix="meme_a1_"))
    SeqIO.write(tp, td/"tp.fa", "fasta"); SeqIO.write(tn, td/"tn.fa", "fasta")
    with open(td/"test.fa", "w") as f:
        for r in tp_test: SeqIO.write(r, f, "fasta")
        for r in tn_test: SeqIO.write(r, f, "fasta")
    
    res = subprocess.run(["streme", "-oc", str(td/"streme"), "-dna", "-minw", "10", "-maxw", "20",
        "-p", str(td/"tp.fa"), "-n", str(td/"tn.fa")], capture_output=True, text=True, timeout=120)
    
    res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
        str(td/"streme"/"streme.txt"), str(td/"test.fa")],
        capture_output=True, text=True, timeout=120)
    
    for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
        try: pv = float(row["p-value"])
        except: continue
        nl = 999.0 if pv <= 0 else -math.log10(pv)
        s = row["sequence_name"]
        if s not in scores_streme or nl > scores_streme[s]: scores_streme[s] = nl
    
    # Save STREME motifs (fold 0 only)
    if fold == 0:
        with open(td/"streme"/"streme.txt") as f:
            txt = f.read()
        motifs = re.findall(r"MOTIF (\S+) STREME-\d+", txt)
        for m in motifs:
            chunk = txt[txt.index(m):txt.index(m)+300]
            e = re.search(r"E=\s*([\d\.e\-\+]+)", chunk)
            ns = re.search(r"nsites=\s*(\d+)", chunk)
            print(f"  {m[:40]}  sites={ns.group(1)}  E={e.group(1)}")
    
    shutil.rmtree(td)

for r in pos + neg:
    if r.id not in scores_streme: scores_streme[r.id] = 0.0


# ═══════════════════════════════════════════════════════════════
# 2. MEME → FIMO → TOMTOM (classic EM, with annotation)
# ═══════════════════════════════════════════════════════════════
print("\n═══ 2. MEME → FIMO → TOMTOM ═══")
DB = ROOT / "tools/meme/motif_databases/unified_prokaryote.meme"
td = Path(tempfile.mkdtemp(prefix="meme_a2_"))

# MEME classic on all positives
res = subprocess.run(["meme", str(POS), "-dna", "-mod", "zoops",
    "-minw", "10", "-maxw", "20", "-oc", str(td/"meme"), "-nostatus"],
    capture_output=True, text=True, timeout=300)

with open(td/"meme"/"meme.txt") as f:
    txt = f.read()
for m in re.findall(r"MOTIF (\S+) MEME-\d+", txt):
    chunk = txt[txt.index(m):txt.index(m)+300]
    e = re.search(r"E-value=\s*([\d\.e\-\+]+)", chunk)
    ns = re.search(r"sites =\s*(\d+)", chunk)
    if e and ns:
        print(f"  {m[:40]}  sites={ns.group(1)}  E={e.group(1)}")

# TOMTOM annotation
print("\n  TOMTOM (unified prokaryotic DB):")
res = subprocess.run(["tomtom", "-no-ssc", "-text", "-min-overlap", "4",
    str(td/"meme"/"meme.xml"), str(DB)],
    capture_output=True, text=True, timeout=120)

tomtom_hits = []
for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
    try: qv = float(row.get("q-value", 1))
    except: continue
    if qv < 0.05 and row.get("Target_ID"):
        tomtom_hits.append((row["Target_ID"], qv))
        print(f"    {row['Target_ID'][:60]}  q={qv:.1e}")

# FIMO scan
combined = td / "all.fa"
with open(combined, "w") as f:
    for r in pos: SeqIO.write(r, f, "fasta")
    for r in neg: SeqIO.write(r, f, "fasta")
res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
    str(td/"meme"/"meme.xml"), str(combined)],
    capture_output=True, text=True, timeout=120)

scores_meme = {}
for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
    try: pv = float(row["p-value"])
    except: continue
    nl = 999.0 if pv <= 0 else -math.log10(pv)
    s = row["sequence_name"]
    if s not in scores_meme or nl > scores_meme[s]: scores_meme[s] = nl

for r in pos + neg:
    if r.id not in scores_meme: scores_meme[r.id] = 0.0

shutil.rmtree(td)


# ═══════════════════════════════════════════════════════════════
# 3. Metrics & AUC
# ═══════════════════════════════════════════════════════════════
print("\n═══ 3. AUC Comparison ═══")

y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])

results = []
for label, scores in [("STREME", scores_streme), ("MEME classic", scores_meme)]:
    sc = np.array([scores[r.id] for r in pos + neg])
    fpr, tpr, thresholds = roc_curve(y, sc)
    opt = np.argmax(tpr - fpr[:len(thresholds)])
    th = thresholds[opt]
    yp = (sc >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
    
    m = {
        "model": label,
        "auc": round(roc_auc_score(y, sc), 4),
        "pr": round(average_precision_score(y, sc), 4),
        "f1": round(f1_score(y, yp), 4),
        "mcc": round(matthews_corrcoef(y, yp), 4),
        "sens": round(recall_score(y, yp), 4),
        "spec": round(tn/(tn+fp), 4) if (tn+fp) > 0 else 0,
        "prec": round(precision_score(y, yp), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "pos_hits": int(sum(1 for s in pos if scores[s.id] > 0)),
        "neg_hits": int(sum(1 for s in neg if scores[s.id] > 0)),
    }
    results.append(m)
    print(f"  {label}: AUC={m['auc']}  F1={m['f1']}  MCC={m['mcc']}  "
          f"TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}")

# Save TSV
with open(OUT_DIR / "meme_metrics.tsv", "w") as f:
    keys = results[0].keys()
    f.write("\t".join(keys) + "\n")
    for r in results:
        f.write("\t".join(str(r[k]) for k in keys) + "\n")
print(f"\n  Saved: {OUT_DIR}/meme_metrics.tsv")


# ═══════════════════════════════════════════════════════════════
# 4. Generate Plots
# ═══════════════════════════════════════════════════════════════
print("\n═══ 4. Generating Plots ═══")
plot_dir = OUT_DIR / "plots"
plot_dir.mkdir(exist_ok=True)

# ROC comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

for label, scores, color, ls in [("STREME", scores_streme, "#002D62", "-"), 
                                   ("MEME classic", scores_meme, "#D95319", "--")]:
    sc = np.array([scores[r.id] for r in pos + neg])
    fpr, tpr, _ = roc_curve(y, sc)
    ax1.plot(fpr, tpr, lw=2, color=color, ls=ls, label=f"{label} (AUC={roc_auc_score(y, sc):.3f})")
    # Score distribution
    pos_sc = [scores[r.id] for r in pos]
    neg_sc = [scores[r.id] for r in neg]
    ax2.hist(pos_sc, bins=50, alpha=0.5, color=color, label=f"{label} pos")
    ax2.hist(neg_sc, bins=50, alpha=0.3, color=color, label=f"{label} neg", histtype="step", lw=2)

ax1.plot([0, 1], [0, 1], "k--", lw=0.5, alpha=0.3)
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC — STREME vs MEME classic"); ax1.legend(fontsize=8)
ax2.set_xlabel("Score (max -log10 p-value)"); ax2.set_ylabel("Count")
ax2.set_title("Score Distribution"); ax2.legend(fontsize=7)
for ax in [ax1, ax2]: ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(plot_dir / "meme_comparison.png", dpi=300)
plt.savefig(plot_dir / "meme_comparison.svg", dpi=300)
plt.close()
print(f"  {plot_dir}/meme_comparison.{{svg,png}}")

# Tomtom summary plot
if tomtom_hits:
    fig, ax = plt.subplots(figsize=(8, len(tomtom_hits[:20])*0.35 + 1), dpi=200)
    names = [h[0][:50] for h in tomtom_hits[:20][::-1]]
    qvals = [-np.log10(h[1]) for h in tomtom_hits[:20][::-1]]
    ax.barh(range(len(names)), qvals, color="#002D62")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("-log10(q-value)"); ax.set_title("Tomtom Annotations (q<0.05)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir / "meme_tomtom.png", dpi=200)
    plt.close()
    print(f"  {plot_dir}/meme_tomtom.png")

print(f"\n═══ DONE ═══")
print(f"Output: {OUT_DIR}")
