#!/usr/bin/env python3
"""
MEME Suite ROC: de novo (STREME+FIMO 2-fold CV) vs zero-shot motif DBs.
Grouped by matrix format: log-odds (orange) vs letter-probability (green).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "output" / "predictions"
OUT_DIR = ROOT / "output" / "plots" / "meme"
OUT_DIR.mkdir(parents=True, exist_ok=True)

curves = []


def add_curve(label, pos_file, neg_file, color, ls, lw):
    pos = pd.read_csv(PRED_DIR / pos_file, sep="\t")
    neg = pd.read_csv(PRED_DIR / neg_file, sep="\t")
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.hstack([pos["PRED"].values, neg["PRED"].values])
    fpr, tpr, _ = roc_curve(y, s)
    curves.append((label, fpr, tpr, auc(fpr, tpr), color, ls, lw))


# ── De novo (learns from S. pneumoniae, 2-fold CV) ──
add_curve("STREME + FIMO (de novo)", "meme_pos.csv", "meme_neg.csv",
          "#1D14C8", "-", 2.2)

# ── Group 2: log-odds matrix (orange) ──
add_curve("DPInteract (68)", "fimo_dpinteract_pos.csv", "fimo_dpinteract_neg.csv",
          "#E07614", "-", 1.5)
add_curve("RegTransBase (141)", "fimo_regtransbase_pos.csv", "fimo_regtransbase_neg.csv",
          "#E07614", "--", 1.2)

# ── Group 1: letter-probability matrix (green) ──
add_curve("PRODORIC 2021.9 (333)", "fimo_prodoric_2021.9_pos.csv", "fimo_prodoric_2021.9_neg.csv",
          "#228B22", "-", 1.5)
add_curve("CollecTF (84)", "fimo_collectf_pos.csv", "fimo_collectf_neg.csv",
          "#228B22", "--", 1.2)
add_curve("SwissRegulon (97)", "fimo_swissregulon_pos.csv", "fimo_swissregulon_neg.csv",
          "#228B22", "-.", 1.0)
add_curve("FAN 2020 (115)", "fimo_fan_2020_pos.csv", "fimo_fan_2020_neg.csv",
          "#228B22", ":", 1.0)

# ── Plot ──
fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=300)
for name, fpr, tpr, a, color, ls, lw in curves:
    ax.plot(fpr, tpr, lw=lw, ls=ls, color=color, alpha=0.85,
            label=f"{name}  (AUC={a:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.3)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("MEME Suite — S. pneumoniae D39V\nDe novo vs Zero-shot Motif DBs",
             fontweight="bold", fontsize=12)
ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_DIR / "meme_zero_shot_roc.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "meme_zero_shot_roc.svg", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: meme_zero_shot_roc.{svg,png}")
for name, _, _, a, *_ in sorted(curves, key=lambda x: -x[3]):
    print(f"  {name:<40} AUC={a:.4f}")
