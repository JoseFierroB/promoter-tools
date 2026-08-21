#!/usr/bin/env python3
"""XGBoost feature comparison: SantaLucia energy vs raw sequence encodings (MLDSPP protocol)."""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from pathlib import Path
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from Bio import SeqIO
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.runners._shared import extract_aligned, STABILITY_MAP

TRAIN_DIR = ROOT / "tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences"
NT_ORDER = "ATCG"
NT2I = {c: i for i, c in enumerate(NT_ORDER)}
DINUC_ORDER = [a + b for a in NT_ORDER for b in NT_ORDER]
DINUC2I = {d: i for i, d in enumerate(DINUC_ORDER)}
RNG = np.random.RandomState(42)


def train_seqs():
    seqs = []
    for f in sorted(TRAIN_DIR.glob("Sequences_80-20_B*.txt")):
        for line in open(f):
            s = line.strip()
            if len(s) >= 100:
                seqs.append(s[20:100])
    return seqs


def feats_santa(seq):
    return np.array([STABILITY_MAP.get(seq[i:i + 2], -1.35) for i in range(79)])


def feats_onehot(seq):
    out = np.zeros(80 * 4)
    for i, c in enumerate(seq):
        if c in NT2I:
            out[i * 4 + NT2I[c]] = 1.0
    return out


def feats_dinuc_pos(seq):
    out = np.zeros(79 * 16)
    for i in range(79):
        d = seq[i:i + 2]
        if d in DINUC2I:
            out[i * 16 + DINUC2I[d]] = 1.0
    return out


def feats_dinuc_freq(seq):
    out = np.zeros(16)
    for i in range(79):
        d = seq[i:i + 2]
        if d in DINUC2I:
            out[DINUC2I[d]] += 1.0
    return out / 79.0


VARIANTS = {
    "SantaLucia (79d)": feats_santa,
    "one-hot nt (320d)": feats_onehot,
    "dinuc posicional (1264d)": feats_dinuc_pos,
    "dinuc frecuencia (16d)": feats_dinuc_freq,
}


def build_matrix(seqs, fn):
    return np.array([fn(s) for s in seqs])


def main():
    seqs = train_seqs()
    print(f"Training seqs (externas): {len(seqs)}")

    pos_fa = list(SeqIO.parse(ROOT / "data/benchmark/d39v/positives_81bp.fasta", "fasta"))
    neg_fa = list(SeqIO.parse(ROOT / "data/benchmark/d39v/negatives_81bp.fasta", "fasta"))
    neg_gc = list(SeqIO.parse(ROOT / "data/benchmark/d39v_gc/negatives_81bp_gc30.fasta", "fasta"))
    test_pos = [str(r.seq).upper()[:80] for r in pos_fa]
    test_neg_cds = [str(r.seq).upper()[:80] for r in neg_fa]
    test_neg_gc = [str(r.seq).upper()[:80] for r in neg_gc]
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg_cds))]

    rows = []
    preds = {}
    for name, fn in VARIANTS.items():
        X_tr = build_matrix(seqs, fn)
        X_neg_tr = np.array([RNG.permutation(row) for row in X_tr])
        X = np.vstack([X_tr, X_neg_tr])
        yt = np.hstack([np.ones(len(X_tr)), np.zeros(len(X_neg_tr))])

        X_tp = build_matrix(test_pos, fn)
        X_tn_cds = build_matrix(test_neg_cds, fn)
        X_tn_gc = build_matrix(test_neg_gc, fn)

        model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42,
                              eval_metric="logloss", verbosity=0, n_jobs=8)
        model.fit(X, yt)
        p_cds = model.predict_proba(np.vstack([X_tp, X_tn_cds]))[:, 1]
        p_gc = model.predict_proba(np.vstack([X_tp, X_tn_gc]))[:, 1]
        auc_cds = roc_auc_score(y, p_cds)
        auc_gc = roc_auc_score(np.r_[np.ones(len(test_pos)), np.zeros(len(test_neg_gc))], p_gc)
        preds[name] = p_cds
        rows.append({"features": name, "dim": X_tr.shape[1], "AUC_d39v_cds": round(auc_cds, 4),
                     "AUC_d39v_gc30": round(auc_gc, 4),
                     "Δ_GC": round(auc_cds - auc_gc, 4)})
        print(f"{name}: AUC cds={auc_cds:.4f}  gc30={auc_gc:.4f}  Δ={auc_cds - auc_gc:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output/gc_analysis/xgboost_features_comparison.tsv", sep="\t", index=False)

    print("\n=== Correlación de predicciones (cds) ===")
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = np.corrcoef(preds[names[i]], preds[names[j]])[0, 1]
            print(f"  {names[i]:<22} vs {names[j]:<22}: r={r:.4f}")

    ref = preds["SantaLucia (79d)"]
    for name in names:
        if name != "SantaLucia (79d)":
            print(f"  |Δscores| {name}: media={np.abs(preds[name] - ref).mean():.4f} max={np.abs(preds[name] - ref).max():.4f}")
    print("\nGuardado:", ROOT / "output/gc_analysis/xgboost_features_comparison.tsv")


if __name__ == "__main__":
    main()