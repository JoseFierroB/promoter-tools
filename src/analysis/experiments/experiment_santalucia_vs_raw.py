#!/usr/bin/env python3
"""SantaLucia vs raw DNA encodings — corrected protocol (council design).
Two negative schemes (sequence uShuffle / feature permutation), 10 seeds,
controls: intra-GC-class table permutation (null) and additive OHE+SantaLucia.
"""
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from pathlib import Path
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from Bio import SeqIO
from collections import Counter
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output/gc_analysis/santalucia_experiment"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = ROOT / "tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences"
NT_ORDER = "ATCG"
NT2I = {c: i for i, c in enumerate(NT_ORDER)}
DINUC_ORDER = [a + b for a in NT_ORDER for b in NT_ORDER]
DINUC2I = {d: i for i, d in enumerate(DINUC_ORDER)}
from src.runners._shared import STABILITY_MAP as SM

N_SEEDS = 10
HP = dict(n_estimators=100, max_depth=6, random_state=0, eval_metric="logloss", verbosity=0, n_jobs=8)


def ushuffle(seq, rng, tries=200):
    for _ in range(tries):
        edges = {v: [] for v in NT_ORDER}
        for i in range(len(seq) - 1):
            edges[seq[i]].append(seq[i + 1])
        cur = seq[0]
        res = [cur]
        total = len(seq) - 1
        while total > 0 and edges[cur]:
            e = edges[cur]
            nxt = e.pop(int(rng.randint(len(e))))
            res.append(nxt)
            cur = nxt
            total -= 1
        if total == 0:
            return "".join(res)
    return seq


def feats_gc(seq):
    return np.array([(seq.count("G") + seq.count("C")) / len(seq)])


def feats_santa(seq, table=SM):
    return np.array([table.get(seq[i:i + 2], -1.35) for i in range(79)])


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


VARIANT_FN = {
    "GC% (1d)": feats_gc,
    "SantaLucia (79d)": feats_santa,
    "one-hot nt (320d)": feats_onehot,
    "dinuc posicional (1264d)": feats_dinuc_pos,
    "dinuc frecuencia (16d)": feats_dinuc_freq,
}
GC_CLASSES = {"AA": 0, "TT": 0, "AT": 0, "TA": 0,
              "AG": 1, "GA": 1, "AC": 1, "CA": 1, "TG": 1, "GT": 1, "TC": 1, "CT": 1,
              "CC": 2, "GG": 2, "CG": 2, "GC": 2}


def delong_full(y, sa, sb):
    pos_sa, neg_sa = sa[y == 1], sa[y == 0]
    pos_sb, neg_sb = sb[y == 1], sb[y == 0]
    n_pos, n_neg = pos_sa.size, neg_sa.size
    v10a = np.where(y == 1, np.array([np.mean(s > neg_sa) + 0.5 * np.mean(s == neg_sa) for s in sa]),
                             np.array([np.mean(s < pos_sa) + 0.5 * np.mean(s == pos_sa) for s in sa]))
    v10b = np.where(y == 1, np.array([np.mean(s > neg_sb) + 0.5 * np.mean(s == neg_sb) for s in sb]),
                             np.array([np.mean(s < pos_sb) + 0.5 * np.mean(s == pos_sb) for s in sb]))
    d = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    va_p, vb_p = np.var(v10a[y == 1], ddof=1), np.var(v10b[y == 1], ddof=1)
    va_n, vb_n = np.var(v10a[y == 0], ddof=1), np.var(v10b[y == 0], ddof=1)
    ca_p = np.cov(v10a[y == 1], v10b[y == 1])[0, 1]
    ca_n = np.cov(v10a[y == 0], v10b[y == 0])[0, 1]
    var = (va_p + vb_p) / n_pos + (va_n + vb_n) / n_neg - 2 * (ca_p / n_pos + ca_n / n_neg)
    se = np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(d) / se)) if se == se and se > 0 else 1.0
    return d, se, p


def main():
    train_raw = []
    for f in sorted(TRAIN_DIR.glob("Sequences_80-20_B*.txt")):
        for line in open(f):
            s = line.strip()
            if len(s) >= 100:
                train_raw.append(s[20:100])
    train_seqs = sorted(set(train_raw))
    print(f"train únicas: {len(train_seqs)} (de {len(train_raw)})")

    pos_fa = [str(r.seq).upper()[:80] for r in SeqIO.parse(ROOT / "data/benchmark/d39v/positives_81bp.fasta", "fasta")]
    neg_fa = {"cds": [str(r.seq).upper()[:80] for r in SeqIO.parse(ROOT / "data/benchmark/d39v/negatives_81bp.fasta", "fasta")],
              "gc30": [str(r.seq).upper()[:80] for r in SeqIO.parse(ROOT / "data/benchmark/d39v_gc/negatives_81bp_gc30.fasta", "fasta")],
              "gc33": [str(r.seq).upper()[:80] for r in SeqIO.parse(ROOT / "data/benchmark/d39v_gc/negatives_81bp_gc33.fasta", "fasta")]}
    y_pos = np.ones(len(pos_fa))

    # precompute feature matrices for test
    test_mats = {}
    for vname, fn in VARIANT_FN.items():
        test_mats[vname] = {k: np.array([fn(s) for s in seqs]) for k, seqs in
                            {"pos": pos_fa, **{f"neg_{k}": v for k, v in neg_fa.items()}}.items()}

    # intra-GC-class permuted tables (null) — 100 variants
    rng_null = np.random.RandomState(777)
    null_tables = []
    for _ in range(100):
        tbl = dict(SM)
        for cls in (0, 1, 2):
            members = [d for d in SM if GC_CLASSES[d] == cls]
            vals = [SM[d] for d in members]
            rng_null.shuffle(vals)
            for d, v in zip(members, vals):
                tbl[d] = v
        null_tables.append(tbl)

    results = []
    ens_preds_all = {}
    for scheme in ["seq_shuffle", "feat_perm"]:
        for vname, fn in VARIANT_FN.items():
            aucs = {k: [] for k in neg_fa}
            ens = {k: [] for k in neg_fa}
            for s in range(N_SEEDS):
                rng_data = np.random.RandomState(1_000_000 + s)
                X_tr = np.array([fn(sq) for sq in train_seqs])
                if scheme == "seq_shuffle":
                    neg_seqs = [ushuffle(sq, rng_data) for sq in train_seqs]
                    X_neg = np.array([fn(sq) for sq in neg_seqs])
                else:
                    X_neg = np.array([rng_data.permutation(row) for row in X_tr])
                X = np.vstack([X_tr, X_neg])
                yt = np.hstack([np.ones(len(X_tr)), np.zeros(len(X_neg))])
                model = XGBClassifier(**{**HP, "random_state": 2_000_000 + s})
                model.fit(X, yt)
                for k, negm in neg_fa.items():
                    p = model.predict_proba(np.vstack([test_mats[vname]["pos"], test_mats[vname][f"neg_{k}"]]))[:, 1]
                    aucs[k].append(roc_auc_score(np.r_[y_pos, np.zeros(len(negm))], p))
                    ens[k].append(p)
            row = {"scheme": scheme, "variant": vname, "dim": X_tr.shape[1]}
            ens_preds_all[(scheme, vname)] = {k: np.mean(np.array(ens[k]), axis=0) for k in neg_fa}
            for k in neg_fa:
                ens_p = np.mean(np.array(ens[k]), axis=0)
                a = roc_auc_score(np.r_[y_pos, np.zeros(len(neg_fa[k]))], ens_p)
                row[f"AUC_{k}"] = round(a, 4)
                row[f"AUC_{k}_sd"] = round(float(np.std(aucs[k])), 4)
            row["Δ_GC"] = round(row["AUC_cds"] - row["AUC_gc30"], 4)
            results.append(row)
            print(f"{scheme:<12} {vname:<24} " + " ".join(f"{k}={row[f'AUC_{k}']:.4f}±{row[f'AUC_{k}_sd']:.4f}" for k in neg_fa) + f" Δ={row['Δ_GC']:+.4f}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "variants_summary.tsv", sep="\t", index=False)

    # Null: intra-GC table permutation (seq_shuffle scheme, gc30 primary)
    print("\n=== CONTROL: tabla SantaLucia permutada intra-clase GC (nulo, 100 permutaciones) ===")
    base_mats = {fn: None for fn in VARIANT_FN}
    null_aucs = []
    X_tr = np.array([feats_santa(sq) for sq in train_seqs])
    for i, tbl in enumerate(null_tables):
        rng = np.random.RandomState(1_000_000)
        neg_seqs = [ushuffle(sq, rng) for sq in train_seqs]
        X_neg = np.array([feats_santa(sq, table=tbl) for sq in neg_seqs])
        m = XGBClassifier(**{**HP, "random_state": 2_000_000})
        m.fit(np.vstack([X_tr, X_neg]), np.hstack([np.ones(len(X_tr)), np.zeros(len(X_neg))]))
        p = m.predict_proba(np.vstack([test_mats["SantaLucia (79d)"]["pos"], test_mats["SantaLucia (79d)"]["neg_gc30"]]))[:, 1]
        null_aucs.append(roc_auc_score(np.r_[y_pos, np.zeros(len(neg_fa["gc30"]))], p))
    null_aucs = np.array(null_aucs)
    real = df[(df["scheme"] == "seq_shuffle") & (df["variant"] == "SantaLucia (79d)")]["AUC_gc30"].iloc[0]
    print(f"nulo: media={null_aucs.mean():.4f} ± {null_aucs.std():.4f} rango=[{null_aucs.min():.4f},{null_aucs.max():.4f}]")
    print(f"real: {real:.4f}  |  percentil del real en nulo: {(null_aucs < real).mean() * 100:.1f}%")
    np.save(OUT / "null_table_aucs.npy", null_aucs)

    # Additive: dinuc OHE + SantaLucia vs dinuc OHE (seq_shuffle, gc30)
    print("\n=== TEST ADITIVO: dinuc posicional + SantaLucia vs dinuc posicional (gc30) ===")
    ens_base = []; ens_add = []
    for s in range(N_SEEDS):
        rng = np.random.RandomState(1_000_000 + s)
        neg_seqs = [ushuffle(sq, rng) for sq in train_seqs]
        X_b = np.vstack([np.array([feats_dinuc_pos(sq) for sq in train_seqs]),
                         np.array([feats_dinuc_pos(sq) for sq in neg_seqs])])
        X_a = np.vstack([np.array([np.r_[feats_dinuc_pos(sq), feats_santa(sq)] for sq in train_seqs]),
                         np.array([np.r_[feats_dinuc_pos(sq), feats_santa(sq)] for sq in neg_seqs])])
        yt = np.hstack([np.ones(len(train_seqs)), np.zeros(len(train_seqs))])
        mb = XGBClassifier(**{**HP, "random_state": 2_000_000 + s}).fit(X_b, yt)
        ma = XGBClassifier(**{**HP, "random_state": 2_000_000 + s}).fit(X_a, yt)
        Xp = np.array([feats_dinuc_pos(sq) for sq in pos_fa]); Xn = np.array([feats_dinuc_pos(sq) for sq in neg_fa["gc30"]])
        Xpa = np.array([np.r_[feats_dinuc_pos(sq), feats_santa(sq)] for sq in pos_fa]); Xna = np.array([np.r_[feats_dinuc_pos(sq), feats_santa(sq)] for sq in neg_fa["gc30"]])
        ens_base.append(mb.predict_proba(np.vstack([Xp, Xn]))[:, 1])
        ens_add.append(ma.predict_proba(np.vstack([Xpa, Xna]))[:, 1])
    pb = np.mean(ens_base, axis=0); pa = np.mean(ens_add, axis=0)
    yt = np.r_[y_pos, np.zeros(len(neg_fa["gc30"]))]
    d, se, pval = delong_full(yt, pa, pb)
    print(f"dinuc: {roc_auc_score(yt, pb):.4f} | +SantaLucia: {roc_auc_score(yt, pa):.4f} | Δ={d:+.4f} (SE={se:.4f}, p={pval:.4f})")

    # DeLong matrix on ensemble preds (seq_shuffle, gc30)
    print("\n=== DeLong pareado (seq_shuffle, gc30), Holm m=6 ===")
    ens_preds = {v: ens_preds_all[("seq_shuffle", v)]["gc30"] for v in VARIANT_FN}
    yt = np.r_[y_pos, np.zeros(len(neg_fa["gc30"]))]
    names = list(VARIANT_FN)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d, se, p = delong_full(yt, ens_preds[names[i]], ens_preds[names[j]])
            pairs.append((names[i], names[j], d, se, p))
    pairs.sort(key=lambda x: x[4])
    for k, (a, b, d, se, p) in enumerate(pairs):
        pholm = min(1.0, p * (len(pairs) - k))
        print(f"  {a:<22} vs {b:<22} Δ={d:+.4f} SE={se:.4f} p={p:.4g} p_holm={pholm:.4g}")
    pd.DataFrame([{"A": a, "B": b, "dAUC": d, "SE": se, "p_raw": p} for a, b, d, se, p in pairs]).to_csv(OUT / "delong_matrix.tsv", sep="\t", index=False)
    print("\nGuardado en:", OUT)


if __name__ == "__main__":
    main()