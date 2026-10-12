#!/usr/bin/env python3
"""Build data/benchmark/annotation_conflicts.tsv: exact-duplicate positives
with differing sigma labels, documented instead of silently resolved.

Sweep: all 1727 canonical positives (D39V+TIGR4), grouped by exact 81bp
sequence (uppercased). Groups whose members carry different Sigma_Factor
values (None = missing label) are reported with:
  - conflict_type: SIGA_VS_SIGX (genuine sigma-vs-sigma) or
    LABELED_VS_NONE (evidence on one side, missing annotation on the other).
  - dedup_kept_id: survivor under the shared keep-first rule (D39V block
    precedes TIGR4 in positives_81bp.fasta), i.e. what 16/17/18 and the default balanced set use.
  - cross-strain note: downstream genes are ortholog pairs (conserved promoter).

Frozen record: data/benchmark/annotation_conflicts.tsv holds the UNRESOLVED
conflicts only (C1/C5 transferred to SigA are omitted, not commented).
Post-transfer reruns of this script must reproduce the same rows.

Usage:
    pixi run python src/dataset/build_annotation_conflicts.py [-o data/benchmark/annotation_conflicts.tsv]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent
CANON = ROOT / "data/benchmark/datasets/2_d39v_tigr4_high_sigma"


def norm(v):
    return "None" if pd.isna(v) else str(v)


# Original pre-transfer numbering (stable IDs; C1/C5 resolved by transfer
# and omitted from output, never renumbered).
ORIGINAL_IDS = {
    "C1": {"TSS_5_D39V_14240_+_3048_SigA", "TSS_878_D39V_1803628_-_3112_SigA",
           "TIGR4_SP_0014_TSS_14436_+_101_None"},
    "C2": {"TSS_49_D39V_88197_+_curated_None", "TIGR4_SP_0090_TSS_92933_+_158_SigA"},
    "C3": {"TSS_120_D39V_220835_+_curated_None", "TIGR4_SP_0248_TSS_217861_+_282_SigA"},
    "C4": {"TSS_242_D39V_486042_+_978_SigA", "TIGR4_SP_0549_TSS_519266_+_7196_ComX"},
    "C5": {"TSS_459_D39V_961366_-_1851_SigA", "TIGR4_SP_1110_TSS_1039154_+_1869_None"},
    "C6": {"TSS_627_D39V_1329764_-_95_SigX", "TIGR4_SP_1480_TSS_1394147_-_7754_None"},
    "C7": {"TSS_730_D39V_1507197_-_curated_None", "TIGR4_SP_1676_TSS_1577989_-_323_SigA"},
}
RESOLVED = {"C1", "C5"}


def main():
    ap = argparse.ArgumentParser(description="Document cross-strain annotation conflicts.")
    ap.add_argument("-o", "--output", default="data/benchmark/annotation_conflicts.tsv")
    args = ap.parse_args()

    meta = pd.read_csv(CANON / "positives_81bp_metadata.tsv",
                       sep="\t", dtype=str).set_index("Sequence_ID")
    groups = defaultdict(list)
    for r in SeqIO.parse(CANON / "positives_81bp.fasta", "fasta"):
        s = str(r.seq).upper()
        if r.id not in groups[s]:
            groups[s].append(r.id)
    file_order = [r.id for r in SeqIO.parse(CANON / "positives_81bp.fasta", "fasta")]
    rank = {i: k for k, i in enumerate(file_order)}

    rows = []
    for s, ids in sorted(groups.items(), key=lambda kv: min(rank[i] for i in kv[1])):
        if len(ids) < 2:
            continue
        sigs = {norm(meta.loc[i, "Sigma_Factor"]) for i in ids}
        if len(sigs) < 2:
            continue
        cid = next((c for c, members in ORIGINAL_IDS.items()
                    if set(members) == set(ids)), None)
        if cid is None or cid in RESOLVED:
            continue  # resolved conflicts are omitted, not commented
        fam = {"SigX" if v in ("SigX", "ComX") else v for v in sigs}
        if len(fam) > 1 and "None" not in fam:
            ctype = "SIGA_VS_SIGX"
        else:
            ctype = "LABELED_VS_NONE"
        kept = min(ids, key=lambda i: rank[i])
        strains = sorted({meta.loc[i, "Chromosome"] for i in ids})
        rows.append({
            "conflict_id": cid,
            "n_members": len(ids),
            "member_ids": ";".join(ids),
            "strains": "+".join(strains),
            "sigma_labels": ";".join(f"{i}={norm(meta.loc[i, 'Sigma_Factor'])}" for i in ids),
            "regulators": ";".join(f"{i}={norm(meta.loc[i, 'Associated_Regulator'])}" for i in ids),
            "downstream_genes": ";".join(f"{i}={norm(meta.loc[i, 'Downstream_Gene'])}" for i in ids),
            "conflict_type": ctype,
            "dedup_kept_id": kept,
            "sequence": s,
        })
    out = ROOT / args.output
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    types = pd.Series([r["conflict_type"] for r in rows]).value_counts().to_dict()
    print(f"WROTE {out}: {len(rows)} conflicts {types}")


if __name__ == "__main__":
    main()
