#!/usr/bin/env python3
"""Extract 100% identical D39V-TIGR4 IGR pairs with full metadata."""
import csv
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

M8 = ROOT / "output/intergenic/mmseqs2/cross/D39V_vs_TIGR4.m8"
D39V_TSV = ROOT / "output/intergenic/d39v/D39V_igrs.tsv"
TIGR4_TSV = ROOT / "output/intergenic/tigr4/TIGR4_igrs.tsv"

# Load IGR metadata
def load_igr(path):
    igrs = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = row["igr_id"].strip("\r")
            igrs[rid] = row
    return igrs

d39v = load_igr(D39V_TSV)
tigr4 = load_igr(TIGR4_TSV)
print(f"D39V: {len(d39v)} IGRs, TIGR4: {len(tigr4)} IGRs")

# Extract 100% pairs
rows = []
with open(M8) as f:
    for line in f:
        parts = line.strip().split("\t")
        if float(parts[2]) != 1.0:
            continue
        q, t = parts[0], parts[1]
        di = d39v.get(q, {})
        ti = tigr4.get(t, {})
        rows.append({
            "query_d39v": q,
            "target_tigr4": t,
            "alnlen": int(parts[3]),
            "mismatches": int(parts[4]),
            "qstart": parts[6],
            "qend": parts[7],
            "tstart": parts[8],
            "tend": parts[9],
            "evalue": parts[10],
            "bitscore": float(parts[11]),
            "len_d39v": di.get("length", ""),
            "orient_d39v": di.get("orientation_type", ""),
            "left_d39v": di.get("left_cds", ""),
            "right_d39v": di.get("right_cds", ""),
            "len_tigr4": ti.get("length", ""),
            "orient_tigr4": ti.get("orientation_type", ""),
            "left_tigr4": ti.get("left_cds", ""),
            "right_tigr4": ti.get("right_cds", ""),
        })

out = OUT_DIR / "identical_pairs_100.tsv"
fieldnames = list(rows[0].keys())
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)

# Stats
print(f"\n{len(rows)} pares 100% identicos")
aln_lens = [r["alnlen"] for r in rows]
unique_d39v = len(set(r["query_d39v"] for r in rows))
unique_tigr4 = len(set(r["target_tigr4"] for r in rows))
orient_match = sum(1 for r in rows
                   if r["orient_d39v"].replace("(++)","").replace("(--)","") ==
                      r["orient_tigr4"].replace("(++)","").replace("(--)",""))

print(f"D39V IGRs unicos: {unique_d39v}")
print(f"TIGR4 IGRs unicos: {unique_tigr4}")
print(f"Long. alineamiento: min={min(aln_lens)} max={max(aln_lens)} mean={sum(aln_lens)/len(aln_lens):.0f}")
print(f"Misma arquitectura genica: {orient_match}/{len(rows)} ({orient_match/len(rows)*100:.1f}%)")
print(f"\nOutput: {out}")

# Top 10
print("\n=== Top 10 por bitscore ===")
for r in sorted(rows, key=lambda x: -x["bitscore"])[:10]:
    print(f"  {r['query_d39v']} <-> {r['target_tigr4']} | {r['alnlen']}bp | "
          f"{r['bitscore']:.0f} bits | {r['orient_d39v']} / {r['orient_tigr4']} | "
          f"D39V:{r['left_d39v']}-{r['right_d39v']} | TIGR4:{r['left_tigr4']}-{r['right_tigr4']}")
