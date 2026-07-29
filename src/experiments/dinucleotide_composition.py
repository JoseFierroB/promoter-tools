#!/usr/bin/env python3
"""
Dinucleotide composition by position — promoter sequences only.
Compares Firmicutes vs non-Firmicutes using SantaLucia stability features.
"""

import numpy as np
from pathlib import Path
from Bio import SeqIO
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

# ── dinucleotides ──
DINUCS = [a + b for a in 'ACGT' for b in 'ACGT']
DINUCS_REV = sorted(set(''.join(sorted(d)) for d in DINUCS))  # 10 pairs

NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# ── paths ──
DATA_DIR = Path("tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences")
POS_81BP = "data/benchmark/positives_81bp.fasta"
OUT_DIR = Path("output/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def dinucleotide_composition(seqs, length=80):
    """Compute dinucleotide frequency matrix (length positions × 16 dinucleotides).
    length = number of dinucleotide positions (i.e., sequence length - 1)."""
    mat = np.zeros((length, len(DINUCS)), dtype=np.float64)
    for s in seqs:
        s = s.upper()
        for i in range(length):
            d = s[i:i+2]
            if d in DINUCS:
                mat[i, DINUCS.index(d)] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums

def group_species(files, length=80):
    """Group external species by phylum. Extract TSS-centered window -60 to +19 from 100bp data."""
    firmicutes = ['B001']
    firm_pos, other_pos = [], []
    firm_names, other_names = [], []
    for f in sorted(files):
        name = f.stem.replace("Sequences_80-20_", "")
        seqs = []
        for l in open(f):
            s = l.strip().upper()
            if len(s) >= 100:
                seqs.append(s[20:100])  # -60 to +19 (80bp)
        if not seqs:
            continue
        mat = dinucleotide_composition(seqs)
        if name in firmicutes:
            firm_pos.append(mat)
            firm_names.append(name)
        else:
            other_pos.append(mat)
            other_names.append(name)
    firm_avg = np.mean(firm_pos, axis=0) if firm_pos else None
    other_avg = np.mean(other_pos, axis=0) if other_pos else None
    return firm_avg, other_avg, firm_names, other_names

# ═══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  DINUCLEOTIDE COMPOSITION BY POSITION")
print("=" * 70)

# S. pneumoniae D39V positives (80bp: -60 to +19, matches external slice)
spn_recs = list(SeqIO.parse(POS_81BP, "fasta"))
spn_seqs = [str(r.seq).upper()[:80] for r in spn_recs]
spn_dinu = dinucleotide_composition(spn_seqs)
print(f"\n  S. pneumoniae D39V: {len(spn_seqs)} positive sequences (80bp, -60 to +19)")

# External species
ext_files = sorted(DATA_DIR.glob("*.txt"))
if not ext_files:
    print(f"  [WARN] No external data found at {DATA_DIR}")
    print("  Using only S. pneumoniae data")
    firm_avg = None
    other_avg = None
    firm_names = []
    other_names = []
else:
    firm_avg, other_avg, firm_names, other_names = group_species(ext_files)
    print(f"  External Firmicutes: {firm_names}")
    print(f"  External non-Firmicutes: {other_names}")

offset = np.arange(-60, 20)  # 80 bp: -60 to +19

# ═══════════════════════════════════════════════════════════
# 2. PLOT: Dinucleotide frequency heatmaps
# ═══════════════════════════════════════════════════════════
print("\n  Generating plots...")

# Sort dinucleotides by GC content for meaningful ordering
dinuc_gc = [d for d in DINUCS if d[0] in 'GC' or d[1] in 'GC']
dinuc_at = [d for d in DINUCS if d not in dinuc_gc]
dinuc_ordered = dinuc_gc + dinuc_at

spn_dinu_ordered = spn_dinu[:, [DINUCS.index(d) for d in dinuc_ordered]]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, dpi=200)

# Panel A: S. pneumoniae absolute frequencies
extent = [offset[0], offset[-1], len(dinuc_ordered) - 0.5, -0.5]
im = axes[0].imshow(spn_dinu_ordered.T, aspect='auto', cmap='YlOrRd',
                     vmin=0, vmax=0.3, extent=extent)
axes[0].set_yticks(range(len(dinuc_ordered)))
axes[0].set_yticklabels(dinuc_ordered, fontsize=7)
axes[0].set_ylabel('Dinucleotide')
axes[0].set_title('S. pneumoniae D39V — Dinucleotide Frequency by Position', fontsize=11)
axes[0].axvline(0, color='navy', ls='--', lw=0.8, alpha=0.5)
axes[0].text(0, -1.2, 'TSS', fontsize=8, ha='center', color='navy')
plt.colorbar(im, ax=axes[0], label='Frequency', shrink=0.8)

# Panel B: Firmicutes vs Others difference (if external data available)
if firm_avg is not None and other_avg is not None:
    spn_firm = (spn_dinu + firm_avg) / 2  # combine both Firmicutes
    diff = spn_firm - other_avg
    diff_ordered = diff[:, [DINUCS.index(d) for d in dinuc_ordered]]

    vmax = max(abs(diff_ordered.min()), abs(diff_ordered.max()))
    im2 = axes[1].imshow(diff_ordered.T, aspect='auto', cmap='RdBu_r',
                          norm=TwoSlopeNorm(0, vmin=-vmax, vmax=vmax),
                          extent=[offset[0], offset[-1], len(dinuc_ordered) - 0.5, -0.5])
    axes[1].set_yticks(range(len(dinuc_ordered)))
    axes[1].set_yticklabels(dinuc_ordered, fontsize=7)
    axes[1].set_xlabel('Position relative to TSS (bp)')
    axes[1].set_ylabel('Dinucleotide')
    axes[1].set_title('Difference: Firmicutes (S. pneumoniae + B. amyloliquefaciens) − Other Bacteria', fontsize=11)
    axes[1].axvline(0, color='navy', ls='--', lw=0.8, alpha=0.5)
    axes[1].text(0, -1.2, 'TSS', fontsize=8, ha='center', color='navy')
    plt.colorbar(im2, ax=axes[1], label='Δ Frequency', shrink=0.8)
else:
    axes[1].text(0.5, 0.5, 'Firmicutes vs Others comparison:\nrequires external MLDSPP data',
                 ha='center', va='center', transform=axes[1].transAxes, fontsize=10, style='italic')
    axes[1].set_xlabel('Position relative to TSS (bp)')

plt.tight_layout()
out1 = OUT_DIR / "dinucleotide_composition_heatmap.png"
fig.savefig(out1, dpi=200)
plt.close()
print(f"  [OK] {out1}")

# ═══════════════════════════════════════════════════════════
# 3. PLOT: Positional frequency profiles for key dinucleotides
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

# Select key dinucleotides
key_gc = ['CG', 'GC']
key_at = ['AT', 'TA']
highlight = key_gc + key_at
colors = ['#D95319', '#E67E22', '#0072B2', '#56B4E9']
labels = ['CG (most stable)', 'GC (most stable)', 'AT (least stable)', 'TA (least stable)']

for d, c, lab in zip(highlight, colors, labels):
    idx = DINUCS.index(d)
    ax.plot(offset, spn_dinu[:, idx], color=c, lw=1.8, label=f'{lab} — S. pneumoniae')

    if firm_avg is not None:
        ax.plot(offset[:firm_avg.shape[0]], firm_avg[:, idx],
                color=c, lw=0.8, ls='--', alpha=0.5)
    if other_avg is not None:
        ax.plot(offset[:other_avg.shape[0]], other_avg[:, idx],
                color=c, lw=0.8, ls=':', alpha=0.5)

# Add a dashed line for TSS and -10 box
ax.axvline(0, color='navy', ls='--', lw=0.8)
ax.annotate('TSS', (0, ax.get_ylim()[1]), fontsize=8, ha='center', color='navy', xytext=(0, 0.005),
            textcoords='offset points')
ax.axvspan(-15, -5, color='red', alpha=0.04)
ax.axvline(-10, color='red', ls=':', lw=0.6)
ax.text(-10, ax.get_ylim()[1], '-10 box', fontsize=7, ha='center', color='red')

ax.set_xlabel('Position relative to TSS (bp)')
ax.set_ylabel('Dinucleotide frequency')
ax.set_title('Positional Dinucleotide Frequency — Promoter Sequences', fontsize=11)
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

out2 = OUT_DIR / "dinucleotide_positional_frequencies.png"
fig.savefig(out2, dpi=200)
plt.close()
print(f"  [OK] {out2}")

# ═══════════════════════════════════════════════════════════
# 4. PLOT: GC/AT ratio by position
# ═══════════════════════════════════════════════════════════
gc_dinucs = [d for d in DINUCS if 'G' in d or 'C' in d]
gc_idxs = [DINUCS.index(d) for d in gc_dinucs]

spn_gc = spn_dinu[:, gc_idxs].sum(axis=1)
if firm_avg is not None:
    firm_gc = firm_avg[:, gc_idxs].sum(axis=1)
if other_avg is not None:
    other_gc = other_avg[:, gc_idxs].sum(axis=1)

fig, ax = plt.subplots(figsize=(12, 5), dpi=200)

ax.plot(offset, spn_gc, lw=2, color='#002D62', label='S. pneumoniae D39V (Firmicutes)')
if firm_avg is not None:
    ax.plot(offset[:firm_avg.shape[0]], firm_gc, lw=1, color='#D95319', ls='--', label='B. amyloliquefaciens (Firmicutes)')
if other_avg is not None:
    ax.plot(offset[:other_avg.shape[0]], other_gc, lw=0.8, color='#999999', label='Other bacteria (n=11)')

ax.axvline(0, color='navy', ls='--', lw=0.8)
ax.axvspan(-15, -5, color='red', alpha=0.04)
ax.axvline(-10, color='red', ls=':', lw=0.6)
ax.text(-10, ax.get_ylim()[1], '-10 box', fontsize=7, ha='center', color='red')

ax.set_xlabel('Position relative to TSS (bp)')
ax.set_ylabel('Fraction of dinucleotides containing G or C')
ax.set_title('GC-Containing Dinucleotide Frequency by Position — Positive Promoters Only', fontsize=11)
ax.legend(frameon=False, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

out3 = OUT_DIR / "dinucleotide_gc_ratio_by_position.png"
fig.savefig(out3, dpi=200)
plt.close()
print(f"  [OK] {out3}")

print(f"\n  Plots saved to {OUT_DIR.resolve()}")
print("=" * 70)
