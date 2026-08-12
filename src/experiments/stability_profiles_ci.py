#!/usr/bin/env python3
"""
Energy profile with 95% CI bands — Firmicutes vs Other bacteria.
Recreates stability_profiles_ci.svg from per-sequence stability values.
"""

import numpy as np
from pathlib import Path
from Bio import SeqIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── SantaLucia stability parameters (kcal/mol) ──
STABILITY = {
    'AA': -1.00, 'TT': -1.00, 'AT': -0.88, 'TA': -0.58,
    'AG': -1.30, 'GA': -1.30, 'AC': -1.45, 'CA': -1.45,
    'TG': -1.44, 'GT': -1.44, 'TC': -1.28, 'CT': -1.28,
    'CC': -1.84, 'GG': -1.84, 'CG': -2.24, 'GC': -2.27,
}

DATA_DIR = Path("tools/MLDSPP-Promoter-prediction/Sample Dataset/Promoter Sequences")
POS_81BP = "data/benchmark/d39v/positives_81bp.fasta"
OUT_DIR = Path("output/plots/stability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def stability_profile(seqs, window=80):
    """Per-sequence SantaLucia stability. Returns (n_seq, window) array."""
    mat = np.zeros((len(seqs), window), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j in range(window):
            mat[i, j] = STABILITY.get(s[j:j+2], -1.35)
    return mat

def bootstrap_ci(data, n_iter=2000, alpha=0.05):
    """Bootstrap 95% CI for each position. data: (n_seq, n_pos)."""
    n = data.shape[0]
    means = np.zeros((n_iter, data.shape[1]))
    rng = np.random.RandomState(42)
    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        means[i] = data[idx].mean(axis=0)
    lo = np.percentile(means, 100 * alpha / 2, axis=0)
    hi = np.percentile(means, 100 * (1 - alpha / 2), axis=0)
    return data.mean(axis=0), lo, hi

# ═══════════════════════════════════════════════════════════
# 1. LOAD & COMPUTE
# ═══════════════════════════════════════════════════════════
print("Loading data...")

# S. pneumoniae D39V — 81bp, extract -60 to +19 (80bp)
spn = [str(r.seq).upper()[:80] for r in SeqIO.parse(POS_81BP, "fasta")]
spn_stab = stability_profile(spn)
spn_mean, spn_lo, spn_hi = bootstrap_ci(spn_stab)
print(f"  S. pneumoniae D39V: {len(spn)} sequences")

# External species — 100bp, extract -60 to +19 (80bp)
ext_files = sorted(DATA_DIR.glob("*.txt"))
firm_profiles, other_profiles = [], []
firm_names, other_names = [], []
for f in ext_files:
    name = f.stem.replace("Sequences_80-20_", "")
    seqs = []
    for l in open(f):
        s = l.strip().upper()
        if len(s) >= 100:
            seqs.append(s[20:100])
    if not seqs:
        continue
    profiles = stability_profile(seqs)
    if name == "B001":  # B. amyloliquefaciens
        firm_profiles.append(profiles)
        firm_names.append(name)
    else:
        other_profiles.append(profiles)
        other_names.append(name)

# Combine Firmicutes (B. amylo + S. pneumoniae)
all_firm = np.vstack([spn_stab] + firm_profiles) if firm_profiles else spn_stab
bamylo_stab = firm_profiles[0] if firm_profiles else np.array([])

# Combine all other bacteria
all_other = np.vstack(other_profiles) if other_profiles else np.array([])

firm_mean, firm_lo, firm_hi = bootstrap_ci(all_firm)
other_mean, other_lo, other_hi = bootstrap_ci(all_other)
bamylo_mean, bamylo_lo, bamylo_hi = bootstrap_ci(bamylo_stab)

# Also get the non-Firmicutes from the "other" group that are most different (E. coli-like)
# B0010 = E. coli MG1655
ecoli_stab = None
for f in ext_files:
    name = f.stem.replace("Sequences_80-20_", "")
    if name == "B0010":  # E. coli
        seqs = [l.strip().upper()[20:100] for l in open(f) if len(l.strip()) >= 100]
        ecoli_stab = stability_profile(seqs)
        break

offset = np.arange(-60, 20)

# ═══════════════════════════════════════════════════════════
# 2. PLOT
# ═══════════════════════════════════════════════════════════
print("Plotting...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

# Firmicutes other than spn (B. amylo)
ax.plot(offset, bamylo_mean, color='#D95319', lw=2.5,
        label=f'B. amyloliquefaciens (Firmicutes, n={bamylo_stab.shape[0]})')
ax.fill_between(offset, bamylo_lo, bamylo_hi, color='#D95319', alpha=0.12)

# S. pneumoniae (Firmicutes)
ax.plot(offset, spn_mean, color='#002D62', lw=2.5,
        label=f'S. pneumoniae D39V (Firmicutes, n={spn_stab.shape[0]})')
ax.fill_between(offset, spn_lo, spn_hi, color='#002D62', alpha=0.12)

# Combined Firmicutes CI
ax.plot(offset, firm_mean, color='#7A2E0D', lw=0.8, ls='--', alpha=0.4)
ax.fill_between(offset, firm_lo, firm_hi, color='#7A2E0D', alpha=0.05)

# Other bacteria
ax.plot(offset, other_mean, color='#666666', lw=2.0,
        label=f'Other bacteria (n={all_other.shape[0]})')
ax.fill_between(offset, other_lo, other_hi, color='#666666', alpha=0.12)

# E. coli specific
if ecoli_stab is not None:
    e_mean, e_lo, e_hi = bootstrap_ci(ecoli_stab)
    ax.plot(offset, e_mean, color='#228B22', lw=1.5, ls='-.',
            label=f'E. coli MG1655 (n={ecoli_stab.shape[0]})')
    ax.fill_between(offset, e_lo, e_hi, color='#228B22', alpha=0.08)

# Annotations
ymin, ymax = ax.get_ylim()
ax.axvspan(-15, -5, color='red', alpha=0.05)
ax.axvline(-10, color='red', ls=':', lw=0.8, alpha=0.5)
ax.annotate('-10 box', (-10, ymax), fontsize=8, color='red', ha='center',
            xytext=(0, 5), textcoords='offset points')

# Mark minimum positions
for name, arr, color in [
    ('B. amylo', bamylo_mean, '#D95319'),
    ('S. pneumo', spn_mean, '#002D62'),
]:
    min_idx = np.argmin(arr)
    ax.scatter(offset[min_idx], arr[min_idx], color=color, s=30, zorder=5)
    ax.annotate(f'{name} min\n({offset[min_idx]:+d})',
                (offset[min_idx], arr[min_idx]),
                fontsize=7, color=color, ha='center',
                xytext=(0, -18), textcoords='offset points')

# Other bacteria min
min_idx = np.argmin(other_mean)
ax.scatter(offset[min_idx], other_mean[min_idx], color='#666666', s=20, zorder=5)
ax.annotate(f'Others min\n({offset[min_idx]:+d})',
            (offset[min_idx], other_mean[min_idx]),
            fontsize=7, color='#666666', ha='center',
            xytext=(0, 12), textcoords='offset points')

ax.set_xlabel('Position relative to TSS (bp)')
ax.set_ylabel('Mean SantaLucia stability (kcal/mol)')
ax.set_title('DNA Stability Profiles with 95% CI — Positive Promoters Only')
ax.legend(frameon=False, fontsize=7, loc='lower right')
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out_svg = OUT_DIR / "stability_profiles_ci.svg"
out_png = OUT_DIR / "stability_profiles_ci.png"
fig.savefig(out_svg, dpi=200)
fig.savefig(out_png, dpi=200)
plt.close()

print(f"  [OK] {out_svg}")
print(f"  [OK] {out_png}")

# ═══════════════════════════════════════════════════════════
# 3. SUMMARY STATS
# ═══════════════════════════════════════════════════════════
print("\nMinimum positions:")
for name, arr in [
    ('B. amyloliquefaciens', bamylo_mean),
    ('S. pneumoniae', spn_mean),
    ('Others', other_mean),
]:
    if arr is not None and len(arr) > 0:
        min_idx = np.argmin(arr)
        print(f"  {name:>25}: {arr[min_idx]:.3f} at offset {offset[min_idx]:+d}")
if ecoli_stab is not None:
    min_idx = np.argmin(e_mean)
    print(f"  {'E. coli MG1655':>25}: {e_mean[min_idx]:.3f} at offset {offset[min_idx]:+d}")

print(f"\nCombined dataset sizes:")
print(f"  Firmicutes (combined): n={all_firm.shape[0]}")
print(f"  Other bacteria:        n={all_other.shape[0]}")
print("Done.")
