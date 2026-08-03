#!/usr/bin/env python3
"""
Generate all MEME plots: logos, optimization, position, genome scan.
Usage: pixi run --manifest-path tools/meme/pixi.toml python src/experiments/meme_all_plots.py
"""
import subprocess, tempfile, shutil, csv, math, re, random, time
from pathlib import Path
import numpy as np
from Bio import SeqIO
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
POS = ROOT / "data/benchmark/positives_81bp.fasta"
NEG = ROOT / "data/benchmark/negatives_81bp.fasta"
PLOT_DIR = ROOT / "output" / "plots" / "meme"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

pos = list(SeqIO.parse(POS, "fasta"))
neg = list(SeqIO.parse(NEG, "fasta"))
print(f"D39V: {len(pos)} pos + {len(neg)} neg")
print()

# 1. STREME motif discovery
print("1. STREME motif discovery...")
tmpdir = Path(tempfile.mkdtemp(prefix="meme_plots_"))
subprocess.run(["streme", "-oc", str(tmpdir/"streme"), "-dna", "-minw", "10", "-maxw", "20",
    "-p", str(POS), "-n", str(NEG)], capture_output=True, timeout=120)

with open(tmpdir/"streme"/"streme.txt") as f:
    streme_txt = f.read()

motifs = []
for m in re.finditer(r"MOTIF (\S+) STREME-\d+", streme_txt):
    chunk = streme_txt[m.start():m.start()+500]
    e = re.search(r"E=\s*([\d\.e\-\+]+)", chunk)
    ns = re.search(r"nsites=\s*(\d+)", chunk)
    w = re.search(r"w=\s*(\d+)", chunk)
    motifs.append((m.group(1), int(w.group(1)), int(ns.group(1)), e.group(1)))

print(f"  Found {len(motifs)} motifs")
for c, w, ns, e in motifs:
    print(f"    {c:<35} w={w:>2}  sites={ns:>3}/988  E={e}")

# 2. FIMO position distribution of Motif 1
print("\n2. Motif position distribution...")
combined = tmpdir / "pos.fa"
with open(combined, "w") as f:
    for r in pos: SeqIO.write(r, f, "fasta")

res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
    str(tmpdir/"streme"/"streme.txt"), str(combined)],
    capture_output=True, text=True, timeout=120)

m1_positions = []
for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
    if row.get("motif_id", "").startswith("1-"):
        m1_positions.append(int(row["start"]))

fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
ax.hist(m1_positions, bins=81, color="#002D62", alpha=0.8, edgecolor="white")
ax.axvline(60, color="black", ls="--", lw=1, alpha=0.5)
ax.text(60, ax.get_ylim()[1]*0.9, "TSS", ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Position in 81bp sequence"); ax.set_ylabel("Motif hits")
ax.set_title(f"Extended -10 Position ({len(m1_positions)} hits)")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "meme_motif_position.png", dpi=300)
plt.savefig(PLOT_DIR / "meme_motif_position.svg", dpi=300)
plt.close()
print(f"  {PLOT_DIR}/meme_motif_position.{{svg,png}}")

# 3. AUC vs Motif Width optimization
print("\n3. Width optimization...")
widths = [(6,9), (6,12), (8,14), (10,20), (10,30)]
results = []

for minw, maxw in widths:
    random.shuffle(pos); random.shuffle(neg)
    mp, mn = len(pos)//2, len(neg)//2
    scores = {}
    for fold in range(2):
        tp = pos[:mp] if fold==0 else pos[mp:]
        tn = neg[:mn] if fold==0 else neg[mn:]
        tpos = pos[mp:] if fold==0 else pos[:mp]
        tneg = neg[mn:] if fold==0 else neg[:mn]
        
        td = Path(tempfile.mkdtemp(prefix="wopt_"))
        SeqIO.write(tp, td/"tp.fa", "fasta"); SeqIO.write(tn, td/"tn.fa", "fasta")
        with open(td/"test.fa", "w") as f:
            for r in tpos: SeqIO.write(r, f, "fasta")
            for r in tneg: SeqIO.write(r, f, "fasta")
        
        subprocess.run(["streme", "-oc", str(td/"s"), "-dna",
            "--minw", str(minw), "--maxw", str(maxw),
            "-p", str(td/"tp.fa"), "-n", str(td/"tn.fa")],
            capture_output=True, timeout=120)
        res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
            str(td/"s"/"streme.txt"), str(td/"test.fa")],
            capture_output=True, text=True, timeout=120)
        
        for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
            try: pv = float(row["p-value"])
            except (ValueError, KeyError, TypeError): continue
            nl = 999.0 if pv <= 0 else -math.log10(pv)
            s = row["sequence_name"]
            if s not in scores or nl > scores[s]: scores[s] = nl
        shutil.rmtree(td)
    
    for r in pos+neg:
        if r.id not in scores: scores[r.id] = 0.0
    y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
    sc = np.array([scores[r.id] for r in pos+neg])
    results.append((f"{minw}-{maxw}", roc_auc_score(y, sc)))

fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
labels = [r[0] for r in results]; aucs = [r[1] for r in results]
ax.bar(range(len(aucs)), aucs, color="#002D62")
ax.set_xticks(range(len(aucs))); ax.set_xticklabels(labels)
ax.set_ylabel("AUC"); ax.set_xlabel("Motif width (min-max)")
ax.set_title("MEME AUC vs STREME Width")
ax.axhline(0.5, color="black", ls=":", alpha=0.3)
for i, a in enumerate(aucs):
    ax.text(i, a+0.01, f"{a:.3f}", ha="center", fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR / "meme_optimization.png", dpi=300)
plt.savefig(PLOT_DIR / "meme_optimization.svg", dpi=300)
plt.close()
print(f"  {PLOT_DIR}/meme_optimization.{{svg,png}}")
print(f"  Best: {results[np.argmax(aucs)][0]} (AUC={max(aucs):.4f})")

# 4. Sequence logos via ceqlogo
print("\n4. Sequence logos...")
for i in range(1, min(7, len(motifs)+1)):
    eps = PLOT_DIR / f"meme_logo{i}.eps"
    png = PLOT_DIR / f"meme_logo{i}.png"
    subprocess.run(["ceqlogo", f"-i{i}", str(tmpdir/"streme"/"streme.txt"),
        "-o", str(eps), "-f", "EPS", "-w", "12", "-h", "6", "-S"],
        capture_output=True, timeout=30)
    if eps.exists():
        subprocess.run(["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
            "-r300", "-dEPSCrop", f"-sOutputFile={png}", str(eps)],
            capture_output=True, timeout=30)
        print(f"  meme_logo{i}.png ✓")

# 4b. Combined logo montage
logo_pngs = sorted(PLOT_DIR.glob("meme_logo[0-9].png"))
if logo_pngs:
    from matplotlib import image as mpimg
    n = len(logo_pngs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.5), dpi=300)
    if rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, png in zip(axes, logo_pngs):
        img = mpimg.imread(str(png))
        ax.imshow(img)
        ax.set_title(png.stem, fontsize=9)
        ax.axis("off")
    for ax in axes[len(logo_pngs):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "meme_all_logos.png", dpi=300, bbox_inches="tight")
    plt.savefig(PLOT_DIR / "meme_all_logos.svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  meme_all_logos.{{png,svg}} ✓")

# 5. Genome-wide scan
print("\n5. Genome-wide scan...")
genome_ref = ROOT / "data/reference/D39V.fna"
if genome_ref.exists():
    genome = list(SeqIO.parse(genome_ref, "fasta"))[0]
    chrom_seq = str(genome.seq).upper()
    windows = [chrom_seq[i:i+81] for i in range(0, min(100000, len(chrom_seq)-81), 40)]
    
    genome_fa = tmpdir / "genome.fa"
    with open(genome_fa, "w") as f:
        for i, s in enumerate(windows):
            f.write(f">window_{i*40}\n{s}\n")
    
    res = subprocess.run(["fimo", "--text", "--skip-matched-sequence",
        str(tmpdir/"streme"/"streme.txt"), str(genome_fa)],
        capture_output=True, text=True, timeout=120)
    
    positions = []; scores_list = []
    for row in csv.DictReader(res.stdout.splitlines(), delimiter="\t"):
        try: pv = float(row["p-value"])
        except (ValueError, KeyError, TypeError): continue
        nl = 999.0 if pv <= 0 else -math.log10(pv)
        start = int(row["sequence_name"].split("_")[1])
        positions.append(start); scores_list.append(nl)
    
    if positions:
        fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
        ax.scatter(np.array(positions)/1e6, scores_list, s=1, alpha=0.3, color="#002D62")
        ax.set_xlabel("Genomic position (Mb)"); ax.set_ylabel("MEME score")
        ax.set_title(f"Genome-wide MEME Scan (first 100K bp, {len(positions)} hits)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "meme_genome_scan.png", dpi=200)
        plt.close()
        print(f"  {PLOT_DIR}/meme_genome_scan.png")

shutil.rmtree(tmpdir)
print(f"\n═══ DONE ═══")
print(f"All plots in: {PLOT_DIR}")
