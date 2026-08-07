#!/usr/bin/env python3
"""
Visualize repetitive/multi-match IGR elements.

Generates:
  1. Sequence logo for top-N multi-match elements (MSA of all copies)
  2. Bipartite network graph D39V ↔ TIGR4 multi-matches
  3. Similarity heatmap + dendrogram of repetitive IGRs

Output: analysis_igr/outputs/plots/repeat_*.svg
"""

import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
M8 = ROOT / "output/intergenic/mmseqs2/cross/D39V_vs_TIGR4.m8"
D39V_TSV = ROOT / "output/intergenic/d39v/D39V_igrs.tsv"
TIGR4_TSV = ROOT / "output/intergenic/tigr4/TIGR4_igrs.tsv"

DNA_COLORS = {"A": "#4CAF50", "C": "#2196F3", "G": "#FF9800", "T": "#F44336",
              "a": "#4CAF50", "c": "#2196F3", "g": "#FF9800", "t": "#F44336"}
GAP_COLOR = "#EEEEEE"


def load_igr_sequences(tsv_path: Path) -> dict:
    """Return dict[igr_id] = seq_string."""
    seqs = {}
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            seqs[row["igr_id"].strip("\r")] = row["sequence"].strip("\r")
    return seqs


def load_multi_matches(m8_path: Path, min_matches: int = 2) -> dict:
    """Return dict[d39v_id] = [(tigr4_id, ident, alnlen, bitscore), ...]."""
    mm = defaultdict(list)
    with open(m8_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            q, t = parts[0], parts[1]
            mm[q].append((t, float(parts[2]), int(parts[3]), float(parts[11])))
    return {k: v for k, v in mm.items() if len(v) >= min_matches}


def star_msa(sequences: list) -> np.ndarray:
    """Star alignment using Biopython PairwiseAligner. Returns numpy array of chars."""
    if len(sequences) < 2:
        return np.array([list(s) for s in sequences])

    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1

    ref = sequences[0]
    aligned = [list(ref)]

    for seq in sequences[1:]:
        alns = aligner.align(ref, seq)
        best = alns[0]
        # Get the aligned query with gaps inserted
        aln_query = str(best).split("\n")[2] if len(str(best).split("\n")) >= 3 else seq
        # Verify length matches reference alignment
        aln_ref = str(best).split("\n")[1]
        if len(aln_query) != len(aln_ref):
            # Fallback: just pad with gaps to match
            aligned.append(list(seq))
            continue
        aligned.append(list(aln_query))

    # Ensure all same length
    max_len = max(len(a) for a in aligned)
    result = []
    for a in aligned:
        if len(a) < max_len:
            a = a + ["-"] * (max_len - len(a))
        result.append(a)
    return np.array(result)


def plot_sequence_logo(msa: np.ndarray, title: str, out_path: Path):
    """Generate sequence logo from MSA."""
    n_seq, n_pos = msa.shape
    if n_pos < 3 or n_seq < 2:
        return

    # Compute letter frequencies per position
    bases = ["A", "C", "G", "T"]
    freq = np.zeros((n_pos, 4))
    for i in range(n_pos):
        col = msa[:, i]
        total = np.sum(col != "-")
        if total > 0:
            for j, b in enumerate(bases):
                freq[i, j] = np.sum(col == b) / total

    # Information content (Shannon)
    bg = 0.25
    heights = np.zeros((n_pos, 4))
    for i in range(n_pos):
        col_freq = freq[i]
        total_h = np.log2(4)
        obs_h = sum(-p * np.log2(p) if p > 0 else 0 for p in col_freq)
        ic = max(0, total_h - obs_h)
        heights[i] = col_freq * ic

    fig, ax = plt.subplots(figsize=(max(6, n_pos * 0.3), 3))
    x = np.arange(n_pos)
    bottom = np.zeros(n_pos)
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    for j, b in enumerate(bases):
        ax.bar(x, heights[:, j], width=0.8, bottom=bottom, color=colors[j],
               edgecolor="white", lw=0.5)
        for i in range(n_pos):
            if heights[i, j] > 0.05:
                ax.text(i, bottom[i] + heights[i, j] / 2, b, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += heights[:, j]

    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(0, 2.2)
    ax.set_ylabel("Information content (bits)")
    ax.set_xlabel("Position")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(n_pos))
    ax.set_xticklabels(range(1, n_pos + 1), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Logo: {out_path.name}")


def plot_msa_matrix(msa: np.ndarray, title: str, out_path: Path):
    """Plot MSA as colored matrix."""
    n_seq, n_pos = msa.shape
    if n_seq > 100:
        msa = msa[:100]  # Cap at 100 sequences for readability

    fig, ax = plt.subplots(figsize=(max(10, n_pos * 0.15), max(4, n_seq * 0.15)))
    img = np.zeros((n_seq, n_pos, 3))
    color_map = {"A": [0.3, 0.7, 0.3], "C": [0.13, 0.55, 0.95],
                 "G": [1.0, 0.6, 0.0], "T": [0.95, 0.26, 0.21],
                 "-": [0.93, 0.93, 0.93]}
    for i in range(n_seq):
        for j in range(n_pos):
            img[i, j] = color_map.get(msa[i, j], [0.9, 0.9, 0.9])
    ax.imshow(img, aspect="auto")
    ax.set_xlabel("Position")
    ax.set_ylabel("Sequence")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  MSA: {out_path.name}")


def plot_network(multi_matches: dict, d39v_seqs: dict, out_path: Path):
    """Plot bipartite network D39V ↔ TIGR4 multi-matches."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Sort D39V IGRs by degree
    d39v_sorted = sorted(multi_matches.items(), key=lambda x: -len(x[1]))
    n_d39v = len(d39v_sorted)

    # Collect all TIGR4 targets
    tigr4_nodes = set()
    for _, targets in multi_matches.items():
        for t, _, _, _ in targets:
            tigr4_nodes.add(t)
    tigr4_list = sorted(tigr4_nodes)
    n_tigr4 = len(tigr4_list)

    # Positions: D39V on left, TIGR4 on right
    d39v_ys = np.linspace(0.9, 0.1, n_d39v)
    tigr4_ys = np.linspace(0.9, 0.1, n_tigr4)
    d39v_x, tigr4_x = 0.15, 0.85

    tigr4_idx = {t: i for i, t in enumerate(tigr4_list)}

    # Draw edges
    max_deg = max(len(t) for _, t in multi_matches.items())
    for i, (q, targets) in enumerate(d39v_sorted):
        deg = len(targets)
        alpha = max(0.1, deg / max_deg * 0.6)
        for t, _, _, bitscore in targets:
            j = tigr4_idx.get(t, 0)
            ax.plot([d39v_x, tigr4_x], [d39v_ys[i], tigr4_ys[j]],
                    color="gray", alpha=alpha, lw=0.5, zorder=1)

    # Draw D39V nodes
    sizes = [max(80, len(t) * 20) for _, t in d39v_sorted]
    ax.scatter([d39v_x] * n_d39v, d39v_ys, s=sizes, c="#2196F3", edgecolors="black",
               linewidth=0.5, zorder=2)
    for i, (q, targets) in enumerate(d39v_sorted):
        deg = len(targets)
        if deg >= 10:
            short = q.replace("IGR_D39V_", "")
            ax.annotate(f"{short} ({deg})", (d39v_x, d39v_ys[i]),
                        xytext=(-60, 0), textcoords="offset points",
                        fontsize=7 if deg < 20 else 8, ha="right", va="center")

    # Draw TIGR4 nodes
    tigr4_deg = defaultdict(int)
    for _, targets in multi_matches.items():
        for t, _, _, _ in targets:
            tigr4_deg[t] += 1
    tigr4_sizes = [max(40, tigr4_deg.get(t, 1) * 15) for t in tigr4_list]
    ax.scatter([tigr4_x] * n_tigr4, tigr4_ys, s=tigr4_sizes, c="#F44336",
               edgecolors="black", linewidth=0.5, zorder=2)
    for t in tigr4_list:
        if tigr4_deg.get(t, 0) >= 5:
            j = tigr4_idx[t]
            short = t.replace("IGR_NC_003028.3_", "")
            ax.annotate(f"{short}", (tigr4_x, tigr4_ys[j]),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=6, ha="left", va="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"Multi-match network: {n_d39v} D39V IGRs ↔ {n_tigr4} TIGR4 IGRs\n"
                 f"Blue = D39V queries | Red = TIGR4 targets",
                 fontsize=12, fontweight="bold")
    legend_elements = [
        plt.scatter([], [], c="#2196F3", s=100, label="D39V IGRs"),
        plt.scatter([], [], c="#F44336", s=100, label="TIGR4 IGRs"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Network: {out_path.name}")


def plot_heatmap(multi_matches: dict, d39v_seqs: dict, out_path: Path):
    """Heatmap of pairwise identity among repetitive D39V IGRs."""
    d39v_list = sorted(multi_matches.keys(), key=lambda k: -len(multi_matches[k]))
    n = len(d39v_list)

    # Compute pairwise identity (fraction of matches in best alignment)
    sim_matrix = np.zeros((n, n))
    from Bio.Align import PairwiseAligner
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -1

    for i in range(n):
        si = d39v_seqs.get(d39v_list[i], "")
        for j in range(i, n):
            sj = d39v_seqs.get(d39v_list[j], "")
            if si and sj:
                try:
                    alns = aligner.align(si, sj)
                    # Get just the first alignment
                    first = next(iter(alns), None)
                    if first:
                        a = str(first).split("\n")
                        if len(a) >= 3:
                            r = a[1]; q = a[2]
                            matches = sum(1 for x, y in zip(r, q) if x == y and x != "-")
                            aln_len = max(len(si), len(sj))
                            sim_matrix[i, j] = matches / max(aln_len, 1)
                except (OverflowError, StopIteration):
                    sim_matrix[i, j] = 0.0
                sim_matrix[j, i] = sim_matrix[i, j]

    # Dendrogram
    if n >= 2:
        dist = 1 - sim_matrix
        np.fill_diagonal(dist, 0)
        condensed = dist[np.triu_indices(n, k=1)]
        Z = linkage(condensed, method="average")

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.25, 0.75], wspace=0.02)

        ax_dendro = fig.add_subplot(gs[0, 0])
        dn = dendrogram(Z, ax=ax_dendro, orientation="left", no_labels=True)
        ax_dendro.axis("off")
        order = dn["leaves"]

        ax_heat = fig.add_subplot(gs[0, 1])
        ordered = sim_matrix[np.ix_(order, order)]
        im = ax_heat.imshow(ordered, cmap="YlOrRd", vmin=0.5, vmax=1.0, aspect="auto")
        labels = [d39v_list[i].replace("IGR_D39V_", "") for i in order]
        ax_heat.set_xticks(range(n))
        ax_heat.set_xticklabels(labels, rotation=90, fontsize=6)
        ax_heat.set_yticks(range(n))
        ax_heat.set_yticklabels(labels, fontsize=6)
        plt.colorbar(im, ax=ax_heat, label="Fraction identity", shrink=0.8)
        ax_heat.set_title(f"Pairwise identity of {n} repetitive D39V IGRs", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"  Heatmap: {out_path.name}")
    else:
        print("  Heatmap: skipped (need >= 2 sequences)")


def main():
    parser = argparse.ArgumentParser(description="Visualize repetitive IGR elements")
    parser.add_argument("--top-n", type=int, default=5, help="Top N to show in MSA/logo")
    parser.add_argument("--min-matches", type=int, default=5, help="Min matches for network/heatmap")
    args = parser.parse_args()

    print("Loading IGR sequences...")
    d39v_seqs = load_igr_sequences(D39V_TSV)
    tigr4_seqs = load_igr_sequences(TIGR4_TSV)

    print("Loading multi-match data...")
    multi = load_multi_matches(M8, min_matches=2)

    # ---- 1. Sequence logo + MSA for top-N ----
    print(f"\n--- Sequence logos for top {args.top_n} ---")
    sorted_multi = sorted(multi.items(), key=lambda x: -len(x[1]))
    for rank, (q_id, targets) in enumerate(sorted_multi[:args.top_n]):
        all_seqs = [d39v_seqs.get(q_id, "")]
        target_ids = []
        for t, _, _, _ in targets:
            ts = tigr4_seqs.get(t, "")
            if ts and ts not in all_seqs:
                all_seqs.append(ts)
                target_ids.append(t)

        if len(all_seqs) < 2:
            continue

        msa = star_msa(all_seqs)
        short_id = q_id.replace("IGR_D39V_", "")
        n_matches = len(targets)

        plot_sequence_logo(
            msa,
            f"D39V_{short_id} ({n_matches} copies in TIGR4, {len(all_seqs)} unique seqs)",
            OUT_DIR / f"repeat_{short_id}_logo.svg"
        )

        if len(all_seqs) <= 50:
            plot_msa_matrix(
                msa,
                f"D39V_{short_id} ({n_matches} copies, {len(all_seqs)} unique)",
                OUT_DIR / f"repeat_{short_id}_msa.svg"
            )

    # ---- 2. Bipartite network ----
    print(f"\n--- Network graph ---")
    multi_net = {k: v for k, v in multi.items() if len(v) >= args.min_matches}
    plot_network(multi_net, d39v_seqs, OUT_DIR / "multimatch_network.svg")

    # ---- 3. Heatmap ----
    print(f"\n--- Similarity heatmap ---")
    multi_hm = {k: v for k, v in multi.items() if len(v) >= args.min_matches}
    if len(multi_hm) >= 2:
        plot_heatmap(multi_hm, d39v_seqs, OUT_DIR / "multimatch_heatmap.svg")
    else:
        print("  Skipped: need >= 2 sequences with >= min_matches")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
