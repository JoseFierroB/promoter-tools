#!/usr/bin/env python3
"""
Agnostic Sigma Factor & Promoter Architecture Assignment Tool for Streptococcus.

Input:
  Any TSS FASTA file (e.g., 81 bp windows with TSS +1 at index 60, or arbitrary lengths).

Pipeline Architecture (Slager et al. 2018):
  1. Computes empirical upstream background frequencies (A, C, G, T) dynamically.
  2. Builds composite & modular PPMs for Streptococcus:
     - 5 RpoD Bipartite Motifs (SP15 to SP19)
     - Extended -10 Motif (TRTGNTATAAT, 11 bp)
     - Standard -10 Motif (TATAAT, 6 bp)
     - ComX CIN-box Motif (TACGAATA, 8 bp)
  3. Extracts 40 bp ([-40, -1]) and 20 bp ([-20, -1]) upstream windows.
  4. Runs two-tier hierarchical classification:
     - ComX CIN-box (p < 0.0005, spacing <= 6 bp)
     - RpoD Bipartite (p <= 5e-5, positive -35 subscore S_-35 > 0, spacing 3-8 bp)
     - Extended -10 Mono-box (p <= 0.001, spacing 3-8 bp)
     - Standard -10 Mono-box (p <= 0.001, spacing 3-8 bp)
     - Orphan / TF-Regulated (Unassigned)
  5. Outputs assignment TSV report, subset FASTAs, and terminal summary.
"""

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# Locate FIMO binary in pixi environment or system PATH
FIMO_BIN = str(ROOT / "tools" / "meme" / ".pixi" / "envs" / "default" / "bin" / "fimo")
if not Path(FIMO_BIN).exists():
    FIMO_BIN = "fimo"

# Position Probability Matrices (PPM) for Streptococcus
PPM_MINUS35 = [
    [0.103877, 0.094572, 0.045572, 0.755979],  # T
    [0.054830, 0.049159, 0.036490, 0.859521],  # T
    [0.169272, 0.096389, 0.610514, 0.123826],  # G
    [0.712415, 0.105471, 0.049205, 0.132908],  # A
    [0.163822, 0.632266, 0.060105, 0.143808],  # C
    [0.746929, 0.076407, 0.087352, 0.089312],  # A
]

PPM_MINUS10 = [
    [0.114776, 0.081857, 0.078270, 0.725098],  # T
    [0.768727, 0.038260, 0.080086, 0.112927],  # A
    [0.118409, 0.076407, 0.054655, 0.750529],  # T
    [0.750562, 0.092756, 0.052838, 0.103844],  # A
    [0.772360, 0.060058, 0.071004, 0.096578],  # A
    [0.120225, 0.069141, 0.049205, 0.761428],  # T
]

PPM_EXT10 = [
    [0.050000, 0.050000, 0.050000, 0.850000],  # T (pos -15)
    [0.450000, 0.050000, 0.450000, 0.050000],  # R (A/G, pos -14)
    [0.050000, 0.050000, 0.050000, 0.850000],  # T (pos -13)
    [0.050000, 0.050000, 0.850000, 0.050000],  # G (pos -12)
    [0.250000, 0.250000, 0.250000, 0.250000],  # N (pos -11)
    [0.050000, 0.050000, 0.050000, 0.850000],  # T (pos -10)
    [0.768727, 0.038260, 0.080086, 0.112927],  # A (pos -9)
    [0.118409, 0.076407, 0.054655, 0.750529],  # T (pos -8)
    [0.750562, 0.092756, 0.052838, 0.103844],  # A (pos -7)
    [0.772360, 0.060058, 0.071004, 0.096578],  # A (pos -6)
    [0.120225, 0.069141, 0.049205, 0.761428],  # T (pos -5)
]

PPM_COMX = [
    [0.050000, 0.050000, 0.050000, 0.850000],  # T
    [0.850000, 0.050000, 0.050000, 0.050000],  # A
    [0.050000, 0.850000, 0.050000, 0.050000],  # C
    [0.050000, 0.050000, 0.850000, 0.050000],  # G
    [0.850000, 0.050000, 0.050000, 0.050000],  # A
    [0.850000, 0.050000, 0.050000, 0.050000],  # A
    [0.050000, 0.050000, 0.050000, 0.850000],  # T
    [0.850000, 0.050000, 0.050000, 0.050000],  # A
]

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def build_meme_file(out_meme_path: Path, bg: dict):
    bg_row = [bg["A"], bg["C"], bg["G"], bg["T"]]
    with open(out_meme_path, "w") as f:
        f.write("MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write(f"Background letter frequencies\n"
                f"A {bg['A']:.4f} C {bg['C']:.4f} G {bg['G']:.4f} T {bg['T']:.4f}\n\n")

        # 5 Bipartite Models (SP15 to SP19)
        for spacer in [15, 16, 17, 18, 19]:
            width = 6 + spacer + 6
            f.write(f"MOTIF RPOD_COMPOSITE_SP{spacer} RpoD_Spacer_{spacer}bp\n")
            f.write(f"letter-probability matrix: alength= 4 w= {width} nsites= 550 E= 1e-100\n")
            for row in PPM_MINUS35:
                f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
            for _ in range(spacer):
                f.write(f"  {bg_row[0]:.6f}  {bg_row[1]:.6f}  {bg_row[2]:.6f}  {bg_row[3]:.6f}\n")
            for row in PPM_MINUS10:
                f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
            f.write("\n")

        # Extended -10
        f.write("MOTIF EXTENDED_MINUS10 Extended_10bp_TRTGNTATAAT\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_EXT10)} nsites= 623 E= 1e-100\n")
        for row in PPM_EXT10:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # Standard -10
        f.write("MOTIF MINUS10_ISOLATED Standard_Minus10_TATAAT\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_MINUS10)} nsites= 550 E= 1e-100\n")
        for row in PPM_MINUS10:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # ComX CIN-box
        f.write("MOTIF COMX_CINBOX ComX_SigX_Motif\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_COMX)} nsites= 21 E= 1e-50\n")
        for row in PPM_COMX:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")


def run_fimo(meme_path: Path, fasta_path: Path, out_dir: Path, p_thresh: float) -> pd.DataFrame:
    cmd = [FIMO_BIN, "--oc", str(out_dir), "--thresh", str(p_thresh), "--norc", str(meme_path), str(fasta_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tsv = out_dir / "fimo.tsv"
    return pd.read_csv(tsv, sep="\t").dropna(subset=["sequence_name"])


def calc_subscore_35(seq_6mer: str, bg_list: list) -> float:
    score = 0.0
    for i, base in enumerate(seq_6mer[:6]):
        if base in BASE_MAP:
            idx = BASE_MAP[base]
            score += np.log2(PPM_MINUS35[i][idx] / bg_list[idx])
    return score


def assign_promoters(
    input_fasta: Path,
    output_dir: Path,
    output_prefix: str = "promoter_assignments",
    tss_idx: int = 60,
    export_fastas: bool = True
) -> pd.DataFrame:
    """Agnostic assignment pipeline for any Streptococcus TSS FASTA."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(SeqIO.parse(input_fasta, "fasta"))
    N_total = len(records)
    if N_total == 0:
        raise ValueError(f"Input FASTA is empty: {input_fasta}")

    first_len = len(records[0].seq)
    if first_len < 40:
        raise ValueError(f"Sequences must be at least 40 bp long (got {first_len} bp).")

    actual_tss_idx = min(tss_idx, first_len)
    start_40 = max(0, actual_tss_idx - 40)
    start_20 = max(0, actual_tss_idx - 20)

    # Compute empirical background frequencies from upstream regions
    counts = {"A": 0, "C": 0, "G": 0, "T": 0}
    tot = 0
    for r in records:
        seq_up = str(r.seq[start_40:actual_tss_idx]).upper()
        for b in seq_up:
            if b in counts:
                counts[b] += 1
                tot += 1
    bg = {b: round(counts[b] / tot, 4) for b in "ACGT"}
    bg_list = [bg["A"], bg["C"], bg["G"], bg["T"]]

    with tempfile.TemporaryDirectory(prefix="strepto_sigma_") as tmpdir:
        tmp_path = Path(tmpdir)
        meme_path = tmp_path / "motifs.meme"
        build_meme_file(meme_path, bg)

        f40_path = tmp_path / "up_40bp.fasta"
        f20_path = tmp_path / "up_20bp.fasta"
        with open(f40_path, "w") as f40, open(f20_path, "w") as f20:
            for r in records:
                s40 = str(r.seq[start_40:actual_tss_idx]).upper()
                s20 = str(r.seq[start_20:actual_tss_idx]).upper()
                f40.write(f">{r.id}\n{s40}\n")
                f20.write(f">{r.id}\n{s20}\n")

        # 40 bp Scan (Bipartite & ComX)
        out_f40 = tmp_path / "fimo_40"
        df_40 = run_fimo(meme_path, f40_path, out_f40, 0.001)
        df_40["dist_to_tss"] = 40 - df_40["stop"]

        # 20 bp Scan (Mono-cajas)
        out_f20 = tmp_path / "fimo_20"
        df_20 = run_fimo(meme_path, f20_path, out_f20, 0.005)
        df_20["dist_to_tss"] = 20 - df_20["stop"]

        # 1. ComX Detection
        df_comx = df_40[(df_40["motif_id"] == "COMX_CINBOX") & (df_40["p-value"] < 0.0005) & (df_40["dist_to_tss"] <= 8)]
        comx_seqs = set(df_comx["sequence_name"].unique())

        # 2. Bipartite Detection with explicit -35 sub-score verification
        df_bip_raw = df_40[(df_40["motif_id"].str.startswith("RPOD_COMPOSITE")) & (df_40["dist_to_tss"].between(3, 8))]
        valid_bip = set()
        for _, row in df_bip_raw.iterrows():
            sid = row["sequence_name"]
            if sid in comx_seqs:
                continue
            p_val = row["p-value"]
            s35 = row["matched_sequence"][:6]
            sc_35 = calc_subscore_35(s35, bg_list)
            if p_val <= 0.00005 and sc_35 > 0:
                valid_bip.add(sid)

        # Extended -10 on 40 bp window (for triple contact subclassification)
        df_ext_40 = df_40[(df_40["motif_id"] == "EXTENDED_MINUS10") & (df_40["dist_to_tss"].between(3, 8))]
        ext_40_seqs = set(df_ext_40["sequence_name"].unique())

        # 3. Isolated Extended -10 on 20 bp window
        df_ext_20 = df_20[(df_20["motif_id"] == "EXTENDED_MINUS10") & (df_20["p-value"] <= 0.001) & (df_20["dist_to_tss"].between(3, 8))]
        ext_20_seqs = set(df_ext_20["sequence_name"].unique()) - valid_bip - comx_seqs

        # 4. Isolated Standard -10 on 20 bp window
        df_std_20 = df_20[(df_20["motif_id"] == "MINUS10_ISOLATED") & (df_20["p-value"] <= 0.001) & (df_20["dist_to_tss"].between(3, 8))]
        std_20_seqs = (set(df_std_20["sequence_name"].unique()) - valid_bip - comx_seqs) - ext_20_seqs

    # Build results table
    results = []
    sig_records = {"SigA": [], "SigX": [], "Bipartite": [], "Monocaja": [], "Orphan": []}

    for r in records:
        sid = r.id
        has_bip = sid in valid_bip
        has_ext_40 = sid in ext_40_seqs
        has_ext_20 = sid in ext_20_seqs
        has_std10 = sid in std_20_seqs
        has_comx = sid in comx_seqs

        if has_comx:
            reg_class = "ComX_SigmaX_Site"
            sigma = "SigX"
            sig_records["SigX"].append(r)
        elif has_bip and has_ext_40:
            reg_class = "RpoD_Complete_Bipartite_With_Extended_Minus10"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Bipartite"].append(r)
        elif has_bip:
            reg_class = "RpoD_Complete_Bipartite_Standard"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Bipartite"].append(r)
        elif has_ext_20:
            reg_class = "RpoD_Extended_Minus10_Only"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Monocaja"].append(r)
        elif has_std10:
            reg_class = "RpoD_Standard_Minus10_Only"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Monocaja"].append(r)
        else:
            reg_class = "Orphan_Unassigned"
            sigma = "Unassigned"
            sig_records["Orphan"].append(r)

        results.append({
            "Sequence_ID": sid,
            "Regulatory_Class": reg_class,
            "Sigma_Factor_Assignment": sigma,
            "Has_Minus35_Signal": has_bip,
            "Has_Extended_Minus10_Signal": (has_ext_40 or has_ext_20),
            "Has_Standard_Minus10_Signal": (has_bip or has_ext_40 or has_ext_20 or has_std10),
            "Has_ComX_CINbox_Signal": has_comx
        })

    df_out = pd.DataFrame(results)
    out_tsv = output_dir / f"{output_prefix}.tsv"
    df_out.to_csv(out_tsv, sep="\t", index=False)

    if export_fastas:
        SeqIO.write(sig_records["SigA"], output_dir / f"{output_prefix}_SigA.fasta", "fasta")
        SeqIO.write(sig_records["SigX"], output_dir / f"{output_prefix}_SigX.fasta", "fasta")
        SeqIO.write(sig_records["Bipartite"], output_dir / f"{output_prefix}_Bipartite.fasta", "fasta")
        SeqIO.write(sig_records["Monocaja"], output_dir / f"{output_prefix}_Monocaja.fasta", "fasta")
        SeqIO.write(sig_records["Orphan"], output_dir / f"{output_prefix}_Orphans.fasta", "fasta")

    # Counts
    n_bip_ext = len(df_out[df_out["Regulatory_Class"] == "RpoD_Complete_Bipartite_With_Extended_Minus10"])
    n_bip_std = len(df_out[df_out["Regulatory_Class"] == "RpoD_Complete_Bipartite_Standard"])
    n_bip_total = n_bip_ext + n_bip_std

    n_ext_only = len(df_out[df_out["Regulatory_Class"] == "RpoD_Extended_Minus10_Only"])
    n_std_only = len(df_out[df_out["Regulatory_Class"] == "RpoD_Standard_Minus10_Only"])
    n_mono_total = n_ext_only + n_std_only

    n_siga = n_bip_total + n_mono_total
    n_comx = len(df_out[df_out["Regulatory_Class"] == "ComX_SigmaX_Site"])
    n_orphans = len(df_out[df_out["Sigma_Factor_Assignment"].str.startswith("Unassigned")])

    print("\n" + "=" * 76)
    print(f"   TRANSCRIPTOME PROMOTER CLASSIFICATION SUMMARY ({input_fasta.name})")
    print("=" * 76)
    print(f"Total Transcriptional Start Sites (TSSs):              {N_total:4d}  (100.0%)")
    print(f"1. RpoD Complete Bipartite (-35 + -10):                {n_bip_total:4d}  ({n_bip_total/N_total*100:5.1f}%)")
    print(f"   - With Extended -10 (Triple Contact):               {n_bip_ext:4d}  ({n_bip_ext/N_total*100:5.1f}%)")
    print(f"   - Standard Bipartite (Dual Contact):                {n_bip_std:4d}  ({n_bip_std/N_total*100:5.1f}%)")
    print(f"2. RpoD Mono-box (-10 / Extended -10 Only):            {n_mono_total:4d}  ({n_mono_total/N_total*100:5.1f}%)")
    print(f"   - Extended -10 Only (TRTG + TATAAT):                {n_ext_only:4d}  ({n_ext_only/N_total*100:5.1f}%)")
    print(f"   - Standard -10 Only (TATAAT):                       {n_std_only:4d}  ({n_std_only/N_total*100:5.1f}%)")
    print(f"----------------------------------------------------------------------------")
    print(f"TOTAL SigA (EσA) FUNCTIONAL PROMOTERS:                 {n_siga:4d}  ({n_siga/N_total*100:5.1f}%)")
    print(f"ComX Sites (CIN-box / σX):                             {n_comx:4d}  ({n_comx/N_total*100:5.1f}%)")
    print(f"TOTAL CLASSIFIED TRANSCRIPTOME (SigA + SigX):          {n_siga + n_comx:4d}  ({(n_siga + n_comx)/N_total*100:5.1f}%)")
    print(f"Orphan / TF-Regulated (Unassigned):                    {n_orphans:4d}  ({n_orphans/N_total*100:5.1f}%)")
    print("=" * 76)
    print(f"Assignments saved to: {out_tsv}")
    print("=" * 76)

    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="Agnostic Promoter & Sigma Factor Assignment Tool for Streptococcus TSSs."
    )
    parser.add_argument("--input-fasta", "-i", required=True, type=Path, help="Input TSS FASTA file.")
    parser.add_argument("--output-dir", "-o", default=Path("output/promoter_assignments"), type=Path, help="Output directory.")
    parser.add_argument("--output-prefix", "-p", default="promoter_assignments", type=str, help="Prefix for output files.")
    parser.add_argument("--tss-pos", type=int, default=60, help="0-based index of TSS +1 in sequence (default: 60 for 81-mers).")
    parser.add_argument("--no-fastas", action="store_true", help="Disable exporting subset FASTAs.")

    args = parser.parse_args()

    assign_promoters(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        tss_idx=args.tss_pos,
        export_fastas=not args.no_fastas
    )


if __name__ == "__main__":
    main()
