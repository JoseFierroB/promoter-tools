#!/usr/bin/env python3
"""
Step 4: Execute FIMO Scanning with Explicit Sub-box Verification and Spatial Spacing Constraints.

Hierarchical Classification Pipeline (Slager et al. 2018):
  1. ComX CIN-box Motif (p < 0.0005, spacing <= 6 bp).
  2. RpoD Composite Motifs (p < 0.001, spacing 3-8 bp) with explicit -35 sub-box verification:
     - Promoters with dual/triple contact (functional -35 AND -10) pass as Bipartite (p <= 5e-5, S_-35 > 0).
     - Promoters with only -10 contact pass to Mono-caja classification.
  3. Extended -10 & Standard -10 Mono-caja scan (p <= 0.001, spacing 3-8 bp).
  4. Unassigned / TF-Regulated (Orphans).
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STEPS_DIR = Path(__file__).resolve().parent
MOTIFS_DIR = Path(__file__).resolve().parent.parent / "motifs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "reference_results"

MEME_FILE = MOTIFS_DIR / "shimada_composite_motifs.meme"
FASTA_40BP = STEPS_DIR / "d39v_tss_40bp_upstream.fasta"
FASTA_20BP = STEPS_DIR / "d39v_tss_20bp_upstream.fasta"
OUT_ASSIGNMENTS = RESULTS_DIR / "d39v_1003_slager_assignments.tsv"

FIMO_BIN = str(ROOT / "tools" / "meme" / ".pixi" / "envs" / "default" / "bin" / "fimo")
if not Path(FIMO_BIN).exists():
    FIMO_BIN = "fimo"

# Upstream background frequencies in D39V
BG = {"A": 0.3155, "C": 0.1768, "G": 0.1958, "T": 0.3120}
BG_LIST = [BG["A"], BG["C"], BG["G"], BG["T"]]
BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

# PPM -35 Box (6 bp: TTGACA) from Shimada et al. (2014)
PPM_MINUS35 = [
    [0.103877, 0.094572, 0.045572, 0.755979],
    [0.054830, 0.049159, 0.036490, 0.859521],
    [0.169272, 0.096389, 0.610514, 0.123826],
    [0.712415, 0.105471, 0.049205, 0.132908],
    [0.163822, 0.632266, 0.060105, 0.143808],
    [0.746929, 0.076407, 0.087352, 0.089312],
]


def calc_subscore_35(seq_6mer: str) -> float:
    """Calculate log-odds score for -35 sub-box (in bits)."""
    score = 0.0
    for i, base in enumerate(seq_6mer[:6]):
        if base in BASE_MAP:
            idx = BASE_MAP[base]
            score += np.log2(PPM_MINUS35[i][idx] / BG_LIST[idx])
    return score


def run_fimo(meme_path: Path, fasta_path: Path, out_dir: Path, p_thresh: float) -> pd.DataFrame:
    cmd = [FIMO_BIN, "--oc", str(out_dir), "--thresh", str(p_thresh), "--norc", str(meme_path), str(fasta_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tsv = out_dir / "fimo.tsv"
    return pd.read_csv(tsv, sep="\t").dropna(subset=["sequence_name"])


def main():
    if not FASTA_40BP.exists() or not FASTA_20BP.exists():
        raise FileNotFoundError(f"Missing fasta files: {FASTA_40BP}, {FASTA_20BP}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fimo_step4_slager_") as tmpdir:
        tmp_path = Path(tmpdir)

        # Stage 1: 40 bp Scan for Bipartite and ComX
        out_fimo_40 = tmp_path / "fimo_40bp"
        df_hits_40 = run_fimo(MEME_FILE, FASTA_40BP, out_fimo_40, 0.001)
        df_hits_40["dist_to_tss"] = 40 - df_hits_40["stop"]

        # Stage 2: 20 bp Scan for Isolated / Extended -10
        out_fimo_20 = tmp_path / "fimo_20bp"
        df_hits_20 = run_fimo(MEME_FILE, FASTA_20BP, out_fimo_20, 0.005)
        df_hits_20["dist_to_tss"] = 20 - df_hits_20["stop"]

        # ComX (p < 0.0005, dist <= 8 bp)
        df_comx = df_hits_40[(df_hits_40["motif_id"] == "COMX_CINBOX") & (df_hits_40["p-value"] < 0.0005) & (df_hits_40["dist_to_tss"] <= 8)]
        detected_comx = set(df_comx["sequence_name"].unique())

        # Explicit Bipartite Verification:
        # Requires composite p <= 5e-5 AND positive -35 sub-score S_-35 > 0, spacing 3-8 bp
        df_bip_raw = df_hits_40[(df_hits_40["motif_id"].str.startswith("RPOD_COMPOSITE")) & (df_hits_40["dist_to_tss"].between(3, 8))]
        
        valid_bip_seqs = set()
        for _, row in df_bip_raw.iterrows():
            sid = row["sequence_name"]
            if sid in detected_comx:
                continue
            matched = row["matched_sequence"]
            p_val = row["p-value"]
            s35 = matched[:6]
            sc_35 = calc_subscore_35(s35)
            if p_val <= 0.00005 and sc_35 > 0:
                valid_bip_seqs.add(sid)

        # Extended -10 on full window (for triple contact subclassification)
        df_ext_40 = df_hits_40[(df_hits_40["motif_id"] == "EXTENDED_MINUS10") & (df_hits_40["dist_to_tss"].between(3, 8))]
        detected_ext_40 = set(df_ext_40["sequence_name"].unique())

        # Isolated Extended -10 (Stage 2, p <= 0.001, spacing 3-8 bp)
        df_ext_20 = df_hits_20[(df_hits_20["motif_id"] == "EXTENDED_MINUS10") & (df_hits_20["p-value"] <= 0.001) & (df_hits_20["dist_to_tss"].between(3, 8))]
        detected_ext_20 = set(df_ext_20["sequence_name"].unique()) - valid_bip_seqs - detected_comx

        # Isolated Standard -10 (Stage 2, p <= 0.001, spacing 3-8 bp)
        df_std_20 = df_hits_20[(df_hits_20["motif_id"] == "MINUS10_ISOLATED") & (df_hits_20["p-value"] <= 0.001) & (df_hits_20["dist_to_tss"].between(3, 8))]
        detected_std_20 = (set(df_std_20["sequence_name"].unique()) - valid_bip_seqs - detected_comx) - detected_ext_20

    all_seq_ids = []
    with open(FASTA_40BP) as f:
        for line in f:
            if line.startswith(">"):
                all_seq_ids.append(line[1:].strip().split()[0])

    results = []
    for sid in all_seq_ids:
        has_bip = sid in valid_bip_seqs
        has_ext_40 = sid in detected_ext_40
        has_ext_20 = sid in detected_ext_20
        has_std10 = sid in detected_std_20
        has_comx = sid in detected_comx

        if has_comx:
            slager_class = "ComX_SigmaX_Site"
            sigma_assignment = "SigX"
        elif has_bip and has_ext_40:
            slager_class = "RpoD_Complete_Bipartite_With_Extended_Minus10"
            sigma_assignment = "SigA"
        elif has_bip:
            slager_class = "RpoD_Complete_Bipartite_Standard"
            sigma_assignment = "SigA"
        elif has_ext_20:
            slager_class = "RpoD_Extended_Minus10_Only"
            sigma_assignment = "SigA"
        elif has_std10:
            slager_class = "RpoD_Standard_Minus10_Only"
            sigma_assignment = "SigA"
        else:
            slager_class = "Orphan_Unassigned"
            sigma_assignment = "Unassigned"

        results.append({
            "Sequence_ID": sid,
            "Slager_2018_Regulatory_Class": slager_class,
            "Sigma_Factor_Assignment": sigma_assignment,
            "Has_Minus35_Signal": has_bip,
            "Has_Extended_Minus10_Signal": (has_ext_40 or has_ext_20),
            "Has_Standard_Minus10_Signal": (has_bip or has_ext_40 or has_ext_20 or has_std10),
            "Has_ComX_CINbox_Signal": has_comx
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_ASSIGNMENTS, sep="\t", index=False)

    n_bip_ext = len(df_out[df_out["Slager_2018_Regulatory_Class"] == "RpoD_Complete_Bipartite_With_Extended_Minus10"])
    n_bip_std = len(df_out[df_out["Slager_2018_Regulatory_Class"] == "RpoD_Complete_Bipartite_Standard"])
    n_bip_total = n_bip_ext + n_bip_std

    n_ext_only = len(df_out[df_out["Slager_2018_Regulatory_Class"] == "RpoD_Extended_Minus10_Only"])
    n_std_only = len(df_out[df_out["Slager_2018_Regulatory_Class"] == "RpoD_Standard_Minus10_Only"])
    n_mono_total = n_ext_only + n_std_only

    n_total_siga = n_bip_total + n_mono_total
    n_comx = len(df_out[df_out["Slager_2018_Regulatory_Class"] == "ComX_SigmaX_Site"])
    n_orphans = len(df_out[df_out["Sigma_Factor_Assignment"].str.startswith("Unassigned")])

    N_total = len(all_seq_ids)

    print("\n" + "=" * 76)
    print("   TRANSCRIPTOME PROMOTER CLASSIFICATION SUMMARY")
    print("=" * 76)
    print(f"Total Transcriptional Start Sites (TSSs):              {N_total:4d}  (100.0%)")
    print(f"1. RpoD Complete Bipartite (-35 + -10):                {n_bip_total:4d}  ({n_bip_total/N_total*100:.1f}%)")
    print(f"   - With Extended -10 (Triple Contact):               {n_bip_ext:4d}  ({n_bip_ext/N_total*100:.1f}%)")
    print(f"   - Standard Bipartite (Dual Contact):                {n_bip_std:4d}  ({n_bip_std/N_total*100:.1f}%)")
    print(f"2. RpoD Mono-box (-10 / Extended -10 Only):            {n_mono_total:4d}  ({n_mono_total/N_total*100:.1f}%)")
    print(f"   - Extended -10 Only (TRTG + TATAAT):                {n_ext_only:4d}  ({n_ext_only/N_total*100:.1f}%)")
    print(f"   - Standard -10 Only (TATAAT):                       {n_std_only:4d}  ({n_std_only/N_total*100:.1f}%)")
    print(f"----------------------------------------------------------------------------")
    print(f"TOTAL SigA (EσA) FUNCTIONAL PROMOTERS:                 {n_total_siga:4d}  ({n_total_siga/N_total*100:.1f}%)")
    print(f"ComX Sites (CIN-box / σX):                             {n_comx:4d}  ({n_comx/N_total*100:.1f}%)")
    print(f"TOTAL CLASSIFIED TRANSCRIPTOME (SigA + SigX):          {n_total_siga + n_comx:4d}  ({(n_total_siga + n_comx)/N_total*100:.1f}%)")
    print(f"Orphan / TF-Regulated (CodY/CcpA/ComE/Unassigned):     {n_orphans:4d}  ({n_orphans/N_total*100:.1f}%)")
    print("=" * 76)
    print(f"Output table saved to: {OUT_ASSIGNMENTS}")
    print("=" * 76)


if __name__ == "__main__":
    main()
