#!/usr/bin/env python3
"""
Step 3: Execute Modular Two-Step Scanning Pipeline.

Phase 1: ComX filter (p < 0.0005, spacing <= 6 bp).
Phase 2 (Sigma-A Step 1): Scan -10 / Extended -10 on 20 bp window (p < 0.001, spacing 3-8 bp).
Phase 3 (Sigma-A Step 2): Scan upstream -35 on 40 bp window (p < 0.001, spacer 15-19 bp from -10).
Phase 4: Compare against Slager sequence.gff3 ground truth.
"""

import subprocess
import tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CURRENT_DIR = Path(__file__).resolve().parent

MEME_FILE = CURRENT_DIR / "modular_motifs.meme"
FASTA_40BP = CURRENT_DIR / "d39v_tss_40bp_upstream.fasta"
FASTA_20BP = CURRENT_DIR / "d39v_tss_20bp_upstream.fasta"
GFF_GROUND_TRUTH = ROOT / "data" / "reference" / "sequence.gff3"
GFF_TSS = ROOT / "data" / "reference" / "D39V_annotation_TSS_Victor.gff"

OUT_TSV = CURRENT_DIR / "modular_two_step_rpod_assignments.tsv"

FIMO_BIN = str(ROOT / "tools" / "meme" / ".pixi" / "envs" / "default" / "bin" / "fimo")
if not Path(FIMO_BIN).exists():
    FIMO_BIN = "fimo"


def run_fimo(meme_path: Path, fasta_path: Path, out_dir: Path, p_thresh: float) -> pd.DataFrame:
    cmd = [FIMO_BIN, "--oc", str(out_dir), "--thresh", str(p_thresh), "--norc", str(meme_path), str(fasta_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tsv = out_dir / "fimo.tsv"
    return pd.read_csv(tsv, sep="\t").dropna(subset=["sequence_name"])


def main():
    print("[STEP 3] Running Modular Two-Step FIMO Scan...")

    with tempfile.TemporaryDirectory(prefix="fimo_modular_") as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Scan 40 bp window (threshold p <= 0.005 to capture all candidates)
        out_fimo_40 = tmp_path / "fimo_40bp"
        df_40 = run_fimo(MEME_FILE, FASTA_40BP, out_fimo_40, 0.005)
        df_40["dist_to_tss"] = 40 - df_40["stop"]

        # 2. Scan 20 bp window (threshold p <= 0.005)
        out_fimo_20 = tmp_path / "fimo_20bp"
        df_20 = run_fimo(MEME_FILE, FASTA_20BP, out_fimo_20, 0.005)
        df_20["dist_to_tss"] = 20 - df_20["stop"]

    # Load all sequence IDs
    all_sids = []
    with open(FASTA_40BP) as f:
        for line in f:
            if line.startswith(">"):
                all_sids.append(line[1:].strip().split()[0])

    # Load GFF Ground Truth for direct validation
    # Parse TSS positions
    tss_info = {}
    with open(GFF_TSS) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.strip().split("\t")
            if parts[2] == "transcription_start_site":
                pos = int(parts[3])
                strand = parts[6]
                attrs = dict(item.split("=", 1) for item in parts[8].split(";") if "=" in item)
                seq_id = attrs.get("ID", f"TSS_{pos}_{strand}")
                tss_info[seq_id] = (pos, strand)

    # Parse GFF signals
    m35_gff = []
    m10_gff = []
    with open(GFF_GROUND_TRUTH) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.strip().split("\t")
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            feat = parts[2]
            note = parts[8]
            if feat == "minus_35_signal":
                m35_gff.append((start, end, strand, note))
            elif feat == "minus_10_signal":
                m10_gff.append((start, end, strand, note))

    # Group FIMO hits by sequence_name
    hits_40_by_seq = df_40.groupby("sequence_name")
    hits_20_by_seq = df_20.groupby("sequence_name")

    results = []

    for sid in all_sids:
        # Check ComX
        has_comx = False
        comx_p = 1.0
        if sid in hits_40_by_seq.groups:
            g40 = hits_40_by_seq.get_group(sid)
            comx_hits = g40[(g40["motif_id"] == "COMX_CINBOX") & (g40["p-value"] < 0.0005) & (g40["dist_to_tss"] <= 8)]
            if not comx_hits.empty:
                has_comx = True
                comx_p = comx_hits["p-value"].min()

        # Step 1: Detect -10 / Extended -10 on 20 bp window
        best_10 = None
        if sid in hits_20_by_seq.groups:
            g20 = hits_20_by_seq.get_group(sid)
            # Filter p <= 0.001 and dist 3-8 bp
            valid_10 = g20[(g20["motif_id"].isin(["EXTENDED_MINUS10", "MINUS10_BOX"])) & 
                           (g20["p-value"] <= 0.001) & 
                           (g20["dist_to_tss"].between(3, 8))].sort_values("p-value")
            if not valid_10.empty:
                best_10 = valid_10.iloc[0]

        # Step 2: Detect coupled -35 on 40 bp window
        has_35 = False
        m35_p = 1.0
        m35_spacer = None
        m35_seq = ""

        if best_10 is not None and sid in hits_40_by_seq.groups:
            g40 = hits_40_by_seq.get_group(sid)
            # Find the position of the -10 in the 40 bp window:
            # 20 bp window corresponds to positions 21..40 of 40 bp window
            m10_start_40 = 20 + best_10["start"]
            
            # The -35 box must stop 15 to 19 bp upstream of m10_start_40:
            # required_stop_40 = m10_start_40 - spacer - 1
            # spacer between 15 and 19 means required_stop_40 in [m10_start_40 - 20, m10_start_40 - 16]
            valid_35 = g40[(g40["motif_id"] == "MINUS35_BOX") & (g40["p-value"] <= 0.001)]
            for _, r35 in valid_35.iterrows():
                spacer = m10_start_40 - r35["stop"] - 1
                if 15 <= spacer <= 19:
                    has_35 = True
                    m35_p = min(m35_p, r35["p-value"])
                    m35_spacer = int(spacer)
                    m35_seq = r35["matched_sequence"]
                    break

        # Ground Truth from GFF3
        gff_has_35 = False
        gff_has_10 = False
        gff_10_type = "None"
        if sid in tss_info:
            pos, strand = tss_info[sid]
            for s, e, st, note in m10_gff:
                if st == strand:
                    dist = pos - e - 1 if strand == "+" else s - pos - 1
                    if 0 <= dist <= 15:
                        gff_has_10 = True
                        gff_10_type = "Extended" if "extended" in note else "Standard"
                        break
            for s, e, st, note in m35_gff:
                if st == strand:
                    dist = pos - e - 1 if strand == "+" else s - pos - 1
                    if 15 <= dist <= 40:
                        gff_has_35 = True
                        break

        if gff_has_35 and gff_has_10:
            gff_class = "Bipartite"
        elif gff_has_10:
            gff_class = f"Mono-caja_{gff_10_type}"
        else:
            gff_class = "Orphan"

        # Predicted Class
        if has_comx:
            pred_class = "ComX_SigmaX"
            pred_sigma = "SigX"
        elif best_10 is not None and has_35:
            if best_10["motif_id"] == "EXTENDED_MINUS10":
                pred_class = "Bipartite_With_Extended_Minus10"
            else:
                pred_class = "Bipartite_Standard"
            pred_sigma = "SigA"
        elif best_10 is not None:
            if best_10["motif_id"] == "EXTENDED_MINUS10":
                pred_class = "Mono-caja_Extended_Minus10"
            else:
                pred_class = "Mono-caja_Standard_Minus10"
            pred_sigma = "SigA"
        else:
            pred_class = "Orphan_Unassigned"
            pred_sigma = "Unassigned"

        results.append({
            "Sequence_ID": sid,
            "Predicted_Class": pred_class,
            "Predicted_Sigma": pred_sigma,
            "Has_ComX": has_comx,
            "Has_Minus10": best_10 is not None,
            "Minus10_Type": best_10["motif_id"] if best_10 is not None else "None",
            "Minus10_pValue": best_10["p-value"] if best_10 is not None else 1.0,
            "Minus10_DistToTSS": best_10["dist_to_tss"] if best_10 is not None else -1,
            "Has_Minus35": has_35,
            "Minus35_pValue": m35_p if has_35 else 1.0,
            "Minus35_Spacer": m35_spacer if has_35 else -1,
            "Minus35_Sequence": m35_seq,
            "GFF3_GroundTruth_Class": gff_class,
            "GFF3_Has_Minus35": gff_has_35,
            "GFF3_Has_Minus10": gff_has_10
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_TSV, sep="\t", index=False)

    n_total = len(df_res)
    print("\n" + "=" * 78)
    print("   MODULAR TWO-STEP SCAN EXPERIMENT: RESULTS & COMPARISON TO SLAGER (2018)")
    print("=" * 78)
    print(f"Total Evaluated TSSs:                                  {n_total:4d}  (100.0%)")
    
    n_bip = len(df_res[df_res["Predicted_Class"].str.startswith("Bipartite")])
    n_bip_ext = len(df_res[df_res["Predicted_Class"] == "Bipartite_With_Extended_Minus10"])
    n_bip_std = len(df_res[df_res["Predicted_Class"] == "Bipartite_Standard"])
    
    n_mono_ext = len(df_res[df_res["Predicted_Class"] == "Mono-caja_Extended_Minus10"])
    n_mono_std = len(df_res[df_res["Predicted_Class"] == "Mono-caja_Standard_Minus10"])
    n_mono = n_mono_ext + n_mono_std
    
    n_siga = n_bip + n_mono
    n_comx = len(df_res[df_res["Predicted_Class"] == "ComX_SigmaX"])
    n_orph = len(df_res[df_res["Predicted_Class"] == "Orphan_Unassigned"])

    print(f"1. RpoD Bipartite Promoters (-35 + -10, p < 0.001 each):{n_bip:4d}  ({n_bip/n_total*100:5.1f}%)  [GFF3 GT: 403]")
    print(f"   - Dual Contact + Extended -10 (Triple Contact):      {n_bip_ext:4d}  ({n_bip_ext/n_total*100:5.1f}%)")
    print(f"   - Standard Bipartite (Dual Contact):                 {n_bip_std:4d}  ({n_bip_std/n_total*100:5.1f}%)")
    print(f"2. RpoD Mono-box Promoters (-10 / Ext -10 Only):        {n_mono:4d}  ({n_mono/n_total*100:5.1f}%)  [GFF3 GT: 467]")
    print(f"   - Extended -10 Only (TRTG + TATAAT):                 {n_mono_ext:4d}  ({n_mono_ext/n_total*100:5.1f}%)")
    print(f"   - Standard -10 Only (TATAAT):                        {n_mono_std:4d}  ({n_mono_std/n_total*100:5.1f}%)")
    print(f"------------------------------------------------------------------------------")
    print(f"TOTAL SigA (EσA) FUNCTIONAL PROMOTERS:                  {n_siga:4d}  ({n_siga/n_total*100:5.1f}%)  [GFF3 GT: 870]")
    print(f"ComX Sites (CIN-box / σX, p < 0.0005):                  {n_comx:4d}  ({n_comx/n_total*100:5.1f}%)  [GFF3 GT:  21]")
    print(f"TOTAL CLASSIFIED TRANSCRIPTOME (SigA + SigX):           {n_siga + n_comx:4d}  ({(n_siga + n_comx)/n_total*100:5.1f}%)")
    print(f"Orphan / TF-Regulated (Unassigned):                     {n_orph:4d}  ({n_orph/n_total*100:5.1f}%)  [GFF3 GT: 133]")
    print("=" * 78)

    # Calculate concordance against GFF3 ground truth
    df_res["GT_is_SigA"] = df_res["GFF3_GroundTruth_Class"].isin(["Bipartite", "Mono-caja_Extended", "Mono-caja_Standard"])
    df_res["Pred_is_SigA"] = df_res["Predicted_Sigma"] == "SigA"
    concordance = (df_res["GT_is_SigA"] == df_res["Pred_is_SigA"]).mean() * 100
    print(f"Concordance with Slager Ground Truth SigA status: {concordance:.2f}%")
    print(f"Output table saved to: {OUT_TSV.name}")
    print("=" * 78)


if __name__ == "__main__":
    main()
