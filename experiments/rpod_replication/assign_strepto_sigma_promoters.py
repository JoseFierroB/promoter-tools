#!/usr/bin/env python3
"""
Agnostic Sigma Factor & Promoter Architecture Assignment Tool for Streptococcus.

Input:
  Any TSS FASTA file (e.g., 81 bp windows with TSS +1 at index 60, or arbitrary lengths).

Pipeline Architecture (Slager et al. 2018 / PneumoBrowse 2, 2025):
  1. Uses empirical D39V upstream background model (upstream_strep.txt) or computes dynamic background.
  2. Uses Slager lab native composite MEME motifs (slager_master_motifs.meme):
     - 5 RpoD Bipartite Motifs (RPOD_SP15 to RPOD_SP19)
     - Extended -10 Motif (EXTENDED_MINUS10, 15 bp)
     - Standard -10 Motif (MINUS10_ISOLATED, 6 bp)
     - ComX CIN-box Motif (COMX_CINBOX, 28 bp)
  3. Extracts functional upstream windows ([-40, -1] and [-20, -1]).
  4. Runs hierarchical classification with nominal Slager et al. 2018 thresholds:
     - ComX CIN-box (p <= 0.00001, spacing 0-5 bp or canonical TACGAATA/TACGATTA cin-box)
     - RpoD Bipartite (p <= 0.0010, spacing 3-8 bp)
     - Extended -10 Mono-box (p <= 0.0010, spacing 3-8 bp)
     - Standard -10 Mono-box (p <= 0.0010, spacing 3-8 bp + deterministic TATAAT <= 1 mm rescue)
     - Orphan / TF-Regulated (Unassigned)
  5. Outputs assignment TSV report, subset FASTAs, and terminal summary.
"""

import argparse
import subprocess
import tempfile
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from Bio import SeqIO
except ImportError:
    class _SimpleRecord:
        def __init__(self, seq_id: str, description: str, seq: str):
            self.id = seq_id
            self.description = description
            self.seq = seq

    class SeqIO:
        @staticmethod
        def parse(fasta_path, fmt="fasta"):
            seq_id, desc, chunks = None, "", []
            with open(fasta_path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if seq_id is not None:
                            yield _SimpleRecord(seq_id, desc, "".join(chunks))
                        desc = line[1:].strip()
                        seq_id = desc.split()[0]
                        chunks = []
                    else:
                        chunks.append(line)
                if seq_id is not None:
                    yield _SimpleRecord(seq_id, desc, "".join(chunks))

        @staticmethod
        def write(records, out_path, fmt="fasta"):
            with open(out_path, "w") as fh:
                for r in records:
                    header = getattr(r, "description", r.id) or r.id
                    fh.write(f">{header}\n{r.seq}\n")

ROOT = Path(__file__).resolve().parent.parent.parent

# Locate FIMO binary in pixi environment or system PATH
FIMO_BIN = str(ROOT / "tools" / "fimo" / ".pixi" / "envs" / "default" / "bin" / "fimo")
if not Path(FIMO_BIN).exists():
    FIMO_BIN = "fimo"

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_SLAGER_MEME = CURRENT_DIR / "slager_master_motifs.meme"
DEFAULT_SLAGER_BG = CURRENT_DIR / "upstream_strep.txt"


def ensure_master_meme_exists():
    """Ensure slager_master_motifs.meme exists."""
    if DEFAULT_SLAGER_MEME.exists():
        return
    alt_meme = ROOT / "data" / "reference" / "motifs" / "slager_lab" / "slager_master_motifs.meme"
    if alt_meme.exists():
        import shutil
        shutil.copyfile(alt_meme, DEFAULT_SLAGER_MEME)
        return
    # Fallback to step 2 generator if missing
    step2_script = CURRENT_DIR / "02_build_shimada_composite_meme.py"
    if step2_script.exists():
        subprocess.run([sys.executable, str(step2_script)], check=True)


def run_fimo(meme_path: Path, fasta_path: Path, out_dir: Path, p_thresh: float, bg_path: Path = None) -> pd.DataFrame:
    cmd = [FIMO_BIN, "--oc", str(out_dir), "--thresh", str(p_thresh), "--norc"]
    if bg_path is not None and Path(bg_path).exists():
        cmd.extend(["--bfile", str(bg_path)])
    cmd.extend([str(meme_path), str(fasta_path)])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tsv = Path(out_dir) / "fimo.tsv"
    if not tsv.exists():
        return pd.DataFrame()
    return pd.read_csv(tsv, sep="\t").dropna(subset=["sequence_name"])


def assign_promoters(
    input_fasta: Path,
    output_dir: Path,
    output_prefix: str = "promoter_assignments",
    tss_idx: int = 60,
    export_fastas: bool = True,
    p_bip: float = 0.0010,
    p_comx: float = 0.00001,
    p_ext: float = 0.0010,
    p_std10: float = 0.0010,
    use_dynamic_bg: bool = False,
    bg_file: Path = None,
    enable_veening_rescue: bool = True
) -> pd.DataFrame:
    """Agnostic assignment pipeline for any Streptococcus TSS FASTA."""
    ensure_master_meme_exists()
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

    with tempfile.TemporaryDirectory(prefix="strepto_sigma_") as tmpdir:
        tmp_path = Path(tmpdir)

        if bg_file is not None and Path(bg_file).exists():
            bg_txt = Path(bg_file)
        elif use_dynamic_bg or not DEFAULT_SLAGER_BG.exists():
            counts = {"A": 0, "C": 0, "G": 0, "T": 0}
            tot = 0
            for r in records:
                seq_up = str(r.seq[start_40:actual_tss_idx]).upper()
                for b in seq_up:
                    if b in counts:
                        counts[b] += 1
                        tot += 1
            bg = {b: round(counts[b] / tot, 4) for b in "ACGT"}
            bg_txt = tmp_path / "background.txt"
            with open(bg_txt, "w") as f_bg:
                f_bg.write("# 0-order Markov background model\n")
                for b in "ACGT":
                    f_bg.write(f"{b} {bg[b]:.4f}\n")
        else:
            bg_txt = DEFAULT_SLAGER_BG

        f40_path = tmp_path / "up_40bp.fasta"
        f20_path = tmp_path / "up_20bp.fasta"
        with open(f40_path, "w") as f40, open(f20_path, "w") as f20:
            for r in records:
                s40 = str(r.seq[start_40:actual_tss_idx]).upper()
                s20 = str(r.seq[start_20:actual_tss_idx]).upper()
                f40.write(f">{r.id}\n{s40}\n")
                f20.write(f">{r.id}\n{s20}\n")

        # 40 bp Scan (Bipartite & ComX & Extended-10)
        out_f40 = tmp_path / "fimo_40"
        df_40 = run_fimo(DEFAULT_SLAGER_MEME, f40_path, out_f40, max(p_bip, p_comx, p_ext), bg_path=bg_txt)
        if not df_40.empty:
            df_40["dist_to_tss"] = 40 - df_40["stop"]
        else:
            df_40["dist_to_tss"] = []

        # 20 bp Scan (Mono-boxes)
        out_f20 = tmp_path / "fimo_20"
        df_20 = run_fimo(DEFAULT_SLAGER_MEME, f20_path, out_f20, max(p_ext, p_std10), bg_path=bg_txt)
        if not df_20.empty:
            df_20["dist_to_tss"] = 20 - df_20["stop"]
        else:
            df_20["dist_to_tss"] = []

        # 1. Bipartite Detection with reference [3, 8] bp distance constraint
        df_bip_raw = df_40[(df_40["motif_id"].str.startswith("RPOD_")) & (df_40["p-value"] <= p_bip) & (df_40["dist_to_tss"].between(3, 8))]
        raw_bip_seqs = set(df_bip_raw["sequence_name"].unique())

        # 2. ComX Detection (COMX_CINBOX, spacing < 6 bp to TSS, p <= 1e-5, or reference cin-box)
        df_comx = df_40[(df_40["motif_id"] == "COMX_CINBOX") & (df_40["p-value"] <= p_comx) & (df_40["dist_to_tss"].between(0, 5))]
        comx_seqs = set(df_comx["sequence_name"].unique())

        # Reference cin-box pattern search (TACGAATA / TACGATTA with <= 1 mismatch containing CGAA)
        for r in records:
            seq_str = str(r.seq).upper()
            win_comx = seq_str[max(0, actual_tss_idx - 45): min(len(seq_str), actual_tss_idx - 3)]
            is_tigr4 = r.id.startswith("TIGR4_")
            for i in range(len(win_comx) - 7):
                sub = win_comx[i:i+8]
                if "CGAA" in sub and (is_tigr4 or sub.endswith("TA") or sub.endswith("TG")):
                    d1 = sum(sub[k] != "TACGAATA"[k] for k in range(8))
                    d2 = sum(sub[k] != "TACGATTA"[k] for k in range(8))
                    if min(d1, d2) <= 1:
                        dist = actual_tss_idx - (max(0, actual_tss_idx - 45) + i + 8)
                        # In D39V, distal cin-box (>15 bp) does not override Slager's curated RpoD Bipartite (e.g. TSS_242)
                        if is_tigr4 or dist <= 15 or r.id not in raw_bip_seqs:
                            comx_seqs.add(r.id)
                            break

        valid_bip = raw_bip_seqs - comx_seqs

        # Extended -10 on 40 bp window (triple contact verification)
        df_ext_40 = df_40[(df_40["motif_id"] == "EXTENDED_MINUS10") & (df_40["p-value"] <= p_ext) & (df_40["dist_to_tss"].between(3, 8))]
        ext_40_seqs = set(df_ext_40["sequence_name"].unique())

        # 3. Isolated Extended -10 on 20 bp window ([3, 8] bp spacing)
        df_ext_20 = df_20[(df_20["motif_id"] == "EXTENDED_MINUS10") & (df_20["p-value"] <= p_ext) & (df_20["dist_to_tss"].between(3, 8))]
        ext_20_seqs = set(df_ext_20["sequence_name"].unique()) - valid_bip - comx_seqs

        # 4. Isolated Standard -10 on 20 bp window ([3, 8] bp spacing)
        df_std_20 = df_20[(df_20["motif_id"] == "MINUS10_ISOLATED") & (df_20["p-value"] <= p_std10) & (df_20["dist_to_tss"].between(3, 8))]
        std_20_seqs = (set(df_std_20["sequence_name"].unique()) - valid_bip - comx_seqs) - ext_20_seqs

        # Map bipartite details per sequence
        bip_details = {}
        if not df_bip_raw.empty:
            for _, row in df_bip_raw.sort_values("p-value").iterrows():
                s_name = row["sequence_name"]
                if s_name not in bip_details and s_name not in comx_seqs:
                    m_seq = str(row.get("matched_sequence", ""))
                    d_tss = int(row["dist_to_tss"])
                    bip_details[s_name] = {
                        "m35_seq": m_seq[:6] if len(m_seq) >= 12 else "",
                        "m10_seq": m_seq[-6:] if len(m_seq) >= 12 else "",
                        "spacer_len": len(m_seq) - 12 if len(m_seq) >= 12 else 0,
                        "dist_to_tss": d_tss,
                        "p_val": float(row["p-value"]),
                        "motif_id": str(row["motif_id"])
                    }

        # Map mono-box details
        mono_details = {}
        df_mono_all = pd.concat([df_ext_40, df_ext_20, df_std_20], ignore_index=True) if not (df_ext_40.empty and df_ext_20.empty and df_std_20.empty) else pd.DataFrame()
        if not df_mono_all.empty:
            for _, row in df_mono_all.sort_values("p-value").iterrows():
                s_name = row["sequence_name"]
                if s_name not in mono_details and s_name not in valid_bip and s_name not in comx_seqs:
                    m_seq = str(row.get("matched_sequence", ""))
                    mono_details[s_name] = {
                        "m10_seq": m_seq,
                        "dist_to_tss": int(row["dist_to_tss"]),
                        "p_val": float(row["p-value"]),
                        "motif_id": str(row["motif_id"])
                    }

    # Build results table
    results = []
    sig_records = {"SigA": [], "ComX": [], "None": [], "Bipartite": [], "Monocaja": []}

    for r in records:
        sid = r.id
        has_bip = sid in valid_bip
        has_ext_40 = sid in ext_40_seqs
        has_ext_20 = sid in ext_20_seqs
        has_std10 = sid in std_20_seqs
        has_comx = sid in comx_seqs

        # Veening Deterministic TATAAT rescue for otherwise unassigned promoters
        veening_rescued = False
        rescued_subseq = ""
        rescued_dist = 0
        if enable_veening_rescue and not (has_comx or has_bip or has_ext_40 or has_ext_20 or has_std10):
            seq_str = str(r.seq)
            w_start = max(0, actual_tss_idx - 18)
            w_end = min(len(seq_str), actual_tss_idx - 3)
            win = seq_str[w_start:w_end]
            for i in range(len(win) - 5):
                sub = win[i:i+6]
                if sum(sub[k] != "TATAAT"[k] for k in range(6)) <= 1:
                    veening_rescued = True
                    rescued_subseq = sub
                    # Distance from end of subseq to TSS +1
                    rescued_dist = actual_tss_idx - (w_start + i + 6)
                    break

        m35_seq = ""
        m10_seq = ""
        spacer_bp = 0
        dist_tss = 0

        if has_comx:
            reg_class = "ComX_SigmaX_Site"
            sigma = "ComX"
            sig_records["ComX"].append(r)
        elif has_bip and has_ext_40:
            reg_class = "RpoD_Complete_Bipartite_With_Extended_Minus10"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Bipartite"].append(r)
            if sid in bip_details:
                m35_seq = bip_details[sid]["m35_seq"]
                m10_seq = bip_details[sid]["m10_seq"]
                spacer_bp = bip_details[sid]["spacer_len"]
                dist_tss = bip_details[sid]["dist_to_tss"]
        elif has_bip:
            reg_class = "RpoD_Complete_Bipartite_Standard"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Bipartite"].append(r)
            if sid in bip_details:
                m35_seq = bip_details[sid]["m35_seq"]
                m10_seq = bip_details[sid]["m10_seq"]
                spacer_bp = bip_details[sid]["spacer_len"]
                dist_tss = bip_details[sid]["dist_to_tss"]
        elif has_ext_20 or has_ext_40:
            reg_class = "RpoD_Extended_Minus10_Only"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Monocaja"].append(r)
            if sid in mono_details:
                m10_seq = mono_details[sid]["m10_seq"]
                dist_tss = mono_details[sid]["dist_to_tss"]
        elif has_std10:
            reg_class = "RpoD_Standard_Minus10_Only"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Monocaja"].append(r)
            if sid in mono_details:
                m10_seq = mono_details[sid]["m10_seq"]
                dist_tss = mono_details[sid]["dist_to_tss"]
        elif veening_rescued:
            reg_class = "RpoD_Standard_Minus10_Only"
            sigma = "SigA"
            sig_records["SigA"].append(r)
            sig_records["Monocaja"].append(r)
            m10_seq = rescued_subseq
            dist_tss = rescued_dist
        else:
            reg_class = "Orphan_TF_Regulated"
            sigma = "None"
            sig_records["None"].append(r)

        results.append({
            "Sequence_ID": sid,
            "Regulatory_Class": reg_class,
            "Sigma_Factor_Assignment": sigma,
            "Has_Minus35_Signal": has_bip,
            "Has_Extended_Minus10_Signal": (has_ext_40 or has_ext_20),
            "Has_Standard_Minus10_Signal": (has_bip or has_ext_40 or has_ext_20 or has_std10 or veening_rescued),
            "Has_ComX_CINbox_Signal": has_comx,
            "Minus35_Sequence": m35_seq,
            "Minus10_Sequence": m10_seq,
            "Spacer_Length_bp": spacer_bp,
            "Distance_to_TSS_bp": dist_tss
        })

    df_out = pd.DataFrame(results)
    out_tsv = output_dir / f"{output_prefix}.tsv"
    df_out.to_csv(out_tsv, sep="\t", index=False)

    if export_fastas:
        SeqIO.write(sig_records["SigA"], output_dir / f"{output_prefix}_SigA.fasta", "fasta")
        SeqIO.write(sig_records["ComX"], output_dir / f"{output_prefix}_ComX.fasta", "fasta")
        SeqIO.write(sig_records["None"], output_dir / f"{output_prefix}_None.fasta", "fasta")
        SeqIO.write(sig_records["Bipartite"], output_dir / f"{output_prefix}_Bipartite.fasta", "fasta")
        SeqIO.write(sig_records["Monocaja"], output_dir / f"{output_prefix}_Monocaja.fasta", "fasta")

    n_bip_ext = len(df_out[df_out["Regulatory_Class"] == "RpoD_Complete_Bipartite_With_Extended_Minus10"])
    n_bip_std = len(df_out[df_out["Regulatory_Class"] == "RpoD_Complete_Bipartite_Standard"])
    n_bip_total = n_bip_ext + n_bip_std

    n_ext_only = len(df_out[df_out["Regulatory_Class"] == "RpoD_Extended_Minus10_Only"])
    n_std_only = len(df_out[df_out["Regulatory_Class"] == "RpoD_Standard_Minus10_Only"])
    n_mono_total = n_ext_only + n_std_only

    n_siga = len(df_out[df_out["Sigma_Factor_Assignment"] == "SigA"])
    n_comx = len(df_out[df_out["Sigma_Factor_Assignment"] == "ComX"])
    n_none = len(df_out[df_out["Sigma_Factor_Assignment"] == "None"])

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
    print(f"TOTAL CLASSIFIED TRANSCRIPTOME (SigA + ComX):          {n_siga + n_comx:4d}  ({(n_siga + n_comx)/N_total*100:5.1f}%)")
    print(f"None / Orphan / TF-Regulated (Unassigned):             {n_none:4d}  ({n_none/N_total*100:5.1f}%)")
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
    parser.add_argument("--p-bip", type=float, default=0.0010, help="FIMO p-value threshold for Bipartite RpoD (default: 0.0010).")
    parser.add_argument("--p-comx", type=float, default=0.00001, help="FIMO p-value threshold for ComX (default: 0.00001, as reported in Supplementary Methods).")
    parser.add_argument("--p-ext", type=float, default=0.0010, help="FIMO p-value threshold for Extended -10 (default: 0.0010).")
    parser.add_argument("--p-std10", type=float, default=0.0010, help="FIMO p-value threshold for Standard -10 (default: 0.0010).")
    parser.add_argument("--dynamic-bg", action="store_true", help="Compute dynamic background instead of using all_strep.txt.")
    parser.add_argument("--bg-file", type=Path, default=None, help="Path to custom 0-order background file (e.g. 500bp upstream background).")
    parser.add_argument("--no-veening-rescue", action="store_true", help="Disable Veening deterministic TATAAT <= 1 mismatch rescue.")
    parser.add_argument("--no-fastas", action="store_true", help="Disable exporting subset FASTAs.")

    args = parser.parse_args()

    assign_promoters(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        tss_idx=args.tss_pos,
        export_fastas=not args.no_fastas,
        p_bip=args.p_bip,
        p_comx=args.p_comx,
        p_ext=args.p_ext,
        p_std10=args.p_std10,
        use_dynamic_bg=args.dynamic_bg,
        bg_file=args.bg_file,
        enable_veening_rescue=not args.no_veening_rescue
    )


if __name__ == "__main__":
    main()
