#!/usr/bin/env python3
"""
Step 2: Assemble Composite Bipartite MEME Motifs (Shimada et al. 2014) + Extended -10 (de Jong et al. 2011).

Constructs:
  1. 5 Bipartite composite PPMs (15, 16, 17, 18, 19 bp spacers) with neutral background spacer (log-odds = 0).
  2. Extended -10 PPM (11 bp: TRTGNTATAAT) from de Jong et al. (2011).
  3. Standard -10 PPM (6 bp: TATAAT) from Shimada et al. (2014).
  4. ComX CIN-box (8 bp: TACGAATA) from Campbell et al. (1998).
"""

import json
from pathlib import Path

MOTIFS_DIR = Path(__file__).resolve().parent.parent / "motifs"
BG_JSON = MOTIFS_DIR / "d39v_500bp_upstream_background.json"
OUT_MEME = MOTIFS_DIR / "shimada_composite_motifs.meme"

# PPM -35 Box (6 bp: TTGACA) from Shimada et al. (2014) gSELEX (N=550)
PPM_MINUS35 = [
    [0.103877, 0.094572, 0.045572, 0.755979],  # T
    [0.054830, 0.049159, 0.036490, 0.859521],  # T
    [0.169272, 0.096389, 0.610514, 0.123826],  # G
    [0.712415, 0.105471, 0.049205, 0.132908],  # A
    [0.163822, 0.632266, 0.060105, 0.143808],  # C
    [0.746929, 0.076407, 0.087352, 0.089312],  # A
]

# PPM -10 Box (6 bp: TATAAT) from Shimada et al. (2014) gSELEX (N=550)
PPM_MINUS10 = [
    [0.114776, 0.081857, 0.078270, 0.725098],  # T
    [0.768727, 0.038260, 0.080086, 0.112927],  # A
    [0.118409, 0.076407, 0.054655, 0.750529],  # T
    [0.750562, 0.092756, 0.052838, 0.103844],  # A
    [0.772360, 0.060058, 0.071004, 0.096578],  # A
    [0.120225, 0.069141, 0.049205, 0.761428],  # T
]

# PPM Extended -10 Box (11 bp: TRTGNTATAAT) from de Jong et al. (2011)
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

# ComX CIN-box (8 bp: TACGAATA)
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


def main():
    print("[STEP 2] Assembling Composite Bipartite + Extended -10 MEME Motifs...")
    if BG_JSON.exists():
        with open(BG_JSON) as f:
            bg = json.load(f)
    else:
        bg = {"A": 0.3155, "C": 0.1768, "G": 0.1958, "T": 0.3120}

    bg_row = [bg["A"], bg["C"], bg["G"], bg["T"]]
    OUT_MEME.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_MEME, "w") as f:
        f.write("MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write(f"Background letter frequencies\n"
                f"A {bg['A']:.4f} C {bg['C']:.4f} G {bg['G']:.4f} T {bg['T']:.4f}\n\n")

        # 1. 5 RpoD Bipartite Motifs (SP15 to SP19)
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

        # 2. Extended -10 Motif (11 bp)
        f.write("MOTIF EXTENDED_MINUS10 Extended_10bp_TRTGNTATAAT\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_EXT10)} nsites= 623 E= 1e-100\n")
        for row in PPM_EXT10:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # 3. Isolated Standard -10 Motif (6 bp: TATAAT)
        f.write("MOTIF MINUS10_ISOLATED Standard_Minus10_TATAAT\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_MINUS10)} nsites= 550 E= 1e-100\n")
        for row in PPM_MINUS10:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # 4. ComX CIN-box
        f.write("MOTIF COMX_CINBOX ComX_SigX_Motif\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_COMX)} nsites= 21 E= 1e-50\n")
        for row in PPM_COMX:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

    print(f"  Saved MEME file with Bipartite + Extended -10 + ComX: {OUT_MEME}")


if __name__ == "__main__":
    main()
