#!/usr/bin/env python3
"""
Build separate modular MEME motifs:
  1. MINUS35_BOX (6 bp: TTGACA) - Shimada et al. (2014)
  2. MINUS10_BOX (6 bp: TATAAT) - Shimada et al. (2014)
  3. EXTENDED_MINUS10 (11 bp: TRTGNTATAAT) - de Jong / Sabelnikov
  4. COMX_CINBOX (8 bp: TACGAATA) - Campbell / Dagkessamanskaia
"""

from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
OUT_MEME = CURRENT_DIR / "modular_motifs.meme"

# Upstream background frequencies in D39V
BG = {"A": 0.3155, "C": 0.1768, "G": 0.1958, "T": 0.3120}

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

# PPM Extended -10 Box (11 bp: TRTGNTATAAT)
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
    print("[STEP 1] Generating modular MEME file with individual sub-motifs...")
    with open(OUT_MEME, "w") as f:
        f.write("MEME version 5\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write(f"Background letter frequencies\n"
                f"A {BG['A']:.4f} C {BG['C']:.4f} G {BG['G']:.4f} T {BG['T']:.4f}\n\n")

        # 1. MINUS35_BOX
        f.write("MOTIF MINUS35_BOX Minus35_TTGACA\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_MINUS35)} nsites= 550 E= 1e-100\n")
        for row in PPM_MINUS35:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # 2. MINUS10_BOX
        f.write("MOTIF MINUS10_BOX Minus10_TATAAT\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_MINUS10)} nsites= 550 E= 1e-100\n")
        for row in PPM_MINUS10:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # 3. EXTENDED_MINUS10
        f.write("MOTIF EXTENDED_MINUS10 Extended_10bp_TRTGNTATAAT\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_EXT10)} nsites= 623 E= 1e-100\n")
        for row in PPM_EXT10:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

        # 4. COMX_CINBOX
        f.write("MOTIF COMX_CINBOX ComX_SigX_Motif\n")
        f.write(f"letter-probability matrix: alength= 4 w= {len(PPM_COMX)} nsites= 21 E= 1e-50\n")
        for row in PPM_COMX:
            f.write(f"  {row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}  {row[3]:.6f}\n")
        f.write("\n")

    print(f"  Saved modular MEME file to: {OUT_MEME}")


if __name__ == "__main__":
    main()
