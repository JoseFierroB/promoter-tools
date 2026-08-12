#!/usr/bin/env python3
"""
Scalability — compute time vs N (log-log, linear projection).
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "output" / "plots" / "benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOOLS = [
    ("MLDSPP XGBoost",      [[100, 0.001], [500, 0.008]], "#942C76", "o"),
    ("PromoterLCNN",        [[100, 0.41],  [500, 0.48],  [1988, 0.40]],   "#228B22", "s"),
    ("PromoTech RF-HOT",    [[1988, 97.4]],                                "#E07614", "D"),
    ("iPro-MP (H. pylori)", [[100, 289.3], [500, 262.7], [1988, 270.5]],   "#3D185A", "^"),
]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

for tool, pts, color, marker in TOOLS:
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    if len(pts) >= 2:
        m, b = np.polyfit(np.log10(x), np.log10(y), 1)
        px = np.logspace(1.5, 3.5, 100)
        py = 10**b * px**m
        ax.plot(px, py, "--", color=color, lw=1.2, alpha=0.5)
    elif len(pts) == 1:
        px = np.array([1, 2500])
        py = px * (y[0] / x[0])
        ax.plot(px, py, "--", color=color, lw=1, alpha=0.3)

    ax.scatter(x, y, color=color, marker=marker, s=70, zorder=5,
               edgecolors="white", linewidth=0.8, label=tool)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of sequences", fontsize=11)
ax.set_ylabel("Compute time (seconds)", fontsize=11)
ax.set_title("Resource Scalability — Compute Time", fontweight="bold", fontsize=12)
ax.set_xlim(80, 2500)
ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT_DIR / "scalability.svg", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "scalability.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: scalability.{svg,png}")
for tool, pts, *_ in TOOLS:
    if len(pts) >= 2:
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        m, _ = np.polyfit(np.log10(x), np.log10(y), 1)
        kind = "sub-linear" if m < 1 else "linear" if abs(m-1) < 0.1 else "super-linear"
        print(f"  {tool:<28} slope={m:.3f}  ({kind})")
