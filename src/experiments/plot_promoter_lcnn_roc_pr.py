#!/usr/bin/env python3
"""
Plots ROC and Precision-Recall Curves for PromoterLCNN Zero-Shot Benchmark.

Reads JSON results from output/plots/promoter_lcnn/promoter_lcnn_zero_shot_results.json
and generates publication-quality plots.

Usage:
    pixi run python src/experiments/plot_promoter_lcnn_roc_pr.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
JSON_PATH = ROOT / "output/plots/promoter_lcnn/promoter_lcnn_zero_shot_results.json"
PLOTS_DIR = ROOT / "output/plots/promoter_lcnn"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "D39V (Cappable-seq Primary)": "#1f77b4",     # Deep Blue
    "TIGR4 (High Conf Primary)": "#2ca02c",       # Forest Green
    "TIGR4 (Extended Primary)": "#ff7f0e",        # Orange
}


def main():
    if not JSON_PATH.exists():
        print(f"[ERROR] Results JSON not found at {JSON_PATH}")
        return

    with open(JSON_PATH) as f:
        data = json.load(f)

    # 1. Plot ROC Curves
    plt.figure(figsize=(8, 7), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for name, res in data.items():
        color = COLORS.get(name, "#333333")
        auc_val = res["roc_auc"]
        plt.plot(
            res["fpr"],
            res["tpr"],
            label=f"{name} (AUC = {auc_val:.4f})",
            color=color,
            linewidth=2.5,
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)", linewidth=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    plt.title("PromoterLCNN Zero-Shot ROC Curves (D39V vs. TIGR4)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    roc_out = PLOTS_DIR / "promoter_lcnn_zero_shot_roc_curves.png"
    plt.savefig(roc_out, dpi=300)
    plt.close()
    print(f"[SUCCESS] ROC curve plot generated ➔ {roc_out}")

    # 2. Plot Precision-Recall Curves
    plt.figure(figsize=(8, 7), dpi=300)

    for name, res in data.items():
        color = COLORS.get(name, "#333333")
        pr_auc_val = res["pr_auc"]
        plt.plot(
            res["recall_curve"],
            res["precision_curve"],
            label=f"{name} (PR-AUC = {pr_auc_val:.4f})",
            color=color,
            linewidth=2.5,
        )

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    plt.title("PromoterLCNN Zero-Shot Precision-Recall Curves", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="none")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    pr_out = PLOTS_DIR / "promoter_lcnn_zero_shot_pr_curves.png"
    plt.savefig(pr_out, dpi=300)
    plt.close()
    print(f"[SUCCESS] PR curve plot generated ➔ {pr_out}")


if __name__ == "__main__":
    main()
