#!/usr/bin/env python3
"""A1: ROC estratificado por sigma factor — IGR dataset (723 pos + 723 neg).

Splits positives into SigA(310) / None(400) / SigX(12) and computes
per-group ROC curves vs the same 723 intergenic negatives, for each tool.
Outputs one PNG+SVG per tool + a summary bar chart.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "output/predictions_igr/d39v"
META = ROOT / "data/benchmark_igr/d39v/positives_81bp_igr_metadata.tsv"
OUT = ROOT / "output/plots/sigma_analysis"
OUT.mkdir(parents=True, exist_ok=True)

TOOLS = {
    "iPro-MP [gLM]":        ("ipromp/ipromp_12_predictions.csv", None),
    "LCNN [CNN]":           ("lcnn/lcnn_pos.csv", "lcnn/lcnn_neg.csv"),
    "PromoTech [RF]":       ("promotech/workdir/hot_pg_pos/sequences_predictions.csv",
                             "promotech/workdir/hot_pg_neg/sequences_predictions.csv"),
    "MLDSPP 0% [BDT]":      ("mldspp_pos.csv", "mldspp_neg.csv"),
    "MEME [De novo]":       ("meme_pos.csv", "meme_neg.csv"),
    "FIMO [PWM]":           ("fimo_prok_pos.csv", "fimo_prok_neg.csv"),
}

GROUP_COLORS = {"SigA (exp)": "#1f77b4", "None (unassigned)": "#ff7f0e", "SigX (n=12)": "#2ca02c"}


def load_scores(pred_dir, pos_rel, neg_rel):
    pos_df = pd.read_csv(pred_dir / pos_rel, sep="\t")
    if neg_rel is None:  # ipromp combined
        col = "Probability" if "Probability" in pos_df.columns else "PRED"
        n = len(pos_df) // 2
        return pos_df[col].values[:n], pos_df[col].values[n:]
    neg_df = pd.read_csv(pred_dir / neg_rel, sep="\t")
    pcol = "PRED" if "PRED" in pos_df.columns else ("score" if "score" in pos_df.columns else pos_df.columns[-1])
    ncol = "PRED" if "PRED" in neg_df.columns else ("score" if "score" in neg_df.columns else neg_df.columns[-1])
    return pos_df[pcol].values, neg_df[ncol].values


def main():
    meta = pd.read_csv(META, sep="\t")
    sigma = meta["Sigma_Factor"].fillna("None").values

    groups = {
        "SigA (exp)": sigma == "SigA",
        "None (unassigned)": sigma == "None",
        "SigX (n=12)": sigma == "SigX",
    }

    summary_rows = []

    for tool_name, (pos_rel, neg_rel) in TOOLS.items():
        try:
            pos_s, neg_s = load_scores(PRED, pos_rel, neg_rel)
        except Exception as e:
            print(f"[SKIP] {tool_name}: {e}")
            continue

        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
        safe_name = tool_name.split("[")[0].strip().replace(" ", "_").replace("/", "_")

        for group_label, mask in groups.items():
            sub_pos = pos_s[mask]
            n_group = len(sub_pos)
            y = np.hstack([np.ones(len(sub_pos)), np.zeros(len(neg_s))])
            s = np.hstack([sub_pos, neg_s])
            fpr, tpr, _ = roc_curve(y, s)
            a = auc(fpr, tpr)

            lw = 2.2 if "SigA" in group_label else (2.0 if "None" in group_label else 1.5)
            ls = "-" if "SigA" in group_label else ("--" if "None" in group_label else ":")
            ax.plot(fpr, tpr, color=GROUP_COLORS[group_label], linewidth=lw, linestyle=ls,
                    label=f"{group_label} (N={n_group}, AUC={a:.3f})")

            summary_rows.append({"tool": tool_name, "group": group_label,
                                 "n_positives": n_group, "AUC": round(a, 4)})

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.3)
        ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
        ax.set_title(f"ROC by Sigma Factor — {tool_name}\nIGR dataset (723 pos / 723 neg)",
                     fontsize=12, fontweight="bold", pad=10)
        ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=0.95)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        for ext in ["png", "svg"]:
            fig.savefig(OUT / f"roc_sigma_{safe_name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {tool_name}")

    # ── Summary bar chart ──
    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT / "auc_by_sigma.tsv", sep="\t", index=False)
    pivot = df.pivot(index="tool", columns="group", values="AUC")

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    x = np.arange(len(pivot))
    w = 0.25
    for i, col_label in enumerate(["SigA (exp)", "None (unassigned)", "SigX (n=12)"]):
        vals = pivot[col_label].values if col_label in pivot.columns else np.zeros(len(pivot))
        bars = ax.bar(x + (i - 1) * w, vals, w, label=col_label,
                      color=GROUP_COLORS[col_label], edgecolor="black", linewidth=0.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("AUC-ROC", fontsize=11, fontweight="bold")
    ax.set_title("Model Performance by Sigma Factor Group — IGR Benchmark",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(frameon=True, fontsize=9.5, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.55, 1.02)
    plt.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"auc_by_sigma_summary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n=== RESUMEN AUC POR GRUPO SIGMA ===")
    print(pivot.to_string())
    print(f"\nPlots guardados en: {OUT}")


if __name__ == "__main__":
    main()