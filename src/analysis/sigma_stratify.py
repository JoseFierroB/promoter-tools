#!/usr/bin/env python3
"""Sigma-stratified evaluation for one CLI run (agnostic).

Classes come from the positives metadata sidecar
(`positives_81bp_metadata.tsv`, column `Sigma_Factor`):
  SigA               -> SigA
  ComX / SigX        -> ComX (SigX == ComX, merged)
  empty / NaN / None -> None (monoboxes + TF-regulated/orphans)

Scores come from the run's canonical predictions
(`1_inference/predictions/{name}_{fam}[.tsv]`, same loaders as
`analyze_run.py`). Positional tools (posneg_dir / ipromp) map scores to
IDs by positives order; ID tools (prompt/prokbert single_tsv) map by ID.

Writes into the run dir (one file per sigma class):
  1_inference/sigma/roc_SigA_{name}.{png,pdf}
  1_inference/sigma/roc_ComX_{name}.{png,pdf}
  1_inference/sigma/roc_None_{name}.{png,pdf}
  3_tables/sigma_factor_benchmark_metrics.tsv

Usage:
    pixi run python src/analysis/sigma_stratify.py \
      --run-dir output/2609xx_runs/4_d39v_tigr4_high_sigma/1cpu-gpu \
      --name 4_d39v_tigr4_high_sigma \
      --metadata data/benchmark/canonical_15_datasets/4_d39v_tigr4_high_sigma/positives_81bp_metadata.tsv
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from analyze_run import TOOLS, FAM, load_all  # noqa: E402

try:
    from bench_labels import PALETTE, TOOL_ORDER
except Exception:
    from src.analysis.bench_labels import PALETTE, TOOL_ORDER

CLASS_ORDER = ["SigA", "ComX", "None"]

SIGMA_OUTPUTS = (
    tuple(f"roc_{s}_{{name}}.{e}" for s in CLASS_ORDER for e in ("png", "pdf"))
)


def _outputs_current(out_dir: Path, tables_path: Path, name: str,
                     newest_input_mtime: float) -> bool:
    """True when all 7 sigma outputs exist and predate no input.

    Used to run sigma once per dataset: sibling regime dirs
    (output/<date>_runs/<name>/<config>/) share inputs, so the first
    regime generates and the rest skip unless --force is given.
    """
    wanted = [out_dir / f.format(name=name) for f in SIGMA_OUTPUTS]
    wanted.append(tables_path)
    if not all(p.is_file() for p in wanted):
        return False
    return min(p.stat().st_mtime for p in wanted) >= newest_input_mtime


def _newest_input_mtime(pred_root: Path, metadata: Path) -> float:
    times = [metadata.stat().st_mtime] if metadata.is_file() else []
    times += [p.stat().st_mtime for p in pred_root.rglob("*") if p.is_file()]
    return max(times) if times else 0.0


def load_classes(meta_path: Path):
    meta = pd.read_csv(meta_path, sep="\t")
    if "Sequence_ID" not in meta.columns or "Sigma_Factor" not in meta.columns:
        raise SystemExit(f"ERROR: {meta_path} needs Sequence_ID + Sigma_Factor columns")
    pos_ids = meta["Sequence_ID"].astype(str).tolist()
    classes = {}
    for _, r in meta.iterrows():
        v = r["Sigma_Factor"]
        v = "" if pd.isna(v) else str(v).strip()
        if v == "SigA":
            classes[str(r["Sequence_ID"])] = "SigA"
        elif v in ("ComX", "SigX"):
            classes[str(r["Sequence_ID"])] = "ComX"
        else:
            classes[str(r["Sequence_ID"])] = "None"
    return pos_ids, classes


def per_id_scores(pred_root: Path, name: str, pos_ids):
    """Return {display_label: ({pid: score}, [neg scores])} using analyze_run loaders."""
    out = {}
    scored = load_all(pred_root, name)
    # need n_pos hint = len(pos_ids) for ipromp (already handled inside load_all)
    for key, (label, y_true, y_score) in scored.items():
        n_pos = int(y_true.sum())
        pos_scores = y_score[:n_pos]
        neg_scores = y_score[n_pos:]
        fam = FAM[key]
        kind = TOOLS[key][1]
        if kind == "single_tsv":
            # ID-grounded mapping (order-independent)
            for f in [pred_root / f"{name}_{fam}.tsv",
                      pred_root / f"{name}_{fam}_{fam}.tsv"]:
                if f.exists():
                    df = pd.read_csv(f, sep="\t")
                    id_col = "ID" if "ID" in df.columns else df.columns[0]
                    m = dict(zip(df[id_col].astype(str), df["PRED"].values))
                    pmap = {pid: float(m[pid]) for pid in pos_ids if pid in m}
                    # negs: LABEL == 0 rows
                    if "LABEL" in df.columns:
                        nscores = df.loc[df["LABEL"] == 0, "PRED"].values.astype(float)
                    else:
                        nscores = np.array(neg_scores, dtype=float)
                    out[label] = (pmap, nscores)
                    break
            else:
                pmap = dict(zip(pos_ids, map(float, pos_scores)))
                out[label] = (pmap, np.array(neg_scores, dtype=float))
        else:
            # positional: prediction order == positives FASTA/metadata order
            pmap = dict(zip(pos_ids, map(float, pos_scores)))
            out[label] = (pmap, np.array(neg_scores, dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser(description="Sigma-stratified ROC/tables for one run.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even when current sigma outputs exist "
                         "(own dir or a sibling regime dir).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_root = run_dir / "1_inference" / "predictions"
    if not pred_root.exists():  # legacy fallback
        pred_root = run_dir / "predictions"
    # Once-per-dataset guard: skip when own or sibling-regime outputs
    # are newer than every input (predictions + metadata).
    if not args.force:
        newest_in = _newest_input_mtime(pred_root, Path(args.metadata))
        own = (run_dir / "1_inference" / "sigma",
               run_dir / "3_tables" / "sigma_factor_benchmark_metrics.tsv")
        if _outputs_current(*own, args.name, newest_in):
            print(f"  [sigma] skip: current outputs in {run_dir} (use --force).")
            return
        for sib in sorted(run_dir.parent.iterdir()):
            if sib == run_dir or not sib.is_dir():
                continue
            sib_out = (sib / "1_inference" / "sigma",
                       sib / "3_tables" / "sigma_factor_benchmark_metrics.tsv")
            if _outputs_current(*sib_out, args.name, newest_in):
                print(f"  [sigma] skip: current outputs in sibling {sib.name} "
                      f"(use --force).")
                return
    pos_ids, classes = load_classes(Path(args.metadata))
    counts = pd.Series([classes[p] for p in pos_ids]).value_counts().to_dict()
    print(f"classes: SigA={counts.get('SigA',0)} ComX={counts.get('ComX',0)} "
          f"None={counts.get('None',0)} total={len(pos_ids)}")

    tool_maps = per_id_scores(pred_root, args.name, pos_ids)
    if not tool_maps:
        raise SystemExit(f"ERROR: no predictions in {pred_root}")

    rows, roc_data = [], {c: {} for c in CLASS_ORDER}
    for label, (pmap, nscores) in tool_maps.items():
        for sig in CLASS_ORDER:
            sub = [p for p in pos_ids if classes.get(p) == sig and p in pmap]
            pscores = [pmap[p] for p in sub]
            y_true = np.array([1] * len(pscores) + [0] * len(nscores))
            y_pred = np.array(list(pscores) + list(map(float, nscores)))
            if len(set(y_pred)) < 2 or len(pscores) == 0:
                fpr, tpr, a = np.array([0., 1.]), np.array([0., 1.]), 0.5
            else:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                a = auc(fpr, tpr)
            roc_data[sig][label] = (fpr, tpr, float(a))
            rows.append({"dataset": args.name, "tool": label, "sigma": sig,
                         "n_pos": len(pscores), "n_neg": len(nscores),
                         "auc": round(float(a), 4)})

    tables = run_dir / "3_tables"
    tables.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame(rows)
    met.to_csv(tables / "sigma_factor_benchmark_metrics.tsv", sep="\t", index=False)
    print(f"  [sigma] metrics: {len(met)} rows -> 3_tables/sigma_factor_benchmark_metrics.tsv")
    print(met.pivot(index="tool", columns="sigma", values="auc").to_string())

    sig_dir = run_dir / "1_inference" / "sigma"
    sig_dir.mkdir(parents=True, exist_ok=True)

    # One ROC file per sigma class (no combined panels, no heatmap, no boxplots)
    order_idx = {t: i for i, t in enumerate(TOOL_ORDER)}
    for sig in CLASS_ORDER:
        fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
        for label, (fpr, tpr, a) in sorted(roc_data[sig].items(),
                                           key=lambda x: order_idx.get(x[0], 999)):
            s = PALETTE.get(label, {"color": "#333", "ls": "-", "lw": 1.8})
            ax.plot(fpr, tpr, color=s["color"], linestyle=s["ls"],
                    linewidth=s["lw"], label=f"{label} (AUC = {a:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=1.2, ls="--", label="Chance (0.500)")
        ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9.2, frameon=True, framealpha=0.95)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(sig_dir / f"roc_{sig}_{args.name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print(f"  [sigma] ROC saved: 1_inference/sigma/roc_{sig}_{args.name}.png/.pdf")


if __name__ == "__main__":
    main()
