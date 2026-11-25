#!/usr/bin/env python3
"""Title-free per-run ROC overlay (one figure, PNG/PDF) and AUC table.

Usage:
    python src/analysis/generate_auc_plots.py <run_dir> [--tools k1,k2]

Accepts a run, its primary alias, or its predictions directory. Reads
1_inference/predictions (legacy fallback supported), and writes:
  1_inference/roc_auc_{dataset}.{png,pdf}
  3_tables/metrics_rows.tsv
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis"))
from analyze_run import analyze_run
from run_layout import RunLayout


def main(run_dir: Path, only=None):
    layout = RunLayout.resolve(run_dir)
    pred_root = Path(run_dir) if Path(run_dir).name == "predictions" else layout.predictions
    if not pred_root.is_dir():
        raise SystemExit(f"ERROR: no predictions in {pred_root}")
    return analyze_run(pred_root, layout.dataset_name, layout.root, only=only)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Per-run title-free ROC + AUC table.")
    p.add_argument("run_dir", help="Run root, primary alias or predictions directory")
    p.add_argument("--tools", default=None, help="Comma-separated registry keys to include")
    p.add_argument("--all-tools", action="store_true",
                   help="Include all 9 registry tools instead of the default 7")
    a = p.parse_args()
    if a.tools:
        only = {t.strip() for t in a.tools.split(",") if t.strip()}
    elif a.all_tools:
        only = None
    else:
        from bench_labels import DEFAULT_TOOLS_7
        only = set(DEFAULT_TOOLS_7)
    main(Path(a.run_dir), only)
