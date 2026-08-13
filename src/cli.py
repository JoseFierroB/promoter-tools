#!/usr/bin/env python3
"""Unified CLI for promoter-tools — benchmarking and results analysis.

Usage:
    pixi run python src/cli.py run lcnn
    pixi run python src/cli.py run --slurm promotech_hot
    pixi run python src/cli.py run lcnn --pos data/tigr4/positives_high_81bp.fasta \\
        --neg data/tigr4/negatives_high_81bp.fasta --output-dir output/tigr4/predictions
    pixi run python src/cli.py results /path/to/output/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def cmd_run(args):
    """Execute benchmark tool(s)."""
    from src.benchmark.tools import PROMOTER_TOOLS, get_enabled_tools, enable

    if args.tools:
        enable(args.tools)

    tools = get_enabled_tools()
    if not tools:
        print("No tools enabled.")
        sys.exit(1)

    if args.slurm:
        from src.backend.slurm import SlurmRunner
        runner = SlurmRunner(pos_fasta=args.pos, neg_fasta=args.neg)
    else:
        from src.backend.local import LocalRunner
        runner = LocalRunner(n_runs=args.runs, output_dir=args.output_dir,
                             pos_fasta=args.pos, neg_fasta=args.neg)

    if not runner.available():
        print(f"Runner '{type(runner).__name__}' not available.")
        sys.exit(1)

    import pandas as pd
    results = []
    for i, tool in enumerate(tools):
        print(f"[{i+1}/{len(tools)}]", end=" ", flush=True)
        m = runner.run(tool)
        results.append(m)
        status = "FAIL" if not m["success"] else ""

    df = pd.DataFrame(results)
    out_tsv = args.output or str(ROOT / "output/tables/resource_metrics.tsv")
    if not args.output and Path(out_tsv).exists():
        prev = pd.read_csv(out_tsv, sep="\t")
        if "tool" in prev.columns and set(prev["tool"]) != set(df["tool"]):
            print(f"  WARNING: {out_tsv} contains {len(prev)} rows from a previous run "
                  f"with different tools — it will be overwritten. Use -o to keep separate files.")
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nMetrics saved: {out_tsv}")


def cmd_results(args):
    """Process benchmark results directory."""
    from src.analysis.process_results import main as process
    sys.argv = ["process_results.py", str(args.directory)]
    process()


def main():
    parser = argparse.ArgumentParser(description="Promoter Tools — Benchmark Pipeline")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Execute benchmark tools")
    p_run.add_argument("tools", nargs="*", help="Tool keys to run (lcnn, promotech_hot, ...)")
    p_run.add_argument("--slurm", action="store_true", help="Use Slurm backend")
    p_run.add_argument("--runs", type=int, default=1, help="Number of independent runs (N≥3 recommended)")
    p_run.add_argument("-o", "--output", help="Output TSV path")
    p_run.add_argument("--output-dir", default=None,
                       help="Predictions output dir (default: output/predictions)")
    p_run.add_argument("--pos", default=None, help="Positive FASTA (default: d39v confirmed)")
    p_run.add_argument("--neg", default=None, help="Negative FASTA (default: d39v confirmed)")

    # results
    p_res = sub.add_parser("results", help="Process benchmark results directory")
    p_res.add_argument("directory", help="Results directory path")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "results":
        cmd_results(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
