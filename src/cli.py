#!/usr/bin/env python3
"""Unified CLI for promoter-tools — dataset generation, benchmarking, results.

Usage:
    pixi run python src/cli.py run lcnn
    pixi run python src/cli.py run --slurm promotech_hot
    pixi run python src/cli.py results /path/to/output/
    pixi run python src/cli.py dataset generate -u 60 -d 20
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
        from src.runner.slurm import SlurmRunner
        runner = SlurmRunner()
    else:
        from src.runner.local import LocalRunner
        runner = LocalRunner(n_runs=args.runs, warmup=not args.no_warmup)

    if not runner.available():
        print(f"Runner '{type(runner).__name__}' not available.")
        sys.exit(1)

    import pandas as pd
    results = []
    for tool in tools:
        m = runner.run(tool)
        results.append(m)
        status = "OK" if m["success"] else "FAIL"
        time_s = f"{m['time_s']:.1f}s" if m.get("time_s") else "N/A"
        print(f"    [{status}] {time_s}")

    df = pd.DataFrame(results)
    out_tsv = args.output or str(ROOT / "output/tables/resource_metrics.tsv")
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nMetrics saved: {out_tsv}")


def cmd_results(args):
    """Process benchmark results directory."""
    from src.analysis.process_results import main as process
    sys.argv = ["process_results.py", str(args.directory)]
    process()


def cmd_dataset(args):
    """Generate positive and negative datasets."""
    import subprocess

    fasta = args.fasta or str(ROOT / "data/reference/D39V.fna")
    gff_tss = args.gff_tss or str(ROOT / "data/reference/D39V_annotation_TSS_Victor.gff")
    gff_cds = args.gff_cds or str(ROOT / "data/reference/sequence.gff3")
    out_dir = args.output or str(ROOT / "output")

    if args.positives:
        cmd = f"pixi run python src/dataset/positive_tss.py --fasta {fasta} --gff {gff_tss} --gff-cds {gff_cds} -u {args.upstream} -d {args.downstream} -o {out_dir}"
        print(f"[RUN] {cmd}")
        subprocess.run(cmd, shell=True, cwd=str(ROOT))

    if args.negatives:
        cmd = f"pixi run python src/dataset/negatives_tss_master.py --gff-cds {gff_cds} --fasta {fasta} --gff-tss {gff_tss} --window {args.upstream + 1 + args.downstream} --limit {args.neg_limit} -o {out_dir}"
        print(f"[RUN] {cmd}")
        subprocess.run(cmd, shell=True, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Promoter Tools — Benchmark Pipeline")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Execute benchmark tools")
    p_run.add_argument("tools", nargs="*", help="Tool keys to run (lcnn, promotech_hot, ...)")
    p_run.add_argument("--slurm", action="store_true", help="Use Slurm backend")
    p_run.add_argument("--runs", type=int, default=1, help="Number of independent runs (N≥3 recommended)")
    p_run.add_argument("--no-warmup", action="store_true", help="Skip warmup inference")
    p_run.add_argument("-o", "--output", help="Output TSV path")

    # results
    p_res = sub.add_parser("results", help="Process benchmark results directory")
    p_res.add_argument("directory", help="Results directory path")

    # dataset
    p_ds = sub.add_parser("dataset", help="Generate positive/negative datasets")
    p_ds.add_argument("action", choices=["generate"], default="generate", nargs="?")
    p_ds.add_argument("-u", "--upstream", type=int, default=60)
    p_ds.add_argument("-d", "--downstream", type=int, default=20)
    p_ds.add_argument("--fasta", help="Genome FASTA path")
    p_ds.add_argument("--gff-tss", help="TSS GFF path")
    p_ds.add_argument("--gff-cds", help="CDS GFF path")
    p_ds.add_argument("-o", "--output", help="Output directory")
    p_ds.add_argument("--positives", action="store_true", default=True, help="Generate positives")
    p_ds.add_argument("--negatives", action="store_true", default=True, help="Generate negatives")
    p_ds.add_argument("--neg-limit", type=int, default=1000, help="Negative sequence limit")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "results":
        cmd_results(args)
    elif args.command == "dataset":
        cmd_dataset(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
