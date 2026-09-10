#!/usr/bin/env python3
"""Unified CLI for promoter-tools — benchmarking and results analysis.

Agnostic pipeline: it only receives data (pos/neg FASTAs or a directory
containing the pair). Dataset creation lives outside the pipeline
(see experiments/canonical_15_datasets/).

Usage:
    pixi run python src/cli.py run mldspp --pos pos.fasta --neg neg.fasta
    pixi run python src/cli.py run mldspp prompt --input-dir data/benchmark/canonical_15_datasets/4_d39v_tigr4_high --name myrun --plots
    pixi run python src/cli.py run prokbert --input-dir <dir> --threads 4 --cpu-only --plots
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}
for _k, _v in _THREAD_CAPS.items():
    os.environ.setdefault(_k, _v)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

# registry key -> (family for canonical layout, single-file output?)
# Canonical layout inside a run: predictions/{name}_{fam}[.tsv]
# New tools: add one line here (dir layout default; single-file if the runner
# writes one combined TSV when -o points at a file).
TOOL_LAYOUT = {
    "prokbert": ("prokbert", True),
    "prompt": ("prompt", True),
    "mldspp": ("mldspp", False),
    "mldspp_75": ("mldspp_75", False),
    "ipromp_sp12": ("ipromp", False),
    "lcnn": ("lcnn", False),
    "promotech_hot": ("promotech", False),
    "promotech_tetra": ("promotech_tetra", False),
    "fimo_prok": ("fimo", False),
    "fimo_db": ("fimo_db", False),
    "meme": ("meme", False),
}


def tool_output_target(pred_root: Path, name: str, short_name: str) -> Path:
    if short_name == "fimo_prok":
        return pred_root / f"{name}_fimo"  # was {name}_fimo_fimo_dir (legacy)
    fam, single = TOOL_LAYOUT.get(short_name, (short_name, False))
    base = pred_root / f"{name}_{fam}"
    return base.with_suffix(".tsv") if single else base


def sanitize_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")
    return s or "results"


def sha_inputs(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.resolve()).encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def resolve_inputs(args):
    """Strict input resolution. Returns (pos, neg, suggested_name)."""
    if args.input_dir:
        d = Path(args.input_dir)
        pos, neg = d / "positives_81bp.fasta", d / "negatives_81bp.fasta"
        if not (pos.exists() and neg.exists()):
            print(f"ERROR: --input-dir '{d}' must contain "
                  f"positives_81bp.fasta and negatives_81bp.fasta.")
            sys.exit(2)
        return pos, neg, sanitize_name(d.name)
    if args.pos and args.neg:
        pos, neg = Path(args.pos), Path(args.neg)
        for p, flag in ((pos, "--pos"), (neg, "--neg")):
            if not p.exists():
                print(f"ERROR: {flag} file not found: {p}")
                sys.exit(2)
        return pos, neg, None
    if args.pos or args.neg:
        print("ERROR: --pos and --neg must be given together (or use --input-dir).")
        sys.exit(2)
    print("ERROR: no input data. Pass --input-dir <dir> or --pos <f> --neg <f>.")
    sys.exit(2)


def derive_configuration(args) -> str:
    """Hardware configuration label from threads + device flags."""
    if args.gpu and args.cpu_only:
        print("ERROR: --gpu and --cpu-only are contradictory.")
        sys.exit(2)
    threads = args.threads or 1
    if args.gpu:
        gpu = True
    elif args.cpu_only:
        gpu = False
    else:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
        except Exception:
            gpu = False
    return f"{threads}cpu" + ("-gpu" if gpu else "")


def read_status(target: Path):
    try:
        return json.loads((target / "STATUS.json").read_text())
    except Exception:
        return None


def prepare_run_dir(base: Path, configuration: str, tool_keys, n_pos, n_neg, sha,
                      pos_src: str = "", neg_src: str = ""):
    """Bump / quarantine / resume logic. Returns (run_root, resumed)."""
    i, candidate = 1, base
    while True:
        target = candidate / configuration
        st = read_status(target)
        if st and st.get("status") == "complete":
            i += 1
            candidate = Path(str(base) + f"_{i:02d}")
            continue
        resumed = False
        if target.exists() and any(target.iterdir()):
            same = (st and st.get("input_sha") == sha
                    and sorted(st.get("tools", [])) == sorted(tool_keys)
                    and st.get("n_pos") == n_pos and st.get("n_neg") == n_neg)
            if same:
                resumed = True
            else:
                stamp = time.strftime("%H%M%S")
                q = Path(str(candidate) + f"_partial_{stamp}") / configuration
                q.parent.mkdir(parents=True, exist_ok=True)
                target.rename(q)
                print(f"  [quarantine] incomplete/stale run moved to {q.parent.name}/{configuration}/")
        target.mkdir(parents=True, exist_ok=True)
        (target / "STATUS.json").write_text(json.dumps({
            "status": "running",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tools": sorted(tool_keys),
            "n_pos": n_pos, "n_neg": n_neg,
            "input_sha": sha, "configuration": configuration,
            "pos_src": pos_src, "neg_src": neg_src,
        }, indent=1))
        return target, resumed


def count_seqs(p: Path) -> int:
    n = 0
    with open(p) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def cmd_run(args):
    """Execute benchmark tool(s)."""
    from src.benchmark.tools import PROMOTER_TOOLS, get_enabled_tools, enable
    import bench_labels as _bl
    run_date = _bl.run_date

    valid = sorted(PROMOTER_TOOLS)
    if not args.tools:
        print("ERROR: specify at least one tool key.\nAvailable tools:\n  " + "\n  ".join(valid))
        sys.exit(2)
    unknown = [t for t in args.tools if t not in PROMOTER_TOOLS]
    if unknown:
        print("ERROR: unknown tool key(s): " + ", ".join(unknown) + "\nAvailable tools:\n  " + "\n  ".join(valid))
        sys.exit(2)

    enable(args.tools)

    if args.cpu_only:
        os.environ["PROMOTER_TOOLS_CPU_ONLY"] = "1"
    if args.threads:
        os.environ["PROMOTER_TOOLS_THREADS"] = str(args.threads)
    if args.no_timeout:
        os.environ["PROMOTER_TOOLS_NO_TIMEOUT"] = "1"

    tools = get_enabled_tools()
    if not tools:
        print("No tools enabled.")
        sys.exit(1)

    # --- strict agnostic inputs: input dir pair or explicit pos+neg ---
    pos_fasta, neg_fasta, suggested = resolve_inputs(args)
    n_pos, n_neg = count_seqs(pos_fasta), count_seqs(neg_fasta)
    name = sanitize_name(args.name) if args.name else (suggested or "results")
    token = name
    sha = sha_inputs(pos_fasta, neg_fasta)

    configuration = derive_configuration(args)
    date = run_date()
    if args.name or suggested:
        base = ROOT / "output" / f"{date}_runs" / name
    else:
        base = ROOT / "output" / f"{date}_results"
    run_root, resumed = prepare_run_dir(base, configuration, args.tools, n_pos, n_neg, sha,
                                          str(pos_fasta.resolve()), str(neg_fasta.resolve()))
    pred_root = Path(args.output_dir) if args.output_dir else run_root / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_root} (name={token}, configuration={configuration}"
          f"{', resumed' if resumed else ''})")

    if args.slurm:
        from src.backend.slurm import SlurmRunner
        runner = SlurmRunner(pos_fasta=pos_fasta, neg_fasta=neg_fasta)
    else:
        from src.backend.local import LocalRunner
        runner = LocalRunner(n_runs=args.runs, output_dir=str(pred_root),
                             pos_fasta=pos_fasta, neg_fasta=neg_fasta)

    if not runner.available():
        print(f"Runner '{type(runner).__name__}' not available.")
        sys.exit(1)

    import pandas as pd
    results = []
    for i, key in enumerate(args.tools):
        tool = PROMOTER_TOOLS[key]
        print(f"[{i+1}/{len(args.tools)}]", end=" ", flush=True)
        if not args.slurm:
            # per-tool namespaced output preserving the canonical
            # {name}_{fam}[.tsv] layout (LocalRunner only)
            runner.output_dir = str(tool_output_target(pred_root, token, key))
        m = runner.run(tool)
        results.append(m)
        if not m["success"]:
            print("FAIL", flush=True)

    df = pd.DataFrame(results)
    df["name"] = token
    df["configuration"] = configuration
    df["run_date"] = run_date()
    out_tsv = args.output or str(run_root / "resource_metrics.tsv")
    if Path(out_tsv).exists() and not args.output:
        prev = pd.read_csv(out_tsv, sep="\t")
        if "tool" in prev.columns and set(prev["tool"]) != set(df["tool"]):
            print(f"  WARNING: {out_tsv} contains {len(prev)} rows from a previous run "
                  f"with different tools — it will be overwritten. Use -o to keep separate files.")
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nMetrics saved: {out_tsv}")

    st = read_status(run_root)
    st = st or {}
    st.update({"status": "complete" if all(r.get("success") for r in results) else "failed",
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (run_root / "STATUS.json").write_text(json.dumps(st, indent=1))

    if args.plots:
        from analyze_run import analyze_run
        analyze_run(pred_root, token, run_root)


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
                       help="Predictions output dir (default: namespaced run dir)")
    p_run.add_argument("--input-dir", default=None,
                       help="Directory with positives_81bp.fasta + negatives_81bp.fasta")
    p_run.add_argument("--name", default=None,
                       help="Run label (default: input dir basename, else dated results)")
    p_run.add_argument("--plots", action="store_true",
                       help="ROC + AUC rows for this run after scoring")
    p_run.add_argument("--pos", default=None, help="Positive FASTA (with --neg)")
    p_run.add_argument("--neg", default=None, help="Negative FASTA (with --pos)")
    p_run.add_argument("--cpu-only", action="store_true",
                       help="Force CPU for all tools (no GPU detection)")
    p_run.add_argument("--gpu", action="store_true",
                       help="Allow GPU for capable tools (with --threads N)")
    p_run.add_argument("--threads", type=int, default=None,
                       help="Max threads per tool (sets OMP/MKL/OPENBLAS threads)")
    p_run.add_argument("--no-timeout", action="store_true",
                       help="Disable per-tool timeouts (run until completion)")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare runs across datasets/days")
    p_cmp.add_argument("runs", nargs="+",
                       help="Run dirs (output/<date>_runs/<name>/<config>/)")
    p_cmp.add_argument("-o", "--output", default=None,
                       help="Comparison dir (default: output/<YYMMDD>_comparison)")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        from datetime import date
        from compare_runs import compare_runs
        compare_runs(args.runs, Path(args.output) if args.output
                     else Path(f"output/{date.today().strftime('%y%m%d')}_comparison"))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
