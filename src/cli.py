#!/usr/bin/env python3
"""Unified CLI for promoter-tools — benchmarking and results analysis.

Agnostic pipeline: it only receives data (pos/neg FASTAs or a directory
containing the pair). Dataset creation lives outside the pipeline
(see experiments/datasets/).

Usage:
    pixi run python src/cli.py run mldspp --pos pos.fasta --neg neg.fasta
    pixi run python src/cli.py run mldspp prompt --input-dir data/benchmark/datasets/2_d39v_tigr4_high --name myrun --plots
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

from run_layout import RunLayout, primary_run_dir

# registry key -> (family for primary layout, single-file output?)
# Primary layout inside a run: 1_inference/predictions/{name}_{fam}[.tsv]
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
    "fimo_prok": ("fimo", False),
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


def _harness_record(args, configuration) -> dict:
    """Harness knobs for STATUS.json traceability.

    Records what the CLI controls; batch size and worker counts live with
    each runner's defaults (see docs/RUNNING.md "Measurement methodology"
    for the per-tool table). Effective device per tool mirrors the runners'
    own rule (cuda iff the run is a -gpu config, the tool is gpu_capable
    and carries a gpu_id, and --cpu-only was not passed); prompt is a known
    exception (always CPU, documented in RUNNING.md).
    """
    from src.benchmark.tools import PROMOTER_TOOLS
    is_gpu_cfg = configuration.endswith("-gpu")
    tools = {}
    for key in (args.tools or []):
        t = PROMOTER_TOOLS.get(key)
        if t is None:
            continue
        dev = ("cuda:" + t.gpu_id) if (not args.cpu_only and is_gpu_cfg
                                       and t.gpu_capable and t.gpu_id) else "cpu"
        tools[key] = {"gpu_id": t.gpu_id, "gpu_capable": t.gpu_capable,
                      "device_effective": dev}
    return {"threads": args.threads or 1, "cpu_only": bool(args.cpu_only),
            "gpu": bool(args.gpu), "no_timeout": bool(args.no_timeout),
            "tools": tools}


def read_status(target: Path):
    try:
        return json.loads((target / "STATUS.json").read_text())
    except Exception:
        return None


def prepare_run_dir(base: Path, configuration: str, tool_keys, n_pos, n_neg, sha,
                      pos_src: str = "", neg_src: str = "", dataset_name: str = None,
                      harness: dict = None):
    """Bump / quarantine / resume logic. Returns (run_root, resumed).

    The dataset name stays independent of the unique run directory and alias.
    `harness` (threads/flags/per-tool device hints) is recorded in STATUS.json
    so two runs are never indistinguishable in metadata.
    """
    base = Path(base)
    date_match = (re.fullmatch(r"(\d{6})_runs", base.parent.name)
                  or re.fullmatch(r"(\d{6})_results", base.name))
    date = date_match.group(1) if date_match else time.strftime("%y%m%d")
    unnamed = bool(re.fullmatch(r"\d{6}_results", base.name))
    name = dataset_name or ("results" if unnamed else base.name)
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
        layout = RunLayout(target)
        layout.ensure_dirs()
        (target / "STATUS.json").write_text(json.dumps({
            "status": "running",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tools": sorted(tool_keys),
            "n_pos": n_pos, "n_neg": n_neg,
            "input_sha": sha, "configuration": configuration,
            "name": name,
            "pos_src": pos_src, "neg_src": neg_src,
            "harness": harness or {},
        }, indent=1))
        instance_name = candidate.name[len(date) + 1:] if unnamed else candidate.name
        alias_base = primary_run_dir(date, instance_name, configuration,
                                       base=ROOT / "output" / "runs")
        alias, alias_index = alias_base, 2
        # Preserve existing aliases, including collisions from truncated names.
        while alias.exists() or alias.is_symlink():
            if alias.resolve() == target.resolve():
                break
            alias = alias_base.with_name(f"{alias_base.name}_{alias_index:02d}")
            alias_index += 1
        if not alias.exists() and not alias.is_symlink():
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(target.resolve(), target_is_directory=True)
        manifest = layout.write_manifest(
            date=date, name=name, config=configuration,
            input_sha=sha, pos_src=pos_src, neg_src=neg_src,
            n_pos=n_pos, n_neg=n_neg, tools=tool_keys,
            legacy_path=str(target.resolve()), primary_path=str(alias),
        )
        manifest["run_id"] = alias.name
        layout.manifest_path.write_text(json.dumps(manifest, indent=2))
        return target, resumed


def count_seqs(p: Path) -> int:
    n = 0
    with open(p) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def _append_run_log(run_root: Path, args, token: str, configuration: str,
                    sha: str, results: list):
    """Append one human-readable block per tool to RUN.log (run root).

    Provenance sidecar: the CLI invocation + relevant env + per-tool
    timings, so a run is reproducible from RUN.log + inputs. TSVs untouched.
    """
    watched = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "PROMOTER_TOOLS_CPU_ONLY",
               "PROMOTER_TOOLS_LCNN_BATCH", "PROMOTER_TOOLS_PROKBERT_BATCH",
               "PROMOTER_TOOLS_PROKBERT_PREFETCH", "IPROMP_SPECIES")
    lines = [f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} {token} [{configuration}]",
             f"# input_sha={sha}",
             "# env: " + ", ".join(f"{k}={os.environ[k]}" for k in watched
                                   if k in os.environ)]
    for r in results:
        lines.append(
            f"{r.get('tool', '?')} wall={r.get('wall_seconds', '?')}s "
            f"train={r.get('train_s', '?')}s infer={r.get('infer_s', '?')}s "
            f"ram={r.get('peak_ram_mb', '?')}MB vram={r.get('peak_vram_mb', '?')}MB "
            f"success={r.get('success', '?')} notes={(r.get('notes') or '')[:120]}")
    with open(run_root / "RUN.log", "a") as fh:
        fh.write("\n".join(lines) + "\n")


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
                                          str(pos_fasta.resolve()), str(neg_fasta.resolve()),
                                          dataset_name=token,
                                          harness=_harness_record(args, configuration))
    layout = RunLayout(run_root)
    # primary predictions lives inside 1_inference/predictions
    if args.output_dir:
        pred_root = Path(args.output_dir).resolve()
    else:
        pred_root = layout.inference / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        manifest = json.loads(layout.manifest_path.read_text())
        manifest["predictions_path"] = str(pred_root)
        layout.manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Run dir: {run_root} (name={token}, configuration={configuration}"
          f"{', resumed' if resumed else ''})")

    if args.slurm:
        from src.backend.slurm import SlurmRunner
        runner = SlurmRunner(pos_fasta=pos_fasta, neg_fasta=neg_fasta,
                             output_dir=str(pred_root))
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
        # Both backends use the primary per-tool {name}_{fam}[.tsv] layout.
        runner.output_dir = str(tool_output_target(pred_root, token, key))
        m = runner.run(tool)
        results.append(m)
        if not m["success"]:
            print("FAIL", flush=True)

    df = pd.DataFrame(results)
    df["name"] = token
    df["configuration"] = configuration
    df["run_date"] = run_date()
    # Keep the capsule complete; -o is an additional export of the same table.
    out_tsv = layout.resources / "resource_metrics.tsv"
    if out_tsv.exists():
        prev = pd.read_csv(out_tsv, sep="\t")
        if "tool" in prev.columns and set(prev["tool"]) != set(df["tool"]):
            print(f"  WARNING: {out_tsv} contains {len(prev)} rows from a previous run "
                  "with different tools — it will be overwritten.")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    layout.link_resource_table(out_tsv)
    print(f"\nMetrics saved: {out_tsv}")
    _append_run_log(run_root, args, token, configuration, sha, results)
    if args.output:
        export = Path(args.output)
        if export.resolve() != out_tsv.resolve():
            export.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(export, sep="\t", index=False)
            print(f"Metrics exported: {export}")

    st = read_status(run_root)
    st = st or {}
    st.update({"status": "complete" if all(r.get("success") for r in results) else "failed",
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    (run_root / "STATUS.json").write_text(json.dumps(st, indent=1))
    manifest = json.loads(layout.manifest_path.read_text())
    manifest.update(status=st["status"], finished=st["finished"])
    layout.manifest_path.write_text(json.dumps(manifest, indent=2))

    if args.plots:
        from analyze_run import analyze_run
        analyze_run(pred_root, token, run_root)
        if getattr(args, "no_compute_plots", False):
            print("  [plots] per-regime compute plots skipped (--no-compute-plots); "
                  "resource_metrics.tsv kept for comparisons.")
        else:
            # A plotting failure must make --plots fail visibly, including import errors.
            try:
                from generate_compute_plots import main_run as _gen_compute
                _gen_compute(run_root)
            except (Exception, SystemExit) as exc:
                raise SystemExit(f"ERROR: compute plots failed for {run_root}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Promoter Tools — Benchmark Pipeline")
    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Execute benchmark tools")
    p_run.add_argument("tools", nargs="*", help="Tool keys to run (lcnn, promotech_hot, ...)")
    p_run.add_argument("--slurm", action="store_true", help="Use Slurm backend")
    p_run.add_argument("--runs", type=int, default=1, help="Number of independent runs (N≥3 recommended)")
    p_run.add_argument("-o", "--output", help="Additional metrics TSV export (primary table is always kept)")
    p_run.add_argument("--output-dir", default=None,
                       help="Predictions output dir (default: namespaced run dir)")
    p_run.add_argument("--input-dir", default=None,
                       help="Directory with positives_81bp.fasta + negatives_81bp.fasta")
    p_run.add_argument("--name", default=None,
                       help="Run label (default: input dir basename, else dated results)")
    p_run.add_argument("--plots", action="store_true",
                       help="ROC + time/RAM plots (PNG/PDF) + AUC table after scoring")
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
    p_run.add_argument("--no-compute-plots", action="store_true",
                       help="With --plots: skip per-regime compute_time/peak_ram plots "
                            "(resource_metrics.tsv is always kept for comparisons)")

    # compare
    p_cmp = sub.add_parser("compare", help="Compare runs across datasets/days")
    p_cmp.add_argument("runs", nargs="+",
                       help="Run dirs (output/<date>_runs/<name>/<config>/)")
    p_cmp.add_argument("-o", "--output", default=None,
                       help="Comparison dir (default: inherited date+names from input runs)")
    p_cmp.add_argument("--dataset", default=None,
                       help="Dataset display name for plot titles")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        from compare_runs import compare_runs, default_comparison_dir
        compare_runs(args.runs, Path(args.output) if args.output
                     else default_comparison_dir(args.runs),
                     dataset_label=args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
