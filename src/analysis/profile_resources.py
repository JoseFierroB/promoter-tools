#!/usr/bin/env python3
"""
Resource profiler — measures RAM, CPU, VRAM time series for each tool.
Spawns tools directly (no pixi run wrapper), samples process tree every 0.1s.

Usage:
    pixi run python src/analysis/profile_resources.py --all
    pixi run python src/analysis/profile_resources.py --tool meme
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import psutil
except ImportError:
    print("ERROR: psutil required. Run: pixi install", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "output" / "profiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_tool_list():
    from src.config import config
    from src.benchmark.tools import PROMOTER_TOOLS, _load_toml_tools
    _load_toml_tools()
    return PROMOTER_TOOLS


def get_vram_mb():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total = 0
        for line in res.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        total += float(parts[1].strip())
                    except ValueError:
                        pass
        return total
    except Exception:
        return 0.0


def sample_tree(root_pid, tree_cache=None):
    try:
        root = psutil.Process(root_pid)
        procs = root.children(recursive=True)
        procs.append(root)
        total_rss = sum(p.memory_info().rss for p in procs)
        total_vms = sum(p.memory_info().vms for p in procs)
        total_cpu = 0.0
        for p in procs:
            try:
                total_cpu += p.cpu_percent()
            except Exception:
                pass
        vram = get_vram_mb()
        return {
            "n_procs": len(procs),
            "rss_mb": round(total_rss / (1024 * 1024), 1),
            "vms_mb": round(total_vms / (1024 * 1024), 1),
            "cpu_pct": round(total_cpu, 1),
            "vram_mb": round(vram, 1),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"n_procs": 0, "rss_mb": 0, "vms_mb": 0, "cpu_pct": 0, "vram_mb": 0}


def profile_tool(tool, pos_fasta, neg_fasta, interval=0.1):
    from src.config import config

    env_path = tool.pixi_env
    feature = "ipro-mp" if "ipromp" in tool.short_name else "default"
    python_bin = config.get_env_python(env_path, feature=feature)
    env_bin = str(python_bin.parent)

    runner_script = ROOT / "src/runners" / f"{tool.short_name}.py"
    if not runner_script.exists():
        print(f"  ✗ {tool.short_name}: runner not found", file=sys.stderr)
        return None, None

    out_dir = ROOT / "output" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(python_bin), str(runner_script),
           "--pos", str(pos_fasta), "--neg", str(neg_fasta),
           "-o", str(out_dir)]
    if tool.short_name == "mldspp_75":
        pos_path = str(pos_fasta)
        if "all_tss" in pos_path:
            split_file = "mldspp_75_split_tigr4_all_tss.npz"
        elif "extended_high" in pos_path:
            split_file = "mldspp_75_split_tigr4_extended_high.npz"
        elif "tigr4_extended" in pos_path or "tigr4_2k" in pos_path:
            split_file = "mldspp_75_split_tigr4_2k.npz"
        elif "tigr4" in pos_path:
            split_file = "mldspp_75_split_tigr4_high.npz"
        else:
            split_file = "mldspp_75_split_d39v.npz"
        cmd += ["--split", split_file]

    run_env = os.environ.copy()
    run_env["PATH"] = f"{env_bin}:{run_env['PATH']}"
    run_env["PYTHONPATH"] = str(ROOT)

    print(f"  [{tool.short_name}] spawning...", file=sys.stderr)
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=run_env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, cwd=str(ROOT))

    # first cpu_percent call returns 0 — need a dummy sample
    sample_tree(proc.pid)

    samples = []
    while proc.poll() is None:
        row = {"elapsed_s": round(time.perf_counter() - t0, 3)}
        row.update(sample_tree(proc.pid))
        samples.append(row)
        time.sleep(interval)

    # final sample
    row = {"elapsed_s": round(time.perf_counter() - t0, 3)}
    row.update(sample_tree(proc.pid))
    samples.append(row)

    wall_s = round(time.perf_counter() - t0, 3)
    proc.wait()

    df = pd.DataFrame(samples)
    if df.empty:
        return None, None

    summary = {
        "tool": tool.short_name,
        "wall_s": wall_s,
        "peak_rss_mb": df["rss_mb"].max(),
        "peak_vms_mb": df["vms_mb"].max(),
        "mean_cpu_pct": round(df["cpu_pct"].mean(), 1),
        "peak_cpu_pct": df["cpu_pct"].max(),
        "peak_vram_mb": df["vram_mb"].max(),
        "max_procs": df["n_procs"].max(),
        "n_samples": len(df),
        "success": proc.returncode == 0,
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Resource profiler for promoter tools")
    parser.add_argument("--tool", default=None, help="Tool key to profile (e.g. meme, lcnn)")
    parser.add_argument("--all", action="store_true", help="Profile all tools")
    parser.add_argument("--pos", default=None, help="Positive FASTA (default: D39V)")
    parser.add_argument("--neg", default=None, help="Negative FASTA (default: D39V)")
    parser.add_argument("-i", "--interval", type=float, default=0.1,
                        help="Sample interval in seconds (default: 0.1)")
    args = parser.parse_args()

    tools = get_tool_list()
    pos = args.pos or str(ROOT / "data/benchmark/d39v/positives_81bp.fasta")
    neg = args.neg or str(ROOT / "data/benchmark/d39v/negatives_81bp.fasta")

    if args.all:
        tool_keys = list(tools.keys())
    elif args.tool:
        tool_keys = [args.tool]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Profiling {len(tool_keys)} tools (interval={args.interval}s)", file=sys.stderr)
    print(f"Pos: {pos}", file=sys.stderr)
    print(f"Neg: {neg}", file=sys.stderr)
    print(file=sys.stderr)

    summaries = []
    for key in tool_keys:
        tool = tools[key]
        ts_df, summary = profile_tool(tool, pos, neg, interval=args.interval)
        if ts_df is not None and ts_df is not None:
            ts_path = OUT_DIR / f"{key}_timeseries.tsv"
            ts_df.to_csv(ts_path, sep="\t", index=False)
            summaries.append(summary)
            print(f"  {key}: {summary['wall_s']:.1f}s, peak={summary['peak_rss_mb']:.0f}MB, "
                  f"cpu={summary['mean_cpu_pct']:.0f}%, vram={summary['peak_vram_mb']:.0f}MB",
                  file=sys.stderr)
        else:
            print(f"  {key}: FAILED", file=sys.stderr)

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(OUT_DIR / "profile_summary.tsv", sep="\t", index=False)
        print(f"\nSummary: {OUT_DIR}/profile_summary.tsv", file=sys.stderr)
        print(f"Timeseries: {OUT_DIR}/<tool>_timeseries.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
