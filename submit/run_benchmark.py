#!/usr/bin/env python3
"""
Simple benchmark runner — reads commands from TSV, executes local or via Slurm.
Pattern: externalize commands, keep the runner simple.
Usage:
  python submit/run_benchmark.py local meme,mldspp,lcnn
  python submit/run_benchmark.py slurm all
"""
import sys, subprocess, csv, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CMDS_FILE = Path(__file__).resolve().parent / "commands.tsv"

def load_commands():
    cmds = {}
    with open(CMDS_FILE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tool = row["tool"]
            mode = row["mode"]
            cmd = row["command"]
            if tool not in cmds:
                cmds[tool] = {}
            cmds[tool][mode] = cmd
    return cmds

def run_local(tools):
    cmds = load_commands()
    out_dir = os.environ.get("BENCH_OUT", str(ROOT / "output" / "predictions"))
    os.makedirs(out_dir, exist_ok=True)
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    combined = Path(f"{tmpdir}/bench_combined.fasta")
    if not combined.exists() and "ipromp" in tools:
        with open(combined, "wb") as out:
            for fname in ["positives_81bp.fasta", "negatives_81bp.fasta"]:
                with open(ROOT / "data/benchmark/d39v" / fname, "rb") as fin:
                    out.write(fin.read())

    for tool in tools:
        cmd = cmds.get(tool, {}).get("local")
        if not cmd:
            print(f"  ✗ {tool}: no local command")
            continue
        cmd = cmd.replace("OUT_DIR", out_dir)
        print(f"  {tool}...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = subprocess.run(cmd, shell=True, cwd=str(ROOT),
                             capture_output=True, text=True, timeout=3600)
        t = time.perf_counter() - t0
        if res.returncode == 0:
            print(f"OK ({t:.1f}s)")
        else:
            print(f"FAIL ({t:.1f}s)")
            if res.stderr:
                print(f"    stderr: {res.stderr[-300:]}")

def run_slurm(tools):
    cmds = load_commands()
    out_dir = os.environ.get("BENCH_OUT", str(ROOT / "output" / "predictions"))
    os.makedirs(out_dir, exist_ok=True)
    job_ids = []
    for tool in tools:
        cmd = cmds.get(tool, {}).get("slurm")
        if not cmd:
            print(f"  ✗ {tool}: no slurm command")
            continue
        cmd = cmd.replace("OUT_DIR", out_dir)
        mem = "32G"
        cpus = 1
        gpu = "--gres=gpu:1" if "ipromp" in tool else ""
        tmpdir = os.environ.get("TMPDIR", "/tmp")
        script = Path(f"{tmpdir}/bench_{tool}.sh")
        pixi_home = os.environ.get("PIXI_HOME", "")
        script.write_text(f"""#!/bin/bash
export PIXI_HOME="{pixi_home}"
export PATH="$PIXI_HOME/bin:$PATH"
{cmd}
""")
        script.chmod(0o755)
        sbatch_cmd = f"sbatch --parsable -t 2:00:00 -c {cpus} --mem={mem} {gpu} {script}"
        res = subprocess.run(sbatch_cmd, shell=True, cwd=str(ROOT),
                             capture_output=True, text=True)
        if res.returncode == 0:
            job_id = res.stdout.strip()
            job_ids.append((tool, job_id))
            print(f"  {tool}: job {job_id}")
        else:
            print(f"  ✗ {tool}: sbatch failed: {res.stderr[-200:]}")

    if job_ids:
        print(f"\nMonitor: squeue -u $USER")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python submit/run_benchmark.py <local|slurm> <tool1,tool2,...|all>")
        print("Tools: meme, mldspp, lcnn, promotech, ipromp")
        sys.exit(1)

    mode = sys.argv[1]
    tools_arg = sys.argv[2]
    all_tools = ["meme", "mldspp", "mldspp_75", "lcnn", "fimo_db", "fimo_prok", "promotech_hot", "promotech_tetra", "ipromp_sp12"]
    tools = all_tools if tools_arg == "all" else tools_arg.split(",")

    print(f"Benchmark: {mode} {' '.join(tools)}")
    print()

    if mode == "local":
        run_local(tools)
    elif mode == "slurm":
        run_slurm(tools)
    else:
        print(f"Unknown mode: {mode}")
