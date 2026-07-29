"""Slurm runner: execute tools via sbatch + sacct on HPC clusters."""
import os, sys, subprocess, time, json
from pathlib import Path
from typing import Optional

from src.runner.base import Runner
from src.benchmark.tools import Tool
from src.config import config

ROOT = config.root


class SlurmRunner(Runner):
    """Run a tool as a Slurm job and collect metrics via sacct."""

    def __init__(self, partition: str = "production",
                 time_limit: str = "2:00:00",
                 cpus: int = 4, mem_gb: int = 16):
        self.partition = partition
        self.time_limit = time_limit
        self.cpus = cpus
        self.mem_gb = mem_gb

    def available(self) -> bool:
        """Check if Slurm is available."""
        try:
            subprocess.run(["sinfo", "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def run(self, tool: Tool) -> dict:
        print(f"  [{tool.name}] Submitting to Slurm...", flush=True)

        job_id = self._submit(tool)
        if not job_id:
            return self._failed(tool, "sbatch failed")

        print(f"    Job ID: {job_id}", flush=True)

        # Poll until complete
        self._wait(job_id)

        # Collect metrics from sacct
        metrics = self._collect_sacct(tool, job_id)
        return metrics

    def _submit(self, tool: Tool) -> Optional[str]:
        """Submit sbatch job, return job ID."""
        script = self._build_script(tool)
        script_path = config.temp_dir / f"slurm_{tool.short_name}.sh"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        script_path.chmod(0o755)

        gpu_flag = "--gres=gpu:1" if tool.gpu_capable else ""
        cmd = ["sbatch", "--parsable",
               "-t", self.time_limit, "-c", str(self.cpus),
               f"--mem={self.mem_gb}G", "-p", self.partition] + \
              ([gpu_flag] if gpu_flag else []) + \
              ["--export=ALL",
               f"--job-name=bench_{tool.short_name[:8]}",
               str(script_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"    sbatch error: {result.stderr}")
            return None
        return result.stdout.strip()

    def _build_script(self, tool: Tool) -> str:
        """Build Slurm batch script from tool definition."""
        env_path = tool.pixi_env
        extra = "-e ipro-mp" if "ipromp" in tool.short_name else ""

        # Reuse runner code from LocalRunner for LCNN/MLDSPP
        from src.runner.local import LocalRunner
        lr = LocalRunner()
        try:
            code = lr._build_code(tool)
        except NotImplementedError:
            code = f'print("Tool {tool.name} — manual run needed")'

        # Escape for bash heredoc
        code_escaped = code.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$")

        return f"""#!/bin/bash
#SBATCH --output={config.temp_dir / f"{tool.short_name}_%j.out"}
#SBATCH --error={config.temp_dir / f"{tool.short_name}_%j.err"}

export PIXI_HOME="{os.environ.get('PIXI_HOME', os.path.expanduser('~/.pixi'))}"
export PATH="$PIXI_HOME/bin:$PATH"

pixi run --manifest-path {env_path} {extra} python -c "{code_escaped}"
"""

    def _wait(self, job_id: str, poll_seconds: int = 10, max_wait: int = 7200):
        """Poll sacct until job completes."""
        elapsed = 0
        while elapsed < max_wait:
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
                capture_output=True, text=True, timeout=10)
            states = result.stdout.strip().split("\n")
            for state in states:
                if state in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED",
                             "OUT_OF_MEMORY", "NODE_FAIL"):
                    return
            time.sleep(poll_seconds)
            elapsed += poll_seconds

    def _collect_sacct(self, tool: Tool, job_id: str) -> dict:
        """Parse sacct output into ResourceMetrics."""
        fields = "JobID,State,CPUTimeRAW,MaxRSS,ReqTRES,AllocTRES%50"
        result = subprocess.run(
            ["sacct", "-j", job_id, f"--format={fields}", "--noheader", "-P"],
            capture_output=True, text=True, timeout=10)

        success = False
        cpu_time = 0.0
        max_rss = 0.0
        gpu_name = ""
        notes = ""

        for line in result.stdout.strip().split("\n"):
            if not line or ".bat+" in line or ".ext+" in line:
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            state = parts[1]
            cpu_time = float(parts[2]) if parts[2] else 0.0
            max_rss_raw = parts[3]
            max_rss = 0.0
            if max_rss_raw and max_rss_raw.endswith("K"):
                max_rss = float(max_rss_raw[:-1]) / 1024.0
            elif max_rss_raw and max_rss_raw.endswith("M"):
                max_rss = float(max_rss_raw[:-1])
            elif max_rss_raw:
                max_rss = float(max_rss_raw) / (1024 * 1024)

            if state == "COMPLETED":
                success = True

        return {
            "tool": tool.name,
            "category": tool.category,
            "wall_seconds": 0,  # sacct has CPUTime, not wall
            "cpu_user_seconds": round(cpu_time, 1),
            "peak_ram_mb": round(max_rss, 1),
            "peak_vram_mb": 0,
            "gpu_name": gpu_name,
            "gpu_available": tool.gpu_capable,
            "model_size_mb": tool.model_size_mb(),
            "intermediate_mb": 0,
            "n_sequences": tool.n_sequences,
            "throughput_seq_s": round(tool.n_sequences / cpu_time, 1) if cpu_time > 0 else None,
            "time_s": round(cpu_time, 1),
            "success": success,
            "notes": notes,
        }

    def _failed(self, tool: Tool, reason: str) -> dict:
        return {
            "tool": tool.name, "category": tool.category,
            "success": False, "notes": reason,
            "n_sequences": tool.n_sequences,
            "time_s": None, "throughput_seq_s": None,
            "wall_seconds": 0, "cpu_user_seconds": 0,
            "peak_ram_mb": 0, "peak_vram_mb": 0,
            "gpu_name": "", "gpu_available": False,
            "model_size_mb": tool.model_size_mb(), "intermediate_mb": 0,
        }
