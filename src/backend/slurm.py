"""Slurm runner: execute tools via sbatch + sacct on HPC clusters."""
import os, sys, subprocess, time
from pathlib import Path
from typing import Optional

from src.backend.base import Runner
from src.benchmark.tools import Tool
from src.config import config

ROOT = config.root


class SlurmRunner(Runner):
    """Run a tool as a Slurm job and collect metrics via sacct."""

    def __init__(self, partition: str = "production",
                 time_limit: str = "2:00:00",
                 cpus: int = 4, mem_gb: int = 16,
                 pos_fasta: Path = None, neg_fasta: Path = None):
        self.partition = partition
        self.time_limit = time_limit
        self.cpus = cpus
        self.mem_gb = mem_gb
        self.pos_fasta = Path(pos_fasta) if pos_fasta else config.pos_fasta
        self.neg_fasta = Path(neg_fasta) if neg_fasta else config.neg_fasta

    def available(self) -> bool:
        """Check if Slurm is available."""
        try:
            subprocess.run(["sinfo", "--version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def run(self, tool: Tool) -> dict:
        print(f"  [{tool.name}] Submitting to Slurm...", flush=True)

        if tool.short_name == "mldspp_75":
            from src.backend.local import _pick_mldspp_split, _count_seqs
            split = _pick_mldspp_split(self.pos_fasta)
            if split is None:
                return {
                    "tool": tool.name, "category": tool.category,
                    "success": False,
                    "notes": (f"No mldspp_75 split for {_count_seqs(self.pos_fasta)} "
                              f"positives — available: "
                              f"{', '.join(s.name for s in (config.data_dir / 'benchmark').glob('mldspp_75_split_*.npz'))}"),
                    "n_sequences": tool.n_sequences,
                    "time_s": None, "throughput_seq_s": None,
                    "wall_seconds": 0, "cpu_user_seconds": 0,
                    "peak_ram_mb": 0, "peak_vram_mb": 0,
                    "gpu_name": "", "gpu_available": False,
                    "model_size_mb": tool.model_size_mb(), "intermediate_mb": 0,
                }

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

        gpu_flag = "--gres=gpu:1" if tool.gpu_capable and not config.force_cpu else ""
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
        runner = ROOT / "src/runners" / f"{tool.short_name}.py"
        env_path = tool.pixi_env
        env_name = "ipro-mp" if "ipromp" in tool.short_name else "default"
        python_bin = str(config.get_env_python(env_path, feature=env_name))

        extra_args = ""
        if "ipromp" in tool.short_name:
            extra_args = f'  -m "{config.ipromp_model_dir}" -d "{config.dnabert_dir}"'
        if "promotech" in tool.short_name:
            n_seqs = _count_seqs(self.pos_fasta) + _count_seqs(self.neg_fasta)
            extra_args += f'  --timeout {_promotech_timeout(n_seqs)}'
        if tool.short_name == "mldspp_75":
            from src.backend.local import _pick_mldspp_split, _count_seqs, _promotech_timeout
            split = _pick_mldspp_split(self.pos_fasta)
            if split:
                extra_args += f'  --split "{split}"'
            else:
                print(f"    WARNING: no mldspp_75 split for {self.pos_fasta.name} — job will be skipped")

        return f"""#!/bin/bash
#SBATCH --output={config.temp_dir / f"{tool.short_name}_%j.out"}
#SBATCH --error={config.temp_dir / f"{tool.short_name}_%j.err"}

export PIXI_HOME="{os.environ.get('PIXI_HOME', os.path.expanduser('~/.pixi'))}"
export PATH="$PIXI_HOME/bin:$PATH"
PYTHONPATH="{ROOT}" {python_bin} "{runner}" \\
  --pos "{self.pos_fasta}" --neg "{self.neg_fasta}" \\
  -o "{ROOT / 'output/predictions'}"{extra_args}
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
        """Parse sacct output → unified metrics schema."""
        from src.utils.metrics import collect_slurm
        return collect_slurm(job_id, tool)

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
