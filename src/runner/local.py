"""Local runner: execute tools via subprocess in any machine."""
import os, sys, subprocess, time, json
from typing import Optional

from src.runner.base import Runner
from src.benchmark.tools import Tool
from src.config import config

ROOT = config.root


class LocalRunner(Runner):
    """Run a tool locally using pixi env subprocess. Works on any machine."""

    def __init__(self, n_runs: int = 1, warmup: bool = True):
        self.n_runs = n_runs
        self.warmup = warmup

    def available(self) -> bool:
        return True  # Always available (no Slurm needed)

    def run(self, tool: Tool) -> dict:
        """Execute tool n_runs times, return aggregate with mean ± SD."""
        runs = []
        for i in range(self.n_runs):
            if self.n_runs > 1:
                print(f"    Run {i+1}/{self.n_runs}...", end=" ", flush=True)
            m = self._run_once(tool)
            runs.append(m)
            if self.n_runs > 1:
                status = "OK" if m["success"] else "FAIL"
                print(f"[{status}] {m.get('time_s', 'N/A')}")

        if len(runs) == 1:
            return runs[0]

        from src.analysis.statistics import aggregate_runs
        return aggregate_runs(runs)

    def _run_once(self, tool: Tool) -> dict:
        # Build command
        env_path = tool.pixi_env
        extra = ["-e", "ipro-mp"] if "ipromp" in tool.short_name else []
        code = self._build_code(tool)
        cmd = ["pixi", "run", "--manifest-path", str(env_path)] + extra + ["python", "-c", code]

        ram_before = self._get_ram()
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=str(ROOT), timeout=600)
        wall = round(time.perf_counter() - t0, 3)
        ram_after = self._get_ram()

        success = result.returncode == 0
        output = result.stdout.strip()
        notes = result.stderr.strip()[-200:] if not success else ""

        time_s, throughput = self._parse_time(output, tool)

        return {
            "tool": tool.name,
            "category": tool.category,
            "wall_seconds": wall,
            "peak_ram_mb": round(ram_after - ram_before, 1),
            "peak_vram_mb": 0,
            "gpu_name": "",
            "gpu_available": False,
            "model_size_mb": tool.model_size_mb(),
            "intermediate_mb": 0,
            "n_sequences": tool.n_sequences,
            "throughput_seq_s": throughput,
            "time_s": time_s,
            "success": success,
            "notes": notes or output[:200],
        }

    def _get_ram(self) -> float:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    def _parse_time(self, output: str, tool: Tool):
        import re
        m = re.search(r"(\d+)\s+seqs\s+in\s+([\d.]+)s", output)
        if m:
            n = int(m.group(1))
            t = float(m.group(2))
            return t, round(n / t, 1) if t > 0 else None
        m = re.search(r"(\d+\.?\d*)ms/seq", output)
        if m:
            tp = round(1000 / float(m.group(1)), 1)
            est = float(m.group(1)) * tool.n_sequences / 1000
            return round(est, 1), tp
        return None, None

    def _build_code(self, tool: Tool) -> str:
        """Build Python code string for each tool type."""
        from src.benchmark.tool_runners import get_runner_code
        code = get_runner_code(
            tool.short_name,
            pos_fasta=str(config.pos_fasta),
            neg_fasta=str(config.neg_fasta),
            combined_fasta=str(config.combined_fasta),
            dnabert_dir=str(config.dnabert_dir),
            ipromp_model_dir=str(config.ipromp_model_dir),
        )
        return code
