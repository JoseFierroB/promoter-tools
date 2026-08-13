"""Local runner: execute tools via subprocess in any machine."""
import os
import subprocess
import time
from typing import Optional

from src.backend.base import Runner
from src.benchmark.tools import Tool
from src.config import config

ROOT = config.root


class LocalRunner(Runner):
    """Run a tool locally using pixi env subprocess. Works on any machine."""

    def __init__(self, n_runs: int = 1, output_dir: str = None):
        self.n_runs = n_runs
        self.output_dir = output_dir or str(ROOT / "output/predictions")

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
        env_path = tool.pixi_env
        env_name = "ipro-mp" if "ipromp" in tool.short_name else "default"
        python_bin = config.get_env_python(env_path, feature=env_name)
        env_bin = str(python_bin.parent)

        runner_script = ROOT / "src/runners" / f"{tool.short_name}.py"
        if not runner_script.exists():
            return {
                "tool": tool.name, "category": tool.category,
                "wall_seconds": 0, "peak_ram_mb": 0, "peak_vram_mb": 0,
                "gpu_name": "", "gpu_available": False, "model_size_mb": 0,
                "intermediate_mb": 0, "n_sequences": tool.n_sequences,
                "throughput_seq_s": None, "time_s": None,
                "success": False,
                "notes": f"No runner: {runner_script}",
            }

        cmd = [
            str(python_bin), str(runner_script),
            "--pos", str(config.pos_fasta),
            "--neg", str(config.neg_fasta),
            "-o", self.output_dir,
        ]
        if "ipromp" in tool.short_name:
            cmd += [
                "-m", str(config.ipromp_model_dir),
                "-d", str(config.dnabert_dir),
            ]
        if tool.short_name == "mldspp_75":
            pos_path = str(config.pos_fasta)
            if "mixed_all" in pos_path:
                split_file = "mldspp_75_split_mixed_all.npz"
            elif "mixed" in pos_path:
                split_file = "mldspp_75_split_mixed.npz"
            elif "all_tss" in pos_path:
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

        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                cwd=str(ROOT), env=run_env)

        # Background psutil sampling while process runs
        samples = []
        import threading
        stop_sampler = threading.Event()

        def _bg_sample():
            try:
                import psutil
                ps_proc = psutil.Process(proc.pid)
                while not stop_sampler.is_set():
                    try:
                        procs = [ps_proc] + ps_proc.children(recursive=True)
                        rss = sum(p.memory_info().rss for p in procs) / (1024 * 1024)
                        cpu = sum(p.cpu_percent() for p in procs)
                        samples.append((rss, cpu))
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.5)
            except Exception:
                pass

        t = threading.Thread(target=_bg_sample, daemon=True)
        t.start()

        stdout, stderr = proc.communicate(timeout=1800)
        stop_sampler.set()
        t.join(timeout=2)
        output = stdout.strip()
        notes = stderr.strip()[-200:] if proc.returncode != 0 else ""

        from src.utils.metrics import collect_local
        result = collect_local(proc, tool, output, t0, samples=samples)
        result["notes"] = notes or output[:200]
        return result

# _get_child_ram and _parse_time moved to src/utils/metrics.py