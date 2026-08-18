"""Local runner: execute tools via subprocess in any machine."""
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.backend.base import Runner
from src.benchmark.tools import Tool
from src.config import config

ROOT = config.root


def _count_seqs(fasta: Path) -> int:
    """Count FASTA headers. Returns 0 if file missing."""
    n = 0
    try:
        with open(fasta) as f:
            for line in f:
                if line.startswith(">"):
                    n += 1
    except OSError:
        return 0
    return n


def _pick_mldspp_split(pos_fasta: Path) -> Optional[str]:
    """Pick the mldspp_75 split built for `pos_fasta` by matching sizes.

    A split covers exactly n_pos positive sequences when len(train_idx) +
    len(test_idx) == n_pos (the 2k split's train_idx ends at 1998, so
    max-index matching is unreliable). Returns the filename, or None if no
    (or ambiguous) split matches.
    """
    n_pos = _count_seqs(pos_fasta)
    matches = []
    for split in sorted((config.data_dir / "benchmark").glob("mldspp_75_split_*.npz")):
        try:
            d = np.load(split)
            covered = len(d["train_idx"]) + len(d["test_idx"])
        except Exception:
            continue
        if covered == n_pos:
            matches.append(split.name)
    return matches[0] if len(matches) == 1 else None


def _promotech_timeout(n_seqs: int) -> int:
    """Per-step subprocess timeout for PromoTech (predict + scan).

    RF-HOT runs single-core at ~20 seq/s; scale with n_seqs so large
    dummy runs don't hit the default 600 s cap.
    """
    return max(600, int(n_seqs / 20.6 * 2.5))


class LocalRunner(Runner):
    """Run a tool locally using pixi env subprocess. Works on any machine."""

    def __init__(self, n_runs: int = 1, output_dir: str = None,
                 pos_fasta: Path = None, neg_fasta: Path = None):
        self.n_runs = n_runs
        self.output_dir = output_dir or str(ROOT / "output/predictions")
        self.pos_fasta = Path(pos_fasta) if pos_fasta else config.pos_fasta
        self.neg_fasta = Path(neg_fasta) if neg_fasta else config.neg_fasta

    def available(self) -> bool:
        return True  # Always available (no Slurm needed)

    def run(self, tool: Tool) -> dict:
        """Execute tool n_runs times, return aggregate with mean ± SD."""
        print(f"  [{tool.name}]", flush=True)
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

        n_seqs = _count_seqs(self.pos_fasta) + _count_seqs(self.neg_fasta)
        if n_seqs > 0:
            tool.n_sequences = n_seqs

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
            "--pos", str(self.pos_fasta),
            "--neg", str(self.neg_fasta),
            "-o", self.output_dir,
        ]
        if "ipromp" in tool.short_name:
            cmd += [
                "-m", str(config.ipromp_model_dir),
                "-d", str(config.dnabert_dir),
            ]
        if "promotech" in tool.short_name:
            cmd += ["--timeout", str(_promotech_timeout(n_seqs))]
        if tool.short_name == "mldspp_75":
            split_file = _pick_mldspp_split(self.pos_fasta)
            if split_file is None:
                return {
                    "tool": tool.name, "category": tool.category,
                    "wall_seconds": 0, "peak_ram_mb": 0, "peak_vram_mb": 0,
                    "gpu_name": "", "gpu_available": False,
                    "model_size_mb": 0, "intermediate_mb": 0,
                    "n_sequences": tool.n_sequences,
                    "throughput_seq_s": None, "time_s": None,
                    "success": False,
                    "notes": (f"No mldspp_75 split for {_count_seqs(self.pos_fasta)} "
                              f"positives — available: "
                              f"{', '.join(s.name for s in (config.data_dir / 'benchmark').glob('mldspp_75_split_*.npz'))}"),
                }
            cmd += ["--split", split_file]
        run_env = os.environ.copy()
        run_env["PATH"] = f"{env_bin}:{run_env['PATH']}"
        run_env["PYTHONPATH"] = str(ROOT)
        run_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # CUDA index == nvidia-smi index
        if config.force_cpu:
            run_env["CUDA_VISIBLE_DEVICES"] = ""
        elif tool.gpu_id:
            run_env["CUDA_VISIBLE_DEVICES"] = tool.gpu_id
            run_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if config.threads > 0:
            run_env["OMP_NUM_THREADS"] = str(config.threads)
            run_env["MKL_NUM_THREADS"] = str(config.threads)
            run_env["OPENBLAS_NUM_THREADS"] = str(config.threads)
            run_env["TF_NUM_INTRAOP_THREADS"] = str(config.threads)
            run_env["TF_NUM_INTEROP_THREADS"] = str(config.threads)

        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                cwd=str(ROOT), env=run_env)

        import threading
        # ── Tee stdout: live terminal output + capture for regex ──
        output_lines = []
        def _tee_stdout():
            for line in iter(proc.stdout.readline, ''):
                print(f"     │ {line.rstrip()}", flush=True)
                output_lines.append(line)

        t_tee = threading.Thread(target=_tee_stdout, daemon=True)
        t_tee.start()

        # ── Background PSS sampling while process runs ──
        samples = []
        gpu_samples = []
        stop_sampler = threading.Event()

        def _bg_sample():
            try:
                import psutil
                from src.utils.metrics import _pss, _gpu_sample
                ps_proc = psutil.Process(proc.pid)
                while not stop_sampler.is_set():
                    try:
                        procs = [ps_proc] + ps_proc.children(recursive=True)
                        pss_kb = sum(_pss(p.pid) for p in procs)
                        ram_mb = pss_kb / 1024.0
                        cpu = sum(p.cpu_percent() for p in procs)
                        samples.append((ram_mb, cpu))
                        if tool.gpu_id:
                            gpu_samples.append(_gpu_sample(tool.gpu_id))
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.5)
            except Exception:
                try:
                    import psutil as _psutil
                    from src.utils.metrics import _gpu_sample
                    ps_proc = _psutil.Process(proc.pid)
                    while not stop_sampler.is_set():
                        try:
                            procs = [ps_proc] + ps_proc.children(recursive=True)
                            rss = sum(p.memory_info().rss for p in procs) / (1024 * 1024)
                            cpu = sum(p.cpu_percent() for p in procs)
                            samples.append((rss, cpu))
                            if tool.gpu_id:
                                gpu_samples.append(_gpu_sample(tool.gpu_id))
                        except _psutil.NoSuchProcess:
                            break
                        time.sleep(0.5)
                except Exception:
                    pass

        t_sampler = threading.Thread(target=_bg_sample, daemon=True)
        t_sampler.start()

        if "promotech" in tool.short_name:
            wait_timeout = _promotech_timeout(n_seqs) * 2 + 300
        elif "fimo" in tool.short_name:
            wait_timeout = max(900, int(n_seqs / 40.0 * 3)) * 2 + 300
        elif "meme" in tool.short_name:
            wait_timeout = max(300, int(n_seqs / 40.0 * 3)) * 2 + 300
        elif "ipromp" in tool.short_name:
            wait_timeout = max(600, int(n_seqs / 3.6 * 2)) + 300
        elif "lcnn" in tool.short_name:
            wait_timeout = max(300, int(n_seqs / 2000 * 2)) + 300
        else:  # mldspp
            wait_timeout = max(300, int(n_seqs / 4000 * 2)) + 300
        try:
            if config.no_timeout:
                proc.wait()
            else:
                proc.wait(timeout=wait_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            stop_sampler.set()
            t_tee.join(timeout=2)
            t_sampler.join(timeout=2)
            return {
                "tool": tool.name, "category": tool.category,
                "wall_seconds": wait_timeout, "peak_ram_mb": 0, "peak_vram_mb": 0,
                "gpu_name": "", "gpu_available": False, "model_size_mb": 0,
                "intermediate_mb": 0, "n_sequences": tool.n_sequences,
                "throughput_seq_s": None, "time_s": None,
                "success": False,
                "notes": f"Timeout after {wait_timeout}s",
            }

        stop_sampler.set()
        t_tee.join(timeout=2)
        t_sampler.join(timeout=2)
        output = ''.join(output_lines)

        from src.utils.metrics import collect_local
        result = collect_local(proc, tool, output, t0, samples=samples,
                               gpu_samples=gpu_samples)
        result["notes"] = output[-200:] if not result["success"] else ""
        extra = []
        if result['peak_ram_mb'] > 0:
            extra.append(f"{result['peak_ram_mb']:.0f}MB")
        if result['peak_vram_mb'] > 0:
            extra.append(f"VRAM {result['peak_vram_mb']:.0f}MB")
        suffix = f"  ({', '.join(extra)})" if extra else ""
        print(f"    [{tool.name}] {result['wall_seconds']:.1f}s{suffix}", flush=True)
        return result

# _get_child_ram and _parse_time moved to src/utils/metrics.py