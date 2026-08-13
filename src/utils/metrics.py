"""Unified resource metrics collector.

Used by LocalRunner (psutil subprocess monitoring) and SlurmRunner (sacct),
ensuring all execution paths produce the same output schema.

Produces dict keys:
    tool, category, wall_seconds, time_s, peak_ram_mb, peak_vram_mb,
    mean_cpu_pct, gpu_name, gpu_available, model_size_mb, intermediate_mb,
    n_sequences, throughput_seq_s, success, notes
"""
import re
import subprocess
import time
from pathlib import Path


def collect_local(proc, tool, output: str, wall_start: float) -> dict:
    """Measure resources while a subprocess.Popen runs (psutil)."""
    wall = round(time.perf_counter() - wall_start, 3)
    time_s, throughput = _parse_runner_output(output, tool)
    ram_mb, cpu_pct = _sample_psutil(proc)
    vram_mb = _query_vram()
    return _build_metrics(tool, wall, time_s, ram_mb, cpu_pct, vram_mb,
                          throughput, success=(proc.returncode == 0))


def collect_slurm(job_id: str, tool) -> dict:
    """Parse sacct output into the same schema as collect_local."""
    fields = "JobID,State,CPUTimeRAW,MaxRSS,ReqTRES,AllocTRES%50"
    result = subprocess.run(
        ["sacct", "-j", job_id, f"--format={fields}", "--noheader", "-P"],
        capture_output=True, text=True, timeout=10)

    success = False
    cpu_time = 0.0
    max_rss = 0.0
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

    throughput = round(tool.n_sequences / cpu_time, 1) if cpu_time > 0 else None
    return _build_metrics(tool, cpu_time, cpu_time, max_rss, 0.0, 0.0,
                          throughput, success=success, notes=notes)


# ── helpers ──

def _parse_runner_output(output: str, tool):
    """Regex: '1988 seqs in 0.425s' → (time_s, throughput)."""
    m = re.search(r"(\d+)\s+seqs\s+in\s+([\d.]+)s", output)
    if m:
        n, t = int(m.group(1)), float(m.group(2))
        return t, round(n / t, 1) if t > 0 else None
    return None, None


def _sample_psutil(proc, interval=0.5):
    """Sample RSS and CPU% of process tree until process exits.
    Falls back to resource.getrusage() if psutil unavailable or process dies."""
    ram_mb = 0.0
    cpu_pct = 0.0
    try:
        import psutil  # noqa: F401
        ps_proc = psutil.Process(proc.pid)
        samples = []
        while proc.poll() is None:
            try:
                procs = [ps_proc] + ps_proc.children(recursive=True)
                rss = sum(p.memory_info().rss for p in procs) / (1024 * 1024)
                cpu = sum(p.cpu_percent() for p in procs)
                samples.append((rss, cpu))
            except psutil.NoSuchProcess:
                break
            time.sleep(interval)
        if samples:
            ram_mb = round(max(s[0] for s in samples), 1)
            cpu_pct = round(sum(s[1] for s in samples) / len(samples), 1)
    except Exception:
        import resource
        ram_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    return ram_mb, cpu_pct


def _query_vram():
    """nvidia-smi VRAM query — returns 0.0 if no GPU or command fails."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return round(sum(float(l) for l in res.stdout.strip().split("\n")
                         if l.strip()), 1)
    except Exception:
        return 0.0


def _build_metrics(tool, wall: float, time_s, ram_mb: float, cpu_pct: float,
                   vram_mb: float, throughput, success: bool = True,
                   notes: str = ""):
    return {
        "tool": tool.name,
        "category": tool.category,
        "wall_seconds": round(wall, 3),
        "time_s": round(time_s or wall, 3),
        "peak_ram_mb": round(ram_mb, 1),
        "peak_vram_mb": round(vram_mb, 1),
        "mean_cpu_pct": round(cpu_pct, 1),
        "gpu_name": "",
        "gpu_available": tool.gpu_capable,
        "model_size_mb": tool.model_size_mb(),
        "intermediate_mb": 0,
        "n_sequences": tool.n_sequences,
        "throughput_seq_s": throughput,
        "success": success,
        "notes": notes,
    }
