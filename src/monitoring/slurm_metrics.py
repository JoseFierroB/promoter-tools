"""Slurm metrics collector: extract resource usage from sacct output.

Agnostic — works with any Slurm job, not just our benchmark.
Provides CPU time, RAM, GPU allocation, and disk usage for temp dirs.
"""
import subprocess, json, re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SlurmJobMetrics:
    """Standardized metrics from a single Slurm job."""
    job_id: str
    tool_name: str = ""
    state: str = ""
    cpu_time_raw: float = 0.0     # seconds
    max_rss_mb: float = 0.0        # peak RAM
    req_gpu: bool = False
    alloc_gpu: str = ""            # e.g. "a100:1"
    wall_seconds: float = 0.0      # from sacct Elapsed
    temp_dir_size_mb: float = 0.0  # intermediate files
    success: bool = False

    def to_dict(self):
        return asdict(self)


def collect_job_metrics(job_id: str, tool_name: str = "",
                        temp_dirs: list[Path] = None) -> SlurmJobMetrics:
    """Collect all metrics for a Slurm job via sacct + filesystem check.

    Args:
        job_id: Slurm job ID
        tool_name: Human-readable tool name
        temp_dirs: Optional list of directories to measure disk usage

    Returns:
        SlurmJobMetrics with all available fields populated
    """
    metrics = SlurmJobMetrics(job_id=job_id, tool_name=tool_name)

    # ── sacct: CPU time, RAM, GPU, state ──
    fields = "JobID,State,Elapsed,CPUTimeRAW,MaxRSS,ReqTRES%60,AllocTRES%60"
    try:
        result = subprocess.run(
            ["sacct", "-j", job_id, f"--format={fields}", "--noheader", "-P", "-n"],
            capture_output=True, text=True, timeout=15)

        for line in result.stdout.strip().split("\n"):
            if not line or ".bat+" in line or ".ext+" in line:
                continue
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue

            metrics.state = parts[1]

            # Wall time: "00:01:23" format
            elapsed = parts[2]
            if elapsed and ":" in elapsed:
                h, m, s = elapsed.split(":")
                metrics.wall_seconds = int(h) * 3600 + int(m) * 60 + int(s)

            # CPU time raw (seconds)
            if parts[3]:
                metrics.cpu_time_raw = float(parts[3])

            # Max RSS: "12345K" or "123.45M"
            if parts[4]:
                rss = parts[4].strip()
                if rss.endswith("K"):
                    metrics.max_rss_mb = float(rss[:-1]) / 1024.0
                elif rss.endswith("M"):
                    metrics.max_rss_mb = float(rss[:-1])
                else:
                    metrics.max_rss_mb = float(rss) / (1024 * 1024)

            # GPU allocation
            alloc = parts[6] if len(parts) > 6 else ""
            req = parts[5] if len(parts) > 5 else ""
            metrics.alloc_gpu = alloc
            metrics.req_gpu = "gres/gpu" in req or "gpu" in alloc.lower()

            if metrics.state == "COMPLETED":
                metrics.success = True

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass  # sacct not available

    # ── Temp directory sizes ──
    if temp_dirs:
        total = 0
        for d in temp_dirs:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        metrics.temp_dir_size_mb = round(total / (1024 * 1024), 1)

    return metrics


def parse_tool_output(out_file: Path) -> Optional[dict]:
    """Extract timing info from a tool's .out file (agnostic parser)."""
    if not out_file.exists():
        return None

    text = out_file.read_text()

    # Pattern: "TOOLNAME: N seqs in X.XXXs"
    m = re.search(r"(\d+)\s+seqs\s+in\s+([\d.]+)s", text)
    if m:
        n_seqs = int(m.group(1))
        t = float(m.group(2))
        return {"time_s": t, "n_seqs": n_seqs,
                "throughput": round(n_seqs / t, 1) if t > 0 else None}

    # Pattern: "X.Xms/seq (N reps on M seqs)"
    m = re.search(r"(\d+\.?\d*)ms/seq\s*\((\d+)\s*reps?\s*on\s*(\d+)\s*seqs?\)", text)
    if m:
        ms = float(m.group(1))
        n_measured = int(m.group(3))
        return {"time_s": round(ms * 1988 / 1000, 1),
                "n_seqs": n_measured,
                "throughput": round(1000 / ms, 1),
                "notes": f"extrapolated from {n_measured} seqs"}

    # Pattern: done (PromoTech)
    if "done" in text.lower() or "complete" in text.lower():
        return {"time_s": None, "n_seqs": 1988, "throughput": None,
                "notes": "use sacct CPU time"}

    return None
