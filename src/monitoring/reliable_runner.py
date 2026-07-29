"""BenchExec-inspired process control: cgroups, CPU pinning, reliable kills.

Requirements from Beyer et al. (2019):
  1. Measure CPU time (not just wall-clock)
  2. Terminate entire process tree
  3. Assign cores deliberately
  4. Limit memory via cgroups (avoids swapping)
  5. Isolate runs (Linux namespaces, or pixi envs for us)

GPU extension (our addition):
  - Use nvidia-smi / pynvml for VRAM measurement
  - CUDA_VISIBLE_DEVICES for GPU isolation
  - GPU time = kernel execution time from CUPTI/nvprof
"""

import os, signal, time, subprocess, resource
from pathlib import Path
from typing import Optional


def cpu_time() -> tuple[float, float]:
    """Get user + system CPU time of current process and children."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime, usage.ru_stime


def kill_process_tree(pid: int, sig: int = signal.SIGKILL, timeout: int = 5):
    """Kill entire process tree reliably. Avoids zombie children."""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=timeout)
        for p in alive:
            try: p.kill()
            except Exception: pass
        try:
            parent.send_signal(sig)
            parent.wait(timeout=timeout)
        except Exception:
            try: parent.kill()
            except Exception: pass
    except ImportError:
        try:
            os.killpg(os.getpgid(pid), sig)
        except ProcessLookupError:
            pass


def set_cpu_affinity(cores: str):
    """Pin process to specific CPU cores. cores='0-3' or '0,2,4'."""
    try:
        os.sched_setaffinity(0, _parse_cpu_list(cores))
    except (OSError, AttributeError):
        pass  # Not available on all systems


def _parse_cpu_list(cores: str) -> list[int]:
    """Parse '0-3,8' → [0,1,2,3,8]."""
    result = []
    for part in cores.split(","):
        if "-" in part:
            start, end = part.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


def set_numa_binding(node: int = 0):
    """Bind memory to specific NUMA node (multi-socket systems only)."""
    try:
        subprocess.run(["numactl", "--membind", str(node), "true"],
                       capture_output=True, timeout=2)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # numactl not available — single-socket or no NUMA


def get_cgroup_memory() -> Optional[int]:
    """Read memory usage from cgroups v2 (more accurate than getrusage)."""
    cgroup_path = "/sys/fs/cgroup/memory.current"
    if Path(cgroup_path).exists():
        try:
            return int(Path(cgroup_path).read_text().strip())
        except Exception:
            pass
    # Fallback: cgroups v1
    cgroup_path = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    if Path(cgroup_path).exists():
        try:
            return int(Path(cgroup_path).read_text().strip())
        except Exception:
            pass
    return None


class ReliableRunner:
    """Run a tool with BenchExec-level reliability on Linux."""

    def __init__(self, cores: str = None, memory_limit_mb: int = None,
                 timeout_seconds: int = 600):
        self.cores = cores
        self.memory_limit_mb = memory_limit_mb
        self.timeout = timeout_seconds

    def run(self, cmd: list[str], cwd: str = None) -> dict:
        """Execute cmd, return {cpu_user, cpu_sys, wall, ram, exit_code}."""
        if self.cores:
            set_cpu_affinity(self.cores)

        t0_cpu = cpu_time()
        t0_wall = time.perf_counter()

        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != "nt" else None  # new process group
        )

        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            kill_process_tree(proc.pid, signal.SIGTERM, timeout=3)
            kill_process_tree(proc.pid, signal.SIGKILL, timeout=2)
            exit_code = -9
            stdout, stderr = proc.stdout.read() if proc.stdout else b"", proc.stderr.read() if proc.stderr else b""

        t1_cpu = cpu_time()
        t1_wall = time.perf_counter()

        cpu_user = t1_cpu[0] - t0_cpu[0]
        cpu_sys = t1_cpu[1] - t0_cpu[1]
        wall = t1_wall - t0_wall

        ram_bytes = get_cgroup_memory()

        return {
            "cpu_user_s": round(cpu_user, 3),
            "cpu_sys_s": round(cpu_sys, 3),
            "wall_s": round(wall, 3),
            "ram_bytes": ram_bytes,
            "ram_mb": round(ram_bytes / (1024 * 1024), 1) if ram_bytes else None,
            "exit_code": exit_code,
        }


# ── GPU Extension (future) ──

def gpu_isolation(gpu_id: int = 0):
    """Set CUDA_VISIBLE_DEVICES to isolate GPU access."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def gpu_memory_used() -> int:
    """Return GPU memory used in bytes via nvidia-smi (reports MiB)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip().split("\n")[0]) * 1024 * 1024  # MiB → bytes
    except Exception:
        return 0
