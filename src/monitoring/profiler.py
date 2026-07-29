"""Monitoring module: GPU-aware resource profiling for any tool."""
import time, os, sys, subprocess, json, threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

try:
    import resource as _resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

@dataclass
class ResourceMetrics:
    tool: str
    category: str = ""  # ML, DL, Other
    wall_seconds: float = 0.0
    cpu_user_seconds: float = 0.0
    cpu_sys_seconds: float = 0.0
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    gpu_name: str = ""
    gpu_available: bool = False
    model_size_mb: float = 0.0
    intermediate_mb: float = 0.0
    n_sequences: int = 0
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class GPUDetector:
    """Auto-detect GPU availability across frameworks."""

    @staticmethod
    def detect() -> dict:
        gpu = {}
        for fw, fn in [("torch", GPUDetector._detect_torch),
                        ("tensorflow", GPUDetector._detect_tf)]:
            try:
                gpu[fw] = fn()
            except Exception:
                gpu[fw] = {"available": False}
        return gpu

    @staticmethod
    def _detect_torch():
        import torch
        avail = torch.cuda.is_available()
        return {"available": avail,
                "name": torch.cuda.get_device_name(0) if avail else None,
                "count": torch.cuda.device_count() if avail else 0}

    @staticmethod
    def _detect_tf():
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        avail = bool(devices)
        return {"available": avail, "count": len(devices),
                "name": devices[0].name if avail else None}


class ResourceProfiler:
    """
    Context manager for measuring resources.

    Usage:
        with ResourceProfiler("promotech_hot") as prof:
            tool.run()
        print(prof.metrics)
    """

    def __init__(self, tool: str, gpu_aware: bool = True):
        self.tool = tool
        self.gpu_aware = gpu_aware
        self._vram_thread = None
        self._vram_samples = []
        self.metrics = ResourceMetrics(tool=tool)

    def __enter__(self):
        self._t0 = time.perf_counter()
        if HAS_RESOURCE:
            self._ru0 = _resource.getrusage(_resource.RUSAGE_SELF)
        if self.gpu_aware:
            self.metrics.gpu_available = self._check_gpu()
            if self.metrics.gpu_available:
                self._start_vram_polling()
        return self

    def __exit__(self, *args):
        self.metrics.wall_seconds = round(time.perf_counter() - self._t0, 3)
        if HAS_RESOURCE:
            ru1 = _resource.getrusage(_resource.RUSAGE_SELF)
            self.metrics.cpu_user_seconds = round(ru1.ru_utime - self._ru0.ru_utime, 3)
            self.metrics.cpu_sys_seconds = round(ru1.ru_stime - self._ru0.ru_stime, 3)
            self.metrics.peak_ram_mb = round(ru1.ru_maxrss / 1024.0, 1)
        if self._vram_thread:
            self._vram_running = False
            self._vram_thread.join(timeout=2)
            if self._vram_samples:
                self.metrics.peak_vram_mb = max(self._vram_samples)
        return False

    def _check_gpu(self) -> bool:
        """Detect CUDA GPU availability. Cached on first call."""
        if not hasattr(self, "_gpu_cached"):
            try:
                import torch
                self._gpu_cached = torch.cuda.is_available()
            except Exception:
                self._gpu_cached = False
        return self._gpu_cached

    def _start_vram_polling(self, interval=0.1):
        self._vram_running = True

        def _poll():
            while self._vram_running:
                try:
                    import torch
                    allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    self._vram_samples.append(allocated)
                except Exception:
                    pass
                time.sleep(interval)

        self._vram_thread = threading.Thread(target=_poll, daemon=True)
        self._vram_thread.start()


def model_size_mb(*paths) -> float:
    """Total size of model files/directories in MB."""
    total = 0
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            total += pp.stat().st_size
        elif pp.is_dir():
            for f in pp.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
    return round(total / (1024 * 1024), 2)


def dir_size_mb(path) -> float:
    """Total size of directory in MB."""
    total = 0
    for f in Path(path).rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return round(total / (1024 * 1024), 2)
