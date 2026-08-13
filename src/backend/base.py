"""Base runner: abstract interface for tool execution backends."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from src.benchmark.tools import Tool


class Runner(ABC):
    """Execute a benchmark tool and return resource metrics.

    Implementations: LocalRunner (subprocess), SlurmRunner (sbatch + sacct).
    """

    @abstractmethod
    def run(self, tool: Tool) -> dict:
        """Execute tool, return dict with keys: tool, category, wall_seconds,
        time_s, peak_ram_mb, peak_vram_mb, mean_cpu_pct, gpu_name,
        gpu_available, model_size_mb, intermediate_mb, n_sequences,
        throughput_seq_s, success, notes."""
        ...

    def available(self) -> bool:
        """Check if this runner's backend is available."""
        return True
