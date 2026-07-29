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
        """Execute tool, return dict with keys: tool, time_s, ram_mb, vram_mb,
        gpu_name, success, notes, n_sequences, throughput_seq_s."""
        ...

    def available(self) -> bool:
        """Check if this runner's backend is available."""
        return True
