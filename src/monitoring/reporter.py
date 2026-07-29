"""Metrics reporter: save to TSV, generate plots."""
import pandas as pd
from pathlib import Path

from src.monitoring.profiler import ResourceMetrics


class MetricsReporter:
    """Collect ResourceMetrics, append to TSV, generate comparison plots."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[ResourceMetrics] = []

    def record(self, metrics: ResourceMetrics):
        self.metrics.append(metrics)

    def save_tsv(self, path: Path = None):
        tsv = path or self.output_dir / "resource_metrics.tsv"
        df = pd.DataFrame([m.to_dict() for m in self.metrics])
        df.to_csv(tsv, sep="\t", index=False)
        print(f"  Metrics saved: {tsv}")
        return df

    def plot(self, path: Path = None, title: str = "Inference Time — Promoter Tools"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        svg = path or self.output_dir / "resource_comparison.svg"
        tools = sorted(self.metrics, key=lambda m: m.wall_seconds)
        names = [m.tool for m in tools]
        times = [m.wall_seconds for m in tools]
        colors = ['#4DAF4A' if m.category == 'ML' else '#377EB8' for m in tools]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.barh(names, times, color=colors, height=0.6)
        for i, t in enumerate(times):
            label = f"{t:.1f}s" if t > 1 else f"{t:.3f}s"
            ax.text(t + max(times) * 0.02, i, label, va="center", fontsize=7)
        ax.set_xlabel("Wall-clock time (seconds)")
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(svg, dpi=300, bbox_inches="tight")
        plt.savefig(str(svg).replace(".svg", ".png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {svg} + .png")

    def print_summary(self):
        print(f"\n{'Tool':<30} {'Time':>8} {'RAM':>8} {'VRAM':>8} {'GPU':>5}")
        print("-" * 65)
        for m in sorted(self.metrics, key=lambda m: m.wall_seconds):
            gpu = "Yes" if m.gpu_available else "No"
            ram = f"{m.peak_ram_mb:.0f}MB" if m.peak_ram_mb else "—"
            vram = f"{m.peak_vram_mb:.0f}MB" if m.peak_vram_mb else "0"
            print(f"{m.tool:<30} {m.wall_seconds:>7.2f}s {ram:>8} {vram:>8} {gpu:>5}")
