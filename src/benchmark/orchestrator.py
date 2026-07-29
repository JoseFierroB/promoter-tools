"""Benchmark orchestrator: runs tools with resource monitoring."""
import sys, time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.profiler import ResourceProfiler, GPUDetector, model_size_mb
from monitoring.reporter import MetricsReporter, ResourceMetrics
from benchmark.tools import PROMOTER_TOOLS, get_enabled_tools, enable, disable

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT / "output"


class Orchestrator:
    """Run enabled tools sequentially with resource monitoring."""

    def __init__(self, monitor: bool = True, force_gpu: bool = False):
        self.monitor = monitor
        self.force_gpu = force_gpu
        self.reporter = MetricsReporter(OUTPUT_DIR / "tables")
        self.gpu_info = GPUDetector.detect()
        self._log_gpu()

    def _log_gpu(self):
        print("=" * 55)
        print("  GPU Detection")
        for fw, info in self.gpu_info.items():
            status = f"{info['name']} ({info['count']} devices)" if info.get("available") else "Not available"
            print(f"  {fw}: {status}")
        print("=" * 55)

    def run_tool(self, tool_key: str):
        """Run a single tool with resource profiling."""
        tool = PROMOTER_TOOLS[tool_key]
        if not tool.enabled:
            print(f"  [{tool.name}] DISABLED — skipping")
            return None

        print(f"\n─── [{tool.name}] ───")
        print(f"  Category: {tool.category} | GPU: {'Yes' if tool.gpu_capable else 'No'}")
        print(f"  Inputs: {[str(p.name) for p in tool.inputs]}")
        print(f"  Model: {tool.model_size_mb():.0f} MB")

        with ResourceProfiler(tool.name, gpu_aware=tool.gpu_capable) as prof:
            success = self._execute(tool)

        if not success:
            print(f"  [{tool.name}] FAILED")
            return None

        metrics = prof.metrics
        metrics.category = tool.category
        metrics.model_size_mb = tool.model_size_mb()
        metrics.n_sequences = tool.n_sequences
        metrics.notes = tool.notes
        if tool.gpu_capable:
            gpu = self.gpu_info.get("torch", self.gpu_info.get("tensorflow", {}))
            metrics.gpu_available = gpu.get("available", False)
            metrics.gpu_name = gpu.get("name", "")

        self.reporter.record(metrics)
        print(f"  Done: {metrics.wall_seconds:.2f}s | RAM: {metrics.peak_ram_mb:.0f}MB | VRAM: {metrics.peak_vram_mb:.0f}MB")
        return metrics

    def _execute(self, tool) -> bool:
        """Execute tool via subprocess in its pixi environment."""
        try:
            return self._run_inline(tool)
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def _run_inline(self, tool) -> bool:
        """Run tool in its pixi env using inline Python with resource measurement."""
        import subprocess

        # PromoTech uses shell subprocess, not inline
        if "PromoTech" in tool.name:
            return self._run_promotech_subprocess(tool)

        from src.benchmark.tool_runners import get_runner_code

        pos_path = str(tool.inputs[0]) if tool.inputs else ""
        neg_path = str(tool.inputs[1]) if len(tool.inputs) > 1 else ""
        combined = pos_path

        code = get_runner_code(tool.short_name, pos_path, neg_path, combined)
        if not code:
            print(f"  No runner for {tool.short_name}")
            return False

        env_path = tool.pixi_env
        cmd = ["pixi", "run", "--manifest-path", str(env_path), "python", "-c", code]
        if tool.short_name.startswith("ipromp"):
            cmd.insert(2, "-e"); cmd.insert(3, "ipro-mp")

        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)
        if "DONE" not in res.stdout and "DONE" not in res.stderr:
            print(f"    stdout: {res.stdout[-200:]}")
            print(f"    stderr: {res.stderr[-200:]}")
            return False
        return True

    def _run_promotech_subprocess(self, tool) -> bool:
        import subprocess
        pt_dir = ROOT / "tools/Promotech"
        model_map = {"promotech_hot": "RF-HOT", "promotech_tetra": "RF-TETRA"}
        model = model_map.get(tool.short_name, "RF-HOT")
        for label, fasta in [("pos", tool.inputs[0]), ("neg", tool.inputs[1])]:
            od = ROOT / f"output/predictions/promotech/workdir/{tool.short_name.replace('promotech_','')}_pg_{label}"
            od.mkdir(parents=True, exist_ok=True)
            for step, args in [
                ("parse", ["-pg", "-m", model, "-f", str(fasta), "-o", str(od)]),
                ("predict", ["-g", "-m", model, "-t", "0.0", "-i", str(od), "-o", str(od)]),
            ]:
                res = subprocess.run(
                    ["pixi", "run", "python", "promotech.py"] + args,
                    capture_output=True, text=True, cwd=str(pt_dir), timeout=600)
                if res.returncode != 0:
                    print(f"    [{tool.name}] {step} failed: {res.stderr[-200:]}")
                    return False
        return True

    def run_all(self, tools: list[str] = None):
        """Run all enabled tools (or specific subset)."""
        if tools:
            enable(tools)
            disable([k for k in PROMOTER_TOOLS if k not in tools])

        enabled = get_enabled_tools()
        print(f"\nRunning {len(enabled)} tools...")
        for tool in enabled:
            self.run_tool(tool.short_name)

        self.reporter.save_tsv()
        self.reporter.plot()
        self.reporter.print_summary()
        return self.reporter


def main():
    """Entry point for standalone benchmark run."""
    import argparse
    p = argparse.ArgumentParser(description="Run benchmark with resource monitoring.")
    p.add_argument("--tools", nargs="*", default=None, help="Tool keys to run (promotech_hot, lcnn, etc.)")
    p.add_argument("--no-monitor", action="store_true", help="Disable resource monitoring")
    p.add_argument("--gpu", action="store_true", help="Force GPU mode")
    args = p.parse_args()

    orch = Orchestrator(monitor=not args.no_monitor, force_gpu=args.gpu)
    orch.run_all(args.tools)


if __name__ == "__main__":
    main()
