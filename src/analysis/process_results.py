#!/usr/bin/env python3
"""Process benchmark results: extract metrics from Slurm job outputs.

Reads tool output files and sacct data from a benchmark run,
compiles into unified resource metrics TSV and generates plots.

Usage:
    pixi run python src/analysis/process_results.py /hps/nobackup/jlees/fierro/tmp/benchmark_9400329/
"""
import sys, re, json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ToolResult:
    tool: str
    category: str  # ML, DL
    n_sequences: int
    time_s: Optional[float] = None
    throughput_seq_s: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    temp_dir_mb: Optional[float] = None
    gpu_available: bool = False
    gpu_name: str = ""
    success: bool = False
    notes: str = ""


def parse_tool_output(out_file: Path) -> Optional[dict]:
    """Extract timing from tool output file.

    Expected formats:
        LCNN: 1988 seqs in 1.027s
        MLDSPP: 1988 seqs in 0.0024s
        iPro-MP: 1988 seqs in 243.639s
        PromoTech: PromoTech RF-HOT done
    """
    if not out_file.exists() or out_file.stat().st_size == 0:
        return None

    text = out_file.read_text()

    # Generic format: "NAME: N seqs in X.XXXs" (LCNN, MLDSPP, iPro-MP, MEME, FIMO)
    m = re.search(r"(\d+)\s+seqs\s+in\s+([\d.]+)s", text)
    if m:
        n = int(m.group(1))
        t = float(m.group(2))
        return {"time_s": round(t, 4), "n_seqs": n,
                "throughput_seq_s": round(n / t, 1) if t > 0 else None}

    # PromoTech format: "PromoTech RF-XXX done" or "PromoTech RF-XXX: N seqs in X.XXXs"
    if "RF-HOT done" in text or "RF-TETRA done" in text:
        return {"time_s": None, "n_seqs": 0, "throughput_seq_s": None,
                "notes": "full pipeline, use sacct CPU time"}

    return None


def map_tool_name(filename: str) -> tuple[str, str]:
    """Map filename to tool display name and category."""
    mapping = {
        "lcnn": ("PromoterLCNN", "DL"),
        "mldspp": ("MLDSPP XGBoost", "ML"),
        "ipromp_sp12": ("iPro-MP sp12 (GPU)", "DL"),
        "promotech_hot": ("PromoTech RF-HOT", "ML"),
        "promotech_tetra": ("PromoTech RF-TETRA", "ML"),
    }
    for key, (name, cat) in mapping.items():
        if key in filename.lower():
            return name, cat
    return filename, "Other"


def _enrich_sacct(results: list, results_dir: Path):
    """Augment results with sacct data (CPU time, RAM, GPU) for each job."""
    try:
        from src.monitoring.slurm_metrics import collect_job_metrics
    except ImportError:
        return

    # Extract job IDs from filenames: toolname_1234567.out
    for r in results:
        for f in results_dir.glob(f"*_*.out"):
            name = f.stem.split("_")
            if len(name) > 1 and name[-1].isdigit():
                job_id = name[-1]
                # Match job to result
                if r.tool.lower().replace(" ", "_").replace("(", "").replace(")", "")[:10] in f.stem.lower():
                    try:
                        sacct = collect_job_metrics(job_id, r.tool)
                        if (r.time_s is None or r.time_s == 0) and sacct.cpu_time_raw > 0:
                            r.time_s = sacct.cpu_time_raw
                        if sacct.max_rss_mb > 0:
                            r.peak_ram_mb = sacct.max_rss_mb
                        if sacct.req_gpu:
                            r.gpu_available = True
                        if sacct.temp_dir_size_mb > 0:
                            r.temp_dir_mb = sacct.temp_dir_size_mb
                        if sacct.cpu_time_raw > 0 and r.n_sequences > 0:
                            r.throughput_seq_s = round(r.n_sequences / sacct.cpu_time_raw, 1)
                    except Exception:
                        pass
                    break


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()

    print(f"Processing results from: {results_dir}")
    if not results_dir.exists():
        print(f"ERROR: Directory not found: {results_dir}")
        sys.exit(1)

    results = []

    # Read tool outputs (skip plots output and scripts dir)
    for out_file in sorted(results_dir.glob("*_*.out")):
        if "plots_" in out_file.stem:
            continue
        tool_name, category = map_tool_name(out_file.stem)
        info = parse_tool_output(out_file)

        if info:
            r = ToolResult(
                tool=tool_name,
                category=category,
                n_sequences=info.get("n_seqs", 0),
                time_s=info.get("time_s"),
                throughput_seq_s=info.get("throughput_seq_s"),
                success=True,
                notes=info.get("notes", ""),
            )
        else:
            # Failed job — check .err
            err_file = out_file.with_suffix(".err")
            err_text = ""
            if err_file.exists():
                err_text = err_file.read_text()[-200:]
            r = ToolResult(
                tool=tool_name,
                category=category,
                n_sequences=0,
                success=False,
                notes=f"FAILED: {err_text.strip()[-80:]}" if err_text else "FAILED (no output)",
            )

        results.append(r)
        status = "OK" if r.success else "FAIL"
        time_str = f"{r.time_s:.3f}s" if r.time_s else "N/A"
        print(f"  [{status}] {r.tool:<25} {time_str}")

    if not results:
        print("No results found.")
        sys.exit(1)

    # Save TSV
    df = pd.DataFrame([asdict(r) for r in results])
    out_tsv = results_dir / "benchmark_metrics.tsv"
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"\nSaved: {out_tsv}")

    # Plot
    successful = [r for r in results if r.success and r.time_s is not None]
    if successful:
        colors = ["#4DAF4A" if r.category == "ML" else "#377EB8" for r in successful]
        names = [r.tool for r in successful]
        times = [r.time_s for r in successful]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        ax.barh(names, times, color=colors, height=0.6)
        for i, (t, r) in enumerate(zip(times, successful)):
            lbl = f"{t:.0f}s" if t > 1 else f"{t:.3f}s"
            ax.text(t + max(times) * 0.02, i, lbl, va="center", fontsize=7)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Benchmark Results — Codon A100 GPU")
        ax.set_xscale("log")
        ax.invert_yaxis()
        plt.tight_layout()
        out_plot = results_dir / "benchmark_results.svg"
        plt.savefig(out_plot, dpi=300, bbox_inches="tight")
        plt.savefig(str(out_plot).replace(".svg", ".png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot: {out_plot}")

    # ── Enrich with slurm_metrics (sacct: CPU time, RAM) ──
    _enrich_sacct(results, results_dir)

    # ── Modular: save JSON for pipeline consumers ──
    metrics_json = results_dir / "benchmark_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"JSON: {metrics_json}")


if __name__ == "__main__":
    main()
