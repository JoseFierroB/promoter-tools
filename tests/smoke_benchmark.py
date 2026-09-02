#!/usr/bin/env python3
"""Smoke integration test: run 2 tools on the canonical d39v dataset and assert outputs."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".pixi/envs/default/bin/python")
POS = ROOT / "data/benchmark/d39v/positives_81bp.fasta"
NEG = ROOT / "data/benchmark/d39v/negatives_81bp.fasta"


def run_tool(tool: str, out_dir: Path):
    m = out_dir / "resource_metrics.tsv"
    subprocess.run([PY, "src/cli.py", "run", tool, "--pos", str(POS), "--neg", str(NEG),
                    "--threads", "1", "--runs", "1", "--output-dir", str(out_dir / "predictions"),
                    "-o", str(m)], cwd=ROOT, check=True)
    return pd.read_csv(m, sep="\t").iloc[0]


def test_cli_guardrails():
    r1 = subprocess.run([PY, "src/cli.py", "run", "tool_inexistente"], cwd=ROOT,
                        capture_output=True, text=True)
    assert r1.returncode == 2, f"invalid tool must exit 2 (got {r1.returncode})"
    assert "unknown tool" in r1.stdout.lower(), "invalid tool message"
    r2 = subprocess.run([PY, "src/cli.py", "run"], cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode == 2, f"sin args debe salir 2 (got {r2.returncode})"
    print("OK CLI guardrails (exit 2 + lista de tools)")


def main():
    test_cli_guardrails()
    out = ROOT / "output/smoke_test"
    out.mkdir(parents=True, exist_ok=True)
    for tool, pos_csv, neg_csv in [
        ("lcnn", "predictions/lcnn/lcnn_pos.csv", "predictions/lcnn/lcnn_neg.csv"),
        ("mldspp", "predictions/mldspp_pos.csv", "predictions/mldspp_neg.csv"),
    ]:
        r = run_tool(tool, out)
        assert bool(r["success"]), f"{tool} success"
        assert float(r["time_s"]) > 0, f"{tool} time_s"
        assert float(r["peak_ram_mb"]) > 0, f"{tool} ram"
        for csv in (pos_csv, neg_csv):
            df = pd.read_csv(out / csv, sep="\t")
            assert "PRED" in df.columns, f"{tool} PRED column"
            assert len(df) > 0 and not df["PRED"].isna().any(), f"{tool} NaN preds"
            assert df["PRED"].between(0, 1).all(), f"{tool} range"
        print(f"OK {tool}: {r['time_s']}s {r['peak_ram_mb']}MB")
    print("SMOKE TEST PASADO")


if __name__ == "__main__":
    main()