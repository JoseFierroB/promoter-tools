"""Runner contract tests (see docs/RUNNER_CONTRACT.md).

Fast, no models, no GPU: --help via subprocess, timing parser +
RUN.log helper as unit tests, analyze BROKEN path on tiny TSVs,
seed hygiene via AST (no global random.seed in src/runners/).
Run: pixi run python -m pytest tests/test_runners_contract.py -q
"""
import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "backend"))

import pandas as pd  # noqa: E402

RUNNERS = {
    "mldspp": "src/runners/mldspp.py",
    "mldspp_75": "src/runners/mldspp_75.py",
    "promotech_hot": "src/runners/promotech_hot.py",
    "lcnn": "src/runners/lcnn.py",
    "prompt": "src/runners/prompt.py",
    "prokbert": "src/runners/prokbert_mini.py",
    "ipromp_sp12": "src/runners/ipromp_sp12.py",
    "fimo_prok": "src/runners/fimo.py",
    "meme": "src/runners/meme.py",
}


class HelpTests(unittest.TestCase):
    def test_help_exits_0_and_declares_required_flags(self):
        try:
            import transformers  # noqa: F401
            transformers_ok = True
        except ImportError:
            transformers_ok = False
        for key, rel in RUNNERS.items():
            with self.subTest(runner=key):
                if key == "prokbert" and not transformers_ok:
                    self.skipTest("transformers missing in test env (use ipro-mp env)")
                p = subprocess.run(
                    [sys.executable, str(ROOT / rel), "--help"],
                    capture_output=True, text=True, timeout=300)
                self.assertEqual(p.returncode, 0, p.stderr[-500:])
                for flag in ("--pos", "--neg", "-o", "--output"):
                    self.assertIn(flag, p.stdout)


class TimingParserTests(unittest.TestCase):
    def test_train_infer_split(self):
        from local import _parse_train_infer
        self.assertEqual(
            _parse_train_infer("MEME: 3465 seqs (train 12.345s / infer 67.890s)"),
            (12.345, 67.89))
        self.assertEqual(
            _parse_train_infer("MLDSPP: 10 seqs (train 1.234s / infer 0.0060s)"),
            (1.234, 0.006))

    def test_no_split_means_pure_inference(self):
        from local import _parse_train_infer
        self.assertEqual(
            _parse_train_infer("ProkBERT: 10 seqs (5 Pos / 5 Neg) in 1.5s [batch=64]"),
            (0.0, None))


class RunLogTests(unittest.TestCase):
    def test_one_line_per_tool_next_to_manifest(self):
        from src.cli import _append_run_log
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _append_run_log(root, None, "demo", "16cpu", "abc123", [
                {"tool": "T", "wall_seconds": 1.5, "train_s": 0.0,
                 "infer_s": 1.2, "peak_ram_mb": 10, "peak_vram_mb": 0,
                 "success": True, "notes": ""}])
            text = (root / "RUN.log").read_text()
            self.assertIn("input_sha=abc123", text)
            self.assertIn("T wall=1.5s train=0.0s infer=1.2s", text)


class AnalyzeBrokenTests(unittest.TestCase):
    def _write(self, pred_dir, rows):
        ids = [f"s{i}" for i in range(len(rows[0]))]
        lab = [1, 1, 1, 0, 0, 0]
        pd.DataFrame({"ID": ids, "LABEL": lab, "PRED": rows[0]}).to_csv(
            pred_dir / "demo_prompt.tsv", sep="\t", index=False)
        pd.DataFrame({"ID": ids, "LABEL": lab, "PRED": rows[1]}).to_csv(
            pred_dir / "demo_prokbert.tsv", sep="\t", index=False)

    def test_constant_scores_marked_broken_and_exit_1(self):
        from analyze_run import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "pred"
            pred.mkdir()
            self._write(pred, [[0.5] * 6, [.9, .8, .4, .6, .2, .1]])
            with self.assertRaises(SystemExit) as cm:
                analyze_run(pred, "demo", Path(tmp) / "out")
            self.assertEqual(cm.exception.code, 1)
            met = pd.read_csv(Path(tmp) / "out" / "3_tables" / "metrics_rows.tsv",
                              sep="\t")
            self.assertEqual(
                met[met.tool == "prompt [NN]"]["status"].iloc[0], "BROKEN")
            self.assertEqual(
                met[met.tool == "ProkBERT-mini [gLM]"]["status"].iloc[0], "ok")

    def test_healthy_run_returns_table(self):
        from analyze_run import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "pred"
            pred.mkdir()
            self._write(pred, [[.9, .7, .6, .4, .3, .1], [.9, .8, .4, .6, .2, .1]])
            met = analyze_run(pred, "demo", Path(tmp) / "out")
            self.assertTrue((met["status"] == "ok").all())


class SeedHygieneTests(unittest.TestCase):
    def test_no_global_seed_calls_in_runners(self):
        offenders = []
        for path in (ROOT / "src" / "runners").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "seed" and isinstance(node.func.value, ast.Name):
                        offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
