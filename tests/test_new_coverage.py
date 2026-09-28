"""Simple tests for new code paths (see docs/RUNNER_CONTRACT.md).

Covers: prefetch waves + numpy IPC (tokenize_texts), strain cuts
(operating_points --neg), --no-compute-plots flag, sigma sibling guard,
DS_DISPLAY 4_ keys. All fast, default env, no GPU/models.
Run: pixi run python -m pytest tests/test_new_coverage.py -q
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "runners"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


def synth_dna(n, seed=7):
    rng = np.random.RandomState(seed)
    return ["".join(rng.choice(list("ACGT"), 81)) for _ in range(n)]


class TokenizeWavesTests(unittest.TestCase):
    def test_order_count_numpy(self):
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from prokbert_mini import tokenize_texts
        texts = synth_dna(3 * 8)  # 3 waves of 8 with batch 8
        out = tokenize_texts(texts, 8, 2)
        self.assertEqual(len(out), 3)
        ref = tokenize_texts(texts, 8, 0)
        for (a_ids, a_m), (b_ids, b_m) in zip(out, ref):
            self.assertIsInstance(a_ids, np.ndarray)
            np.testing.assert_array_equal(a_ids, b_ids)
            np.testing.assert_array_equal(a_m, b_m)


class StrainCutsTests(unittest.TestCase):
    def test_d39v_tigr4_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = root / "run" / "1_inference" / "predictions"
            pred.mkdir(parents=True)
            ids = [f"P{i}" for i in range(6)]
            pd.DataFrame({"ID": ids, "LABEL": [1, 1, 1, 0, 0, 0],
                          "PRED": [.9, .8, .7, .3, .2, .1]}).to_csv(
                pred / "demo_prompt.tsv", sep="\t", index=False)
            meta = pd.DataFrame({
                "Sequence_ID": ["P0", "P1", "P2"],
                "Sigma_Factor": ["SigA", "SigA", "None"],
                "Chromosome": ["D39V", "NC_003028.3", "D39V"]})
            meta_path = root / "meta.tsv"
            meta.to_csv(meta_path, sep="\t", index=False)
            neg = root / "neg.fasta"
            neg.write_text(">NEG_A_D39V_1_+\n" + "A" * 81 + "\n"
                           ">NEG_B_D39V_2_-\n" + "C" * 81 + "\n"
                           ">NEG_C_TIGR4_3_+\n" + "G" * 81 + "\n")
            p = subprocess.run(
                [sys.executable, str(ROOT / "src/analysis/operating_points.py"),
                 "--run-dir", str(root / "run"), "--name", "demo",
                 "--metadata", str(meta_path), "--neg", str(neg)],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            st = pd.read_csv(root / "run" / "3_tables" /
                             "operating_point_metrics_strain.tsv", sep="\t")
            d39v = st[st["cut"] == "d39v"].iloc[0]
            tigr4 = st[st["cut"] == "tigr4"].iloc[0]
            self.assertEqual((d39v["n_pos"], d39v["n_neg"]), (2, 2))
            self.assertEqual((tigr4["n_pos"], tigr4["n_neg"]), (1, 1))


class FlagsTests(unittest.TestCase):
    def test_no_compute_plots_flag(self):
        p = subprocess.run(
            [sys.executable, str(ROOT / "src/cli.py"), "run", "--help"],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0)
        self.assertIn("--no-compute-plots", p.stdout)

    def test_ds_display_4_keys(self):
        from bench_labels import DS_DISPLAY
        self.assertEqual(DS_DISPLAY["4_d39v_tigr4_high_sigma"], "D39V+TIGR4 high")
        self.assertEqual(DS_DISPLAY["4_d39v_tigr4_high"], "D39V+TIGR4 high")


class SiblingGuardTests(unittest.TestCase):
    def _skeleton(self, root, cfg, with_sigma):
        run = root / "ds" / cfg
        pred = run / "1_inference" / "predictions"
        pred.mkdir(parents=True)
        ids = [f"P{i}" for i in range(4)]
        pd.DataFrame({"ID": ids, "LABEL": [1, 1, 0, 0],
                      "PRED": [.9, .8, .3, .2]}).to_csv(
            pred / "demo_prompt.tsv", sep="\t", index=False)
        meta = pd.DataFrame({"Sequence_ID": ["P0", "P1"],
                             "Sigma_Factor": ["SigA", "None"]})
        meta_path = root / "meta.tsv"
        meta.to_csv(meta_path, sep="\t", index=False)
        if with_sigma:
            sig = run / "1_inference" / "sigma"
            sig.mkdir(parents=True)
            (sig / "roc_SigA_demo.png").write_text("x")
            (sig / "roc_SigA_demo.pdf").write_text("x")
            (sig / "roc_ComX_demo.png").write_text("x")
            (sig / "roc_ComX_demo.pdf").write_text("x")
            (sig / "roc_None_demo.png").write_text("x")
            (sig / "roc_None_demo.pdf").write_text("x")
            tab = run / "3_tables"
            tab.mkdir(parents=True, exist_ok=True)
            (tab / "sigma_factor_benchmark_metrics.tsv").write_text("x")
            # outputs newer than inputs
            now = 2000000000
            for f in list(sig.iterdir()) + [tab / "sigma_factor_benchmark_metrics.tsv",
                                            meta_path] + list(pred.iterdir()):
                os.utime(f, (now, now))
        return run, meta_path

    def test_sibling_skip_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._skeleton(root, "1cpu", True)
            run2, meta = self._skeleton(root, "16cpu", False)
            p = subprocess.run(
                [sys.executable, str(ROOT / "src/analysis/sigma_stratify.py"),
                 "--run-dir", str(run2), "--name", "demo",
                 "--metadata", str(meta)],
                capture_output=True, text=True, timeout=300)
            self.assertEqual(p.returncode, 0, p.stderr[-500:])
            self.assertIn("skip", p.stdout)
            self.assertFalse((run2 / "1_inference" / "sigma").exists())


if __name__ == "__main__":
    unittest.main()
