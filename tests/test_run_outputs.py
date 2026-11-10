"""Output contracts, using small fixtures and mocked scoring (no model execution)."""
import argparse
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src/analysis"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import run_layout
from run_layout import RunLayout
import generate_auc_plots as auc_plots
import generate_compute_plots as compute
import compute_metrics
import compare_runs
import sigma_stratify
from src import cli


class RunOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.addCleanup(plt.close, "all")

    def fixture(self, date="260917", cfg="1cpu-gpu", seconds=5.0):
        layout = RunLayout(self.root / f"{date}_runs" / "demo" / cfg)
        layout.ensure_dirs()
        pd.DataFrame({"LABEL": [1, 1, 1, 0, 0, 0],
                      "PRED": [.9, .8, .4, .6, .2, .1]}).to_csv(
            layout.inference / "predictions/demo_prompt.tsv", sep="\t", index=False)
        pd.DataFrame([{"tool": "Prompt MLP (B. subtilis 168)", "name": "demo",
                       "configuration": cfg, "wall_seconds": seconds,
                       "peak_ram_mb": 200., "n_sequences": 6, "success": True}]).to_csv(
            layout.resources / "resource_metrics.tsv", sep="\t", index=False)
        return layout

    def test_compute_import_has_no_file_io_or_directory_creation(self):
        with patch.object(Path, "mkdir", side_effect=AssertionError("import writes directory")), \
                patch.object(pd, "read_csv", side_effect=AssertionError("import reads suite")):
            importlib.reload(compute)

    def test_run_resolution_metadata_legacy_and_long_identifiers(self):
        layout = self.fixture()
        alias = self.root / "260917_demo_1cpu-gpu"
        alias.symlink_to(layout.root, target_is_directory=True)
        for path in (layout.root, alias, alias / "1_inference/predictions"):
            resolved = RunLayout.resolve(path)
            self.assertEqual(resolved.root, layout.root)
            self.assertEqual(resolved.dataset_name, "demo")
        empty = RunLayout(self.root / "old")
        empty.ensure_dirs()
        (empty.root / "predictions").mkdir()
        (empty.root / "predictions/present.tsv").touch()
        self.assertEqual(empty.predictions, empty.root / "predictions")
        long_name = "dataset" * 30
        a = run_layout.primary_run_dir("260917", long_name, "1cpu-gpu")
        b = run_layout.primary_run_dir("260917", long_name, "16cpu-gpu")
        c = run_layout.primary_run_dir("260917", long_name + "_02", "1cpu-gpu")
        self.assertEqual(len({a, b, c}), 3)
        self.assertLessEqual(len(a.name), 100)
        self.assertTrue(a.name.endswith("_1cpu-gpu"))

    def test_canonical_metrics_win_and_table_link_is_valid(self):
        layout = self.fixture()
        (layout.root / "resource_metrics.tsv").write_text("stale legacy copy")
        self.assertEqual(layout.find_resource_metrics(), layout.resources / "resource_metrics.tsv")
        link = layout.tables / "resource_metrics.tsv"
        link.symlink_to("../missing.tsv")
        layout.link_resource_table()
        self.assertTrue(link.is_file())
        self.assertEqual(link.resolve(), (layout.resources / "resource_metrics.tsv").resolve())

    def test_title_free_png_pdf_exports_and_no_nested_inference(self):
        layout = self.fixture()
        saved = []
        save = Figure.savefig

        def checked_save(fig, path, *args, **kwargs):
            self.assertTrue(all(not ax.get_title() for ax in fig.axes))
            self.assertIsNone(fig._suptitle)
            self.assertEqual(len(fig.axes), 1)
            self.assertIn(Path(path).suffix, (".png", ".pdf"))
            saved.append(Path(path).relative_to(layout.root))
            return save(fig, path, *args, **kwargs)

        with patch.object(Figure, "savefig", checked_save):
            auc_plots.main(layout.inference / "predictions")
            compute.main_run(layout.root)
        self.assertEqual(len(saved), 6)  # one ROC, time and RAM, two formats each
        self.assertTrue((layout.tables / "metrics_rows.tsv").is_file())
        self.assertTrue((layout.tables / "resource_metrics.tsv").is_file())
        self.assertEqual({p.name for p in layout.root.iterdir()},
                         {"1_inference", "2_resources", "3_tables"})
        self.assertFalse((layout.inference / "1_inference").exists())

    def test_confusion_table_uses_canonical_predictions_and_tables(self):
        layout = self.fixture()
        result = compute_metrics.main(layout.inference / "predictions", n_bootstrap=10)
        self.assertEqual(len(result), 1)
        self.assertTrue((layout.tables / "metrics_table.tsv").is_file())
        self.assertFalse((layout.root / "metrics_table.tsv").exists())

    def test_compare_title_free_and_distinguishes_same_config_other_dates(self):
        a, b = self.fixture(), self.fixture("260918", seconds=10.)
        output = self.root / "comparison"
        heights = []

        def inspect(fig, path, **kwargs):
            self.assertFalse(fig.axes[0].get_title())
            self.assertIn(Path(path).suffix, (".png", ".pdf"))
            if Path(path).stem == "compare_time":
                heights.append([bar.get_height() for bar in fig.axes[0].patches])

        with patch.object(Figure, "savefig", inspect):
            compare_runs.compare_runs([a.root, b.root, a.root], output)
        table = pd.read_csv(output / "3_tables/resources_compare.tsv", sep="\t")
        self.assertEqual(table["run"].nunique(), 2)
        self.assertEqual(heights, [[5., 10.], [5., 10.]])
        self.assertEqual(set(table["speedup_vs_slowest"]), {1., 2.})

    def test_compare_skips_failed_rows_and_empty_tables(self):
        layout = self.fixture()
        table = layout.resources / "resource_metrics.tsv"
        frame = pd.read_csv(table, sep="\t")
        failed = frame.copy()
        failed["success"] = False
        failed["wall_seconds"] = 900
        pd.concat([frame, failed]).to_csv(table, sep="\t", index=False)
        self.assertEqual(compare_runs.load_run(layout.root)[1]["wall_seconds"].tolist(), [5.])
        frame.iloc[:0].to_csv(table, sep="\t", index=False)
        self.assertIsNone(compare_runs.load_run(layout.root))
        table.write_text("")
        self.assertIsNone(compare_runs.load_run(layout.root))

    def test_compare_rejects_overwriting_a_source_capsule(self):
        layout = self.fixture()
        before = sorted(p.relative_to(layout.root).as_posix()
                        for p in layout.root.rglob("*"))
        with self.assertRaises(ValueError):
            compare_runs.compare_runs([layout.root], layout.root)
        with self.assertRaises(ValueError):
            compare_runs.compare_runs([layout.root], layout.root.parent)
        after = sorted(p.relative_to(layout.root).as_posix()
                       for p in layout.root.rglob("*"))
        self.assertEqual(before, after)

    def test_sigma_exports_are_individual_and_title_free(self):
        layout = self.fixture()
        preds = layout.inference / "predictions/demo_prompt.tsv"
        frame = pd.read_csv(preds, sep="\t")
        frame["ID"] = ["p1", "p2", "p3", "n1", "n2", "n3"]
        frame.to_csv(preds, sep="\t", index=False)
        metadata = self.root / "metadata.tsv"
        pd.DataFrame({"Sequence_ID": ["p1", "p2", "p3"],
                      "Sigma_Factor": ["SigA", "SigX", "None"]}).to_csv(
            metadata, sep="\t", index=False)
        saved = []
        def check(fig, path, **kwargs):
            self.assertEqual(len(fig.axes), 1)
            self.assertFalse(fig.axes[0].get_title())
            self.assertIsNone(fig._suptitle)
            saved.append(Path(path).name)
        with patch.object(sys, "argv", ["sigma", "--run-dir", str(layout.root),
                                        "--name", "demo", "--metadata", str(metadata)]), \
                patch.object(Figure, "savefig", check):
            sigma_stratify.main()
        self.assertEqual(set(saved), {f"roc_{sig}_demo.{ext}" for sig in ("SigA", "SigX", "None")
                                     for ext in ("png", "pdf")})

    def test_cli_mock_run_plots_and_repeat_are_self_contained(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "positives_81bp.fasta").write_text(">p1\nAAA\n>p2\nAAC\n>p3\nAAT\n")
        (inputs / "negatives_81bp.fasta").write_text(">n1\nCCC\n>n2\nCCG\n>n3\nCCT\n")
        args = argparse.Namespace(tools=["prompt"], input_dir=str(inputs), pos=None, neg=None,
                                  name="demo", threads=1, gpu=False, cpu_only=True,
                                  no_timeout=False, slurm=False, runs=1, plots=True,
                                  output=str(self.root / "export.tsv"),
                                  output_dir=str(self.root / "external_predictions"))

        class FakeRunner:
            def __init__(self, **kwargs):
                self.output_dir = kwargs["output_dir"]
            def available(self):
                return True
            def run(self, tool):
                pd.DataFrame({"LABEL": [1, 1, 1, 0, 0, 0],
                              "PRED": [.9, .8, .4, .6, .2, .1]}).to_csv(
                    self.output_dir, sep="\t", index=False)
                return {"tool": tool.name, "wall_seconds": 1., "peak_ram_mb": 20.,
                        "n_sequences": 6, "success": True}

        from src.backend import local
        import bench_labels
        with patch.object(cli, "ROOT", self.root), \
                patch.object(bench_labels, "run_date", return_value="260917"), \
                patch.object(local, "LocalRunner", FakeRunner), \
                patch.object(Figure, "savefig"):
            cli.cmd_run(args)
            ext_files = list((self.root / "external_predictions").glob("*prompt*"))
            self.assertTrue(ext_files, "runner output missing in --output-dir")
            # A repeat overwrites the same run dir (no bumps, no aliases).
            args.output_dir = None
            args.output = None
            cli.cmd_run(args)
        run = self.root / "output/260917_runs/demo/1cpu"
        self.assertFalse((self.root / "output/260917_runs/demo_02").exists())
        self.assertFalse((self.root / "output/runs").exists())
        layout = RunLayout(run)
        self.assertEqual(layout.dataset_name, "demo")
        self.assertTrue(layout.find_resource_metrics().is_file())
        self.assertTrue((layout.tables / "resource_metrics.tsv").is_file())
        self.assertTrue((layout.tables / "metrics_rows.tsv").is_file())
        self.assertFalse((run / "resources").exists())
        self.assertFalse((run / "metrics_rows.tsv").exists())
        self.assertFalse((run / "STATUS.json").exists())
        self.assertFalse((run / "MANIFEST.json").exists())
        self.assertEqual(layout.predictions, run / "1_inference/predictions")


if __name__ == "__main__":
    unittest.main()
