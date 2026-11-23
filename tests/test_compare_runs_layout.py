"""Rendering regressions for crowded resource comparisons (no model execution)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "analysis"))
import compare_runs as plots


def assert_readable(test, ax):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    texts = [t.get_window_extent(renderer) for t in ax.texts]
    bars = [p.get_window_extent(renderer) for p in ax.patches
            if np.isfinite(p.get_height()) and p.get_height() > 0]
    for i, box in enumerate(texts):
        test.assertGreaterEqual(box.y0, frame.y0)
        test.assertLessEqual(box.y1, frame.y1 + 1)
        test.assertGreaterEqual(box.x0, frame.x0)
        test.assertLessEqual(box.x1, frame.x1)
        for other in texts[i + 1:]:
            test.assertFalse(box.overlaps(other), "value labels overlap")
        for bar in bars:
            test.assertFalse(box.overlaps(bar), "value label obscured by a bar")
    ticks = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    for i, box in enumerate(ticks):
        for other in ticks[i + 1:]:
            test.assertFalse(box.overlaps(other), "tool names overlap")
    legend = ax.get_legend()
    if legend:
        test.assertFalse(legend.get_window_extent(renderer).overlaps(frame))


class ComparisonLayoutTests(unittest.TestCase):
    def tearDown(self):
        plots.plt.close("all")

    def test_dense_regimes_preserve_every_value_and_method_order(self):
        tools = list(reversed(plots.PRIMARY_ORDER))
        for n in (2, 3, 4):
            for metric, formatter in (("time", plots.fmt_time_short),
                                      ("ram", plots.fmt_ram_value),
                                      ("speedup", lambda v: f"{v:.2f}x")):
                with self.subTest(regimes=n, metric=metric):
                    runs = [f"run {i}" for i in range(n)]
                    values = {(t, r): (i + 1) * 0.12 + 1.6
                              for t in tools for i, r in enumerate(runs)}
                    values[(tools[2], runs[0])] = {
                        "time": 2363.516, "ram": 7386.8, "speedup": 119.77}[metric]
                    if metric == "ram":
                        for r in runs:
                            values[(tools[6], r)] = 7303.1  # equal tall bars
                    values[(tools[0], runs[0])] = np.nan
                    values[(tools[1], runs[0])] = 0.0
                    fig, ax = plots._comparison_figure(len(tools), n)
                    plots.regime_bars(ax, tools, runs, values, metric,
                                      "D39V+TIGR4 high (N=3,465)", formatter, {},
                                      annotation_rotation=45)
                    self.assertTrue(all(t.get_rotation() == 45 for t in ax.texts))
                    self.assertEqual(len(ax.texts), len(values) - 1)
                    self.assertEqual([t.get_text().replace("\n", " ")
                                      for t in ax.get_xticklabels()], plots.PRIMARY_ORDER)
                    for series, run in zip(ax.containers, runs):
                        for bar, tool in zip(series, plots.PRIMARY_ORDER):
                            np.testing.assert_equal(bar.get_height(), values[(tool, run)])
                    assert_readable(self, ax)
                    plots.plt.close(fig)

    def test_vram_reference_ignores_cpu_gpu_monitor_samples(self):
        df = pd.DataFrame({"run": ["cpu", "gpu16", "gpu1"],
                           "peak_vram_mb": [497.1, 3285.3, 4273.3]})
        regimes = {"cpu": "1cpu", "gpu16": "16cpu-gpu", "gpu1": "1cpu-gpu"}
        selected, name = plots._vram_reference(df, regimes)
        self.assertEqual(name, "gpu1")
        self.assertEqual(selected["peak_vram_mb"].tolist(), [4273.3])
        selected, name = plots._vram_reference(df.iloc[:1], {"cpu": "1cpu"})
        self.assertTrue(selected.empty)
        self.assertIsNone(name)


if __name__ == "__main__":
    unittest.main()
