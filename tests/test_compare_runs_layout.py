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
    # No value labels on bars by design: exact numbers live in the TSVs.
    test.assertEqual(len(ax.texts), 0)
    ticks = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    for i, box in enumerate(ticks):
        for other in ticks[i + 1:]:
            test.assertFalse(box.overlaps(other), "tool names overlap")
    legend = ax.get_legend()
    if legend:
        # Legend in the upper-right corner, inside the axes frame.
        lbox = legend.get_window_extent(renderer)
        test.assertGreaterEqual(lbox.x0, frame.x0 - 1)
        test.assertLessEqual(lbox.x1, frame.x1 + 1)
        test.assertGreaterEqual(lbox.y0, frame.y0 - 1)
        test.assertLessEqual(lbox.y1, frame.y1 + 1)
        cx = (lbox.x0 + lbox.x1) / 2
        cy = (lbox.y0 + lbox.y1) / 2
        test.assertGreater(cx, (frame.x0 + frame.x1) / 2, "legend not on the right")
        test.assertGreater(cy, (frame.y0 + frame.y1) / 2, "legend not in the top")


class ComparisonLayoutTests(unittest.TestCase):
    def tearDown(self):
        plots.plt.close("all")

    def test_dense_regimes_keep_method_order_and_inside_legend(self):
        tools = list(reversed(plots.PRIMARY_ORDER))
        for n in (2, 3, 4):
            for metric in ("time", "ram", "speedup"):
                with self.subTest(regimes=n, metric=metric):
                    runs = [f"run {i}" for i in range(n)]
                    values = {(t, r): (i + 1) * 0.12 + 1.6
                              for t in tools for i, r in enumerate(runs)}
                    values[(tools[0], runs[0])] = np.nan
                    values[(tools[1], runs[0])] = 0.0
                    fig, ax = plots._comparison_figure(len(tools), n)
                    plots.regime_bars(ax, tools, runs, values, metric,
                                      {}, run_display={r: r for r in runs})
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
