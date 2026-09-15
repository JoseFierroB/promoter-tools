#!/usr/bin/env python3
"""Centralized run layout — single source of truth for all output paths.

Reversible design: keeps legacy `output/<YYMMDD>_runs/<name>/<config>/`
working while introducing canonical `output/runs/<YYMMDD>_<name>_<config>/`
capsule with `MANIFEST.json`. All path logic lives here so other modules
do not hard-code `"output/..."` strings.

Capsule (one run = one dir):
  output/runs/<YYMMDD>_<name>_<config>/   # canonical
  output/<YYMMDD>_runs/<name>/<config>/   # legacy (symlink or same content)
    STATUS.json          # running/complete, tools, input_sha, pos_src/neg_src, n_pos/n_neg
    MANIFEST.json        # {run_id, date, dataset, config, input_sha, n, tools, layout_version, created_at, legacy_path, canonical_path}
    1_inference/         # roc_auc_{name}.{png,pdf,svg}, metrics_rows.tsv, predictions/ subdir
      predictions/       # {name}_{fam}[.tsv] (canonical) + legacy fallback at root/predictions
    2_resources/         # resource_metrics.tsv, compute_time/peak_ram.{png,pdf}
    3_tables/            # metrics_rows.tsv, resource_metrics.tsv (symlinks/copies), benchmark_metrics.tsv
    # no root/predictions, root/plots, root/resources, root/*.tsv — all inside 1/2/3

Comparisons / benchmarks (lightweight views, no prediction copies):
  output/comparisons/<YYMMDD>_<tokens>/   # canonical
  output/<YYMMDD>_comparison_*            # legacy (symlink)
    MANIFEST.json  # {sources: [canonical run paths], dataset, regimes, input_shas}
    2_resources/compare_{time,ram,speedup,vram}.{png,pdf}
    3_tables/resources_compare.tsv
  output/benchmarks/<YYMMDD>_<Ndatasets>datasets/
    MANIFEST.json
    roc/  atlas/  benchmark_metrics.tsv

Revert: `git checkout -- src/analysis/run_layout.py src/cli.py src/analysis/compare_runs.py src/analysis/generate_benchmark_suite.py src/analysis/generate_compute_plots.py src/analysis/generate_auc_plots.py src/analysis/analyze_run.py`
and `rm -rf output/runs output/comparisons output/benchmarks` — legacy paths remain untouched.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional

# bump when layout changes
LAYOUT_VERSION = "1.0"

# canonical roots (new)
RUNS_ROOT = Path("output/runs")
COMPARISONS_ROOT = Path("output/comparisons")
BENCHMARKS_ROOT = Path("output/benchmarks")

# legacy roots (kept for compat)
LEGACY_RUNS_PREFIX = "output"  # + "/<YYMMDD>_runs/<name>/<config>/"

def _sanitize(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip()).strip("_")
    return s or "run"

def canonical_run_dir(date: str, name: str, config: str, base: Path = RUNS_ROOT) -> Path:
    """output/runs/<YYMMDD>_<name>_<config>/"""
    token = f"{date}_{_sanitize(name)}_{_sanitize(config)}"
    # truncate to stay filesystem-friendly
    if len(token) > 100:
        token = token[:100].rstrip("_")
    return base / token

def legacy_run_dir(date: str, name: str, config: str, base: Path = Path("output")) -> Path:
    """output/<YYMMDD>_runs/<name>/<config>/"""
    return base / f"{date}_runs" / _sanitize(name) / _sanitize(config)

def comparison_dir(date: str, tokens: List[str], base: Path = COMPARISONS_ROOT) -> Path:
    stem = f"{date}_comparison_" + "__".join(_sanitize(t) for t in tokens)
    if len(stem) > 100:
        stem = stem[:100].rstrip("_")
    return base / stem

def benchmark_dir(date: str, n_datasets: int, base: Path = BENCHMARKS_ROOT) -> Path:
    return base / f"{date}_{n_datasets}datasets"

class RunLayout:
    """Helper wrapping a single run's directory tree."""

    def __init__(self, run_root: Path):
        self.root = Path(run_root)

    @classmethod
    def from_canonical(cls, date: str, name: str, config: str, base: Path = RUNS_ROOT):
        return cls(canonical_run_dir(date, name, config, base))

    @classmethod
    def from_legacy(cls, date: str, name: str, config: str, base: Path = Path("output")):
        return cls(legacy_run_dir(date, name, config, base))

    @classmethod
    def resolve(cls, path: Path) -> "RunLayout":
        """Accept either canonical or legacy path (or predictions subdir) and return layout."""
        p = Path(path)
        # if pointing at predictions/, lift to parent
        if p.name == "predictions":
            p = p.parent
        # if legacy has date prefix in parent, keep as is
        return cls(p)

    @property
    def predictions(self) -> Path:
        # canonical: 1_inference/predictions — legacy fallback: root/predictions
        cand = self.root / "1_inference" / "predictions"
        if cand.exists():
            return cand
        # also check legacy
        return self.root / "predictions"

    @property
    def inference(self) -> Path:
        # new canonical
        return self.root / "1_inference"

    @property
    def resources(self) -> Path:
        return self.root / "2_resources"

    @property
    def tables(self) -> Path:
        return self.root / "3_tables"

    @property
    def plots(self) -> Path:
        # legacy `plots/` symlink -> 1_inference (deprecated, kept for fallback)
        p = self.root / "plots"
        if p.exists():
            return p
        return self.root / "1_inference"

    @property
    def legacy_predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def legacy_plots(self) -> Path:
        return self.root / "plots"

    @property
    def legacy_resources(self) -> Path:
        return self.root / "resources"

    @property
    def status_path(self) -> Path:
        return self.root / "STATUS.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "MANIFEST.json"

    def ensure_dirs(self):
        # canonical hierarchy only — no legacy symlinks/files at root
        (self.root / "1_inference" / "predictions").mkdir(parents=True, exist_ok=True)
        self.resources.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        # note: no creation of root/predictions, root/plots, root/resources — deprecated
        # migration will move existing legacy dirs into canonical locations

    def write_manifest(self, *, date: str, name: str, config: str,
                       input_sha: str = "", pos_src: str = "", neg_src: str = "",
                       n_pos: int = 0, n_neg: int = 0, tools: Optional[List[str]] = None,
                       legacy_path: str = "", canonical_path: str = ""):
        manifest = {
            "layout_version": LAYOUT_VERSION,
            "run_id": f"{date}_{name}_{config}",
            "date": date,
            "dataset": name,
            "config": config,
            "input_sha": input_sha,
            "pos_src": pos_src,
            "neg_src": neg_src,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_total": (n_pos + n_neg) if (n_pos or n_neg) else 0,
            "tools": sorted(tools or []),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "canonical_path": canonical_path or str(self.root),
            "legacy_path": legacy_path,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        # also ensure STATUS.json has same core fields (written by cli)
        return manifest

    def write_comparison_manifest(self, out_dir: Path, *, sources: List[str],
                                  dataset: str = "", regimes: List[str] = None,
                                  input_shas: List[str] = None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        m = {
            "layout_version": LAYOUT_VERSION,
            "type": "comparison",
            "date": time.strftime("%y%m%d"),
            "dataset": dataset,
            "sources": sources,
            "regimes": regimes or [],
            "input_shas": input_shas or [],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / "MANIFEST.json").write_text(json.dumps(m, indent=2))
        # legacy symlink for old comparison path users
        return m

    def write_benchmark_manifest(self, out_dir: Path, *, source_runs: List[str],
                                 datasets: List[str] = None, n_tools: int = 0):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        m = {
            "layout_version": LAYOUT_VERSION,
            "type": "benchmark",
            "date": time.strftime("%y%m%d"),
            "source_runs": source_runs,
            "datasets": datasets or [],
            "n_tools": n_tools,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / "MANIFEST.json").write_text(json.dumps(m, indent=2))
        return m


def ensure_legacy_symlink(canonical: Path, legacy: Path):
    """Create legacy symlink to canonical for backward compat (no data duplication)."""
    try:
        if legacy.exists():
            return
        legacy.parent.mkdir(parents=True, exist_ok=True)
        # use relative symlink if possible
        try:
            rel = canonical.resolve().relative_to(legacy.parent.resolve())
            legacy.symlink_to(rel)
        except Exception:
            legacy.symlink_to(canonical.resolve())
    except Exception:
        pass

def migrate_legacy_to_canonical(date: str, name: str, config: str,
                                dry_run: bool = True) -> dict:
    """Plan (and optionally execute) migration of one legacy run to canonical."""
    legacy = legacy_run_dir(date, name, config)
    canonical = canonical_run_dir(date, name, config)
    plan = {
        "legacy": str(legacy),
        "canonical": str(canonical),
        "legacy_exists": legacy.exists(),
        "canonical_exists": canonical.exists(),
        "action": "none",
    }
    if not legacy.exists():
        plan["action"] = "no_legacy"
        return plan
    if canonical.exists():
        plan["action"] = "exists"
        return plan
    plan["action"] = "symlink" if dry_run else "migrated"
    if not dry_run:
        ensure_legacy_symlink(canonical, legacy)
        # if canonical does not exist, copy structure via symlink
        # actually we want canonical to be the real data, legacy symlink to it
        # so we need to move legacy content to canonical then symlink back
        if not canonical.exists():
            legacy.rename(canonical)
            ensure_legacy_symlink(canonical, legacy)
            plan["action"] = "migrated"
    else:
        # dry-run: just report what would happen
        plan["action"] = "would_symlink"
    return plan
