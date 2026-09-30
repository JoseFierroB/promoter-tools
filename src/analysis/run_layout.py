#!/usr/bin/env python3
"""Centralized run layout — single source of truth for all output paths.

Reversible design: keeps legacy `output/<YYMMDD>_runs/<name>/<config>/`
working while introducing primary `output/runs/<YYMMDD>_<name>_<config>/`
capsule with `MANIFEST.json`. All path logic lives here so other modules
do not hard-code `"output/..."` strings.

Capsule (one run = one dir):
  output/runs/<YYMMDD>_<name>_<config>/   # primary
  output/<YYMMDD>_runs/<name>/<config>/   # legacy (symlink or same content)
    STATUS.json          # running/complete, tools, input_sha, pos_src/neg_src, n_pos/n_neg
    MANIFEST.json        # {run_id, date, dataset, config, input_sha, n, tools, layout_version, created_at, legacy_path, primary_path}
    1_inference/         # title-free roc_auc_{name}.{png,pdf}, predictions/ subdir
      predictions/       # {name}_{fam}[.tsv] (primary) + legacy fallback at root/predictions
    2_resources/         # resource_metrics.tsv, compute_time/peak_ram.{png,pdf}
    3_tables/            # metrics_rows.tsv, metrics_table.tsv, resource_metrics.tsv (link)
    # no root/predictions, root/plots, root/resources, root/*.tsv — all inside 1/2/3

Comparisons / benchmarks (lightweight views, no prediction copies):
  output/comparisons/<YYMMDD>_<tokens>/   # primary
  output/<YYMMDD>_comparison_*            # legacy (symlink)
    MANIFEST.json  # {sources: [primary run paths], dataset, regimes, input_shas}
    2_resources/compare_{time,ram,speedup,vram}.{png,pdf}
    3_tables/resources_compare.tsv
  output/benchmarks/<YYMMDD>_<Ndatasets>datasets/
    MANIFEST.json
    roc/  atlas/  benchmark_metrics.tsv

Legacy paths remain readable; new per-run exports stay inside 1/2/3.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional

# bump when layout changes
LAYOUT_VERSION = "1.0"

# primary roots (new)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "output/runs"
COMPARISONS_ROOT = PROJECT_ROOT / "output/comparisons"
BENCHMARKS_ROOT = PROJECT_ROOT / "output/benchmarks"

# legacy roots (kept for compat)
LEGACY_RUNS_PREFIX = "output"  # + "/<YYMMDD>_runs/<name>/<config>/"

def _sanitize(s: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip()).strip("_")
    return s or "run"

def primary_run_dir(date: str, name: str, config: str, base: Path = RUNS_ROOT) -> Path:
    """output/runs/<YYMMDD>_<name>_<config>/"""
    import hashlib
    prefix, suffix = f"{date}_", f"_{_sanitize(config)}"
    dataset = _sanitize(name)
    if len(prefix + dataset + suffix) > 100:
        digest = hashlib.sha256(name.encode()).hexdigest()[:10]
        dataset = dataset[:100 - len(prefix + suffix) - 11].rstrip("_") + "_" + digest
    token = prefix + dataset + suffix
    return base / token

def legacy_run_dir(date: str, name: str, config: str, base: Path = PROJECT_ROOT / "output") -> Path:
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
    def from_primary(cls, date: str, name: str, config: str, base: Path = RUNS_ROOT):
        return cls(primary_run_dir(date, name, config, base))

    @classmethod
    def from_legacy(cls, date: str, name: str, config: str, base: Path = PROJECT_ROOT / "output"):
        return cls(legacy_run_dir(date, name, config, base))

    @classmethod
    def resolve(cls, path: Path) -> "RunLayout":
        """Accept either primary or legacy path (or predictions subdir) and return layout."""
        p = Path(path)
        # if pointing at predictions/, lift to parent
        if p.name == "predictions":
            p = p.parent
            if p.name == "1_inference":
                p = p.parent
        p = p.resolve()
        # if legacy has date prefix in parent, keep as is
        return cls(p)

    @property
    def predictions(self) -> Path:
        declared = self.metadata.get("predictions_path")
        if declared:
            path = Path(declared)
            return path if path.is_absolute() else self.root / path
        # primary: 1_inference/predictions — legacy fallback: root/predictions
        cand = self.root / "1_inference" / "predictions"
        if cand.is_dir() and any(cand.iterdir()):
            return cand
        # also check legacy
        legacy = self.root / "predictions"
        return legacy if legacy.exists() else cand

    @property
    def metadata(self) -> dict:
        """Manifest overrides status; both may be absent in old runs."""
        result = {}
        for filename in ("STATUS.json", "MANIFEST.json"):
            path = self.root / filename
            if path.is_file():
                result.update(json.loads(path.read_text()))
        return result

    @property
    def dataset_name(self) -> str:
        """Do not confuse a primary run ID or a repeat suffix with its dataset."""
        import csv
        import re
        name = self.metadata.get("dataset") or self.metadata.get("name")
        if name:
            return str(name)
        metrics = self.find_resource_metrics()
        if metrics:
            with metrics.open() as stream:
                row = next(csv.DictReader(stream, delimiter="\t"), {})
            if row.get("name") or row.get("db"):
                return row.get("name") or row["db"]
        if re.fullmatch(r"\d+cpu(?:-gpu)?", self.root.name):
            return self.root.parent.name
        match = re.fullmatch(r"\d{6}_(.+)_\d+cpu(?:-gpu)?", self.root.name)
        return match.group(1) if match else self.root.name

    def find_resource_metrics(self) -> Optional[Path]:
        """Prefer primary metrics over stale legacy copies; reject ambiguity."""
        directories = (self.resources, self.tables, self.root / "resources", self.root)
        for directory in directories:
            exact = directory / "resource_metrics.tsv"
            if exact.is_file():
                return exact
        for directory in directories:
            candidates = sorted(p for p in directory.glob("resource_metrics*.tsv") if p.is_file())
            if len(candidates) > 1:
                raise ValueError(f"Ambiguous resource metrics in {directory}: {candidates}")
            if candidates:
                return candidates[0]
        return None

    def link_resource_table(self, source: Optional[Path] = None):
        """Expose the actual metrics in tables without duplicating or moving data."""
        import os
        source = source or self.find_resource_metrics()
        if source is None:
            return
        self.tables.mkdir(parents=True, exist_ok=True)
        target = self.tables / "resource_metrics.tsv"
        if target.exists():
            return
        if target.is_symlink():  # repair a dangling legacy link only
            target.unlink()
        target.symlink_to(os.path.relpath(Path(source).resolve(), self.tables.resolve()))

    @property
    def inference(self) -> Path:
        # new primary
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
        # primary hierarchy only — no legacy symlinks/files at root
        (self.root / "1_inference" / "predictions").mkdir(parents=True, exist_ok=True)
        self.resources.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        # note: no creation of root/predictions, root/plots, root/resources — deprecated
        # migration will move existing legacy dirs into primary locations

    def write_manifest(self, *, date: str, name: str, config: str,
                       input_sha: str = "", pos_src: str = "", neg_src: str = "",
                       n_pos: int = 0, n_neg: int = 0, tools: Optional[List[str]] = None,
                       legacy_path: str = "", primary_path: str = ""):
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
            "primary_path": primary_path or str(self.root),
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


def ensure_legacy_symlink(primary: Path, legacy: Path):
    """Create legacy symlink to primary for backward compat (no data duplication)."""
    try:
        if legacy.exists():
            return
        legacy.parent.mkdir(parents=True, exist_ok=True)
        # use relative symlink if possible
        try:
            rel = primary.resolve().relative_to(legacy.parent.resolve())
            legacy.symlink_to(rel)
        except Exception:
            legacy.symlink_to(primary.resolve())
    except Exception:
        pass

def migrate_legacy_to_primary(date: str, name: str, config: str,
                                dry_run: bool = True) -> dict:
    """Plan (and optionally execute) migration of one legacy run to primary."""
    legacy = legacy_run_dir(date, name, config)
    primary = primary_run_dir(date, name, config)
    plan = {
        "legacy": str(legacy),
        "primary": str(primary),
        "legacy_exists": legacy.exists(),
        "primary_exists": primary.exists(),
        "action": "none",
    }
    if not legacy.exists():
        plan["action"] = "no_legacy"
        return plan
    if primary.exists():
        plan["action"] = "exists"
        return plan
    plan["action"] = "symlink" if dry_run else "migrated"
    if not dry_run:
        ensure_legacy_symlink(primary, legacy)
        # if primary does not exist, copy structure via symlink
        # actually we want primary to be the real data, legacy symlink to it
        # so we need to move legacy content to primary then symlink back
        if not primary.exists():
            primary.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(primary)
            ensure_legacy_symlink(primary, legacy)
            plan["action"] = "migrated"
    else:
        # dry-run: just report what would happen
        plan["action"] = "would_symlink"
    return plan
