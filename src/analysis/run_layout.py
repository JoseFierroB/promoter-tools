#!/usr/bin/env python3
"""Run layout — single source of truth for all output paths.

One run = one dir:
  output/<YYMMDD>_runs/<name>/<config>/
    1_inference/         # title-free roc_auc_{name}.{png,pdf}, predictions/ subdir
      predictions/       # {name}_{fam}[.tsv]
    2_resources/         # resource_metrics.tsv, compute_time/peak_ram.{png,pdf}
    3_tables/            # metrics_rows.tsv, metrics_table.tsv

Comparisons (lightweight views, no prediction copies):
  output/comparisons/<YYMMDD>_<tokens>/
    2_resources/compare_{time,ram,speedup,vram}.{png,pdf}
    3_tables/resources_compare.tsv

No JSON manifests: runs are plain directories; reruns overwrite.
Old STATUS.json/MANIFEST.json files are still read when present
(backward compat) but nothing writes them anymore.
"""
from __future__ import annotations
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "output/runs"
COMPARISONS_ROOT = PROJECT_ROOT / "output/comparisons"

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

class RunLayout:
    """Helper wrapping a single run's directory tree."""

    def __init__(self, run_root: Path):
        self.root = Path(run_root)

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
            if not path.is_absolute():
                path = self.root / path
            # manifests written on another machine/layout may declare a stale
            # path: trust it only when it exists and is non-empty.
            if path.is_dir() and any(path.iterdir()):
                return path
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

    def ensure_dirs(self):
        (self.root / "1_inference" / "predictions").mkdir(parents=True, exist_ok=True)
        self.resources.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
