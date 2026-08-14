#!/usr/bin/env python3
"""FIMO + E. coli DB — zero-shot promoter prediction (wrapper for fimo.py)."""
import sys
from pathlib import Path

from fimo import main as _fimo_main

ECOLI_DB = Path(__file__).resolve().parent.parent.parent / "tools/meme/motif_databases/ecoli_combined.meme"

sys.argv += ["--db", str(ECOLI_DB), "--label", "fimo_db", "--tag", "FIMO_DB"]
_fimo_main()