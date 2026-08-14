#!/usr/bin/env python3
"""FIMO + Prokaryote DB — zero-shot (838 motifs) (wrapper for fimo.py)."""
import sys
from pathlib import Path

from fimo import main as _fimo_main

PROK_DB = Path(__file__).resolve().parent.parent.parent / "tools/meme/motif_databases/unified_prokaryote.meme"

sys.argv += ["--db", str(PROK_DB), "--label", "fimo_prok", "--tag", "FIMO_PROK"]
_fimo_main()