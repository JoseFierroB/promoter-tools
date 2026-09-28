#!/usr/bin/env python3
"""FIMO + Prokaryote DB — zero-shot (838 motifs) (wrapper for fimo.py)."""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in os.environ.get("PYTHONPATH", ""):
    sys.path.insert(0, str(_ROOT))

from src.runners.fimo import main as _fimo_main

PROK_DB = Path(__file__).resolve().parent.parent.parent / "tools/meme/motif_databases/unified_prokaryote.meme"


def main():
    sys.argv += ["--db", str(PROK_DB), "--label", "fimo_prok", "--tag", "FIMO_PROK"]
    _fimo_main()


if __name__ == "__main__":
    main()
