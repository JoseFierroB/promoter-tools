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
    # Forward user flags first, then wrapper defaults (same precedence as
    # the old sys.argv += [...]: wrapper values win; --help still works).
    _fimo_main(sys.argv[1:] + ["--db", str(PROK_DB), "--label", "fimo_prok",
                               "--tag", "FIMO_PROK"])


if __name__ == "__main__":
    main()
