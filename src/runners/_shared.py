"""Shared helpers for tool runners (deduplicated)."""
import numpy as np

STABILITY_MAP = {
    "AA": -1.00, "TT": -1.00,
    "AT": -0.88, "TA": -0.58,
    "AG": -1.30, "GA": -1.30,
    "AC": -1.45, "CA": -1.45,
    "TG": -1.44, "GT": -1.44,
    "TC": -1.28, "CT": -1.28,
    "CC": -1.84, "GG": -1.84,
    "CG": -2.24, "GC": -2.27,
}


def extract_aligned(seq):
    """SantaLucia dinucleotide stability: 80bp window (middle for long seqs,
    start for short). 79 features."""
    s = seq.upper()
    if len(s) >= 100:
        s = s[20:100]
    else:
        s = s[:80]
    return np.array([STABILITY_MAP.get(s[i:i + 2], -1.35) for i in range(79)])


def fimo_score_merge(stdout: str) -> dict:
    """Merge FIMO --text output: max -log10(p-value) per sequence name."""
    import csv
    import math

    scores = {}
    for row in csv.DictReader(stdout.splitlines(), delimiter="\t"):
        try:
            pv = float(row["p-value"])
        except (ValueError, KeyError, TypeError):
            continue
        nl = 999.0 if pv <= 0 else -math.log10(pv)
        s = row["sequence_name"]
        if s not in scores or nl > scores[s]:
            scores[s] = nl
    return scores


def get_promotech_python(promotech_dir):
    """Resolve the python binary from PromoTech's pixi environment."""
    from src.config import config
    return str(config.get_env_python(promotech_dir / "pixi.toml"))