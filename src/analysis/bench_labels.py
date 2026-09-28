#!/usr/bin/env python3
"""Shared benchmark labels: numberless display names for tables/legends/titles.

Rules:
- Dataset numbers live ONLY in directory/file names (e.g. "2_d39v_tigr4_high").
  Tables, legends and titles use DS_DISPLAY (no numbers).
"""
from datetime import date

DS_DISPLAY = {
    "1_d39v": "D39V",
    "2_d39v_tigr4_high": "D39V+TIGR4 high",
    "2_d39v_tigr4_high_sigma": "D39V+TIGR4 high",
    "4_d39v_tigr4_high": "D39V+TIGR4 high",
    "4_d39v_tigr4_high_sigma": "D39V+TIGR4 high",
    "3_d39v_tigr4_all": "D39V+TIGR4 all",
    "scale_10k": "Scale 10k",
    "scale_30k": "Scale 30k",
    "scale_60k": "Scale 60k",
    "scale_100k": "Scale 100k",
    "scale_200k": "Scale 200k",
    "scale_10k_9tools": "Scale 10k",
    "scale_30k_9tools": "Scale 30k",
    "scale_60k_9tools": "Scale 60k",
    "scale_100k_9tools": "Scale 100k",
}

# ---------------------------------------------------------------------------
# Primary tool palette & order — single source of truth for all plots.
# Grouped by method family (BDT → RF → CNN → NN → gLM → motif/PWM)
# so bars/lines/legends are always in the same, non-metric order.
# ---------------------------------------------------------------------------
TOOL_ORDER = [
    "MLDSPP 0% [BDT]",
    "MLDSPP 75% [BDT]",
    "PromoTech RF-HOT [RF]",
    "PromoterLCNN [CNN]",
    "prompt [NN]",
    "ProkBERT-mini [gLM]",
    "iPro-MP [gLM]",
    "FIMO ProkDB [PWMs]",
    "STREME+FIMO [motif]",
]

# exhaustive style map (color + line style for ROC, color for bars)
PALETTE = {
    "MLDSPP 0% [BDT]":       {"color": "#F81D54", "ls": "-",  "lw": 1.8, "method": "BDT"},
    "MLDSPP 75% [BDT]":      {"color": "#55001C", "ls": "-",  "lw": 1.8, "method": "BDT"},
    "PromoTech RF-HOT [RF]": {"color": "#F1A52B", "ls": "-",  "lw": 1.9, "method": "RF"},
    "PromoterLCNN [CNN]":    {"color": "#973C04", "ls": "-",  "lw": 1.9, "method": "CNN"},
    "prompt [NN]":           {"color": "#0060E6", "ls": "-",  "lw": 1.8, "method": "NN"},
    "ProkBERT-mini [gLM]":   {"color": "#2D004D", "ls": "-",  "lw": 2.3, "method": "gLM"},
    "iPro-MP [gLM]":         {"color": "#BA9CEE", "ls": "-",  "lw": 2.1, "method": "gLM"},
    "FIMO ProkDB [PWMs]":    {"color": "#002616", "ls": "--", "lw": 1.7, "method": "PWMs"},
    "STREME+FIMO [motif]":   {"color": "#18974C", "ls": ":",  "lw": 1.7, "method": "motif"},
}

# flat color map for bar plots (subset of PALETTE)
TOOL_COLORS = {k: v["color"] for k, v in PALETTE.items()}


def run_date() -> str:
    """YYMMDD prefix for output directories (run date, never hardcoded)."""
    return date.today().strftime("%y%m%d")
