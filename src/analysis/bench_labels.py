#!/usr/bin/env python3
"""Shared benchmark labels: numberless display names for tables/legends/titles.

Rules:
- Dataset numbers live ONLY in directory/file names (e.g. "4_d39v_tigr4_high").
  Tables, legends and titles use DS_DISPLAY (no numbers).
"""
from datetime import date

DS_DISPLAY = {
    "1_d39v": "D39V baseline",
    "2_tigr4_high": "TIGR4 high",
    "3_tigr4_all": "TIGR4 all",
    "4_d39v_tigr4_high": "D39V+TIGR4 high",
    "5_d39v_tigr4_all": "D39V+TIGR4 all",
    "6_d39v_igr": "D39V IGR",
    "7_tigr4_igr_high": "TIGR4 IGR high",
    "8_tigr4_igr_all": "TIGR4 IGR all",
    "9_d39v_igr_tigr4_igr_high": "D39V+TIGR4 IGR high",
    "10_d39v_igr_tigr4_igr_all": "D39V+TIGR4 IGR all",
    "11_d39v_cds": "D39V CDS",
    "12_tigr4_cds_high": "TIGR4 CDS high",
    "13_tigr4_cds_all": "TIGR4 CDS all",
    "14_d39v_cds_tigr4_cds_high": "D39V+TIGR4 CDS high",
    "15_d39v_cds_tigr4_cds_all": "D39V+TIGR4 CDS all",
}

# ---------------------------------------------------------------------------
# Canonical tool palette & order — single source of truth for all plots.
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
    "MLDSPP 0% [BDT]":       {"color": "#880E4F", "ls": "-",  "lw": 1.8, "method": "BDT"},
    "MLDSPP 75% [BDT]":      {"color": "#C2185B", "ls": "-",  "lw": 1.8, "method": "BDT"},
    "PromoTech RF-HOT [RF]": {"color": "#EF6C00", "ls": "-",  "lw": 1.9, "method": "RF"},
    "PromoterLCNN [CNN]":    {"color": "#2E7D32", "ls": "-",  "lw": 1.9, "method": "CNN"},
    "prompt [NN]":           {"color": "#0288D1", "ls": "-",  "lw": 1.8, "method": "NN"},
    "ProkBERT-mini [gLM]":   {"color": "#D81B60", "ls": "-",  "lw": 2.3, "method": "gLM"},
    "iPro-MP [gLM]":         {"color": "#7E57C2", "ls": "-",  "lw": 2.1, "method": "gLM"},
    "FIMO ProkDB [PWMs]":    {"color": "#00897B", "ls": "--", "lw": 1.7, "method": "PWMs"},
    "STREME+FIMO [motif]":   {"color": "#795548", "ls": ":",  "lw": 1.7, "method": "motif"},
}

# flat color map for bar plots (subset of PALETTE)
TOOL_COLORS = {k: v["color"] for k, v in PALETTE.items()}


def run_date() -> str:
    """YYMMDD prefix for output directories (run date, never hardcoded)."""
    return date.today().strftime("%y%m%d")
