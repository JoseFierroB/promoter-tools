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

def run_date() -> str:
    """YYMMDD prefix for output directories (run date, never hardcoded)."""
    return date.today().strftime("%y%m%d")
