#!/usr/bin/env python3
"""
Preflight checks — run before any experiment to catch recurring pitfalls.
Usage: pixi run python src/utils/preflight.py [--strict]
"""

import sys
import hashlib
import re
from pathlib import Path
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent.parent

# ── RULE 1: Data integrity ──
BENCHMARK_HASHES = {
    "data/benchmark/d39v/positives_81bp.fasta": "bf85ed392b0bd3ab9f62ce9c35da2cf6",
    "data/benchmark/d39v/negatives_81bp.fasta": "193cdb5e265519149ca07b6c475cae7a",
}

def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

# ── RULE 2: Sample size minimums ──
MIN_SAMPLE = 50

# ── RULE 3: Strand verification ──
def verify_strands(fasta_path, strict=False):
    """Check that minus-strand sequences are reverse-complemented correctly.
    Returns (plus_ok, minus_ok, issues)."""
    issues = []
    n_plus = n_minus = 0
    plus_ok = minus_ok = 0
    
    for r in SeqIO.parse(fasta_path, "fasta"):
        parts = r.id.split("_")
        strand = None
        for i, p in enumerate(parts[:-1]):
            if parts[i+1] in ("+", "-"):
                strand = parts[i+1]
                break
        
        if strand == "+":
            n_plus += 1
            # Plus strand: TSS (+1) should be A or G (purine preference)
            if len(r.seq) >= 61 and r.seq[60] in "AG":
                plus_ok += 1
        elif strand == "-":
            n_minus += 1
            # Minus strand: after revcomp, TSS should also be purine
            if len(r.seq) >= 61 and r.seq[60] in "AG":
                minus_ok += 1
    
    total = n_plus + n_minus
    if total == 0:
        return True, True, ["No sequences found"]
    
    plus_ratio = plus_ok / n_plus if n_plus else 1.0
    minus_ratio = minus_ok / n_minus if n_minus else 1.0
    
    if plus_ratio < 0.5:
        issues.append(f"Low purine at TSS+1 on plus strand: {plus_ratio:.1%}")
    if minus_ratio < 0.5:
        issues.append(f"Low purine at TSS+1 on minus strand: {minus_ratio:.1%} — possible strand inversion")
    
    return plus_ratio >= 0.5, minus_ratio >= 0.5, issues


# ── RULE 4: Data leakage ──
def check_leakage(train_fasta, test_fasta):
    """Verify no overlapping sequence IDs between train and test."""
    train_ids = set()
    for r in SeqIO.parse(train_fasta, "fasta"):
        train_ids.add(r.id)
    test_ids = set()
    for r in SeqIO.parse(test_fasta, "fasta"):
        test_ids.add(r.id)
    overlap = train_ids & test_ids
    return len(overlap) == 0, len(train_ids), len(test_ids), len(overlap)


# ── RULE 5: Reference consistency ──
def check_benchmark_consistency():
    """Verify benchmark regenerates identically."""
    import subprocess, tempfile, shutil
    
    tmpdir = Path(tempfile.mkdtemp(prefix="preflight_"))
    try:
        res = subprocess.run([
            "pixi", "run", "python", "src/dataset/positive_tss_d39v.py",
            "--gff", "data/reference/D39V_annotation_TSS_Victor.gff",
            "--fasta", "data/reference/D39V.fna",
            "--gff-cds", "data/reference/sequence.gff3",
            "-u", "60", "-d", "20",
            "-o", str(tmpdir / "test")
        ], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
        
        if res.returncode != 0:
            return False, f"Regeneration failed: {res.stderr[-200:]}"
        
        new_hash = md5(tmpdir / "test.fasta")
        bench_hash = md5(ROOT / "data/benchmark/d39v/positives_81bp.fasta")
        
        if new_hash == bench_hash:
            return True, "Benchmark regenerates identically"
        else:
            # Check per-sequence match
            new_seqs = {r.id: str(r.seq) for r in SeqIO.parse(tmpdir / "test.fasta", "fasta")}
            bench_seqs = {r.id: str(r.seq) for r in SeqIO.parse(ROOT / "data/benchmark/d39v/positives_81bp.fasta", "fasta")}
            shared = set(new_seqs) & set(bench_seqs)
            diffs = sum(1 for k in shared if new_seqs[k] != bench_seqs[k])
            only_new = len(new_seqs) - len(shared)
            only_bench = len(bench_seqs) - len(shared)
            return False, f"Benchmark mismatch: {diffs} diffs, {only_new} only-new, {only_bench} only-bench"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── RULE 6: Git history check ──
def check_git_history(keyword):
    """Check if this topic was already worked on."""
    import subprocess
    res = subprocess.run(
        ["git", "log", "--oneline", "--all", f"--grep={keyword}"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=10
    )
    return res.stdout.strip()


# ═══════════════════════════════════════════════════════
def run_all(strict=False):
    """Run all preflight checks. Returns (pass, report)."""
    report = []
    all_ok = True
    
    # R1: Data files exist + hash check
    print("═══ PREFLIGHT CHECKS ═══")
    print()
    
    for label, path in [
        ("Positives (81bp)", "data/benchmark/d39v/positives_81bp.fasta"),
        ("Negatives (81bp)", "data/benchmark/d39v/negatives_81bp.fasta"),
    ]:
        p = ROOT / path
        if p.exists():
            n = sum(1 for _ in SeqIO.parse(p, "fasta"))
            h = md5(p)
            expected = BENCHMARK_HASHES.get(path)
            if expected and h != expected:
                print(f"  ✗ {label}: {n} seqs, HASH MISMATCH (got {h[:12]}, expected {expected[:12]})")
                all_ok = False
            else:
                status = "✓" if expected else "○"
                print(f"  {status} {label}: {n} seqs, md5={h[:12]}")
        else:
            print(f"  ✗ {label}: MISSING")
            all_ok = False
    
    # R2: Strand verification
    pos_ok, neg_ok, strand_issues = verify_strands(ROOT / "data/benchmark/d39v/positives_81bp.fasta")
    print(f"  {'✓' if pos_ok and neg_ok else '✗'} Strand check: +{pos_ok}/-{neg_ok}")
    for issue in strand_issues:
        print(f"     ⚠  {issue}")
        if strict and "inversion" in issue:
            all_ok = False
    
    # R3: Sample sizes
    pos_n = sum(1 for _ in SeqIO.parse(ROOT / "data/benchmark/d39v/positives_81bp.fasta", "fasta"))
    print(f"  {'✓' if pos_n >= MIN_SAMPLE else f'✗ n={pos_n} < {MIN_SAMPLE}'} Sample size: n={pos_n}")
    
    # R4: Data leakage check (if train/test files specified)
    # Skipped — requires user to specify files
    
    # R5: Benchmark consistency (slow — only with --strict)
    if strict:
        ok, msg = check_benchmark_consistency()
        print(f"  {'✓' if ok else '✗'} Benchmark consistency: {msg}")
        if not ok:
            all_ok = False
    
    print()
    if all_ok:
        print("✓ ALL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED — review before proceeding")
    
    return all_ok


# ═══════════════════════════════════════════════════════
# RULES REFERENCE (for AI and humans)
# ═══════════════════════════════════════════════════════
RULES = """
RECURRING PITFALLS — CHECK BEFORE EVERY EXPERIMENT:

1. DATA LEAKAGE
   - Never use same sequences for motif discovery AND scoring
   - Split train/test BEFORE calling STREME/FIMO
   - For n < 50, do NOT report AUC (report qualitative only)
   - STREME uses pos+neg → FIMO must scan a DIFFERENT set

2. DATA INTEGRITY
   - Always verify data source: data/benchmark/ (correct) vs output/ (may be stale)
   - Run preflight --strict before starting any experiment
   - Check strand handling: minus-strand TSS must be reverse-complemented

3. GIT HISTORY FIRST
   - Before coding: git log --oneline --grep=<keyword>
   - Before claiming "new finding": check if already documented
   - Check git diff before committing: no stale/debug files

4. NO DUPLICATED CODE
   - Tool runners: one source of truth (not local.py + orchestrator.py copies)
   - Plot generation: one script per plot type, not inline one-offs

5. SAMPLE SIZE
   - n < 50 → qualitative only, no AUC/Spearman/p-values
   - n 50-200 → report metrics with bootstrap CI
   - n > 200 → standard metrics OK

6. PLOT WORKFLOW
   - Generate to /tmp/ first
   - Verify visually and numerically
   - Only then copy to output/plots/<tool>/
   - Delete stale/erroneous plots immediately

7. BEFORE COMMIT
   - git diff --stat → review all changed files
   - No /tmp/ files, no .pyc, no debug output
   - All plots regenerated from current code
   - Commit message describes WHAT and WHY
"""

if __name__ == "__main__":
    strict = "--strict" in sys.argv
    if "--rules" in sys.argv:
        print(RULES)
    else:
        ok = run_all(strict=strict)
        sys.exit(0 if ok else 1)
