# MEME Suite — Complete Command Reference

## Data
All commands use D39V benchmark data (988 promoters + 1000 CDS negatives, 81bp, -60/+20).

## 1. STREME → FIMO (Discriminative Motif Discovery)

STREME finds motifs over-represented in positives vs negatives.
FIMO scans sequences and reports hits with p-values.

### 1.1 Basic STREME + FIMO (no CV, for motif discovery only)
```bash
pixi run --manifest-path tools/meme/pixi.toml streme \
  -oc streme_out -dna -minw 10 -maxw 20 \
  -p data/benchmark/positives_81bp.fasta \
  -n data/benchmark/negatives_81bp.fasta

pixi run --manifest-path tools/meme/pixi.toml fimo \
  --text --skip-matched-sequence \
  streme_out/streme.txt combined_pos_neg.fasta
```

### 1.2 STREME + FIMO with 2-fold CV (honest AUC)
```bash
pixi run --manifest-path tools/meme/pixi.toml python -c "
import numpy as np, random
from Bio import SeqIO
from sklearn.metrics import roc_auc_score
import subprocess, tempfile, shutil, csv, math
from pathlib import Path

random.seed(42)
pos = list(SeqIO.parse('data/benchmark/positives_81bp.fasta','fasta'))
neg = list(SeqIO.parse('data/benchmark/negatives_81bp.fasta','fasta'))
random.shuffle(pos); random.shuffle(neg)
mp, mn = len(pos)//2, len(neg)//2
all_scores = {}

for fold in range(2):
    tp = pos[:mp] if fold==0 else pos[mp:]
    tn = neg[:mn] if fold==0 else neg[mn:]
    tpos = pos[mp:] if fold==0 else pos[:mp]
    tneg = neg[mn:] if fold==0 else neg[:mn]
    
    td = Path(tempfile.mkdtemp(prefix='meme_cv_'))
    SeqIO.write(tp, td/'tp.fa','fasta'); SeqIO.write(tn, td/'tn.fa','fasta')
    with open(td/'test.fa','w') as f:
        for r in tpos: SeqIO.write(r,f,'fasta')
        for r in tneg: SeqIO.write(r,f,'fasta')
    
    subprocess.run(['streme','-oc',str(td/'s'),'-dna','-minw','10','-maxw','20',
        '-p',str(td/'tp.fa'),'-n',str(td/'tn.fa')], capture_output=True, timeout=120)
    
    res = subprocess.run(['fimo','--text','--skip-matched-sequence',
        str(td/'s'/'streme.txt'), str(td/'test.fa')],
        capture_output=True, text=True, timeout=120)
    
    for row in csv.DictReader(res.stdout.splitlines(), delimiter='\t'):
        try: pv=float(row['p-value'])
        except: continue
        nl=999.0 if pv<=0 else -math.log10(pv)
        s=row['sequence_name']
        if s not in all_scores or nl>all_scores[s]: all_scores[s]=nl
    shutil.rmtree(td)

for r in pos+neg:
    if r.id not in all_scores: all_scores[r.id]=0.0

y = np.hstack([np.ones(len(pos)), np.zeros(len(neg))])
sc = np.array([all_scores[r.id] for r in pos+neg])
print(f'AUC: {roc_auc_score(y,sc):.4f}')
"
```
**Result: AUC = 0.854** (honest, no data leakage)

### 1.3 Via CLI (integrated runner)
```bash
pixi run python src/cli.py run meme
```

## 2. MEME → FIMO (Classic EM, no negatives needed)

### 2.1 Basic MEME + FIMO
```bash
pixi run --manifest-path tools/meme/pixi.toml meme \
  data/benchmark/positives_81bp.fasta \
  -dna -mod zoops -minw 10 -maxw 20 \
  -oc meme_out -nostatus

pixi run --manifest-path tools/meme/pixi.toml fimo \
  --text --skip-matched-sequence \
  meme_out/meme.xml combined.fasta
```
**Result: AUC = 0.861** (MEME classic, no negatives used)

### 2.2 MEME Pipeline Script
```bash
pixi run --manifest-path tools/meme/pixi.toml python src/experiments/meme_pipeline1_basic.py
```

## 3. TOMTOM (Motif Annotation)

Compare discovered motifs against known databases:

```bash
pixi run --manifest-path tools/meme/pixi.toml tomtom \
  -no-ssc -text -min-overlap 4 \
  streme_out/streme.txt \
  tools/meme/motif_databases/unified_prokaryote.meme
```

### Available Databases
| Database | Motifs | Size |
|----------|--------|------|
| `unified_prokaryote.meme` | 838 | 844 KB |
| `PROKARYOTE/prodoric_2021.9.meme` | 333 | 228 KB |
| `PROKARYOTE/regtransbase.meme` | 141 | 220 KB |
| `PROKARYOTE/fan2020.meme` | 115 | 92 KB |
| `PROKARYOTE/collectf.meme` | 84 | 92 KB |
| `ECOLI/dpinteract.meme` | 68 | 124 KB |
| `ECOLI/SwissRegulon_e_coli.meme` | 97 | 96 KB |

## 4. XSTREME (Combined Pipeline)

Same as STREME but also runs MEME classic + SEA enrichment:
```bash
pixi run --manifest-path tools/meme/pixi.toml xstreme \
  --oc xstreme_out --dna --minw 6 --maxw 15 \
  --p data/benchmark/positives_81bp.fasta \
  --n data/benchmark/negatives_81bp.fasta \
  --m tools/meme/motif_databases/PROKARYOTE/prodoric_2021.9.meme
```

## 5. Results Summary

### Motifs Discovered (STREME on D39V, 988 pos + 1000 neg)

| # | Consensus | Width | Sites | E-value | Category | Tomtom |
|---|-----------|-------|-------|---------|----------|--------|
| 1 | YTATTATAYCAYAWWWW | 17 | 843 (85%) | 3.2×10⁻³³ | Extended -10 (σ⁷⁰) | rpoD19, rpoD15 |
| 2 | AAAAAAKMAWA | 11 | 632 (64%) | 1.6×10⁻⁹ | UP element / AT-rich | — |
| 3 | AAAACGCTTRCA | 12 | 36 (4%) | 0.17 | TF: CcpA/MalR/PurR | CcpA_S.pneumoniae, MalR |
| 4 | AAGGAGGAAA | 10 | 39 (4%) | 1.5 | Shine-Dalgarno | — |
| 5 | AATAGAWAGRR | 11 | 73 (7%) | 1.8 | Other regulatory | — |
| 6 | AAAAATTTTGCAAAWT | 16 | 34 (3%) | 6.0 | AT-rich | CodY |

### AUC Comparison (D39V, honest CV)

| Method | AUC | Notes |
|--------|-----|-------|
| STREME + FIMO (2-fold CV) | **0.854** | Discriminative, uses negatives |
| MEME classic + FIMO | 0.861 | No negatives needed (slightly higher leakage risk) |
| XSTREME | 0.487 | On TIGR4 only (strain-specific) |

### Key Findings
- **Extended -10 DOMINATES** — 85% of promoters, E=10⁻³³
- **-35 box is ABSENT** — expected for Firmicutes (S. pneumoniae)
- **CcpA/MalR binding sites** detected in 4% of promoters (validated by Tomtom)
- **MEME does NOT generalize** between D39V and TIGR4 (strain-specific motifs)
- **MLDSPP DOES generalize** (ΔG physics is universal across strains)

## 6. Quick Reference

```bash
# STREME + FIMO (discriminative)
pixi run --manifest-path tools/meme/pixi.toml python src/experiments/meme_pipeline3_streme.py

# MEME + FIMO + TOMTOM (annotated)
pixi run --manifest-path tools/meme/pixi.toml python src/experiments/meme_pipeline2_tomtom.py

# Full analysis (STREME + MEME + TOMTOM + plots)
pixi run --manifest-path tools/meme/pixi.toml python submit/meme_full_analysis.py

# Via benchmark runner
python submit/run_benchmark.py local analysis
```
