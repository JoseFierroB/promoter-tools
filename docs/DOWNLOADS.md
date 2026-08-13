# Model Downloads

Some tools require large model files not tracked in git. Follow these instructions.

---

## Self-Contained Tools (No Downloads)

These work immediately after `git clone` + `pixi install`:

| Tool | What's in git |
|------|---------------|
| MEME (STREME+FIMO) | Motif databases (`tools/meme/motif_databases/`) |
| FIMO + Prokaryote DB | `unified_prokaryote.meme` (838 motifs) |
| MLDSPP XGBoost (0%) | Training data (12 species, `tools/MLDSPP-Promoter-prediction/Sample Dataset/`) |
| MLDSPP XGBoost (75%) | Split indices (`data/benchmark/mldspp_75_split_*.npz`) |
| PromoterLCNN | Keras SavedModel weights (`tools/Promoters/weights/PromoterLCNN/`, ~8 MB) |

---

## PromoTech (RF-HOT)

**Source repo:** https://github.com/BioinformaticsLabAtMUN/Promotech

**Download URL:** http://www.cs.mun.ca/~lourdes/public/PromoTech_models/

```bash
mkdir -p tools/Promotech/models
cd tools/Promotech/models

# RF-HOT (959 MB compressed, ~3.5 GB uncompressed)
wget http://www.cs.mun.ca/~lourdes/public/PromoTech_models/RF-HOT.zip
unzip RF-HOT.zip && rm RF-HOT.zip
```

Note: RF-TETRA is **not** used in the benchmark (excluded from the analysis) — no need to download it.

---

## iPro-MP (sp 12 — H. pylori)

**Source repo:** https://github.com/Jackie-Suv/iPro-MP

### A. DNABERT-6 weights (~340 MB)

```bash
mkdir -p tools/iPro-MP/DNABERT-6
cd tools/iPro-MP/DNABERT-6

wget https://huggingface.co/zhihan1996/DNABERT-6/resolve/main/pytorch_model.bin
```

Note: tokenizer config files (`config.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`) are already in git.

### B. iPro-MP trained folds (species 12 = Helicobacter pylori)

**Download URL:** https://doi.org/10.5281/zenodo.15180138

The Zenodo record contains a single `model.zip` (~38 GB) with fine-tuned models for all 23 species. Species ID 12 corresponds to *Helicobacter pylori* strain 26695 (the one used in this benchmark).

```bash
mkdir -p tools/iPro-MP/07-final
cd tools/iPro-MP/07-final

# Download the full archive (~38 GB) — or copy only the 12_fold_*.pth files
# from the HPC cluster:
#   /nfs/research/jlees/fierro/models/07-final/12_fold_{1..5}.pth
```

The runner expects 5 fold files: `12_fold_1.pth` … `12_fold_5.pth` (~343 MB each).

> **Environment note**: `pixi install` in `tools/iPro-MP/` installs `torch`
> via pip (PyPI wheel bundles CUDA, ~2 GB). The conda `pytorch` channel
> stops at 2.5.1, which lacks Blackwell (sm_120) support.

---

## PromoterLCNN

**Weights are tracked in git** (`tools/Promoters/weights/PromoterLCNN/`, ~8 MB — SavedModel for the IsPromoter_fold_5 classifier used by the runner).

**Source repo:** https://github.com/occasumlux/Promoters

**Note:** The original repo also distributes iPromoter-BnCNN weights (`*.h5`, ~113 MB) via Google Drive. These are **not** needed for this benchmark and are not tracked in git. If you want them:

- Lightweight (PromoterLCNN only): https://drive.google.com/file/d/1D1XOIAUDMv04sZUIvgdfBAgL75lW8AgW/view
- Full (LCNN + iPromoter-BnCNN + pcPromoter-CNN): https://drive.google.com/file/d/1awsszk6905sVzetdgcQe5kOVTv4n70up/view

---

## Quick Checklist

After downloads, verify with:

```bash
# PromoTech (expected: ~3.5 GB)
ls -lh tools/Promotech/models/RF-HOT.model    # 3.6 GB

# iPro-MP folds (expected: ~343 MB each, ~1.7 GB total)
ls -lh tools/iPro-MP/07-final/12_fold_{1..5}.pth

# iPro-MP DNABERT-6 (expected: ~341 MB)
ls -lh tools/iPro-MP/DNABERT-6/pytorch_model.bin

# PromoterLCNN weights (already in git, ~4 MB SavedModel)
ls -lh tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5/saved_model.pb
```

## Verify Before Running

To confirm all models are present before starting a benchmark run:

```bash
python -c "
from pathlib import Path
ROOT = Path('.')
checks = [
    ('PromoTech HOT', ROOT/'tools/Promotech/models/RF-HOT.model'),
    ('iPro-MP fold 1', ROOT/'tools/iPro-MP/07-final/12_fold_1.pth'),
    ('DNABERT-6', ROOT/'tools/iPro-MP/DNABERT-6/pytorch_model.bin'),
    ('LCNN weights', ROOT/'tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5/saved_model.pb'),
]
missing = []
for name, path in checks:
    if path.exists():
        print(f'  ✅ {name}: {path.stat().st_size/1e9:.1f} GB')
    else:
        print(f'  ❌ {name}: MISSING — {path}')
        missing.append(name)
if missing:
    print(f'\n⚠️  Missing {len(missing)} model(s). Run the commands above.')
else:
    print(f'\n✅ All models present.')
"
```
