# Model Downloads

Some tools require large model files not tracked in git. Follow these instructions.

---

## Self-Contained Tools (No Downloads)

These work immediately after `git clone` + `pixi install`:

| Tool | What's in git |
|------|---------------|
| MEME (STREME+FIMO) | Motif databases (`tools/meme/motif_databases/`) |
| FIMO + E. coli DB | `ecoli_combined.meme` (165 motifs) |
| FIMO + Prokaryote DB | `unified_prokaryote.meme` (838 motifs) |
| MLDSPP XGBoost (0%) | Training data (12 species, `tools/MLDSPP-Promoter-prediction/Sample Dataset/`) |
| MLDSPP XGBoost (75%) | Split indices (`data/benchmark/mldspp_75_split_*.npz`) |

---

## PromoterLCNN

**Files needed:** TensorFlow 1.x SavedModel (keras_metadata.pb, saved_model.pb, variables/).

**Where they live:** These weights are **not in git** (removed for size). They are available on the HPC cluster at:

```
/hps/software/users/jlees/fierro/promoter-tools/tools/Promoters/weights/
/nfs/research/jlees/fierro/promoter-tools/tools/Promoters/weights/
```

Or re-download from the original [PromoterLCNN](https://github.com/WangLabTHU/PromoterLCNN) repository.

**Setup:**
```bash
mkdir -p tools/Promoters/weights/PromoterLCNN/IsPromoter_fold_5
# Copy the SavedModel files into the directory above
```

---

## PromoTech (RF-HOT / RF-TETRA)

**Files needed (each ~3.5 GB):**

```
tools/Promotech/models/RF-HOT.model
tools/Promotech/models/RF-TETRA.model
```

**Where they live:** These are scikit-learn Random Forest models **trained with our pipeline** on E. coli promoters. They are not publicly downloadable — they live on the HPC cluster:

```
/nfs/research/jlees/fierro/models/promotech/RF-HOT.model
/nfs/research/jlees/fierro/models/promotech/RF-TETRA.model
```

**Setup:**
```bash
mkdir -p tools/Promotech/models
scp <cluster>:/nfs/research/jlees/fierro/models/promotech/RF-HOT.model tools/Promotech/models/
scp <cluster>:/nfs/research/jlees/fierro/models/promotech/RF-TETRA.model tools/Promotech/models/
```

To retrain from scratch, use the [PromoTech](https://github.com/A-Londo/promotech) pipeline.

---

## iPro-MP (sp 12)

**Files needed (~2 GB total):**

### A. DNABERT-6 weights (~340 MB) — downloadable

```bash
mkdir -p tools/iPro-MP/DNABERT-6
cd tools/iPro-MP/DNABERT-6

# Download pytorch_model.bin from HuggingFace
wget https://huggingface.co/zhihan1996/DNABERT-6/resolve/main/pytorch_model.bin
```

Note: tokenizer config files (`config.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`) are already in git.

### B. iPro-MP trained folds (~1.7 GB) — NOT downloadable

These are PyTorch model weights **fine-tuned by us** on H. pylori promoters (species fold 12) using the [iPro-MP](https://github.com/linxi159/iPro-MP) training code. They live on the HPC cluster:

```
/nfs/research/jlees/fierro/models/07-final/12_fold_{1..5}.pth
```

**Setup:**
```bash
mkdir -p tools/iPro-MP/07-final
scp <cluster>:'/nfs/research/jlees/fierro/models/07-final/12_fold_*.pth' tools/iPro-MP/07-final/
```

To retrain from scratch, use `tools/iPro-MP/iPro-MP_train.py` (in git).

---

## Quick Checklist

After downloads, verify with:

```bash
# PromoTech (expected: ~3.5 GB each)
ls -lh tools/Promotech/models/RF-HOT.model    # 3.6 GB
ls -lh tools/Promotech/models/RF-TETRA.model  # 3.5 GB

# iPro-MP folds (expected: ~343 MB each, ~1.7 GB total)
ls -lh tools/iPro-MP/07-final/12_fold_{1..5}.pth

# iPro-MP DNABERT-6 (expected: ~341 MB)
ls -lh tools/iPro-MP/DNABERT-6/pytorch_model.bin

# PromoterLCNN weights (expected: ~8 MB SavedModel + ~4 MB variables)
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
    ('PromoTech TETRA', ROOT/'tools/Promotech/models/RF-TETRA.model'),
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
