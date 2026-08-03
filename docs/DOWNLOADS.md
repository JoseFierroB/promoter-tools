# Model Downloads

Some tools require large model files not tracked in git. Follow these instructions.

---

## Self-Contained Tools (No Downloads)

These work immediately after `git clone` + `pixi install`:

| Tool | What's in git |
|------|---------------|
| MEME (STREME+FIMO) | Motif databases (`tools/meme/motif_databases/`) |
| FIMO + E. coli DB | `ecoli_combined.meme` (165 motifs) |
| MLDSPP XGBoost | Training data (12 species, `tools/MLDSPP-Promoter-prediction/Sample Dataset/`) |
| PromoterLCNN | Keras weights (`tools/Promoters/weights/`, ~121 MB) |

---

## PromoTech (RF-HOT / RF-TETRA)

**Files needed (each ~3.5 GB):**

```
tools/Promotech/models/RF-HOT.model
tools/Promotech/models/RF-TETRA.model
```

**Source:** These are scikit-learn Random Forest models trained with the [PromoTech](https://github.com/A-Londo/promotech) pipeline on E. coli promoters.

**Setup:**
```bash
mkdir -p tools/Promotech/models
# Download RF-HOT.model and RF-TETRA.model into tools/Promotech/models/
```

---

## iPro-MP (sp 12)

**Files needed (~2 GB total):**

### A. DNABERT-6 weights (~340 MB)

```bash
mkdir -p tools/iPro-MP/DNABERT-6
cd tools/iPro-MP/DNABERT-6

# Download pytorch_model.bin from HuggingFace
wget https://huggingface.co/zhihan1996/DNABERT-6/resolve/main/pytorch_model.bin
```

Note: tokenizer config files (`config.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`) are already in git.

### B. iPro-MP trained folds (~1.7 GB)

```bash
mkdir -p tools/iPro-MP/07-final
cd tools/iPro-MP/07-final

# Download the 5 fold model files:
#   12_fold_1.pth
#   12_fold_2.pth
#   12_fold_3.pth
#   12_fold_4.pth
#   12_fold_5.pth
```

**Source:** These are PyTorch model weights from iPro-MP [paper](https://github.com/linxi159/iPro-MP), fine-tuned on H. pylori promoters (species fold 12).

---

## Quick Checklist

After downloads, verify with:

```bash
# PromoTech
ls -lh tools/Promotech/models/RF-HOT.model tools/Promotech/models/RF-TETRA.model

# iPro-MP
ls -lh tools/iPro-MP/07-final/12_fold_1.pth
ls -lh tools/iPro-MP/DNABERT-6/pytorch_model.bin
```
