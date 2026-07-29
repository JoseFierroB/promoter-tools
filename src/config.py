"""Central configuration with environment-variable overrides and Codon fallbacks.

Usage:
    from src.config import Config
    cfg = Config()
    print(cfg.pos_fasta)  # → data/benchmark/positives_81bp.fasta
    print(cfg.combined_fasta)  # → auto-generated combined pos+neg

Override any path via env var:
    PROMOTECH_DIR=/custom/path pixi run python ...
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Lazy-loaded config with env var overrides and Codon NFS fallbacks."""

    # ── Core paths ──
    @property
    def root(self) -> Path:
        return ROOT

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def output_dir(self) -> Path:
        p = os.environ.get("PROMOTER_OUTPUT_DIR", str(ROOT / "output"))
        return Path(p)

    # ── Dataset paths ──
    @property
    def pos_fasta(self) -> Path:
        return Path(os.environ.get("POS_FASTA", str(ROOT / "data/benchmark/positives_81bp.fasta")))

    @property
    def neg_fasta(self) -> Path:
        return Path(os.environ.get("NEG_FASTA", str(ROOT / "data/benchmark/negatives_81bp.fasta")))

    @property
    def combined_fasta(self) -> Path:
        """Auto-generate combined pos+neg if not exists."""
        combined = self.output_dir / "predictions" / "combined_81bp.fa"
        combined.parent.mkdir(parents=True, exist_ok=True)
        if not combined.exists():
            with open(self.pos_fasta) as f:
                pos_content = f.read()
            with open(self.neg_fasta) as f:
                neg_content = f.read()
            combined.write_text(pos_content.strip() + "\n" + neg_content.strip() + "\n")
        return combined

    # ── Tool directories ──
    @property
    def promotech_dir(self) -> Path:
        return Path(os.environ.get("PROMOTECH_DIR", str(ROOT / "tools/Promotech")))

    @property
    def promoters_dir(self) -> Path:
        return Path(os.environ.get("PROMOTERS_DIR", str(ROOT / "tools/Promoters")))

    @property
    def ipromp_dir(self) -> Path:
        return Path(os.environ.get("IPROMP_DIR", str(ROOT / "tools/iPro-MP")))

    @property
    def mldspp_dir(self) -> Path:
        return Path(os.environ.get("MLDSPP_DIR", str(ROOT / "tools/MLDSPP-Promoter-prediction")))

    # ── Model paths (Codon NFS → local fallback) ──
    @property
    def ipromp_model_dir(self) -> Path:
        p = os.environ.get("IPROMP_MODEL_DIR", "")
        if p:
            return Path(p)
        for candidate in [
            "/nfs/research/jlees/fierro/models/07-final",
            "/hps/nobackup/jlees/fierro/models/07-final",
            "/hps/software/users/jlees/fierro/promoter-tools/models",
            str(ROOT / "tools/iPro-MP/07-final"),
        ]:
            if Path(candidate).exists():
                return Path(candidate)
        return ROOT / "tools/iPro-MP/07-final"

    @property
    def dnabert_dir(self) -> Path:
        p = os.environ.get("DNABERT_DIR", "")
        if p:
            return Path(p)
        for candidate in [
            "/nfs/research/jlees/fierro/DNABERT-6",
            "/hps/nobackup/jlees/fierro/DNABERT-6",
            "/hps/software/users/jlees/fierro/promoter-tools/DNABERT-6",
            str(ROOT / "tools/iPro-MP/DNABERT-6"),
        ]:
            if Path(candidate).exists():
                return Path(candidate)
        return ROOT / "tools/iPro-MP/DNABERT-6"

    @property
    def promotech_models_dir(self) -> Path:
        p = os.environ.get("PROMOTECH_MODELS_DIR", "")
        if p:
            return Path(p)
        for candidate in [
            "/nfs/research/jlees/fierro/models/promotech",
            str(ROOT / "tools/Promotech/models"),
        ]:
            if Path(candidate).exists():
                return Path(candidate)
        return ROOT / "tools/Promotech/models"

    # ── Temp / scratch ──
    @property
    def temp_dir(self) -> Path:
        p = os.environ.get("TMPDIR", "")
        if p and Path(p).parent.exists():
            return Path(p)
        for candidate in [
            "/hps/nobackup/jlees/fierro/tmp",
            "/nfs/research/jlees/fierro/tmp",
            str(ROOT / "output" / "tmp"),
        ]:
            parent = Path(candidate).parent
            if parent.exists() and os.access(str(parent), os.W_OK):
                return Path(candidate)
        return ROOT / "output" / "tmp"

    # ── Pixi home (for Slurm) ──
    @property
    def pixi_home(self) -> Path:
        p = os.environ.get("PIXI_HOME", os.path.expanduser("~/.pixi"))
        return Path(p)

    # ── Convenience ──
    @property
    def n_positives(self) -> int:
        return 988

    @property
    def n_negatives(self) -> int:
        return 1000

    @property
    def n_total(self) -> int:
        return self.n_positives + self.n_negatives


# Singleton
config = Config()
