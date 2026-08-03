"""Declarative tool definitions for the unified benchmark."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.config import config

ROOT = config.root


@dataclass
class Tool:
    """A benchmark tool with its runtime configuration."""
    name: str
    short_name: str
    category: str                      # "ML", "DL", "Other"
    pixi_env: Path                     # path to pixi.toml
    input_fasta: Path = None           # single combined FASTA (pos+neg)
    inputs: list[Path] = field(default_factory=list)  # deprecated, use input_fasta
    outputs: list[Path] = field(default_factory=list)
    model_paths: list[Path] = field(default_factory=list)
    gpu_capable: bool = False
    n_sequences: int = 1988
    enabled: bool = True
    notes: str = ""

    def __post_init__(self):
        if self.input_fasta is None:
            self.input_fasta = config.combined_fasta

    def model_size_mb(self) -> float:
        total = 0
        for p in self.model_paths:
            pp = Path(p)
            if pp.is_file():
                total += pp.stat().st_size
            elif pp.is_dir():
                for f in pp.rglob("*"):
                    if f.is_file():
                        total += f.stat().st_size
        return round(total / (1024 * 1024), 2)


# ════════════════════════════════════════════════════════════════
# Tool Registry — single combined FASTA for all
# ════════════════════════════════════════════════════════════════

PROMOTER_TOOLS = {
    "promotech_hot": Tool(
        name="PromoTech RF-HOT (PG Max)",
        short_name="promotech_hot",
        category="ML",
        pixi_env=config.promotech_dir / "pixi.toml",
        outputs=[
            ROOT / "output/predictions/promotech/workdir/hot_pg_pos/genome_predictions.csv",
            ROOT / "output/predictions/promotech/workdir/hot_pg_pos/sequences_predictions.csv",
            ROOT / "output/predictions/promotech/workdir/hot_pg_neg/genome_predictions.csv",
            ROOT / "output/predictions/promotech/workdir/hot_pg_neg/sequences_predictions.csv",
        ],
        model_paths=[config.promotech_models_dir / "RF-HOT.model"],
        gpu_capable=False,
    ),
    "promotech_tetra": Tool(
        name="PromoTech RF-TETRA (PG Max)",
        short_name="promotech_tetra",
        category="ML",
        pixi_env=config.promotech_dir / "pixi.toml",
        outputs=[
            ROOT / "output/predictions/promotech/workdir/tetra_pg_pos/genome_predictions.csv",
            ROOT / "output/predictions/promotech/workdir/tetra_pg_pos/sequences_predictions.csv",
            ROOT / "output/predictions/promotech/workdir/tetra_pg_neg/genome_predictions.csv",
            ROOT / "output/predictions/promotech/workdir/tetra_pg_neg/sequences_predictions.csv",
        ],
        model_paths=[config.promotech_models_dir / "RF-TETRA.model"],
        gpu_capable=False,
    ),
    "lcnn": Tool(
        name="PromoterLCNN",
        short_name="lcnn",
        category="DL",
        pixi_env=config.promoters_dir / "pixi.toml",
        outputs=[
            ROOT / "output/predictions/lcnn/lcnn_pos.csv",
            ROOT / "output/predictions/lcnn/lcnn_neg.csv",
        ],
        model_paths=[config.promoters_dir / "weights/PromoterLCNN/IsPromoter_fold_5"],
        gpu_capable=True,
    ),
    "mldspp": Tool(
        name="MLDSPP XGBoost",
        short_name="mldspp",
        category="ML",
        pixi_env=config.mldspp_dir / "pixi.toml",
        outputs=[
            ROOT / "output/predictions/mldspp_pos.csv",
            ROOT / "output/predictions/mldspp_neg.csv",
        ],
        model_paths=[],
        gpu_capable=False,
        notes="Model is tiny (<1MB), trained on-the-fly",
    ),
    "mldspp_75": Tool(
        name="MLDSPP XGBoost (75% spn)",
        short_name="mldspp_75",
        category="ML",
        pixi_env=config.mldspp_dir / "pixi.toml",
        outputs=[
            ROOT / "output/predictions/mldspp_75spn_pos.csv",
            ROOT / "output/predictions/mldspp_75spn_neg.csv",
        ],
        model_paths=[],
        gpu_capable=False,
        notes="75% S. pneumoniae in training — data leakage, for reference only",
    ),
    "ipromp_sp12": Tool(
        name="iPro-MP (H. pylori)",
        short_name="ipromp_sp12",
        category="DL",
        pixi_env=config.ipromp_dir / "pixi.toml",
        outputs=[ROOT / "output/predictions/ipromp/ipromp_12_predictions.csv"],
        model_paths=[config.ipromp_model_dir, config.dnabert_dir],
        gpu_capable=True,
        notes="DNABERT-6 transformer. Heavy on CPU, fast on GPU.",
    ),
}


def _load_toml_tools():
    """Auto-register tools from tools.d/*.toml files. Non-breaking add-on."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return  # No TOML parser available, skip

    tools_dir = Path(__file__).parent / "tools.d"
    if not tools_dir.exists():
        return

    for toml_file in sorted(tools_dir.glob("*.toml")):
        try:
            data = tomllib.loads(toml_file.read_text())
        except Exception:
            continue

        short = data.get("short", toml_file.stem)
        if short in PROMOTER_TOOLS:
            continue  # Don't override Python-defined tools

        pixi_env_raw = Path(data.get("pixi_env", "pixi.toml"))
        if not pixi_env_raw.is_absolute():
            pixi_env_raw = ROOT / pixi_env_raw

        PROMOTER_TOOLS[short] = Tool(
            name=data.get("name", short),
            short_name=short,
            category=data.get("category", "Other"),
            pixi_env=pixi_env_raw,
            model_paths=[Path(p) for p in data.get("model_paths", [])],
            gpu_capable=data.get("gpu_capable", False),
            n_sequences=data.get("n_sequences", 1988),
            notes=data.get("notes", ""),
        )


# Load TOML overrides after Python registry
_load_toml_tools()


def get_enabled_tools():
    """Return all enabled tools, ready to run."""
    return [t for t in PROMOTER_TOOLS.values() if t.enabled]


def enable(tool_names: list[str]):
    """Enable only the specified tools, disable others."""
    for k in PROMOTER_TOOLS:
        PROMOTER_TOOLS[k].enabled = k in tool_names


def disable(tool_names: list[str]):
    """Disable specified tools."""
    for k in tool_names:
        if k in PROMOTER_TOOLS:
            PROMOTER_TOOLS[k].enabled = False
