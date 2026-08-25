"""Shared conservation class assignment for analysis scripts."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def build_conservation_classes(metadata_tsv: Path) -> pd.DataFrame:
    """Assign conserved/nonconserved/intragenic class to each D39V positive."""
    meta = pd.read_csv(metadata_tsv, sep="\t")
    cls_tab = pd.read_csv(ROOT / "output/tables/tss_position_classification.tsv", sep="\t")
    cls_tab = cls_tab[cls_tab["strain"] == "D39V"].set_index("tss_id")
    val = pd.read_csv(ROOT / "output/tables/conserved_igrs_tss_validation.tsv", sep="\t")
    hit_igrs = set(val["query_d39v"].astype(str))

    classes = []
    for _, r in meta.iterrows():
        c = cls_tab.loc[r["Sequence_ID"], "classification"]
        if c in ("CDS_deep", "CDS_near_start"):
            classes.append("intragenic")
        elif c.startswith("IGR"):
            gid = cls_tab.loc[r["Sequence_ID"], "igr_id"]
            classes.append("conserved" if str(gid) in hit_igrs else "nonconserved")
        else:
            classes.append("other")
    meta["class_cons"] = classes
    return meta
