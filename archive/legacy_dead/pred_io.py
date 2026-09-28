"""Shared prediction-file I/O for analysis scripts."""
import pandas as pd
from pathlib import Path

_PATS = {
    "lcnn": ("lcnn/lcnn_pos.csv", "lcnn/lcnn_neg.csv"),
    "ipromp": ("ipromp/ipromp_12_predictions.csv",),
    "mldspp": ("mldspp_pos.csv", "mldspp_neg.csv"),
    "mldspp_75": ("mldspp_75spn_pos.csv", "mldspp_75spn_neg.csv"),
    "promotech": ("promotech/workdir/hot_pg_pos/sequences_predictions.csv",
                  "promotech/workdir/hot_pg_neg/sequences_predictions.csv"),
    "fimo": ("fimo_prok_pos.csv", "fimo_prok_neg.csv"),
    "meme": ("meme_pos.csv", "meme_neg.csv"),
}


def load_preds(root, key, npos=None):
    """Load (pos, neg) prediction arrays for a tool key.

    root: predictions directory; key: one of _PATS; npos: exact number of
    positives (required for ipromp, where pos/neg share one file).
    """
    p = Path(root)
    if key == "ipromp":
        df = pd.read_csv(p / _PATS[key][0], sep="\t")
        col = "PRED" if "PRED" in df.columns else "Probability"
        if npos is None:
            npos = len(df) // 2
        return df[col].values[:npos], df[col].values[npos:]
    a, b = _PATS[key]
    return pd.read_csv(p / a, sep="\t")["PRED"].values, pd.read_csv(p / b, sep="\t")["PRED"].values