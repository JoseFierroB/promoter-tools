"""Statistical methods for rigorous benchmark comparison.

Provides bootstrap confidence intervals for AUC,
DeLong test for pairwise AUC comparison, and N≥3 run aggregation.
"""
import numpy as np
from typing import Optional


def bootstrap_auc_ci(y_true: np.ndarray, y_scores: np.ndarray,
                     n_bootstrap: int = 1000, alpha: float = 0.05) -> dict:
    """Compute bootstrap 95% CI for ROC AUC.

    The reported "auc" is the plug-in estimate on the full sample;
    the bootstrap resampling is used only for the interval.

    Args:
        y_true: ground truth labels
        y_scores: prediction scores
        n_bootstrap: resampling iterations (default 1000)
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        {"auc": point estimate, "ci_lower": X, "ci_upper": Y, ...}
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    aucs = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)
    for i in range(n_bootstrap):
        idx_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        idx_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
        aucs[i] = auc(fpr, tpr)

    ci_lower = np.percentile(aucs, 100 * alpha / 2)
    ci_upper = np.percentile(aucs, 100 * (1 - alpha / 2))

    return {
        "auc": round(float(roc_auc_score(y_true, y_scores)), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "n_bootstrap": n_bootstrap,
        "auc_median": round(float(np.median(aucs)), 4),
    }


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x (ties get the average rank), O(n log n)."""
    order = np.argsort(x)
    ranks_sorted = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        xs = x[order]
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks_sorted
    return out


def delong_test(y_true: np.ndarray, scores_a: np.ndarray,
                scores_b: np.ndarray) -> dict:
    """Fast DeLong test for comparing two correlated ROC curves.

    Vectorized via midranks/searchsorted (Sun & Xu 2014): O(n log n).

    Returns:
        {"p_value", "significant", "auc_a", "auc_b", "delta_auc", "delta_auc_se"}
    """
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, scores_a)
    auc_b = roc_auc_score(y_true, scores_b)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    def structural_components(scores):
        """V10 per subject: positives vs sorted negatives and vice versa."""
        v10 = np.zeros(len(scores))
        pos_mask = y_true == 1
        neg_mask = ~pos_mask
        neg_sorted = np.sort(scores[neg_mask])
        pos_sorted = np.sort(scores[pos_mask])

        s_pos = scores[pos_mask]
        left = np.searchsorted(neg_sorted, s_pos, side="left")
        right = np.searchsorted(neg_sorted, s_pos, side="right")
        # mean(s > neg) + 0.5 * mean(s == neg)
        v10[pos_mask] = (right + left) / 2.0 / n_neg

        s_neg = scores[neg_mask]
        left = np.searchsorted(pos_sorted, s_neg, side="left")
        right = np.searchsorted(pos_sorted, s_neg, side="right")
        # mean(s < pos) + 0.5 * mean(s == pos)
        v10[neg_mask] = ((n_pos - right) + 0.5 * (right - left)) / n_pos
        return v10

    v10_a = structural_components(scores_a)
    v10_b = structural_components(scores_b)

    s_aa = float(np.var(v10_a[y_true == 1], ddof=1)) if n_pos > 1 else 0.0
    s_bb = float(np.var(v10_b[y_true == 1], ddof=1)) if n_pos > 1 else 0.0
    s_ab = float(np.cov(v10_a[y_true == 1], v10_b[y_true == 1])[0, 1]) if n_pos > 1 else 0.0

    s_aa_n = float(np.var(v10_a[y_true == 0], ddof=1)) if n_neg > 1 else 0.0
    s_bb_n = float(np.var(v10_b[y_true == 0], ddof=1)) if n_neg > 1 else 0.0
    s_ab_n = float(np.cov(v10_a[y_true == 0], v10_b[y_true == 0])[0, 1]) if n_neg > 1 else 0.0

    var_diff = (s_aa / n_pos + s_bb / n_pos +
                s_aa_n / n_neg + s_bb_n / n_neg -
                2 * s_ab / n_pos - 2 * s_ab_n / n_neg)

    if var_diff <= 0:
        p_value = 1.0
        se = np.nan
    else:
        se = np.sqrt(var_diff)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(auc_a - auc_b) / se))

    return {
        "auc_a": round(float(auc_a), 4),
        "auc_b": round(float(auc_b), 4),
        "delta_auc": round(float(auc_a - auc_b), 4),
        "delta_auc_se": round(float(se), 4) if se == se else None,
        "p_value": round(float(p_value), 4),
        "significant": bool(p_value < 0.05),
    }


def aggregate_runs(runs: list[dict], metric_keys: list[str] = None) -> dict:
    """Aggregate N independent runs into mean ± SD.

    Args:
        runs: list of dicts, each from one run (e.g., LocalRunner output)
        metric_keys: which keys to aggregate (default: time_s, throughput_seq_s)

    Returns:
        dict with mean ± SD for each metric
    """
    if not runs:
        return {}

    if metric_keys is None:
        metric_keys = ["time_s", "throughput_seq_s"]

    result = {"n_runs": len(runs)}
    for key in metric_keys:
        values = [r.get(key) for r in runs if r.get(key) is not None and r.get(key) == r.get(key)]
        if values:
            result[key] = round(float(np.mean(values)), 4)
            result[f"{key}_std"] = round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4)
        else:
            result[key] = None
            result[f"{key}_std"] = None

    # Success rate
    successes = [r.get("success", False) for r in runs]
    result["success_rate"] = round(np.mean(successes), 2)
    result["success"] = result["success_rate"] == 1.0

    # Pick best run's other fields
    for key in ["tool", "category", "n_sequences", "peak_ram_mb", "peak_vram_mb",
                "model_size_mb", "mean_cpu_pct", "gpu_util_pct", "wall_seconds"]:
        for r in runs:
            if key in r and r[key] is not None:
                result[key] = r[key]
                break

    return result
