"""Statistical methods for rigorous benchmark comparison.

Provides bootstrap confidence intervals for AUC,
DeLong test for pairwise AUC comparison, and N≥3 run aggregation.
"""
import numpy as np
from typing import Optional


def bootstrap_auc_ci(y_true: np.ndarray, y_scores: np.ndarray,
                     n_bootstrap: int = 1000, alpha: float = 0.05) -> dict:
    """Compute bootstrap 95% CI for ROC AUC.

    Args:
        y_true: ground truth labels
        y_scores: prediction scores
        n_bootstrap: resampling iterations (default 1000)
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        {"auc": mean, "ci_lower": X, "ci_upper": Y, "n_bootstrap": N}
    """
    from sklearn.metrics import roc_curve, auc

    n = len(y_true)
    aucs = np.zeros(n_bootstrap)
    rng = np.random.RandomState(42)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    for i in range(n_bootstrap):
        idx_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        idx_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(idx)
        fpr, tpr, _ = roc_curve(y_true[idx], y_scores[idx])
        aucs[i] = auc(fpr, tpr)

    ci_lower = np.percentile(aucs, 100 * alpha / 2)
    ci_upper = np.percentile(aucs, 100 * (1 - alpha / 2))

    return {
        "auc": round(float(np.mean(aucs)), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "n_bootstrap": n_bootstrap,
        "auc_median": round(float(np.median(aucs)), 4),
    }


def delong_test(y_true: np.ndarray, scores_a: np.ndarray,
                scores_b: np.ndarray) -> dict:
    """DeLong test for comparing two correlated ROC curves.

    Returns p-value: if p < 0.05, AUC difference is statistically significant.

    Args:
        y_true: ground truth labels
        scores_a: predictions from model A
        scores_b: predictions from model B

    Returns:
        {"p_value": X, "significant": bool, "auc_a": X, "auc_b": X}
    """
    from sklearn.metrics import roc_auc_score

    auc_a = roc_auc_score(y_true, scores_a)
    auc_b = roc_auc_score(y_true, scores_b)

    # DeLong's method: compute covariance of AUC components
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # Structural components for DeLong
    def compute_v10(scores, y_true):
        pos_scores = scores[y_true == 1]
        neg_scores = scores[y_true == 0]
        v10 = np.zeros(len(scores))
        for i, s in enumerate(scores):
            if y_true[i] == 1:
                v10[i] = np.mean(s > neg_scores) + 0.5 * np.mean(s == neg_scores)
            else:
                v10[i] = np.mean(s < pos_scores) + 0.5 * np.mean(s == pos_scores)
        return v10

    v10_a = compute_v10(scores_a, y_true)
    v10_b = compute_v10(scores_b, y_true)

    # Covariance matrix
    s_aa = float(np.var(v10_a[y_true == 1], ddof=1)) if n_pos > 1 else 0.0
    s_bb = float(np.var(v10_b[y_true == 1], ddof=1)) if n_pos > 1 else 0.0
    if n_pos > 1:
        s_ab = np.cov(v10_a[y_true == 1], v10_b[y_true == 1])[0, 1] if len(v10_a[y_true == 1]) > 1 else 0
    else:
        s_ab = 0

    s_aa_n = float(np.var(v10_a[y_true == 0], ddof=1)) if n_neg > 1 else 0.0
    s_bb_n = float(np.var(v10_b[y_true == 0], ddof=1)) if n_neg > 1 else 0.0
    if n_neg > 1:
        s_ab_n = np.cov(v10_a[y_true == 0], v10_b[y_true == 0])[0, 1] if len(v10_a[y_true == 0]) > 1 else 0
    else:
        s_ab_n = 0

    # Variance of difference
    var_diff = (float(s_aa) / n_pos +
                float(s_bb) / n_pos +
                float(s_aa_n) / n_neg +
                float(s_bb_n) / n_neg -
                2 * s_ab / n_pos - 2 * s_ab_n / n_neg)

    if var_diff <= 0:
        p_value = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z)))

    return {
        "auc_a": round(float(auc_a), 4),
        "auc_b": round(float(auc_b), 4),
        "delta_auc": round(float(auc_a - auc_b), 4),
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
        values = [r.get(key) for r in runs if r.get(key) is not None]
        if values:
            result[key] = round(float(np.mean(values)), 4)
            result[f"{key}_std"] = round(float(np.std(values)), 4)
        else:
            result[key] = None
            result[f"{key}_std"] = None

    # Success rate
    successes = [r.get("success", False) for r in runs]
    result["success_rate"] = round(np.mean(successes), 2)

    # Pick best run's other fields
    for key in ["tool", "category", "n_sequences", "peak_ram_mb", "model_size_mb"]:
        for r in runs:
            if key in r and r[key] is not None:
                result[key] = r[key]
                break

    return result
