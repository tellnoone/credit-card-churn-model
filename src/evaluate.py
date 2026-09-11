"""Metrics, bootstrap intervals and cross-validation helpers.

The metric set follows ANALYSIS_PLAN.md section 3:

* **PR-AUC (average precision)** - primary. Always reported next to the base
  rate, which is what a random model scores, so the number cannot flatter itself.
* **ROC-AUC** - reported because stakeholders expect it, not because it decides.
* **Brier score** - calibration. The profit model multiplies by P(churn) as a
  probability, so being right *on average* matters as much as ranking.
* **precision@k / lift@k** - what a fixed retention budget actually buys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold

from .config import (CV_FOLDS, CV_REPEATS, N_BOOTSTRAP, PRECISION_AT_K,
                     RANDOM_SEED)


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Share of the top-k highest-scoring customers who actually churned.

    This is what a campaign with budget for k contacts experiences.
    """
    k = min(k, len(y_score))
    if k == 0:
        return float("nan")
    top = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[top]))


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """precision@k divided by the base rate.

    Lift of 3.0 means "three times as many churners as picking at random".
    """
    base = float(np.mean(y_true))
    if base == 0:
        return float("nan")
    return precision_at_k(y_true, y_score, k) / base


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    ks: Iterable[int] = PRECISION_AT_K,
) -> dict[str, float]:
    """All headline metrics for one set of predictions."""
    out: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
        out[f"lift_at_{k}"] = lift_at_k(y_true, y_score, k)
    return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    ks: Iterable[int] = PRECISION_AT_K,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap 95% CIs for every metric.

    Resamples customers with replacement. A resample that happens to contain no
    churners is skipped rather than scored, since the metrics are undefined
    there; with ~325 positives that is vanishingly rare, but it would crash.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    collected: dict[str, list[float]] = {}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() < 2 or yt.sum() == len(yt):
            continue
        for key, value in compute_metrics(yt, ys, ks=ks).items():
            collected.setdefault(key, []).append(value)

    summary: dict[str, dict[str, float]] = {}
    point = compute_metrics(y_true, y_score, ks=ks)
    for key, values in collected.items():
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "point": float(point[key]),
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
            "se": float(np.std(arr, ddof=1)),
        }
    return summary


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """Per-fold metrics for one model on one feature set."""

    label: str
    model_kind: str
    strict: bool
    fold_metrics: list[dict[str, float]] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_metrics)

    def mean(self, metric: str) -> float:
        return float(self.frame()[metric].mean())

    def std(self, metric: str) -> float:
        return float(self.frame()[metric].std(ddof=1))

    def summary(self, metrics: Iterable[str]) -> dict[str, float]:
        out: dict[str, float] = {"label": self.label,          # type: ignore[dict-item]
                                 "model_kind": self.model_kind,  # type: ignore[dict-item]
                                 "strict": self.strict}          # type: ignore[dict-item]
        for m in metrics:
            out[f"{m}_mean"] = self.mean(m)
            out[f"{m}_std"] = self.std(m)
        return out


def repeated_cv(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    label: str,
    model_kind: str,
    strict: bool,
    n_splits: int = CV_FOLDS,
    n_repeats: int = CV_REPEATS,
    seed: int = RANDOM_SEED,
    ks: Iterable[int] = PRECISION_AT_K,
) -> CVResult:
    """Repeated stratified k-fold CV, scoring every fold on the full metric set.

    Averaging over n_splits x n_repeats fits is what makes a *comparison*
    between models stable. Per the plan's section 10 deviation, the test set is
    not used for this -- at ~325 test positives it cannot separate models whose
    PR-AUC differs by less than roughly 0.05.

    The whole pipeline is cloned and refitted inside each fold, so imputation,
    scaling and encoding are all learned from training data only.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    result = CVResult(label=label, model_kind=model_kind, strict=strict)

    for train_idx, valid_idx in cv.split(X, y):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y[train_idx])
        scores = model.predict_proba(X.iloc[valid_idx])[:, 1]
        result.fold_metrics.append(compute_metrics(y[valid_idx], scores, ks=ks))

    return result


def paired_fold_difference(a: CVResult, b: CVResult, metric: str) -> dict[str, float]:
    """Compare two models fold by fold on identical splits.

    Both CVResults must come from the same seed and fold structure, which
    `repeated_cv` guarantees. Pairing removes fold-to-fold difficulty variation,
    so the comparison is far tighter than comparing two independent means -- the
    same reason a paired t-test beats an unpaired one.
    """
    x = a.frame()[metric].to_numpy()
    y = b.frame()[metric].to_numpy()
    if len(x) != len(y):
        raise ValueError("CV results have different fold counts; cannot pair")
    diff = x - y
    return {
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        # Percentile interval over folds. Descriptive, not a significance test:
        # repeated CV folds overlap heavily, so they are not independent and a
        # p-value computed from them would be anti-conservative.
        "diff_lo": float(np.percentile(diff, 2.5)),
        "diff_hi": float(np.percentile(diff, 97.5)),
        "a_wins_share": float((diff > 0).mean()),
        "n_folds": int(len(diff)),
    }
