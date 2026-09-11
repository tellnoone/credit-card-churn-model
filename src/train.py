"""Model construction: preprocessing pipelines and estimators.

Two preprocessing routes, because linear models and trees want different things:

* **linear** - median-impute, standardise, one-hot. Logistic regression needs
  finite, comparably-scaled inputs, and its coefficients are only interpretable
  if the inputs share a scale.
* **tree** - one-hot only. LightGBM handles NaN natively and is invariant to
  monotone rescaling, so imputing would invent values and scaling would buy
  nothing. Letting the tree learn its own split for "missing" is strictly more
  informative than replacing it with a median.

Everything is wrapped in a Pipeline so that imputation and scaling are fitted
**inside** each cross-validation fold. Fitting a scaler on the full dataset
before splitting is a quiet, common form of leakage, and a Pipeline makes it
structurally impossible.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_SEED, TARGET
from .features import categorical_names, feature_names, numeric_names

ModelKind = Literal["logistic", "lightgbm"]
Route = Literal["linear", "tree"]

ROUTE_FOR: dict[ModelKind, Route] = {"logistic": "linear", "lightgbm": "tree"}


def make_preprocessor(*, strict: bool, route: Route) -> ColumnTransformer:
    """Build the column transformer for a feature set and model family."""
    numeric = numeric_names(strict=strict)
    categorical = categorical_names(strict=strict)

    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           drop=None, min_frequency=None)

    if route == "linear":
        numeric_pipe: Any = Pipeline([
            # Median, not mean: several features are heavily right-skewed
            # (credit_limit, transaction amounts), where the mean is dragged by
            # the tail and is not a typical value.
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        # LightGBM: pass numerics through untouched, NaN included.
        numeric_pipe = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", onehot, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(kind: ModelKind, **overrides: Any) -> Any:
    """Build an estimator with sensible, stated defaults."""
    if kind == "logistic":
        params: dict[str, Any] = dict(
            # L2 is the default; naming `penalty` explicitly is deprecated in
            # scikit-learn 1.8+, so the default is relied on instead.
            C=1.0,
            solver="lbfgs",
            max_iter=2_000,
            # Not class_weight="balanced": that distorts predicted probabilities
            # away from the true base rate, and the profit calculation in stage 6
            # needs calibrated probabilities rather than balanced ones. Imbalance
            # is handled by the choice of metric and threshold, not by reweighting.
            class_weight=None,
            random_state=RANDOM_SEED,
        )
        params.update(overrides)
        return LogisticRegression(**params)

    if kind == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=1,          # determinism over speed; the data is small
            verbose=-1,
        )
        params.update(overrides)
        return LGBMClassifier(**params)

    raise ValueError(f"unknown model kind: {kind}")


def make_pipeline(kind: ModelKind, *, strict: bool, **overrides: Any) -> Pipeline:
    """Preprocessor + estimator as one fittable object."""
    return Pipeline([
        ("prep", make_preprocessor(strict=strict, route=ROUTE_FOR[kind])),
        ("model", make_estimator(kind, **overrides)),
    ])


def xy(df: pd.DataFrame, *, strict: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a frame into the feature matrix for a variant, and the target."""
    cols = feature_names(strict=strict)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"features missing from frame: {missing}")
    return df[cols].copy(), df[TARGET].to_numpy()


def feature_names_out(pipeline: Pipeline) -> list[str]:
    """Column names after preprocessing, for coefficient and SHAP tables."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

SEARCH_SPACES: dict[ModelKind, dict[str, Any]] = {
    # Only the regularisation strength is searched. The solver and penalty are
    # fixed because changing them changes what the coefficients MEAN, and the
    # logistic model earns its place here by being readable.
    "logistic": {
        "model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
    # Ranges chosen for a small, imbalanced tabular problem: shallow trees and
    # strong minimum-leaf sizes, because ~975 positives in a training fold will
    # let a deep tree memorise individuals.
    "lightgbm": {
        "model__n_estimators": [200, 300, 400, 600, 800],
        "model__learning_rate": [0.01, 0.02, 0.05, 0.1],
        "model__num_leaves": [7, 15, 31, 63],
        "model__max_depth": [3, 4, 5, 6, -1],
        "model__min_child_samples": [10, 20, 40, 80],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_lambda": [0.0, 1.0, 5.0, 20.0],
    },
}


def tune_model(
    kind: ModelKind,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    strict: bool,
    n_iter: int = 25,
    cv_splits: int = 5,
    seed: int = RANDOM_SEED,
    scoring: str = "average_precision",
) -> tuple[Pipeline, dict[str, Any], float]:
    """Randomised search on the TRAINING set only.

    Scored by average precision (PR-AUC), the plan's primary metric, so the
    search optimises the thing the project says it cares about rather than
    accuracy.

    Deliberately run on train alone, not train+validation: the validation set
    has to stay unused by any fitting decision so it can calibrate the final
    model and choose an operating threshold.

    Returns:
        (fitted best pipeline, best params, best CV score)
    """
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    space = SEARCH_SPACES[kind]
    base = make_pipeline(kind, strict=strict)

    # An exhaustive grid is cheaper than sampling when the space is tiny.
    n_combos = 1
    for values in space.values():
        n_combos *= len(values)
    n_iter = min(n_iter, n_combos)

    search = RandomizedSearchCV(
        base,
        param_distributions=space,
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def calibrate(estimator: Any, X_val: pd.DataFrame, y_val: np.ndarray,
              *, method: str = "isotonic") -> Any:
    """Wrap a fitted estimator in a calibrator fitted on held-out data.

    The estimator is frozen first, so calibration learns only the mapping from
    its scores to probabilities and cannot refit the underlying model. Fitting
    the calibrator on data the model was trained on would produce a mapping that
    looks perfect in training and is wrong everywhere else.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    calibrated.fit(X_val, y_val)
    return calibrated


class HeuristicRanker:
    """Rank customers by a single column, no fitting involved.

    The bar a model has to clear to justify its existence: this is what a
    competent analyst produces in an afternoon with a SQL query and no model.
    """

    def __init__(self, column: str, *, ascending: bool = True) -> None:
        self.column = column
        self.ascending = ascending
        self._lo: float = 0.0
        self._hi: float = 1.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "HeuristicRanker":
        values = X[self.column].astype(float)
        self._lo, self._hi = float(values.min()), float(values.max())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.column].astype(float).to_numpy()
        span = self._hi - self._lo
        scaled = (values - self._lo) / span if span else np.zeros_like(values)
        # `ascending=True` means a LOW value indicates high churn risk.
        score = 1.0 - scaled if self.ascending else scaled
        score = np.clip(np.nan_to_num(score, nan=0.5), 0.0, 1.0)
        return np.column_stack([1.0 - score, score])
