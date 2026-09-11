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
