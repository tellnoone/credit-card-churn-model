"""Tests for the data layer: the feature table and the splits.

These need the DuckDB build, so they skip with a clear message if it is absent
rather than failing confusingly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DUCKDB_PATH, ID_COL, RANDOM_SEED, TARGET
from src.data import load_features, make_splits, split_summary
from src.features import feature_names

pytestmark = pytest.mark.skipif(
    not DUCKDB_PATH.exists(),
    reason="DuckDB not built. Run `python run_all.py` (or `python -m src.data`) first.",
)

EXPECTED_ROWS = 10_127
EXPECTED_CHURN_RATE = 0.1607


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_features()


def test_row_count_survives_the_pipeline(df: pd.DataFrame) -> None:
    """Raw -> staging -> features must be 1:1. A drop means a silent bug."""
    assert len(df) == EXPECTED_ROWS


def test_customer_id_is_a_key(df: pd.DataFrame) -> None:
    assert df[ID_COL].is_unique
    assert df[ID_COL].notna().all()


def test_target_is_binary_and_the_base_rate_is_right(df: pd.DataFrame) -> None:
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df[TARGET].notna().all()
    assert df[TARGET].mean() == pytest.approx(EXPECTED_CHURN_RATE, abs=5e-4)


def test_no_leaked_columns_reached_the_feature_table(df: pd.DataFrame) -> None:
    """The single most important test in the file."""
    assert not [c for c in df.columns if "naive_bayes" in c.lower()]


def test_every_declared_feature_exists_as_a_column(df: pd.DataFrame) -> None:
    """Catches a feature renamed in SQL but not in features.py, or vice versa."""
    missing = [n for n in feature_names(strict=False) if n not in df.columns]
    assert not missing, f"declared but absent from the feature table: {missing}"


def test_unknown_is_preserved_not_imputed(df: pd.DataFrame) -> None:
    """Plan section 2: 'Unknown' is kept as a real level."""
    assert (df["education_level"] == "Unknown").sum() == 1_519
    assert (df["income_category"] == "Unknown").sum() == 1_112
    assert (df["marital_status"] == "Unknown").sum() == 749


def test_unknown_flags_agree_with_the_string_values(df: pd.DataFrame) -> None:
    """The boolean flag and the category must never disagree."""
    for col, flag in [("education_level", "education_is_unknown"),
                      ("income_category", "income_is_unknown"),
                      ("marital_status", "marital_is_unknown")]:
        assert (df[flag].astype(bool) == (df[col] == "Unknown")).all(), col


def test_ordinal_encoding_is_null_exactly_where_unknown(df: pd.DataFrame) -> None:
    assert (df["education_level_ord"].isna() == (df["education_level"] == "Unknown")).all()
    assert (df["income_category_ord"].isna() == (df["income_category"] == "Unknown")).all()


def test_card_category_ordering_is_monotonic(df: pd.DataFrame) -> None:
    """Blue < Silver < Gold < Platinum, as a business fact."""
    mapping = (df[["card_category", "card_category_ord"]]
               .drop_duplicates().set_index("card_category")["card_category_ord"].to_dict())
    assert mapping["Blue"] < mapping["Silver"] < mapping["Gold"] < mapping["Platinum"]


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def test_splits_partition_the_data_exactly(df: pd.DataFrame) -> None:
    splits = make_splits(df)
    total = sum(len(p) for p in splits.values())
    assert total == len(df), "splits do not account for every customer"


def test_splits_do_not_overlap(df: pd.DataFrame) -> None:
    """A customer in two splits would leak the test set into training."""
    splits = make_splits(df)
    ids = {k: set(v[ID_COL]) for k, v in splits.items()}
    assert ids["train"] & ids["val"] == set()
    assert ids["train"] & ids["test"] == set()
    assert ids["val"] & ids["test"] == set()


def test_stratification_holds_in_every_split(df: pd.DataFrame) -> None:
    for _, row in split_summary(make_splits(df)).iterrows():
        assert row["churn_rate"] == pytest.approx(EXPECTED_CHURN_RATE, abs=2e-3), row["split"]


def test_splits_are_reproducible(df: pd.DataFrame) -> None:
    """Same seed, same split. Without this, no result is reproducible."""
    a = make_splits(df, seed=RANDOM_SEED)
    b = make_splits(df, seed=RANDOM_SEED)
    for key in ("train", "val", "test"):
        assert list(a[key][ID_COL]) == list(b[key][ID_COL])


def test_a_different_seed_gives_a_different_split(df: pd.DataFrame) -> None:
    """Guards against the seed being ignored, which would fake reproducibility."""
    a = make_splits(df, seed=RANDOM_SEED)
    b = make_splits(df, seed=RANDOM_SEED + 1)
    assert list(a["test"][ID_COL]) != list(b["test"][ID_COL])


def test_split_proportions_are_roughly_60_20_20(df: pd.DataFrame) -> None:
    splits = make_splits(df)
    n = len(df)
    assert len(splits["train"]) / n == pytest.approx(0.60, abs=0.01)
    assert len(splits["val"]) / n == pytest.approx(0.20, abs=0.01)
    assert len(splits["test"]) / n == pytest.approx(0.20, abs=0.01)
