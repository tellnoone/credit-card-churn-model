"""Build the SQL layer and load the feature table into pandas.

The dbt project owns the cleaning and feature SQL. This module runs it and
hands back a DataFrame, so Python never re-implements a transformation that
already exists in SQL.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Literal

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (DBT_DIR, DUCKDB_PATH, FEATURE_TABLE, RANDOM_SEED, TARGET,
                     TEST_SIZE, VAL_SIZE)
from .features import LEAKED_COLUMNS

Split = Literal["train", "val", "test"]


def _dbt_executable() -> str:
    """Path to dbt inside the project venv, falling back to PATH."""
    candidate = DBT_DIR.parent / ".venv" / "Scripts" / "dbt.exe"
    if candidate.exists():
        return str(candidate)
    candidate = DBT_DIR.parent / ".venv" / "bin" / "dbt"
    if candidate.exists():
        return str(candidate)
    return "dbt"


def build_sql_layer(*, run_tests: bool = True) -> None:
    """Run `dbt run` and (by default) `dbt test`.

    Raises:
        RuntimeError: if dbt reports a failure. A failing data test must stop
            the pipeline -- silently modelling on data that failed validation is
            the exact habit this project is arguing against.
    """
    dbt = _dbt_executable()
    commands = [["run"]] + ([["test"]] if run_tests else [])
    for cmd in commands:
        full = [dbt, *cmd, "--profiles-dir", str(DBT_DIR), "--project-dir", str(DBT_DIR)]
        print(f"$ dbt {' '.join(cmd)}")
        # cwd MUST be the dbt directory: both the DuckDB path in profiles.yml and
        # the CSV path in _sources.yml are relative, and relative paths resolve
        # against the working directory, not the project directory.
        result = subprocess.run(full, capture_output=True, text=True, cwd=DBT_DIR)
        tail = "\n".join(result.stdout.strip().splitlines()[-3:])
        print(tail)
        if result.returncode != 0:
            raise RuntimeError(
                f"dbt {' '.join(cmd)} failed (exit {result.returncode}).\n"
                f"{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
            )


def _normalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable extension dtypes to numpy-backed equivalents.

    DuckDB hands back nullable Int32/boolean columns that carry ``pd.NA``.
    scikit-learn's ColumnTransformer refuses those in a numeric passthrough,
    because ``pd.NA`` does not survive conversion to a numpy array. Nullable
    integers become float64 (so missing stays missing as NaN, which LightGBM
    handles natively) and booleans become int8.

    Done here, at the data boundary, so no downstream module has to know.
    """
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if isinstance(dtype, pd.BooleanDtype):
            out[col] = out[col].astype("float64").fillna(0).astype("int8")
        elif pd.api.types.is_bool_dtype(dtype):
            out[col] = out[col].astype("int8")
        elif isinstance(dtype, pd.api.extensions.ExtensionDtype) and                 pd.api.types.is_numeric_dtype(dtype):
            out[col] = out[col].astype("float64")
    return out


def load_features() -> pd.DataFrame:
    """Read the dbt feature table into pandas.

    Raises:
        RuntimeError: if a known-leaked column somehow appears. This should be
            impossible -- staging selects columns explicitly -- which is exactly
            why it is worth asserting rather than trusting.
    """
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"{DUCKDB_PATH} not found. Run build_sql_layer() first "
            "(or `python run_all.py`)."
        )
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        df = con.execute(f"select * from {FEATURE_TABLE}").fetchdf()
    finally:
        con.close()

    df = _normalise_dtypes(df)

    lowered = {c.lower() for c in df.columns}
    for leaked in LEAKED_COLUMNS:
        if leaked.lower() in lowered:
            raise RuntimeError(f"Leaked column reached the feature table: {leaked}")
    if any("naive_bayes" in c for c in lowered):
        raise RuntimeError("A Naive_Bayes_* column reached the feature table.")

    return df


def make_splits(
    df: pd.DataFrame, *, seed: int = RANDOM_SEED
) -> dict[Split, pd.DataFrame]:
    """Stratified 60/20/20 train/validation/test split.

    Stratified on the target so each split holds ~16.07% churners. Done in two
    steps because sklearn splits in two at a time: first carve off the test set,
    then split the remainder into train and validation.
    """
    y = df[TARGET]
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=seed
    )
    # VAL_SIZE is a share of the FULL dataset, so rescale it to a share of the
    # remainder: 0.20 of the whole is 0.25 of the 80% that is left.
    val_share_of_remainder = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_share_of_remainder,
        stratify=train_val[TARGET],
        random_state=seed,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_summary(splits: dict[Split, pd.DataFrame]) -> pd.DataFrame:
    """Row counts and churn rate per split, for the record."""
    rows = []
    for name, part in splits.items():
        rows.append({
            "split": name,
            "n": len(part),
            "churned": int(part[TARGET].sum()),
            "churn_rate": float(part[TARGET].mean()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    build_sql_layer()
    frame = load_features()
    print(f"\nloaded {len(frame):,} rows x {frame.shape[1]} columns")
    print(split_summary(make_splits(frame)).to_string(index=False))
    sys.exit(0)
