"""Single source of truth for paths, seeds and split sizes.

Anything that would change a result if edited belongs here, so a reader can find
every knob in one file rather than hunting through scripts.
"""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]

# --- Paths -----------------------------------------------------------------
DATA_RAW: Path = ROOT / "data" / "raw" / "BankChurners.csv"
DUCKDB_PATH: Path = ROOT / "data" / "churn.duckdb"
DBT_DIR: Path = ROOT / "dbt"

OUTPUTS: Path = ROOT / "outputs"
FIGURES: Path = OUTPUTS / "figures"
TABLES: Path = OUTPUTS / "tables"
MODELS: Path = OUTPUTS / "models"

CONFIG_DIR: Path = ROOT / "config"
POLICY_CONFIG: Path = CONFIG_DIR / "policy.yaml"

FEATURE_TABLE: str = "main_features.fct_customer_features"

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED: int = 20260911

# --- Split sizes (ANALYSIS_PLAN.md section 4) ------------------------------
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20          # of the full dataset, taken from the remainder

# --- Evaluation ------------------------------------------------------------
TARGET: str = "is_attrited"
ID_COL: str = "customer_id"
N_BOOTSTRAP: int = 2_000
CV_FOLDS: int = 5
CV_REPEATS: int = 5             # 5x5 repeated CV -- see plan section 10 deviation
PRECISION_AT_K: tuple[int, ...] = (100, 250, 500)


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (FIGURES, TABLES, MODELS):
        d.mkdir(parents=True, exist_ok=True)
