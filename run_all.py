"""Reproduce every output in this repository from one command.

    python run_all.py

Runs the SQL layer, then each analysis stage in the order the plan requires.
Stage 4 is the slow one (hyperparameter search); the whole run takes a few
minutes.

The pipeline stops on the first failure. A failing dbt data test or a failing
stage must not leave stale outputs sitting next to fresh ones, because the
README quotes those outputs and verify_report.py checks them.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Prefer the project venv, so `python run_all.py` works from any interpreter.
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PY) if VENV_PY.exists() else sys.executable

STAGES: list[tuple[str, str]] = [
    ("src/data.py", "SQL layer: dbt run + dbt test, then the splits"),
    ("scripts/03_leakage_audit.py", "Stage 3: leakage audit, all vs strict"),
    ("scripts/04_modelling.py", "Stage 4: tuning, calibration, single test pass"),
    ("scripts/05_explainability.py", "Stage 5: SHAP and fairness"),
    ("scripts/06_policy.py", "Stage 6: targeting policy and profit curve"),
    ("scripts/07_experiment_design.py", "Stage 7: experiment design"),
    ("scripts/08_one_pager_chart.py", "Stage 8: the one-pager chart"),
]


def run(script: str, label: str) -> float:
    print(f"\n{'=' * 78}\n>>> {script}\n    {label}\n{'=' * 78}", flush=True)
    started = time.time()
    if script == "src/data.py":
        cmd = [PYTHON, "-m", "src.data"]
    else:
        cmd = [PYTHON, str(ROOT / script)]
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise SystemExit(
            f"\n*** {script} failed (exit {result.returncode}). "
            f"Stopping: later stages read its outputs."
        )
    print(f"\n[ok] {script}  ({elapsed:.1f}s)", flush=True)
    return elapsed


def main() -> int:
    print(f"Python: {PYTHON}")
    total = 0.0
    for script, label in STAGES:
        total += run(script, label)

    print(f"\n{'=' * 78}")
    print(f"All {len(STAGES)} stages completed in {total / 60:.1f} minutes.")
    print("\nNext:")
    print("  python -m pytest        # tests over features, splits, policy, power")
    print("  python verify_report.py # check every number in the README")
    print("\nOutputs in outputs/tables/ and outputs/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
