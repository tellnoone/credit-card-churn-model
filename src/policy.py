"""Turning churn scores into a retention targeting decision.

The bridge from model to business, per ANALYSIS_PLAN.md section 7:

    expected_value = P(churn) x P(offer saves them) x customer_value - offer_cost

Target a customer when that is positive. Because expected value is monotonically
increasing in P(churn), "target everyone with positive expected value" is
identical to "target everyone above a probability threshold", and that threshold
has a closed form:

    break_even_p = offer_cost / (save_rate x customer_value)

Two things this module is careful about:

1. **The offer cost is paid for every contact**, churner or not, saved or not.
   Getting this wrong is the most common way a retention business case
   accidentally becomes profitable.
2. **P(churn) must be a calibrated probability**, not a ranking score. Stage 4
   checked that for the recommended model; if it had failed, this whole layer
   would have to fall back to top-k ranking instead of a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .config import POLICY_CONFIG


@dataclass(frozen=True)
class Assumptions:
    """The three business parameters the data cannot supply."""

    customer_value: float
    save_rate: float
    offer_cost: float
    horizon_years: float = 1.0
    currency: str = "GBP"

    def __post_init__(self) -> None:
        if not 0.0 <= self.save_rate <= 1.0:
            raise ValueError(f"save_rate must be a probability, got {self.save_rate}")
        if self.customer_value < 0 or self.offer_cost < 0:
            raise ValueError("customer_value and offer_cost must be non-negative")

    @property
    def value_per_save(self) -> float:
        """Margin earned when an offer actually retains someone."""
        return self.customer_value * self.horizon_years

    def replace(self, **changes: Any) -> "Assumptions":
        from dataclasses import replace as _replace
        return _replace(self, **changes)


def load_assumptions(path: Path | None = None) -> tuple[Assumptions, dict]:
    """Load central assumptions and the full config from policy.yaml."""
    path = path or POLICY_CONFIG
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    central = Assumptions(
        customer_value=float(raw["customer_value"]["central"]),
        save_rate=float(raw["save_rate"]["central"]),
        offer_cost=float(raw["offer_cost"]["central"]),
        horizon_years=float(raw.get("horizon_years", 1.0)),
        currency=str(raw.get("currency", "GBP")),
    )
    return central, raw


# ---------------------------------------------------------------------------
# Core arithmetic
# ---------------------------------------------------------------------------

def expected_value(p_churn: np.ndarray | float, a: Assumptions) -> np.ndarray | float:
    """Expected value of contacting a customer with churn probability `p_churn`.

    The offer cost is subtracted unconditionally: it is spent on contact, not on
    success.
    """
    return np.asarray(p_churn) * a.save_rate * a.value_per_save - a.offer_cost


def break_even_probability(a: Assumptions) -> float:
    """Churn probability at which contacting a customer breaks even.

    Returns inf when the offer cannot pay for itself at any probability, which
    happens when save_rate or customer_value is zero.
    """
    denominator = a.save_rate * a.value_per_save
    if denominator <= 0:
        return float("inf")
    return a.offer_cost / denominator


def campaign_value(
    p_churn: np.ndarray,
    a: Assumptions,
    *,
    threshold: float | None = None,
    max_contacts: int | None = None,
) -> dict[str, float]:
    """Evaluate a targeting policy over a population of scored customers.

    Args:
        p_churn: calibrated churn probabilities, one per customer.
        a: business assumptions.
        threshold: contact customers at or above this probability. None means
            use the break-even threshold, which maximises total net value.
        max_contacts: optional budget cap. When it binds, the highest-scoring
            customers are chosen, because expected value rises with p_churn.

    Returns:
        Counts and pounds for the policy.
    """
    p = np.asarray(p_churn, dtype=float)
    if threshold is None:
        threshold = break_even_probability(a)

    selected = p >= threshold
    if max_contacts is not None and selected.sum() > max_contacts:
        # Keep the top `max_contacts` by score among those above threshold.
        cutoff_idx = np.argsort(p)[::-1][:max_contacts]
        keep = np.zeros_like(selected)
        keep[cutoff_idx] = True
        selected = selected & keep

    n_targeted = int(selected.sum())
    gross = float(np.sum(p[selected]) * a.save_rate * a.value_per_save)
    cost = float(n_targeted * a.offer_cost)
    return {
        "threshold": float(threshold),
        "n_targeted": n_targeted,
        "share_targeted": float(n_targeted / len(p)) if len(p) else 0.0,
        "expected_saves": float(np.sum(p[selected]) * a.save_rate),
        "gross_value": gross,
        "offer_cost_total": cost,
        "net_value": gross - cost,
        "net_value_per_contact": (gross - cost) / n_targeted if n_targeted else 0.0,
    }


def target_everyone(p_churn: np.ndarray, a: Assumptions) -> dict[str, float]:
    """The 'just contact the whole book' policy."""
    return campaign_value(p_churn, a, threshold=-np.inf)


def target_nobody(p_churn: np.ndarray, a: Assumptions) -> dict[str, float]:
    """Doing nothing. Net value is zero by definition, and it is a real option."""
    return {
        "threshold": float("inf"), "n_targeted": 0, "share_targeted": 0.0,
        "expected_saves": 0.0, "gross_value": 0.0, "offer_cost_total": 0.0,
        "net_value": 0.0, "net_value_per_contact": 0.0,
    }


def profit_curve(
    p_churn: np.ndarray, a: Assumptions, *, n_points: int = 200
) -> pd.DataFrame:
    """Net campaign value across the full range of possible thresholds."""
    p = np.asarray(p_churn, dtype=float)
    thresholds = np.linspace(p.min(), p.max(), n_points)
    rows = [campaign_value(p, a, threshold=t) for t in thresholds]
    return pd.DataFrame(rows)


def optimal_threshold(p_churn: np.ndarray, a: Assumptions) -> dict[str, float]:
    """Best achievable policy, and confirmation it matches the closed form.

    The empirical optimum should equal the break-even probability. Computing both
    is a check on the arithmetic rather than a redundancy.
    """
    curve = profit_curve(p_churn, a)
    best = curve.loc[curve["net_value"].idxmax()].to_dict()
    best["break_even_probability"] = break_even_probability(a)
    return best


# ---------------------------------------------------------------------------
# Sensitivity and break-even on the assumptions themselves
# ---------------------------------------------------------------------------

def sensitivity_grid(
    p_churn: np.ndarray,
    base: Assumptions,
    *,
    values: Iterable[float],
    save_rates: Iterable[float],
    offer_cost: float | None = None,
) -> pd.DataFrame:
    """Optimal policy across a grid of customer_value x save_rate."""
    rows = []
    cost = base.offer_cost if offer_cost is None else offer_cost
    for v in values:
        for s in save_rates:
            a = base.replace(customer_value=v, save_rate=s, offer_cost=cost)
            out = campaign_value(p_churn, a)
            rows.append({
                "customer_value": v, "save_rate": s, "offer_cost": cost,
                "break_even_p": break_even_probability(a),
                "n_targeted": out["n_targeted"],
                "net_value": out["net_value"],
                "pays": out["net_value"] > 0,
            })
    return pd.DataFrame(rows)


def minimum_viable(
    p_churn: np.ndarray,
    base: Assumptions,
    parameter: str,
    *,
    lo: float,
    hi: float,
    tol: float = 1e-4,
) -> float | None:
    """Smallest (or for offer_cost, largest) value of one parameter that still pays.

    Bisects on the parameter holding the other two at their central values. This
    is the "how wrong can I be before this stops working" number, and it is the
    single most useful output for a sceptical stakeholder.

    Returns None when the policy pays nowhere in the range, or everywhere.
    """
    decreasing = parameter == "offer_cost"

    def pays(x: float) -> bool:
        a = base.replace(**{parameter: x})
        return campaign_value(p_churn, a)["net_value"] > 0

    pays_lo, pays_hi = pays(lo), pays(hi)
    if pays_lo == pays_hi:
        return None                       # no crossing inside the range

    # Arrange so that `lo` fails and `hi` pays, then bisect the boundary.
    if decreasing:
        lo, hi = hi, lo
    while abs(hi - lo) > tol:
        mid = (lo + hi) / 2
        if pays(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def bootstrap_net_value(
    p_churn: np.ndarray,
    a: Assumptions,
    *,
    n_boot: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile CI on net campaign value, resampling customers.

    This captures sampling variation in WHO is in the population. It does not
    capture uncertainty in the assumptions themselves -- that is what the
    sensitivity grid is for, and it is much the larger source of doubt.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p_churn, dtype=float)
    n = len(p)
    threshold = break_even_probability(a)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        sample = p[rng.integers(0, n, n)]
        draws[i] = campaign_value(sample, a, threshold=threshold)["net_value"]
    return {
        "point": campaign_value(p, a, threshold=threshold)["net_value"],
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "share_positive": float((draws > 0).mean()),
    }
