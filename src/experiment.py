"""Sample-size and power arithmetic for the retention-offer experiment.

Implemented directly from the normal approximation rather than pulled from
statsmodels, to avoid adding a dependency for three formulas -- and because the
formula is worth having visible in a project whose argument is that the
reasoning should be checkable.

The quantity the experiment exists to measure is the **save rate**: the causal
effect of the offer on someone who would otherwise have churned. Stage 6 showed
that the entire business case turns on it, and that the data cannot supply it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ExperimentDesign:
    """A two-arm randomised test of a retention offer."""

    baseline_churn: float        # churn rate in the targeted segment, no offer
    save_rate_mde: float         # smallest save rate worth detecting
    alpha: float = 0.05
    power: float = 0.80
    allocation: float = 0.5      # share assigned to treatment

    def __post_init__(self) -> None:
        for name, v in (("baseline_churn", self.baseline_churn),
                        ("save_rate_mde", self.save_rate_mde),
                        ("alpha", self.alpha), ("power", self.power),
                        ("allocation", self.allocation)):
            if not 0.0 < v < 1.0:
                raise ValueError(f"{name} must be strictly between 0 and 1, got {v}")

    @property
    def treatment_churn(self) -> float:
        """Churn rate under the offer, if the save rate is exactly the MDE."""
        return self.baseline_churn * (1.0 - self.save_rate_mde)

    @property
    def absolute_effect(self) -> float:
        """Percentage-point reduction in churn the test must detect."""
        return self.baseline_churn - self.treatment_churn


def sample_size_per_arm(design: ExperimentDesign) -> float:
    """Customers needed per arm, two-sided two-proportion test.

    n = (z_a * sqrt(2 * p_bar * (1 - p_bar))
         + z_b * sqrt(p1(1-p1) + p2(1-p2)))^2 / (p1 - p2)^2

    The pooled term belongs to the null (where both arms share a rate) and the
    unpooled term to the alternative. Using one for both is a common shortcut
    that misstates the requirement.
    """
    p1, p2 = design.baseline_churn, design.treatment_churn
    delta = p1 - p2
    if delta <= 0:
        return float("inf")
    p_bar = (p1 + p2) / 2.0
    z_a = stats.norm.ppf(1.0 - design.alpha / 2.0)
    z_b = stats.norm.ppf(design.power)
    numerator = (z_a * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
                 + z_b * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))) ** 2
    return float(numerator / delta ** 2)


def total_sample_size(design: ExperimentDesign) -> float:
    """Customers needed across both arms."""
    return 2.0 * sample_size_per_arm(design)


def power_at_n(design: ExperimentDesign, n_per_arm: float) -> float:
    """Power actually achieved with a given number of customers per arm."""
    p1, p2 = design.baseline_churn, design.treatment_churn
    delta = p1 - p2
    if delta <= 0 or n_per_arm <= 0:
        return 0.0
    p_bar = (p1 + p2) / 2.0
    z_a = stats.norm.ppf(1.0 - design.alpha / 2.0)
    se_null = np.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_arm)
    se_alt = np.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_per_arm)
    return float(stats.norm.cdf((delta - z_a * se_null) / se_alt))


def detectable_save_rate(design: ExperimentDesign, n_per_arm: float) -> float:
    """Smallest save rate detectable at the design's power, given n.

    The retrospective counterpart of sample_size_per_arm: it answers "what could
    this experiment actually have seen" rather than "how big must it be".
    """
    lo, hi = 1e-6, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2.0
        trial = ExperimentDesign(design.baseline_churn, mid, design.alpha,
                                 design.power, design.allocation)
        if power_at_n(trial, n_per_arm) >= design.power:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0
