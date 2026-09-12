"""Tests for the experiment sample-size arithmetic."""

from __future__ import annotations

import pytest

from src.experiment import (ExperimentDesign, detectable_save_rate, power_at_n,
                            sample_size_per_arm, total_sample_size)

D = ExperimentDesign(baseline_churn=0.30, save_rate_mde=0.25)


def test_treatment_churn_follows_from_the_save_rate() -> None:
    # 30% churn, a quarter of them saved -> 22.5%
    assert D.treatment_churn == pytest.approx(0.225)
    assert D.absolute_effect == pytest.approx(0.075)


def test_sample_size_against_a_known_reference() -> None:
    """0.30 vs 0.225 at alpha 0.05, 80% power needs roughly 540-560 per arm.

    Cross-checked against the standard two-proportion formula. Pinned loosely
    because textbook tables differ slightly on the pooled-variance term.
    """
    n = sample_size_per_arm(D)
    assert 520 < n < 600, n
    assert total_sample_size(D) == pytest.approx(2 * n)


def test_smaller_effects_need_bigger_samples() -> None:
    small = ExperimentDesign(baseline_churn=0.30, save_rate_mde=0.10)
    assert sample_size_per_arm(small) > sample_size_per_arm(D)


def test_halving_the_effect_roughly_quadruples_the_sample() -> None:
    """The relationship every stakeholder asking for 'a smaller test' needs."""
    half = ExperimentDesign(baseline_churn=0.30, save_rate_mde=0.125)
    ratio = sample_size_per_arm(half) / sample_size_per_arm(D)
    assert 3.5 < ratio < 4.5, ratio


def test_higher_baseline_churn_needs_fewer_customers() -> None:
    """Targeting a higher-risk segment is itself a way to shrink the test."""
    high = ExperimentDesign(baseline_churn=0.50, save_rate_mde=0.25)
    assert sample_size_per_arm(high) < sample_size_per_arm(D)


def test_power_at_the_required_n_is_the_target_power() -> None:
    n = sample_size_per_arm(D)
    assert power_at_n(D, n) == pytest.approx(D.power, abs=0.02)


def test_power_rises_with_sample_size() -> None:
    n = sample_size_per_arm(D)
    assert power_at_n(D, n / 4) < power_at_n(D, n) < power_at_n(D, n * 4)


def test_power_is_bounded() -> None:
    assert 0.0 <= power_at_n(D, 10) <= 1.0
    assert power_at_n(D, 10_000_000) == pytest.approx(1.0, abs=1e-6)


def test_detectable_save_rate_inverts_the_sample_size_calculation() -> None:
    """The retrospective MDE must round-trip with the prospective one."""
    n = sample_size_per_arm(D)
    assert detectable_save_rate(D, n) == pytest.approx(D.save_rate_mde, abs=0.01)


def test_a_small_experiment_can_only_see_a_large_effect() -> None:
    assert detectable_save_rate(D, 100) > D.save_rate_mde


def test_invalid_designs_are_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentDesign(baseline_churn=0.0, save_rate_mde=0.25)
    with pytest.raises(ValueError):
        ExperimentDesign(baseline_churn=0.3, save_rate_mde=1.5)
    with pytest.raises(ValueError):
        ExperimentDesign(baseline_churn=0.3, save_rate_mde=0.25, power=1.0)
