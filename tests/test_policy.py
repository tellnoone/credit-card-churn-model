"""Tests for the profit calculation.

This arithmetic decides how much money the business spends, so it is tested
against hand-computed values rather than against itself. Several of these pin
mistakes that are easy to make and hard to notice:

* charging the offer cost only to customers who churn (inflates profit)
* multiplying by lifetime value while calling it annual (inflates profit)
* letting the optimal threshold drift away from the closed-form break-even
"""

from __future__ import annotations

import numpy as np
import pytest

from src.policy import (Assumptions, break_even_probability, bootstrap_net_value,
                        campaign_value, expected_value, load_assumptions,
                        minimum_viable, optimal_threshold, profit_curve,
                        target_everyone, target_nobody)

# save_rate 0.25, value 300 -> 75 per expected save; cost 40 per contact.
A = Assumptions(customer_value=300.0, save_rate=0.25, offer_cost=40.0)


# ---------------------------------------------------------------------------
# Expected value, by hand
# ---------------------------------------------------------------------------

def test_expected_value_matches_hand_calculation() -> None:
    # p=0.8: 0.8 * 0.25 * 300 = 60 gross, minus 40 cost = +20
    assert expected_value(0.8, A) == pytest.approx(20.0)
    # p=0.4: 0.4 * 0.25 * 300 = 30 gross, minus 40 cost = -10
    assert expected_value(0.4, A) == pytest.approx(-10.0)


def test_offer_cost_is_charged_even_at_zero_churn_probability() -> None:
    """The classic error: only charging for customers who would have churned.

    A customer certain to stay still costs the full offer to contact.
    """
    assert expected_value(0.0, A) == pytest.approx(-A.offer_cost)


def test_expected_value_is_monotonic_in_churn_probability() -> None:
    """Justifies treating 'positive EV' and 'above a threshold' as the same rule."""
    p = np.linspace(0, 1, 50)
    ev = np.asarray(expected_value(p, A))
    assert np.all(np.diff(ev) > 0)


# ---------------------------------------------------------------------------
# Break-even threshold
# ---------------------------------------------------------------------------

def test_break_even_probability_matches_hand_calculation() -> None:
    # 40 / (0.25 * 300) = 40 / 75 = 0.5333...
    assert break_even_probability(A) == pytest.approx(40.0 / 75.0)


def test_expected_value_is_zero_at_the_break_even_point() -> None:
    assert expected_value(break_even_probability(A), A) == pytest.approx(0.0)


def test_break_even_is_infinite_when_the_offer_cannot_work() -> None:
    assert break_even_probability(A.replace(save_rate=0.0)) == float("inf")
    assert break_even_probability(A.replace(customer_value=0.0)) == float("inf")


def test_cheaper_offers_lower_the_bar() -> None:
    assert break_even_probability(A.replace(offer_cost=10.0)) < break_even_probability(A)


def test_more_valuable_customers_lower_the_bar() -> None:
    assert break_even_probability(A.replace(customer_value=800.0)) < break_even_probability(A)


# ---------------------------------------------------------------------------
# Campaign aggregation
# ---------------------------------------------------------------------------

def test_campaign_value_on_a_tiny_hand_computable_population() -> None:
    p = np.array([0.9, 0.6, 0.1])
    # Break-even is 0.5333, so 0.9 and 0.6 are contacted.
    # gross = (0.9 + 0.6) * 0.25 * 300 = 1.5 * 75 = 112.5
    # cost  = 2 * 40 = 80  ->  net = 32.5
    out = campaign_value(p, A)
    assert out["n_targeted"] == 2
    assert out["gross_value"] == pytest.approx(112.5)
    assert out["offer_cost_total"] == pytest.approx(80.0)
    assert out["net_value"] == pytest.approx(32.5)
    assert out["expected_saves"] == pytest.approx(1.5 * 0.25)


def test_net_value_equals_gross_minus_cost() -> None:
    p = np.random.default_rng(0).random(500)
    out = campaign_value(p, A)
    assert out["net_value"] == pytest.approx(out["gross_value"] - out["offer_cost_total"])


def test_target_nobody_is_exactly_zero() -> None:
    p = np.random.default_rng(1).random(200)
    out = target_nobody(p, A)
    assert out["net_value"] == 0.0
    assert out["n_targeted"] == 0


def test_target_everyone_contacts_everyone() -> None:
    p = np.random.default_rng(2).random(200)
    out = target_everyone(p, A)
    assert out["n_targeted"] == 200
    assert out["offer_cost_total"] == pytest.approx(200 * A.offer_cost)


def test_target_everyone_loses_money_on_a_low_risk_book() -> None:
    """With every p below break-even, contacting the whole book must lose."""
    p = np.full(100, 0.1)
    assert target_everyone(p, A)["net_value"] < 0


def test_budget_cap_selects_the_highest_scorers() -> None:
    p = np.array([0.95, 0.90, 0.85, 0.80, 0.75])
    out = campaign_value(p, A, threshold=0.0, max_contacts=2)
    assert out["n_targeted"] == 2
    # gross from the top two only: (0.95 + 0.90) * 75
    assert out["gross_value"] == pytest.approx((0.95 + 0.90) * 75.0)


def test_budget_cap_does_nothing_when_it_does_not_bind() -> None:
    p = np.array([0.9, 0.8])
    capped = campaign_value(p, A, threshold=0.0, max_contacts=10)
    uncapped = campaign_value(p, A, threshold=0.0)
    assert capped == uncapped


# ---------------------------------------------------------------------------
# The optimum
# ---------------------------------------------------------------------------

def test_optimal_threshold_is_the_break_even_threshold() -> None:
    """The empirical optimum must agree with the closed form.

    If these ever disagree, either the curve or the formula is wrong.
    """
    p = np.random.default_rng(3).beta(2, 6, 3_000)
    out = optimal_threshold(p, A)
    assert out["threshold"] == pytest.approx(out["break_even_probability"], abs=0.02)


def test_break_even_policy_beats_both_trivial_policies() -> None:
    p = np.random.default_rng(4).beta(2, 3, 2_000)
    best = campaign_value(p, A)["net_value"]
    assert best >= target_nobody(p, A)["net_value"]
    assert best >= target_everyone(p, A)["net_value"]


def test_profit_curve_never_exceeds_the_optimum() -> None:
    p = np.random.default_rng(5).beta(2, 4, 1_000)
    curve = profit_curve(p, A)
    assert curve["net_value"].max() <= campaign_value(p, A)["net_value"] + 1e-6


def test_optimal_net_value_is_never_negative() -> None:
    """Targeting only positive-EV customers cannot lose money in expectation.

    An all-low-risk book should simply target nobody.
    """
    p = np.full(500, 0.01)
    assert campaign_value(p, A)["net_value"] >= 0.0


# ---------------------------------------------------------------------------
# Assumption break-even
# ---------------------------------------------------------------------------

def test_minimum_viable_customer_value_is_a_real_boundary() -> None:
    p = np.random.default_rng(6).beta(2, 8, 2_000)
    boundary = minimum_viable(p, A, "customer_value", lo=1.0, hi=5_000.0)
    assert boundary is not None
    just_below = campaign_value(p, A.replace(customer_value=boundary * 0.9))["net_value"]
    just_above = campaign_value(p, A.replace(customer_value=boundary * 1.1))["net_value"]
    assert just_below <= 0 < just_above


def test_minimum_viable_returns_none_when_there_is_no_crossing() -> None:
    """A population that always pays has no lower boundary in range."""
    p = np.full(100, 0.99)
    assert minimum_viable(p, A, "customer_value", lo=200.0, hi=1_000.0) is None


def test_maximum_viable_offer_cost_is_a_real_boundary() -> None:
    p = np.random.default_rng(7).beta(2, 6, 2_000)
    boundary = minimum_viable(p, A, "offer_cost", lo=1.0, hi=1_000.0)
    assert boundary is not None
    cheaper = campaign_value(p, A.replace(offer_cost=boundary * 0.9))["net_value"]
    dearer = campaign_value(p, A.replace(offer_cost=boundary * 1.1))["net_value"]
    assert dearer <= 0 < cheaper


# ---------------------------------------------------------------------------
# Guards and config
# ---------------------------------------------------------------------------

def test_invalid_assumptions_are_rejected() -> None:
    with pytest.raises(ValueError):
        Assumptions(customer_value=300.0, save_rate=1.5, offer_cost=40.0)
    with pytest.raises(ValueError):
        Assumptions(customer_value=-1.0, save_rate=0.25, offer_cost=40.0)


def test_horizon_scales_value_and_is_not_silently_ignored() -> None:
    """Guards against a lifetime-value figure being reported as annual."""
    annual = Assumptions(customer_value=300.0, save_rate=0.25, offer_cost=40.0,
                         horizon_years=1.0)
    three_year = annual.replace(horizon_years=3.0)
    assert three_year.value_per_save == pytest.approx(3 * annual.value_per_save)
    assert break_even_probability(three_year) < break_even_probability(annual)


def test_config_file_loads_and_is_internally_consistent() -> None:
    central, raw = load_assumptions()
    assert central.customer_value == raw["customer_value"]["central"]
    assert central.save_rate == raw["save_rate"]["central"]
    assert central.offer_cost == raw["offer_cost"]["central"]
    for key in ("customer_value", "save_rate", "offer_cost"):
        block = raw[key]
        assert block["low"] <= block["central"] <= block["high"], key
        assert block["reasoning"].strip(), f"{key} has no stated reasoning"


def test_bootstrap_net_value_brackets_the_point_estimate() -> None:
    p = np.random.default_rng(8).beta(2, 5, 1_500)
    out = bootstrap_net_value(p, A, n_boot=300, seed=1)
    assert out["lo"] <= out["point"] <= out["hi"]
    assert 0.0 <= out["share_positive"] <= 1.0
