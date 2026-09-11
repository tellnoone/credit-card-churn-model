"""Tests for the feature inventory and the leakage classification.

These are pure-logic tests: they need no database and run in milliseconds. They
guard the property the whole project rests on -- that a SUSPECT feature cannot
get into the strict model by accident.
"""

from __future__ import annotations

import pytest

from src.features import (FEATURES, LEAKED_COLUMNS, NON_FEATURE_COLUMNS,
                          Feature, Leakage, categorical_names, feature_names,
                          features_by_class, numeric_names)


def test_no_duplicate_feature_names() -> None:
    names = [f.name for f in FEATURES]
    assert len(names) == len(set(names)), "a feature is declared twice"


def test_every_feature_has_a_rationale() -> None:
    """A classification without an argument is an assertion, not an audit."""
    for f in FEATURES:
        assert f.rationale.strip(), f"{f.name} has no rationale"
        assert len(f.rationale) > 40, f"{f.name}'s rationale is too thin to defend"


def test_strict_is_a_subset_of_all() -> None:
    strict = set(feature_names(strict=True))
    everything = set(feature_names(strict=False))
    assert strict <= everything
    assert strict < everything, "strict should drop something, or the audit is moot"


def test_strict_contains_only_safe_features() -> None:
    """The property that makes the strict model meaningful."""
    safe = {f.name for f in features_by_class(Leakage.SAFE)}
    assert set(feature_names(strict=True)) == safe


def test_suspect_features_are_excluded_from_strict() -> None:
    strict = set(feature_names(strict=True))
    for f in features_by_class(Leakage.SUSPECT):
        assert f.name not in strict, f"SUSPECT feature {f.name} leaked into strict"


def test_leaked_features_never_returned() -> None:
    """LEAKED columns must not appear in either variant."""
    for strict in (True, False):
        returned = set(feature_names(strict=strict))
        for leaked in LEAKED_COLUMNS:
            assert leaked not in returned
        assert not any("naive_bayes" in n.lower() for n in returned)


def test_no_feature_is_classified_leaked_in_the_inventory() -> None:
    """LEAKED columns are dropped in SQL, so they should never be modelled at all.

    If someone adds one to FEATURES, this fails loudly rather than relying on
    feature_names() to filter it.
    """
    assert features_by_class(Leakage.LEAKED) == []


def test_target_and_ids_are_not_features() -> None:
    names = set(feature_names(strict=False))
    for col in NON_FEATURE_COLUMNS:
        assert col not in names, f"{col} must not be used as a feature"


def test_numeric_and_categorical_partition_the_feature_set() -> None:
    """Every feature is encoded exactly one way -- no gaps, no double counting."""
    for strict in (True, False):
        everything = set(feature_names(strict=strict))
        num = set(numeric_names(strict=strict))
        cat = set(categorical_names(strict=strict))
        assert num | cat == everything, "a feature has no encoding route"
        assert num & cat == set(), "a feature is both numeric and categorical"


def test_kind_values_are_known() -> None:
    for f in FEATURES:
        assert f.kind in {"numeric", "categorical", "binary"}, f.name


@pytest.mark.parametrize("strict", [True, False])
def test_feature_lists_are_non_empty_and_stable(strict: bool) -> None:
    first = feature_names(strict=strict)
    assert first, "no features returned"
    assert first == feature_names(strict=strict), "feature order is not deterministic"


def test_q4_change_ratios_are_suspect() -> None:
    """The sharpest leakage candidates must not drift into SAFE.

    Pinned explicitly: these are the features whose classification most changes
    the headline result, so a future edit should have to argue with a test.
    """
    suspect = {f.name for f in features_by_class(Leakage.SUSPECT)}
    for name in ("total_amt_chng_q4_q1", "total_ct_chng_q4_q1",
                 "q4_activity_collapsed", "months_inactive_12_mon",
                 "total_trans_ct", "total_trans_amt"):
        assert name in suspect, f"{name} must stay SUSPECT"


def test_demographics_are_safe_for_leakage_purposes() -> None:
    """Age and gender are a FAIRNESS question, not a leakage one.

    Keeping the two concerns separate matters: conflating them would let a
    fairness decision be smuggled in as a leakage argument.
    """
    safe = {f.name for f in features_by_class(Leakage.SAFE)}
    assert {"customer_age", "gender"} <= safe


def test_derived_features_inherit_the_worst_input_class() -> None:
    """revolving_to_limit mixes a SAFE and a SUSPECT input; it must be SUSPECT."""
    by_name: dict[str, Feature] = {f.name: f for f in FEATURES}
    assert by_name["revolving_to_limit"].leakage is Leakage.SUSPECT
    assert by_name["credit_limit"].leakage is Leakage.SAFE      # the SAFE input
    assert by_name["total_revolving_bal"].leakage is Leakage.SUSPECT  # the SUSPECT one
