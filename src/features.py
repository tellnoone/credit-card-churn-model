"""Feature inventory and the SAFE / SUSPECT leakage classification.

This module is the authority on which features may be used in which model. The
dbt layer builds the columns; this decides what they mean and whether they can
be trusted. Stage 3's audit reads its classification from here, so the argument
and the code cannot drift apart.

The classification rests on one fact about the data: the snapshot has **no
dates**. There is no churn date and no observation window, so for any feature
measured over "the last 12 months" we cannot tell whether that window ended
before the customer churned or after. A feature is SUSPECT when a plausible
reading of the data generating process has it being measured *during or after*
the churn it is supposed to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Leakage(str, Enum):
    """How much a feature can be trusted to exist before churn happens."""

    SAFE = "safe"
    SUSPECT = "suspect"
    LEAKED = "leaked"


@dataclass(frozen=True)
class Feature:
    """One model input, with the argument for its leakage classification."""

    name: str
    kind: str                  # "numeric" | "categorical" | "binary"
    leakage: Leakage
    rationale: str


FEATURES: tuple[Feature, ...] = (
    # ---------------------------------------------------------------- SAFE --
    Feature(
        "customer_age", "numeric", Leakage.SAFE,
        "Age is a property of the person, not of their account behaviour. Known "
        "at application and updated by the calendar alone. Cannot be caused by "
        "churning. (Whether it should be USED is a fairness question, not a "
        "leakage one -- see plan section 8.)",
    ),
    Feature(
        "gender", "categorical", Leakage.SAFE,
        "Collected at application and immutable for this purpose, so it cannot be "
        "a consequence of churn. Same fairness caveat as age: SAFE here is a "
        "statement about leakage only, not about whether it should be used.",
    ),
    Feature(
        "dependent_count", "numeric", Leakage.SAFE,
        "Household composition, captured at application. Slow-moving and cannot "
        "plausibly be caused by a decision to close a credit card account.",
    ),
    Feature(
        "education_level", "categorical", Leakage.SAFE,
        "Captured at application and effectively fixed thereafter, so it cannot be "
        "caused by the churn it is used to predict. 15.0% 'Unknown', kept as its "
        "own level rather than imputed.",
    ),
    Feature(
        "marital_status", "categorical", Leakage.SAFE,
        "Captured at application and rarely refreshed afterwards, so it is stale "
        "rather than leaked -- it cannot be a consequence of churning. 7.4% "
        "'Unknown'.",
    ),
    Feature(
        "income_category", "categorical", Leakage.SAFE,
        "Stated income at application, banded rather than continuous, so it is "
        "coarse but stable and not refreshed by recent behaviour. 11.0% "
        "'Unknown'.",
    ),
    Feature(
        "card_category", "categorical", Leakage.SAFE,
        "Product held. Changes only on an explicit upgrade/downgrade event, "
        "which is a deliberate action rather than a symptom of disengagement.",
    ),
    Feature(
        "card_category_ord", "numeric", Leakage.SAFE,
        "Ordinal encoding of card_category on the issuer's own product ladder: "
        "Blue < Silver < Gold < Platinum. Inherits card_category's SAFE class.",
    ),
    Feature(
        "months_on_book", "numeric", Leakage.SAFE,
        "Tenure. Increases with the calendar. A churner's tenure stops growing "
        "at churn, which is a mild concern, but tenure is knowable at any point "
        "before churn so a model can use it prospectively.",
    ),
    Feature(
        "tenure_years", "numeric", Leakage.SAFE,
        "months_on_book expressed in years, which reads more naturally in a "
        "coefficient table. Same SAFE reasoning as months_on_book itself.",
    ),
    Feature(
        "total_relationship_count", "numeric", Leakage.SAFE,
        "Number of products held. A customer winding down MIGHT close products "
        "first, which would make this a symptom. Classified SAFE because product "
        "holdings are a deliberate cross-sell outcome and are the standard "
        "pre-period feature in retention models -- but this is the weakest SAFE "
        "call in the list, and is flagged as such rather than hidden.",
    ),
    Feature(
        "credit_limit", "numeric", Leakage.SAFE,
        "Set by the issuer's own risk policy at origination and on periodic review, "
        "not by the customer's recent activity, so it is an input to behaviour "
        "rather than an output of it.",
    ),
    Feature(
        "products_per_tenure_year", "numeric", Leakage.SAFE,
        "How fast the relationship deepened: products held per year of tenure. "
        "Derived from total_relationship_count and months_on_book, both SAFE.",
    ),
    Feature(
        "education_level_ord", "numeric", Leakage.SAFE,
        "Ordinal encoding of the education ladder; NULL where 'Unknown', with the "
        "signal carried by education_is_unknown instead.",
    ),
    Feature(
        "income_category_ord", "numeric", Leakage.SAFE,
        "Ordinal encoding of the stated-income bands; NULL where 'Unknown', with "
        "the signal carried by income_is_unknown instead.",
    ),
    Feature(
        "education_is_unknown", "binary", Leakage.SAFE,
        "Missingness indicator for education. Carries the signal that the ordinal "
        "encoding drops when it emits NULL. Refusal to state a field is itself "
        "weakly informative (+0.93pp churn, measured).",
    ),
    Feature(
        "income_is_unknown", "binary", Leakage.SAFE,
        "Missingness indicator for stated income (+0.84pp churn, measured). May "
        "also proxy an acquisition channel that did not collect it.",
    ),
    Feature(
        "marital_is_unknown", "binary", Leakage.SAFE,
        "Missingness indicator for marital status (+1.25pp churn, measured -- the "
        "strongest of the three, though all are weak).",
    ),

    # ------------------------------------------------------------- SUSPECT --
    Feature(
        "total_trans_ct", "numeric", Leakage.SUSPECT,
        "Transactions over 'the last 12 months'. If that window runs up to the "
        "snapshot date, then for a customer who churned six months ago it "
        "includes the months they had already stopped transacting. The feature "
        "would then be measuring the churn, not anticipating it.",
    ),
    Feature(
        "total_trans_amt", "numeric", Leakage.SUSPECT,
        "Spend over the same undated 12-month window as total_trans_ct, and so "
        "carries exactly the same risk of straddling the churn event.",
    ),
    Feature(
        "months_inactive_12_mon", "numeric", Leakage.SUSPECT,
        "Months with no activity. For a churned customer, inactivity IS the "
        "churn. Also quirky: despite the name it maxes at 6 in this snapshot, an "
        "undocumented ceiling that is its own reason for caution.",
    ),
    Feature(
        "contacts_count_12_mon", "numeric", Leakage.SUSPECT,
        "Contacts with the bank. Plausibly a leading indicator (complaints "
        "before leaving) but equally plausibly the closure call itself. Without "
        "dates or contact reasons the two cannot be separated.",
    ),
    Feature(
        "total_revolving_bal", "numeric", Leakage.SUSPECT,
        "Revolving balance at snapshot. A closing customer pays down to zero, so "
        "a low balance may be a consequence of leaving rather than a cause.",
    ),
    Feature(
        "avg_open_to_buy", "numeric", Leakage.SUSPECT,
        "credit_limit minus revolving balance. Inherits the balance problem: "
        "arithmetically it is almost the complement of total_revolving_bal.",
    ),
    Feature(
        "avg_utilization_ratio", "numeric", Leakage.SUSPECT,
        "Balance over limit. Same reasoning as total_revolving_bal.",
    ),
    Feature(
        "total_amt_chng_q4_q1", "numeric", Leakage.SUSPECT,
        "Q4-vs-Q1 spend ratio. The sharpest case in the dataset: a collapse in "
        "recent spend versus earlier spend is close to a DEFINITION of "
        "disengagement rather than a predictor of it.",
    ),
    Feature(
        "total_ct_chng_q4_q1", "numeric", Leakage.SUSPECT,
        "Q4-vs-Q1 transaction-count ratio. Same argument as the amount ratio: a "
        "collapse in recent versus earlier activity may BE the churn signal "
        "rather than precede it.",
    ),
    Feature(
        "avg_trans_amt", "numeric", Leakage.SUSPECT,
        "Mean value per transaction. Derived from total_trans_amt and "
        "total_trans_ct, both SUSPECT, so it cannot be safer than its inputs.",
    ),
    Feature(
        "revolving_to_limit", "numeric", Leakage.SUSPECT,
        "Derived from total_revolving_bal (SUSPECT) and credit_limit (SAFE). A "
        "derived feature inherits the WORST class of its inputs.",
    ),
    Feature(
        "spend_to_limit", "numeric", Leakage.SUSPECT,
        "Spend as a share of the credit line. Derived from total_trans_amt, which "
        "is SUSPECT, so this inherits that class.",
    ),
    Feature(
        "trans_per_active_month", "numeric", Leakage.SUSPECT,
        "Transactions per month the customer was actually active. Derived from two "
        "SUSPECT columns (total_trans_ct and months_inactive_12_mon), so it "
        "compounds their window problem rather than escaping it.",
    ),
    Feature(
        "q4_activity_collapsed", "binary", Leakage.SUSPECT,
        "Binary flag on total_ct_chng_q4_q1 < 0.5. Included precisely so the "
        "audit can show what a near-definition-of-churn feature does to the "
        "metrics.",
    ),
)

# Columns that must never be modelled. Dropped in dbt staging; listed here so a
# test can assert they never reappear.
LEAKED_COLUMNS: tuple[str, ...] = (
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon"
    "_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon"
    "_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
)

# Non-feature columns present in the feature table.
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "customer_id", "is_attrited", "attrition_flag_raw",
)


def feature_names(*, strict: bool) -> list[str]:
    """Return the feature list for a model variant.

    Args:
        strict: If True, return only SAFE features -- the model the plan
            pre-commits to recommending. If False, return SAFE + SUSPECT (the
            "all features" variant). LEAKED is never returned either way.
    """
    allowed = {Leakage.SAFE} if strict else {Leakage.SAFE, Leakage.SUSPECT}
    return [f.name for f in FEATURES if f.leakage in allowed]


def features_by_class(leakage: Leakage) -> list[Feature]:
    """All features in one leakage class."""
    return [f for f in FEATURES if f.leakage is leakage]


def categorical_names(*, strict: bool) -> list[str]:
    """Categorical feature names for a variant (need encoding)."""
    names = set(feature_names(strict=strict))
    return [f.name for f in FEATURES if f.name in names and f.kind == "categorical"]


def numeric_names(*, strict: bool) -> list[str]:
    """Numeric and binary feature names for a variant."""
    names = set(feature_names(strict=strict))
    return [f.name for f in FEATURES
            if f.name in names and f.kind in ("numeric", "binary")]
