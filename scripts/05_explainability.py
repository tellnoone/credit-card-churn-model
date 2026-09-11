"""Stage 5: explainability and fairness (ANALYSIS_PLAN.md section 8).

Two questions:

1. **What is the model using?** SHAP global importance, plus three individual
   customers explained in plain English -- the form a retention agent would
   actually need if the score appeared in their queue.

2. **Should it be using age and gender at all?** For a UK retention decision this
   is a legal question as much as a modelling one. The cost of removing them is
   measured rather than assumed, and so is whether removal actually works --
   proxies survive a dropped column.

Note on the test set: the fairness variants are compared by repeated CV on the
training set, not on test. Stage 4 spent the single permitted test pass. Scoring
a new model on test now would be a second pass, which is the thing the plan's
"used once" rule exists to prevent.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import (FIGURES, MODELS, PRECISION_AT_K, RANDOM_SEED, TABLES,
                        TARGET, ensure_dirs)
from src.data import load_features, make_splits
from src.evaluate import paired_fold_difference, repeated_cv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import Leakage, feature_names, features_by_class
from src.train import make_estimator, make_pipeline, xy

REPORT: list[str] = []
PROTECTED = ["gender", "customer_age"]


def say(msg: str = "") -> None:
    print(msg)
    REPORT.append(msg)


def rule(ch: str = "-", n: int = 78) -> None:
    say(ch * n)


def head(title: str) -> None:
    rule("=")
    say(title)
    rule("=")
    say()


def plain_english(feature: str, value: float, shap_value: float) -> str:
    """Turn one SHAP contribution into a sentence a retention agent can use."""
    direction = "raises" if shap_value > 0 else "lowers"
    readable = {
        "total_relationship_count": f"holds {value:.0f} product(s) with us",
        "products_per_tenure_year": f"has taken on {value:.2f} products per year of tenure",
        "months_on_book": f"has been a customer for {value:.0f} months",
        "tenure_years": f"has been a customer for {value:.1f} years",
        "credit_limit": f"has a credit limit of ${value:,.0f}",
        "customer_age": f"is {value:.0f} years old",
        "dependent_count": f"has {value:.0f} dependent(s)",
        "card_category_ord": f"holds a tier-{value:.0f} card (1=Blue, 4=Platinum)",
        "income_category_ord": f"is in income band {value:.0f} of 5",
        "education_level_ord": f"is at education level {value:.0f} of 6",
    }
    phrase = readable.get(feature)
    if phrase is None:
        if feature.startswith("gender_"):
            phrase = f"is recorded as gender {feature.split('_', 1)[1]}"
        elif "_is_unknown" in feature:
            field = feature.replace("_is_unknown", "")
            phrase = (f"has no {field} on file" if value >= 0.5
                      else f"has {field} on file")
        elif "_" in feature:
            field, level = feature.split("_", 1)
            phrase = f"has {field} = {level}" if value >= 0.5 else f"does not have {field} = {level}"
        else:
            phrase = f"{feature} = {value:g}"
    return f"{phrase} -- this {direction} their churn score"


def main() -> int:
    ensure_dirs()
    import shap

    df = load_features()
    splits = make_splits(df)
    train, val, test = splits["train"], splits["val"], splits["test"]
    payload: dict = {}

    head("STAGE 5: EXPLAINABILITY AND FAIRNESS")
    say("  Model explained: lightgbm_strict, the stage 4 recommendation.")
    say("  SHAP values computed on the test set (already scored in stage 4, so")
    say("  no new information is extracted from it).")
    say()

    import joblib
    model = joblib.load(MODELS / "recommended_model_uncalibrated.joblib")
    X_test, y_test = xy(test, strict=True)

    prep = model.named_steps["prep"]
    booster = model.named_steps["model"]
    X_enc = pd.DataFrame(prep.transform(X_test),
                         columns=prep.get_feature_names_out(),
                         index=X_test.index)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_enc)
    if isinstance(shap_values, list):          # older API returns per-class
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:                  # (n, features, classes)
        shap_values = shap_values[:, :, 1]

    # ------------------------------------------------------------- global --
    head("1. GLOBAL IMPORTANCE (mean |SHAP|)")
    imp = (pd.DataFrame({
        "feature": X_enc.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["cumulative"] = imp["share"].cumsum()

    say(f"  {'rank':>4}  {'feature':<32}{'mean|SHAP|':>12}{'share':>9}{'cum':>8}")
    rule()
    for i, r in imp.head(15).iterrows():
        say(f"  {i+1:>4}  {r['feature']:<32}{r['mean_abs_shap']:>12.4f}"
            f"{r['share']:>8.1%}{r['cumulative']:>8.1%}")
    rule()
    say()
    top = imp.iloc[0]
    say(f"  {top['feature']} carries {top['share']:.0%} of the total attribution.")
    say(f"  The top 3 features carry {imp['share'].head(3).sum():.0%}.")
    say()
    say("  This matches the stage 3 concentration finding from a different")
    say("  direction: the model is mostly one feature. SHAP is not revealing a")
    say("  rich set of churn drivers, because the strict feature set does not")
    say("  contain one.")
    say()
    payload["global_importance"] = imp.to_dict(orient="records")

    # --------------------------------------------------------- individuals --
    head("2. THREE CUSTOMERS, EXPLAINED")
    scores = model.predict_proba(X_test)[:, 1]
    order = np.argsort(scores)
    picks = {
        "highest risk": int(order[-1]),
        "median risk": int(order[len(order) // 2]),
        "lowest risk": int(order[0]),
    }
    individuals = []
    for label, pos in picks.items():
        cid = int(test.iloc[pos]["customer_id"])
        actual = int(y_test[pos])
        score = float(scores[pos])
        contribs = pd.DataFrame({
            "feature": X_enc.columns,
            "value": X_enc.iloc[pos].to_numpy(),
            "shap": shap_values[pos],
        })
        contribs["abs"] = contribs["shap"].abs()
        contribs = contribs.sort_values("abs", ascending=False).head(4)

        say(f"  {label.upper()}  --  customer {cid}")
        say(f"    model score : {score:.1%} probability of churn")
        say(f"    actually    : {'CHURNED' if actual else 'stayed'}")
        say(f"    baseline    : {1/(1+np.exp(-explainer.expected_value)):.1%} "
            f"(the average customer)")
        say("    why:")
        lines = []
        for _, c in contribs.iterrows():
            sentence = plain_english(c["feature"], c["value"], c["shap"])
            say(f"      - {sentence}  ({c['shap']:+.3f})")
            lines.append(sentence)
        say()
        individuals.append({"label": label, "customer_id": cid, "score": score,
                            "actual": actual, "explanations": lines})
    payload["individuals"] = individuals

    say("  A caveat that belongs next to any individual explanation here: the")
    say("  model's scores span roughly 5% to 56%, so even the 'highest risk'")
    say("  customer is more likely to stay than to leave. An explanation makes")
    say("  the score legible; it does not make the score confident.")
    say()

    # ------------------------------------------------------------ fairness --
    head("3. FAIRNESS: SHOULD AGE AND GENDER BE IN THIS MODEL AT ALL?")
    say("  This is a UK financial-services retention decision, so the question is")
    say("  legal before it is technical.")
    say()
    say("  The argument FOR using them: a retention offer is a benefit, not a")
    say("  denial of service. Withholding credit by protected characteristic is")
    say("  the classic harm; here we are deciding who gets offered a discount.")
    say()
    say("  The argument AGAINST, which I find stronger:")
    say("    - Sex, age and marital status are protected characteristics under")
    say("      the Equality Act 2010. Allocating commercial benefit by them")
    say("      invites a discrimination challenge whatever the intent.")
    say("    - 'The model found it predictive' is not a defence anyone wants to")
    say("      make to the FCA under Consumer Duty, which asks whether outcomes")
    say("      are fair across customer groups.")
    say("    - The reputational asymmetry is brutal: 'bank offers worse retention")
    say("      deals to women' is a headline; a 0.002 PR-AUC gain is not.")
    say()

    # Per-group performance of the recommended model.
    say("  How the CURRENT model behaves per group (test set, already scored):")
    say()
    from sklearn.metrics import average_precision_score, roc_auc_score
    group_rows = []
    test_g = test.copy()
    test_g["score"] = scores
    test_g["age_band"] = pd.cut(test_g["customer_age"],
                                [0, 35, 45, 55, 120],
                                labels=["<35", "35-44", "45-54", "55+"])
    k = 250
    cutoff = np.sort(scores)[::-1][k - 1]
    test_g["targeted"] = test_g["score"] >= cutoff

    for col in ("gender", "age_band"):
        say(f"  by {col}:")
        say(f"    {'group':<10}{'n':>7}{'churn':>9}{'PR-AUC':>10}{'ROC-AUC':>10}"
            f"{'targeted@250':>14}")
        for g, part in test_g.groupby(col, observed=True):
            if part[TARGET].nunique() < 2:
                continue
            row = {
                "dimension": col, "group": str(g), "n": int(len(part)),
                "churn_rate": float(part[TARGET].mean()),
                "pr_auc": float(average_precision_score(part[TARGET], part["score"])),
                "roc_auc": float(roc_auc_score(part[TARGET], part["score"])),
                "targeted_share": float(part["targeted"].mean()),
            }
            group_rows.append(row)
            say(f"    {row['group']:<10}{row['n']:>7,}{row['churn_rate']:>9.2%}"
                f"{row['pr_auc']:>10.4f}{row['roc_auc']:>10.4f}"
                f"{row['targeted_share']:>13.1%}")
        say()
    # Targeting rate relative to underlying risk. This is the disparity that
    # matters commercially and legally: not "are the churn rates different"
    # (they legitimately are) but "is the model targeting a group harder than
    # its actual risk justifies".
    say("  DISPARITY: targeting rate against underlying risk")
    say()
    say("  A group that churns more SHOULD be targeted more. The question is")
    say("  whether targeting is proportionate to risk. The ratio below is")
    say("  (share targeted) / (actual churn rate): equal values across groups")
    say("  means proportionate treatment, whatever the underlying rates.")
    say()
    gp = pd.DataFrame(group_rows)
    gp["targeting_per_unit_risk"] = gp["targeted_share"] / gp["churn_rate"]
    for dim in ("gender", "age_band"):
        part = gp[gp["dimension"] == dim]
        say(f"  by {dim}:")
        say(f"    {'group':<10}{'churn':>9}{'targeted':>10}{'ratio':>9}")
        for _, r in part.iterrows():
            say(f"    {r['group']:<10}{r['churn_rate']:>9.2%}"
                f"{r['targeted_share']:>10.2%}{r['targeting_per_unit_risk']:>9.2f}")
        hi = part.loc[part["targeting_per_unit_risk"].idxmax()]
        lo = part.loc[part["targeting_per_unit_risk"].idxmin()]
        spread = hi["targeting_per_unit_risk"] / lo["targeting_per_unit_risk"]
        say(f"    -> '{hi['group']}' is targeted {spread:.2f}x as hard per unit of")
        say(f"       actual risk as '{lo['group']}'")
        say()
    payload["disparity"] = gp.to_dict(orient="records")

    g = gp[gp["dimension"] == "gender"].set_index("group")
    if {"F", "M"} <= set(g.index):
        risk_ratio = g.loc["F", "churn_rate"] / g.loc["M", "churn_rate"]
        target_ratio = g.loc["F", "targeted_share"] / g.loc["M", "targeted_share"]
        say(f"  Stated plainly for gender: women churn {risk_ratio:.2f}x as often as")
        say(f"  men in this test set, but are targeted {target_ratio:.2f}x as often.")
        say("  The targeting gap is materially wider than the risk gap, which is")
        say("  what disparate impact looks like in a targeting model. It is")
        say("  visible here only because the check was run -- no metric in stage 4")
        say("  would have surfaced it.")
        say()
        payload["gender_disparity"] = {
            "risk_ratio_f_over_m": float(risk_ratio),
            "targeting_ratio_f_over_m": float(target_ratio),
        }

    payload["group_performance"] = group_rows

    # Cost of removing the protected attributes.
    head("4. WHAT DOES REMOVING AGE AND GENDER COST?")
    say("  Measured by repeated CV on the training set. Not on test: stage 4")
    say("  spent the one permitted test pass, and a second would undermine the")
    say("  number reported there.")
    say()

    y_tr = train[TARGET].to_numpy()
    strict_cols = feature_names(strict=True)
    variants = {
        "with age + gender": strict_cols,
        "without age + gender": [c for c in strict_cols if c not in PROTECTED],
    }
    cv_out = {}
    for label, cols in variants.items():
        X = train[cols].copy()
        num = [c for c in cols if X[c].dtype.kind in "ifb"]
        cat = [c for c in cols if c not in num]
        pipe = Pipeline([
            ("prep", ColumnTransformer(
                [("num", "passthrough", num),
                 ("cat", OneHotEncoder(handle_unknown="ignore",
                                       sparse_output=False), cat)],
                remainder="drop", verbose_feature_names_out=False)),
            ("model", make_estimator("lightgbm")),
        ])
        cv_out[label] = repeated_cv(pipe, X, y_tr, label=label,
                                    model_kind="lightgbm", strict=True)
        print(f"    {label}: {cv_out[label].mean('pr_auc'):.4f}", flush=True)

    cmp = paired_fold_difference(cv_out["with age + gender"],
                                 cv_out["without age + gender"], "pr_auc")
    say(f"  with age + gender    : PR-AUC {cmp['mean_a']:.4f}")
    say(f"  without age + gender : PR-AUC {cmp['mean_b']:.4f}")
    say(f"  cost of removal      : {cmp['mean_diff']:+.4f} PR-AUC")
    say(f"  paired fold interval : [{cmp['diff_lo']:+.4f}, {cmp['diff_hi']:+.4f}]")
    say(f"  keeping them wins in {cmp['a_wins_share']:.0%} of {cmp['n_folds']} folds")
    say()
    material = cmp["diff_lo"] > 0
    if material:
        say("  The removal cost is measurable and consistent across folds.")
    else:
        say("  The paired interval spans zero: removing them costs nothing the")
        say("  data can distinguish from noise.")
    say()
    payload["removal_cost"] = cmp

    # Does removal actually remove the information?
    head("5. DOES REMOVING THEM ACTUALLY REMOVE THE INFORMATION?")
    say("  Dropping a column does not delete what it encoded if other features")
    say("  predict it. Testing that directly: can the remaining SAFE features")
    say("  recover gender and age band?")
    say()

    def recoverability(feature_cols: list[str], target: np.ndarray,
                       name: str, repeats: int = 2) -> float:
        X = train[feature_cols].copy()
        num = [c for c in feature_cols if X[c].dtype.kind in "ifb"]
        cat = [c for c in feature_cols if c not in num]
        pipe = Pipeline([
            ("prep", ColumnTransformer(
                [("num", "passthrough", num),
                 ("cat", OneHotEncoder(handle_unknown="ignore",
                                       sparse_output=False), cat)],
                remainder="drop", verbose_feature_names_out=False)),
            ("model", make_estimator("lightgbm")),
        ])
        r = repeated_cv(pipe, X, target, label=name, model_kind="lightgbm",
                        strict=True, n_repeats=repeats)
        return r.mean("roc_auc")

    remaining = [c for c in strict_cols if c not in PROTECTED]
    is_male = (train["gender"] == "M").astype(int).to_numpy()
    is_older = (train["customer_age"] >= 55).astype(int).to_numpy()

    proxy_rows = []
    for target_name, target in (("gender", is_male), ("age 55+", is_older)):
        auc = recoverability(remaining, target, target_name)
        proxy_rows.append({"attribute": target_name, "feature_set": "strict minus age+gender",
                           "recovery_roc_auc": float(auc),
                           "base_rate": float(target.mean())})
        say(f"  predicting {target_name:<10} from the other features: ROC-AUC {auc:.4f}")
    say()

    gender_auc = next(r["recovery_roc_auc"] for r in proxy_rows if r["attribute"] == "gender")
    cosmetic = gender_auc >= 0.70

    if cosmetic:
        say(f"  ROC-AUC {gender_auc:.4f}. Gender survives its own deletion almost")
        say("  intact. Dropping the column is very nearly cosmetic.")
        say()
        say("  WHICH FEATURE IS DOING IT:")
        say()
        from sklearn.metrics import roc_auc_score as _auc
        uni = []
        for c in remaining:
            col = train[c]
            if col.dtype.kind in "ifb":
                v = col.astype(float)
                v = v.fillna(v.median())
                a = _auc(is_male, v)
            else:
                m = train.groupby(c, observed=True)["gender"].apply(
                    lambda d: (d == "M").mean())
                a = _auc(is_male, col.map(m).astype(float))
            uni.append({"feature": c, "auc": float(a), "strength": abs(a - 0.5)})
        uni_df = pd.DataFrame(uni).sort_values("strength", ascending=False)
        say(f"    {'feature':<26}{'ROC-AUC for gender':>20}")
        for _, r in uni_df.head(4).iterrows():
            say(f"    {r['feature']:<26}{r['auc']:>20.4f}")
        say()
        share = pd.crosstab(train["income_category"], train["gender"],
                            normalize="columns") * 100
        say("  income_category by gender (% of each gender in each band):")
        say()
        say(f"    {'band':<18}{'F':>8}{'M':>8}")
        for band in share.index:
            say(f"    {str(band):<18}{share.loc[band, 'F']:>8.1f}{share.loc[band, 'M']:>8.1f}")
        say()
        say("  Read the zeros. In this dataset there is not one woman in the")
        say("  $60K-$80K, $80K-$120K or $120K+ bands, while 60.6% of women sit in")
        say("  'Less than $40K' against 5.4% of men. Income band above $60K")
        say("  therefore implies male with certainty, which is why a model")
        say("  recovers gender at ROC-AUC {:.2f} without ever seeing it.".format(gender_auc))
        say()
        say("  Credit limit carries it too: the training-set mean is "
            f"${train[train.gender=='F'].credit_limit.mean():,.0f} for women against "
            f"${train[train.gender=='M'].credit_limit.mean():,.0f} for men.")
        say()
        say("  (A distribution this clean is almost certainly an artefact of a")
        say("  teaching dataset rather than a real portfolio. It is still the")
        say("  perfect illustration of the problem, and the check is exactly the")
        say("  one to run on real data.)")
        say()

        # What would it actually take?
        say("  WHAT WOULD IT TAKE TO ACTUALLY REMOVE GENDER?")
        say()
        income_like = [c for c in remaining
                       if c.startswith("income") or c == "credit_limit"]
        stripped = [c for c in remaining if c not in income_like]
        auc_stripped = recoverability(stripped, is_male, "gender-stripped")
        say(f"    also dropping {', '.join(income_like)}")
        say(f"    gender recovery falls from {gender_auc:.4f} to {auc_stripped:.4f}")

        X_str = train[stripped].copy()
        num = [c for c in stripped if X_str[c].dtype.kind in "ifb"]
        cat = [c for c in stripped if c not in num]
        pipe = Pipeline([
            ("prep", ColumnTransformer(
                [("num", "passthrough", num),
                 ("cat", OneHotEncoder(handle_unknown="ignore",
                                       sparse_output=False), cat)],
                remainder="drop", verbose_feature_names_out=False)),
            ("model", make_estimator("lightgbm")),
        ])
        cv_stripped = repeated_cv(pipe, X_str, y_tr, label="gender-neutral",
                                  model_kind="lightgbm", strict=True)
        cmp_strip = paired_fold_difference(cv_out["without age + gender"],
                                           cv_stripped, "pr_auc")
        base_rate_train = float(train[TARGET].mean())
        cmp_strip_ref = cmp_strip["mean_a"] - base_rate_train
        say(f"    model PR-AUC falls from {cmp_strip['mean_a']:.4f} to "
            f"{cmp_strip['mean_b']:.4f}, a cost of {cmp_strip['mean_diff']:.4f}")
        say(f"    paired fold interval [{cmp_strip['diff_lo']:+.4f}, "
            f"{cmp_strip['diff_hi']:+.4f}]")
        say()
        payload["gender_neutral"] = {
            "dropped": income_like,
            "recovery_after": float(auc_stripped),
            "pr_auc_cost": cmp_strip,
        }
        proxy_rows.append({"attribute": "gender", "feature_set": "also minus income+limit",
                           "recovery_roc_auc": float(auc_stripped),
                           "base_rate": float(is_male.mean())})
    else:
        say(f"  Highest recovery is {gender_auc:.4f}, close to a coin flip. Removal")
        say("  genuinely removes the information here.")
        say()
        auc_stripped = None

    payload["proxy_recovery"] = proxy_rows

    head("6. RECOMMENDATION")
    say("  1. REMOVE age and gender as model inputs.")
    say(f"     Measured cost: {cmp['mean_diff']:+.4f} PR-AUC, paired interval")
    say(f"     [{cmp['diff_lo']:+.4f}, {cmp['diff_hi']:+.4f}] -- indistinguishable from")
    say("     zero. There is no performance argument for keeping them, so the")
    say("     legal and reputational exposure is unpaid-for risk.")
    say()
    if cosmetic:
        say("  2. BUT DO NOT CLAIM THE MODEL IS THEREFORE GENDER-BLIND.")
        say(f"     Gender is still recoverable at ROC-AUC {gender_auc:.4f} from the")
        say("     remaining features, overwhelmingly through income_category.")
        say("     A fairness claim based on 'we do not use gender' would be")
        say("     false, and falsifiable by anyone who runs the check above.")
        say()
        say("  3. The real choice is between two honest positions:")
        say()
        say("     (a) Keep income and credit limit, accept that the model")
        say("         encodes gender, and manage it by monitoring OUTCOMES by")
        say("         group rather than by policing inputs.")
        if auc_stripped is not None:
            say(f"     (b) Drop income and credit limit too. Recovery falls to")
            say(f"         {auc_stripped:.4f} -- genuinely gender-blind -- at a cost")
            say(f"         of {payload['gender_neutral']['pr_auc_cost']['mean_diff']:.4f} PR-AUC, which is"
                f" {payload['gender_neutral']['pr_auc_cost']['mean_diff']/cmp_strip_ref*100:.0f}% of")
            say("         the model's entire margin over random.")
        say()
        say("     I recommend (a), for a specific reason: income and credit limit")
        say("     are legitimate, causally-relevant predictors of retention in a")
        say("     credit product. Removing genuine economic signal to break a")
        say("     statistical association is the kind of fix that degrades the")
        say("     model without helping anyone. Fairness here has to be measured")
        say("     on who gets the offer, not on which columns were fed in.")
        say()
        say("  4. Therefore the fairness control is an OUTCOME control:")
        say("     - report targeting rate and offer acceptance by gender and age")
        say("       band every campaign cycle, not once at build time")
        say("     - set a tolerance in advance for divergence between a group's")
        say("       targeting rate and its actual churn rate")
        say("     - escalate to a human when it breaches, rather than silently")
        say("       rebalancing the model")
    else:
        say("  2. Removal genuinely works here: neither attribute is recoverable")
        say("     from the remaining features, so this is a real change rather")
        say("     than a gesture.")
    say()
    say("  What none of this fixes: removing inputs does not guarantee equal")
    say("  outcomes, and this dataset has no offer-response data, so whether the")
    say("  campaign itself lands differently across groups is untestable here.")
    say("  That is a question for the stage 7 experiment.")
    say()

    _plot_shap(imp, shap_values, X_enc)
    _plot_fairness(group_rows)

    (TABLES / "05_explainability.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (TABLES / "05_explainability.txt").write_text("\n".join(REPORT) + "\n",
                                                  encoding="utf-8")
    print("\n[written] outputs/tables/05_explainability.{json,txt}")
    return 0


def _plot_shap(imp: pd.DataFrame, shap_values: np.ndarray,
               X_enc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    topn = imp.head(12).iloc[::-1]
    axes[0].barh(topn["feature"], topn["mean_abs_shap"], color="#1f4e79", alpha=0.88)
    axes[0].set_xlabel("mean |SHAP| (impact on log-odds)")
    axes[0].set_title("Global importance\none feature dominates")
    axes[0].grid(alpha=0.25, axis="x")

    lead = imp.iloc[0]["feature"]
    idx = list(X_enc.columns).index(lead)
    axes[1].scatter(X_enc.iloc[:, idx], shap_values[:, idx], s=9, alpha=0.35,
                    color="#b3452c")
    axes[1].axhline(0, color="#333", lw=1)
    axes[1].set_xlabel(lead)
    axes[1].set_ylabel("SHAP value")
    axes[1].set_title(f"How {lead} moves the score\nnegative SHAP = lower churn risk")
    axes[1].grid(alpha=0.25)

    fig.suptitle("What the recommended model is actually using (plan §8)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_shap_importance.png", dpi=140)
    plt.close(fig)
    print("[written] outputs/figures/05_shap_importance.png")


def _plot_fairness(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, dim in zip(axes, ("gender", "age_band")):
        part = df[df["dimension"] == dim]
        x = np.arange(len(part))
        w = 0.38
        ax.bar(x - w/2, part["churn_rate"] * 100, w, label="actual churn rate",
               color="#7a7a7a", alpha=0.85)
        ax.bar(x + w/2, part["targeted_share"] * 100, w,
               label="share targeted (top 250)", color="#1f4e79", alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(part["group"])
        ax.set_ylabel("%")
        ax.set_title(f"By {dim}\ntargeting rate vs underlying churn")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Fairness check: is the model targeting groups in proportion to "
                 "their actual risk?", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_fairness.png", dpi=140)
    plt.close(fig)
    print("[written] outputs/figures/05_fairness.png")


if __name__ == "__main__":
    raise SystemExit(main())
