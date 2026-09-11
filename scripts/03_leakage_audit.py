"""Stage 3: leakage audit (ANALYSIS_PLAN.md section 5).

Trains two feature sets -- `all` (SAFE + SUSPECT) and `strict` (SAFE only) --
with two model families each, and reports the gap.

The framing matters and is fixed in the plan before any of this ran: **the gap
is the cost of caution, not evidence about whether the suspect features are
safe.** A larger gap is equally consistent with "these features are genuinely
predictive" and "these features contain the answer". The two are empirically
indistinguishable, so only reasoning about how the data was generated can settle
it -- and with no dates in the snapshot, that reasoning points to caution.

Uses repeated stratified CV over train+validation. The test set is not touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CV_FOLDS, CV_REPEATS, TABLES, TARGET, ensure_dirs
from src.data import load_features, make_splits
from src.evaluate import paired_fold_difference, repeated_cv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import FEATURES, Leakage, feature_names, features_by_class
from src.train import make_estimator, make_pipeline, xy

REPORT: list[str] = []


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


def main() -> int:
    ensure_dirs()
    df = load_features()
    splits = make_splits(df)

    # Model selection and the audit run on train+validation. The test set is
    # reserved for stage 4's single final evaluation.
    dev = pd.concat([splits["train"], splits["val"]], ignore_index=True)

    head("STAGE 3: LEAKAGE AUDIT")
    say(f"  Rows used (train + validation) : {len(dev):,}")
    say(f"  Churn rate                     : {dev[TARGET].mean():.4%}")
    say(f"  Test set                       : held back, {len(splits['test']):,} rows")
    say(f"  Scheme                         : {CV_FOLDS}-fold x {CV_REPEATS} repeats "
        f"= {CV_FOLDS * CV_REPEATS} fits per model")
    say()

    # ---------------------------------------------------------------- audit --
    head("1. FEATURE-BY-FEATURE CLASSIFICATION")
    safe = features_by_class(Leakage.SAFE)
    suspect = features_by_class(Leakage.SUSPECT)
    say(f"  SAFE    : {len(safe):>2} features -> used by BOTH models")
    say(f"  SUSPECT : {len(suspect):>2} features -> used by `all` only")
    say(f"  LEAKED  :  2 columns  -> dropped in dbt staging, never modelled")
    say()
    say("  The question asked of every feature: at the moment we would need to")
    say("  score a customer, does this value already exist, and is it free of")
    say("  information about the churn we are trying to predict?")
    say()

    for cls, items in (("SAFE", safe), ("SUSPECT", suspect)):
        rule()
        say(f"  {cls}")
        rule()
        for f in items:
            say(f"  * {f.name}")
            words = f.rationale.split()
            line = "      "
            for w in words:
                if len(line) + len(w) + 1 > 78:
                    say(line)
                    line = "      " + w
                else:
                    line = f"{line} {w}" if line.strip() else line + w
            if line.strip():
                say(line)
            say()

    # -------------------------------------------------------------- fitting --
    head("2. PERFORMANCE: `all` vs `strict`")
    say("  Repeated stratified CV. Identical folds for every model, so the")
    say("  comparisons below are paired.")
    say()

    results = {}
    for kind in ("logistic", "lightgbm"):
        for strict in (False, True):
            variant = "strict" if strict else "all"
            label = f"{kind}_{variant}"
            X, y = xy(dev, strict=strict)
            print(f"    fitting {label} ({X.shape[1]} features)...", flush=True)
            results[label] = repeated_cv(
                make_pipeline(kind, strict=strict), X, y,
                label=label, model_kind=kind, strict=strict,
            )
    say()

    base_rate = float(dev[TARGET].mean())
    say(f"  {'model':<20}{'features':>9}{'PR-AUC':>16}{'ROC-AUC':>16}{'Brier':>12}")
    rule()
    for label, res in results.items():
        n_feat = len(feature_names(strict=res.strict))
        say(f"  {label:<20}{n_feat:>9}"
            f"{res.mean('pr_auc'):>10.4f} ±{res.std('pr_auc'):<5.3f}"
            f"{res.mean('roc_auc'):>10.4f} ±{res.std('roc_auc'):<5.3f}"
            f"{res.mean('brier'):>12.4f}")
    rule()
    say(f"  {'random baseline':<20}{'-':>9}{base_rate:>10.4f}{'':6}{0.5:>10.4f}")
    say()
    say("  (± is the standard deviation across folds, not a confidence interval.)")
    say()

    # ----------------------------------------------------------------- gaps --
    head("3. THE COST OF CAUTION")
    gaps = {}
    for kind in ("logistic", "lightgbm"):
        a, b = results[f"{kind}_all"], results[f"{kind}_strict"]
        g = paired_fold_difference(a, b, "pr_auc")
        gaps[kind] = g
        rel = g["mean_diff"] / g["mean_b"] * 100 if g["mean_b"] else float("nan")
        say(f"  {kind}")
        say(f"    all    PR-AUC : {g['mean_a']:.4f}")
        say(f"    strict PR-AUC : {g['mean_b']:.4f}")
        say(f"    gap           : {g['mean_diff']:+.4f}  "
            f"({rel:+.1f}% relative to strict)")
        say(f"    paired fold interval : [{g['diff_lo']:+.4f}, {g['diff_hi']:+.4f}]")
        say(f"    `all` wins in {g['a_wins_share']:.0%} of {g['n_folds']} folds")
        say()

    # ------------------------------------------------------- single-feature --
    head("4. WHICH SUSPECT FEATURE CARRIES THE GAP?")
    say("  Each SUSPECT feature added to the strict set on its own, to see how")
    say("  much of the gap one column explains. A single feature that recovers")
    say("  most of the gap is the one to be most suspicious of.")
    say()

    strict_cols = feature_names(strict=True)
    y_dev = dev[TARGET].to_numpy()
    baseline_pr = results["lightgbm_strict"].mean("pr_auc")
    full_pr = results["lightgbm_all"].mean("pr_auc")
    total_gap = full_pr - baseline_pr

    single = []
    for f in suspect:
        cols = strict_cols + [f.name]
        X = dev[cols].copy()
        num = [c for c in cols if X[c].dtype.kind in "ifb"]
        cat = [c for c in cols if c not in num]
        prep = ColumnTransformer(
            [("num", "passthrough", num),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
            remainder="drop", verbose_feature_names_out=False)
        pipe = Pipeline([("prep", prep), ("model", make_estimator("lightgbm"))])
        res = repeated_cv(pipe, X, y_dev, label=f.name, model_kind="lightgbm",
                          strict=False, n_repeats=2)   # fewer repeats: 14 features
        pr = res.mean("pr_auc")
        single.append({
            "feature": f.name,
            "pr_auc": pr,
            "gain_over_strict": pr - baseline_pr,
            "share_of_total_gap": (pr - baseline_pr) / total_gap if total_gap else np.nan,
        })
        print(f"    +{f.name}: {pr:.4f}", flush=True)

    single_df = pd.DataFrame(single).sort_values("gain_over_strict", ascending=False)
    say()
    say(f"  strict-only LightGBM PR-AUC : {baseline_pr:.4f}")
    say(f"  all-features LightGBM PR-AUC: {full_pr:.4f}")
    say(f"  total gap                   : {total_gap:+.4f}")
    say()
    say(f"  {'feature added to strict':<28}{'PR-AUC':>10}{'gain':>10}{'% of gap':>11}")
    rule()
    for _, r in single_df.iterrows():
        say(f"  {r['feature']:<28}{r['pr_auc']:>10.4f}{r['gain_over_strict']:>+10.4f}"
            f"{r['share_of_total_gap']*100:>10.1f}%")
    rule()
    say()
    say("  NOTE: these use 5x2 CV rather than 5x5, for runtime. They are")
    say("  indicative of ordering, not precise estimates.")
    say()

    # ------------------------------------------- concentration within strict --
    head("4b. HOW MUCH OF THE STRICT MODEL IS ONE FEATURE?")
    say("  The strict model is weak. Before recommending it, it is worth asking")
    say("  where what little signal it has actually comes from.")
    say()

    relationship_cols = {"total_relationship_count", "products_per_tenure_year"}
    conc = {}
    for cols, key, desc in (
        (strict_cols, "strict_full", "strict (all SAFE features)"),
        ([c for c in strict_cols if c not in relationship_cols], "strict_minus_rel",
         "strict MINUS relationship-count features"),
        (["total_relationship_count"], "rel_only", "total_relationship_count ALONE"),
    ):
        X = dev[cols].copy()
        num = [c for c in cols if X[c].dtype.kind in "ifb"]
        cat = [c for c in cols if c not in num]
        prep = ColumnTransformer(
            [("num", "passthrough", num),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
            remainder="drop", verbose_feature_names_out=False)
        pipe = Pipeline([("prep", prep), ("model", make_estimator("lightgbm"))])
        r = repeated_cv(pipe, X, y_dev, label=key, model_kind="lightgbm",
                        strict=True, n_repeats=3)
        conc[key] = {"n_features": len(cols), "pr_auc": r.mean("pr_auc"),
                     "roc_auc": r.mean("roc_auc"), "description": desc}
        print(f"    {desc}: {r.mean('pr_auc'):.4f}", flush=True)

    say(f"  {'model':<44}{'features':>9}{'PR-AUC':>10}{'ROC-AUC':>10}")
    rule()
    say(f"  {'random baseline':<44}{'-':>9}{base_rate:>10.4f}{0.5:>10.4f}")
    for key in ("strict_full", "strict_minus_rel", "rel_only"):
        c = conc[key]
        say(f"  {c['description']:<44}{c['n_features']:>9}"
            f"{c['pr_auc']:>10.4f}{c['roc_auc']:>10.4f}")
    rule()
    say()
    lost = conc["strict_full"]["pr_auc"] - conc["strict_minus_rel"]["pr_auc"]
    alone_share = conc["rel_only"]["pr_auc"] / conc["strict_full"]["pr_auc"]
    say(f"  total_relationship_count on its own reaches "
        f"{conc['rel_only']['pr_auc']:.4f} PR-AUC --")
    say(f"  {alone_share:.0%} of what the full 18-feature strict model achieves.")
    say(f"  Removing it (and its derivative) costs {lost:.4f} and leaves the model")
    say(f"  at ROC-AUC {conc['strict_minus_rel']['roc_auc']:.4f}, which is close to random.")
    say()
    say("  This is uncomfortable and worth saying plainly: the strict model is")
    say("  essentially ONE feature, and that feature is the one flagged in")
    say("  src/features.py as the weakest SAFE call in the list. A customer who")
    say("  is winding down their banking relationship may well close other")
    say("  products BEFORE closing the card -- in which case product count is a")
    say("  symptom too, and the strict model's small advantage over random is")
    say("  partly leakage of a subtler kind.")
    say()
    say("  The remaining 17 SAFE features -- every demographic, tenure, income")
    say(f"  and card-tier field -- add {conc['strict_full']['pr_auc'] - conc['rel_only']['pr_auc']:.4f} PR-AUC between them.")
    say("  Demographics barely predict churn here. That is a real finding, not a")
    say("  modelling failure, and it is consistent with what retention teams")
    say("  generally observe: behaviour predicts churn, attributes do not.")
    say()

    # ----------------------------------------------------------- conclusion --
    head("5. RECOMMENDATION")
    lgb_gap = gaps["lightgbm"]["mean_diff"]
    lgb_rel = lgb_gap / gaps["lightgbm"]["mean_b"] * 100
    say(f"  Dropping the {len(suspect)} SUSPECT features costs {lgb_gap:+.4f} PR-AUC")
    say(f"  ({lgb_rel:+.1f}% relative) on the LightGBM model.")
    say()
    say("  RECOMMENDED MODEL: strict.")
    say()
    say("  The reasoning, stated so it can be argued with:")
    say()
    say("  1. The gap does NOT tell us whether the suspect features are safe.")
    say("     A big gap is exactly what genuine predictive power looks like, and")
    say("     also exactly what leakage looks like. The two are empirically")
    say("     indistinguishable from performance alone.")
    say()
    say("  2. What settles it is how the data was generated, and there the")
    say("     snapshot is silent: no churn date, no observation window. We")
    say("     cannot demonstrate that a '12-month' aggregate ends before the")
    say("     churn it predicts. `months_inactive_12_mon` is the clearest case")
    say("     -- for a churned customer, inactivity IS the churn.")
    say()
    say("  3. The asymmetry of being wrong. A strict model that is too cautious")
    say("     under-performs by a known, measured amount. An `all` model built")
    say("     on leakage looks excellent in validation and fails silently in")
    say("     production, after the campaign budget has been committed.")
    say()
    say("  4. The strict model is deployable as specified. Every feature in it")
    say("     is available at scoring time for a customer who has not churned.")
    say()
    rule()
    say("  BUT THE RECOMMENDATION COMES WITH A SERIOUS CAVEAT")
    rule()
    say()
    say(f"  The strict model is weak: PR-AUC {conc['strict_full']['pr_auc']:.4f} against a "
        f"{base_rate:.4f} base rate,")
    say(f"  ROC-AUC {conc['strict_full']['roc_auc']:.4f}. And per section 4b it is effectively one")
    say("  feature, which is itself of contestable safety.")
    say()
    say("  So the honest headline is NOT 'here is a churn model'. It is:")
    say()
    say("    This dataset cannot support a deployable churn model built only on")
    say("    features that are demonstrably known before churn. The 0.97 PR-AUC")
    say("    version is almost certainly reading the answer; the leak-free")
    say("    version barely beats guessing.")
    say()
    say("  That is a finding about the DATA, and it is the correct thing to")
    say("  report to a business. Shipping the 0.97 model would mean committing a")
    say("  retention budget against a number that will not survive contact with")
    say("  production.")
    say()
    say("  What a real deployment would do instead: keep the behavioural")
    say("  features, but compute them over a window that provably CLOSES before")
    say("  the prediction date -- transactions in months 1-6 predicting churn in")
    say("  months 7-12. That is standard practice and it works. It is impossible")
    say("  here only because this snapshot has no dates. The strict model is")
    say("  therefore a pessimistic lower bound, not an estimate of what a")
    say("  properly-built model would achieve.")
    say()
    say("  What would change this recommendation: a data dictionary showing the")
    say("  behavioural window closes before the churn observation window opens.")
    say("  That is a question for the data owner, not for the model.")
    say()

    # ------------------------------------------------------------- persist --
    payload = {
        "scheme": {"folds": CV_FOLDS, "repeats": CV_REPEATS,
                   "rows_dev": len(dev), "base_rate": base_rate},
        "counts": {"safe": len(safe), "suspect": len(suspect),
                   "total_features": len(FEATURES)},
        "cv": {label: res.summary(["pr_auc", "roc_auc", "brier"])
               for label, res in results.items()},
        "gaps": gaps,
        "single_feature": single_df.to_dict(orient="records"),
        "concentration": conc,
        "recommendation": "strict",
    }
    out = TABLES / "03_leakage_audit.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (TABLES / "03_leakage_audit.txt").write_text("\n".join(REPORT) + "\n",
                                                 encoding="utf-8")
    print(f"\n[written] {out.relative_to(TABLES.parents[1])}")
    print(f"[written] {(TABLES / '03_leakage_audit.txt').relative_to(TABLES.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
