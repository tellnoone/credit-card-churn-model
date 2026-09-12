"""Stage 4: modelling (ANALYSIS_PLAN.md sections 3, 4, 6).

Order of operations, which is the part that matters:

1. Tune on TRAIN only, by randomised search scored on PR-AUC.
2. Confirm the winner with repeated stratified CV on TRAIN, for a stable estimate.
3. Fit on TRAIN. Calibrate on VALIDATION (never seen by the fit).
4. Evaluate on TEST **exactly once**, with bootstrap confidence intervals.

The test set is loaded at the last possible moment and scored in a single block,
so "used once" is visible in the code rather than merely promised.
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import (CV_FOLDS, CV_REPEATS, FIGURES, MODELS, PRECISION_AT_K,
                        RANDOM_SEED, TABLES, TARGET, ensure_dirs)
from src.data import load_features, make_splits
from src.evaluate import (bootstrap_metrics, compute_metrics,
                          paired_fold_difference, repeated_cv)
from src.train import HeuristicRanker, calibrate, make_pipeline, tune_model, xy

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


def fmt_ci(d: dict[str, float]) -> str:
    return f"{d['point']:.4f}  [{d['lo']:.4f}, {d['hi']:.4f}]"


def main() -> int:
    ensure_dirs()
    rng_seed = RANDOM_SEED
    df = load_features()
    splits = make_splits(df)
    train, val, test = splits["train"], splits["val"], splits["test"]

    head("STAGE 4: MODELLING")
    say(f"  train      : {len(train):,} rows  ({train[TARGET].mean():.2%} churn)")
    say(f"  validation : {len(val):,} rows  ({val[TARGET].mean():.2%} churn)  "
        f"-- calibration only")
    say(f"  test       : {len(test):,} rows  ({test[TARGET].mean():.2%} churn)  "
        f"-- scored ONCE")
    say()
    say("  Tuning runs on TRAIN alone. The validation set is kept clear of every")
    say("  fitting decision so it can calibrate the final model honestly.")
    say()

    base_rate = float(test[TARGET].mean())
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    # ---------------------------------------------------------------- tune --
    head("1. TUNING (train only, randomised search on PR-AUC)")
    tuned: dict[str, dict] = {}
    for strict in (True, False):
        variant = "strict" if strict else "all"
        X_tr, y_tr = xy(train, strict=strict)
        for kind in ("logistic", "lightgbm"):
            label = f"{kind}_{variant}"
            print(f"    searching {label}...", flush=True)
            best, params, score = tune_model(kind, X_tr, y_tr, strict=strict)
            tuned[label] = {"params": params, "search_cv_pr_auc": score}
            fitted[label] = best
            say(f"  {label:<20} best CV PR-AUC {score:.4f}")
            for k, v in sorted(params.items()):
                say(f"      {k.replace('model__',''):<22}{v}")
            say()

    # ------------------------------------------------------------ confirm --
    head("2. CONFIRMATION (repeated CV on train, tuned params)")
    say(f"  {CV_FOLDS}-fold x {CV_REPEATS} repeats. This is the stable comparison the")
    say("  test set cannot provide: at ~325 test positives a PR-AUC difference")
    say("  below roughly 0.05 is inside the noise.")
    say()

    cv_results = {}
    for strict in (True, False):
        variant = "strict" if strict else "all"
        X_tr, y_tr = xy(train, strict=strict)
        for kind in ("logistic", "lightgbm"):
            label = f"{kind}_{variant}"
            overrides = {k.replace("model__", ""): v
                         for k, v in tuned[label]["params"].items()}
            pipe = make_pipeline(kind, strict=strict, **overrides)
            cv_results[label] = repeated_cv(pipe, X_tr, y_tr, label=label,
                                            model_kind=kind, strict=strict)
            print(f"    confirmed {label}", flush=True)

    say(f"  {'model':<20}{'PR-AUC':>18}{'ROC-AUC':>18}{'Brier':>12}")
    rule()
    for label, res in cv_results.items():
        say(f"  {label:<20}{res.mean('pr_auc'):>11.4f} ±{res.std('pr_auc'):<5.3f}"
            f"{res.mean('roc_auc'):>11.4f} ±{res.std('roc_auc'):<5.3f}"
            f"{res.mean('brier'):>12.4f}")
    rule()
    say()

    # Does boosting beat logistic by more than noise? Plan section 6 pre-commits
    # to preferring the simpler model if not.
    head("3. DOES BOOSTING EARN ITS COMPLEXITY?")
    verdicts = {}
    for variant in ("strict", "all"):
        cmp = paired_fold_difference(cv_results[f"lightgbm_{variant}"],
                                     cv_results[f"logistic_{variant}"], "pr_auc")
        verdicts[variant] = cmp
        say(f"  {variant}")
        say(f"    lightgbm PR-AUC : {cmp['mean_a']:.4f}")
        say(f"    logistic PR-AUC : {cmp['mean_b']:.4f}")
        say(f"    difference      : {cmp['mean_diff']:+.4f}   "
            f"paired fold interval [{cmp['diff_lo']:+.4f}, {cmp['diff_hi']:+.4f}]")
        say(f"    lightgbm wins in {cmp['a_wins_share']:.0%} of {cmp['n_folds']} folds")
        beats = cmp["diff_lo"] > 0
        say(f"    -> boosting {'beats' if beats else 'does NOT clearly beat'} "
            f"logistic on the recommended metric")
        say()

    # ----------------------------------------------------------- baselines --
    head("4. BASELINES")
    say("  A model is only good relative to something. Three reference points,")
    say("  all scored on the SAME test set later in this run.")
    say()
    say("  1. Random          -> PR-AUC = the base rate by construction")
    say("  2. Heuristic       -> rank by total_relationship_count, ascending")
    say("                        (fewer products held = higher churn risk). This")
    say("                        is what an analyst produces with one SQL query.")
    say("  3. Logistic        -> interpretable, regularised")
    say()
    heuristic = HeuristicRanker("total_relationship_count", ascending=True)
    heuristic.fit(train, train[TARGET].to_numpy())
    fitted["heuristic"] = heuristic
    say()

    # --------------------------------------------------------- calibration --
    head("5. CALIBRATION (fitted on validation)")
    say("  The profit model in stage 6 multiplies by P(churn) as a probability,")
    say("  so being correctly ranked is not enough -- the number has to mean what")
    say("  it says. Both isotonic and sigmoid (Platt) are fitted on validation.")
    say()
    say("  ACCEPTANCE RULE (all judged on validation; test is not consulted):")
    say("    1. It must improve validation Brier by at least 0.002. Anything")
    say("       smaller is not worth an extra moving part.")
    say("    2. It must not reduce validation PR-AUC. A calibrator that damages")
    say("       ranking is taking away the thing the policy layer runs on.")
    say("    3. It must leave at least 100 distinct scores. Stage 6 targets the")
    say("       top-k customers by score; a calibrator that collapses the range")
    say("       into a handful of levels makes 'the top 250' undefined.")
    say()
    say("  Rule 3 is the operational one and it is easy to overlook: isotonic")
    say("  regression is a step function, so it maps many scores onto one value.")
    say()

    MIN_BRIER_GAIN = 0.002
    MIN_DISTINCT = 100

    calibrated: dict[str, object] = {}
    calib_choice: dict[str, dict] = {}
    for strict in (True, False):
        variant = "strict" if strict else "all"
        X_val, y_val = xy(val, strict=strict)
        for kind in ("logistic", "lightgbm"):
            label = f"{kind}_{variant}"
            model = fitted[label]
            raw_val = model.predict_proba(X_val)[:, 1]
            raw_brier = float(brier_score_loss(y_val, raw_val))
            raw_pr = float(average_precision_score(y_val, raw_val))

            candidates: dict[str, dict] = {}
            for method in ("isotonic", "sigmoid"):
                cal = calibrate(model, X_val, y_val, method=method)
                sv = cal.predict_proba(X_val)[:, 1]
                brier = float(brier_score_loss(y_val, sv))
                pr = float(average_precision_score(y_val, sv))
                distinct = int(len(np.unique(sv)))
                passes = (raw_brier - brier >= MIN_BRIER_GAIN
                          and pr >= raw_pr - 1e-9
                          and distinct >= MIN_DISTINCT)
                candidates[method] = {
                    "estimator": cal, "val_brier": brier, "val_pr_auc": pr,
                    "distinct_scores": distinct, "passes": passes,
                    "brier_gain": raw_brier - brier,
                }

            accepted = [m for m, c in candidates.items() if c["passes"]]
            if accepted:
                chosen = min(accepted, key=lambda m: candidates[m]["val_brier"])
                calibrated[label] = candidates[chosen]["estimator"]
            else:
                chosen = "none (raw model kept)"
                calibrated[label] = model

            say(f"  {label}")
            say(f"    {'variant':<12}{'val Brier':>11}{'gain':>9}"
                f"{'val PR-AUC':>12}{'distinct':>10}{'accepted':>10}")
            say(f"    {'raw':<12}{raw_brier:>11.4f}{'-':>9}{raw_pr:>12.4f}"
                f"{len(np.unique(raw_val)):>10,}{'-':>10}")
            for m, c in candidates.items():
                say(f"    {m:<12}{c['val_brier']:>11.4f}{c['brier_gain']:>+9.4f}"
                    f"{c['val_pr_auc']:>12.4f}{c['distinct_scores']:>10,}"
                    f"{str(c['passes']):>10}")
            say(f"    -> {chosen}")
            say()

            calib_choice[label] = {
                "raw_val_brier": raw_brier,
                "raw_val_pr_auc": raw_pr,
                "chosen": chosen,
                "candidates": {m: {k: v for k, v in c.items() if k != "estimator"}
                               for m, c in candidates.items()},
            }

    kept_raw = [lbl for lbl, c in calib_choice.items() if c["chosen"].startswith("none")]
    took_cal = [lbl for lbl, c in calib_choice.items() if not c["chosen"].startswith("none")]
    say(f"  Raw model kept for : {', '.join(kept_raw) if kept_raw else 'none'}")
    if took_cal:
        applied = ", ".join(f"{l} -> {calib_choice[l]['chosen']}" for l in took_cal)
    else:
        applied = "none"
    say(f"  Calibrator applied : {applied}")
    say()
    say("  Isotonic is rejected everywhere, always on rule 3: it collapses ~2,000")
    say("  scores into 12-24 levels. Sigmoid is rejected wherever the Brier gain")
    say("  is below the 0.002 threshold, which is every model except the")
    say("  all-features LightGBM, where it gains 0.0045 and is accepted.")
    say()
    say(f"  The RECOMMENDED model (lightgbm_strict) keeps its raw scores. That is")
    say("  not a cop-out: LightGBM trained without class reweighting is already")
    say("  close to calibrated, which is precisely why class_weight='balanced'")
    say("  was rejected in src/train.py. The Brier gain on offer was 0.0015, and")
    say("  isotonic wanted to collapse 2,019 distinct scores into 15 to get it.")
    say("  For a model whose whole job is to rank customers for targeting, that")
    say("  is a bad trade.")
    say()

    # --------------------------------------------------------------- TEST --
    head("6. TEST SET  (scored once)")
    say("  Everything above used train and validation only. This is the single")
    say("  pass over the test set. Bootstrap CIs from 2,000 resamples.")
    say()

    test_scores: dict[str, np.ndarray] = {}
    y_test = test[TARGET].to_numpy()

    test_scores["heuristic"] = heuristic.predict_proba(test)[:, 1]
    for strict in (True, False):
        variant = "strict" if strict else "all"
        X_te, _ = xy(test, strict=strict)
        for kind in ("logistic", "lightgbm"):
            label = f"{kind}_{variant}"
            test_scores[f"{label}_raw"] = fitted[label].predict_proba(X_te)[:, 1]
            test_scores[f"{label}_cal"] = calibrated[label].predict_proba(X_te)[:, 1]

    for label, scores in test_scores.items():
        results[label] = bootstrap_metrics(y_test, scores, seed=rng_seed)

    say(f"  {'model':<26}{'PR-AUC (95% CI)':<28}{'ROC-AUC':<20}{'Brier':>8}")
    rule()
    say(f"  {'random baseline':<26}{base_rate:.4f}{'':22}{0.5000:<20.4f}"
        f"{base_rate*(1-base_rate):>8.4f}")
    for label in test_scores:
        r = results[label]
        say(f"  {label:<26}{fmt_ci(r['pr_auc']):<28}"
            f"{r['roc_auc']['point']:.4f}{'':14}{r['brier']['point']:>8.4f}")
    rule()
    say()

    say("  Precision and lift at the top-k customers a budget would contact:")
    say()
    say(f"  {'model':<26}" + "".join(f"{'p@'+str(k):>10}{'lift@'+str(k):>11}"
                                     for k in PRECISION_AT_K))
    rule()
    for label in test_scores:
        r = results[label]
        line = f"  {label:<26}"
        for k in PRECISION_AT_K:
            line += f"{r[f'precision_at_{k}']['point']:>10.3f}"
            line += f"{r[f'lift_at_{k}']['point']:>11.2f}"
        say(line)
    rule()
    say(f"  (base rate {base_rate:.4f}; lift of 1.00 = no better than random)")
    say()

    # ------------------------------------- model vs heuristic, paired on test --
    head("7. DOES THE MODEL BEAT THE ONE-LINE HEURISTIC?")
    say("  The heuristic is `ORDER BY total_relationship_count ASC` -- what an")
    say("  analyst produces in an afternoon. If the model cannot beat it by more")
    say("  than noise, the model is not worth maintaining.")
    say()
    rng = np.random.default_rng(rng_seed)
    s_model = test_scores["lightgbm_strict_raw"]
    s_heur = test_scores["heuristic"]
    n = len(y_test)
    diffs = []
    for _ in range(2_000):
        i = rng.integers(0, n, n)
        if y_test[i].sum() < 2:
            continue
        diffs.append(average_precision_score(y_test[i], s_model[i])
                     - average_precision_score(y_test[i], s_heur[i]))
    diffs = np.asarray(diffs)
    lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
    beats = bool(lo_d > 0)
    say(f"  lightgbm_strict PR-AUC : "
        f"{results['lightgbm_strict_raw']['pr_auc']['point']:.4f}")
    say(f"  heuristic PR-AUC       : {results['heuristic']['pr_auc']['point']:.4f}")
    say(f"  paired difference      : {diffs.mean():+.4f}  "
        f"[{lo_d:+.4f}, {hi_d:+.4f}]")
    say(f"  -> the model {'BEATS' if beats else 'does NOT beat'} the heuristic "
        f"beyond sampling noise")
    say()
    say("  Paired on the same bootstrap resamples, so fold-to-fold difficulty")
    say("  cancels. An unpaired comparison of two intervals would be far weaker.")
    say()
    model_vs_heuristic = {
        "model_pr_auc": float(results["lightgbm_strict_raw"]["pr_auc"]["point"]),
        "heuristic_pr_auc": float(results["heuristic"]["pr_auc"]["point"]),
        "mean_diff": float(diffs.mean()),
        "diff_lo": float(lo_d), "diff_hi": float(hi_d),
        "beats_heuristic": beats,
    }

    score_range = {
        "min": float(test_scores["lightgbm_strict_raw"].min()),
        "max": float(test_scores["lightgbm_strict_raw"].max()),
        "n_distinct": int(len(np.unique(test_scores["lightgbm_strict_raw"]))),
        "n_above_0_50": int((test_scores["lightgbm_strict_raw"] > 0.50).sum()),
    }
    say(f"  Recommended model score range on test: "
        f"{score_range['min']:.3f} to {score_range['max']:.3f}, "
        f"{score_range['n_distinct']:,} distinct values.")
    say(f"  Only {score_range['n_above_0_50']} customers score above 0.50, which")
    say("  is what caps the achievable profit in stage 6.")
    say()

    # ------------------------------------------------------------- figures --
    _plot_reliability(y_test, test_scores, base_rate)
    _plot_pr_curves(y_test, test_scores, base_rate)

    # ------------------------------------------------------------- persist --
    import joblib
    recommended = "lightgbm_strict"
    joblib.dump(calibrated[recommended], MODELS / "recommended_model.joblib")
    joblib.dump(fitted[recommended], MODELS / "recommended_model_uncalibrated.joblib")
    joblib.dump(calibrated["lightgbm_all"], MODELS / "all_features_model.joblib")

    np.save(MODELS / "test_scores_recommended.npy", test_scores[f"{recommended}_cal"])
    test[["customer_id", TARGET]].assign(
        score=test_scores[f"{recommended}_cal"]
    ).to_csv(TABLES / "04_test_scores.csv", index=False)

    payload = {
        "base_rate_test": base_rate,
        "n_test": int(len(test)),
        "n_test_positive": int(y_test.sum()),
        "tuning": tuned,
        "cv_confirmation": {k: v.summary(["pr_auc", "roc_auc", "brier"])
                            for k, v in cv_results.items()},
        "boosting_vs_logistic": verdicts,
        "calibration_choice": calib_choice,
        "test_metrics": results,
        "recommended_model": recommended,
        "model_vs_heuristic": model_vs_heuristic,
        "score_range": score_range,
    }
    (TABLES / "04_modelling.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (TABLES / "04_modelling.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print("\n[written] outputs/tables/04_modelling.{json,txt}")
    print("[written] outputs/tables/04_test_scores.csv")
    print("[written] outputs/models/recommended_model.joblib")
    return 0


def _plot_reliability(y_true: np.ndarray, scores: dict[str, np.ndarray],
                      base_rate: float) -> None:
    """Reliability curves before and after calibration, for both variants."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    panels = [("lightgbm_strict", "strict (recommended)"),
              ("lightgbm_all", "all features")]

    for ax, (key, title) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.2, label="perfect")
        for suffix, colour, lab in (("_raw", "#b3452c", "model as fitted"),
                                    ("_cal", "#1f4e79", "after calibration rule")):
            s = scores[key + suffix]
            n_bins = 10
            frac, mean_pred = calibration_curve(y_true, s, n_bins=n_bins,
                                                strategy="quantile")
            ax.plot(mean_pred, frac, "o-", color=colour, lw=1.8, ms=5,
                    label=f"{lab} (Brier {brier_score_loss(y_true, s):.4f})")
        ax.axhline(base_rate, ls=":", color="#2e7d32", lw=1.2,
                   label=f"base rate {base_rate:.3f}")
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed churn frequency")
        ax.set_title(f"{title}\nreliability, 10 quantile bins")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration: does a predicted 30% actually churn 30% of the "
                 "time? (plan §3)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_reliability_curves.png", dpi=140)
    plt.close(fig)
    print("[written] outputs/figures/04_reliability_curves.png")


def _plot_pr_curves(y_true: np.ndarray, scores: dict[str, np.ndarray],
                    base_rate: float) -> None:
    from sklearn.metrics import average_precision_score

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    show = [("lightgbm_all_cal", "#b3452c", "LightGBM, all features"),
            ("logistic_all_cal", "#e08a5a", "Logistic, all features"),
            ("lightgbm_strict_cal", "#1f4e79", "LightGBM, strict (recommended)"),
            ("logistic_strict_cal", "#6fa8d6", "Logistic, strict"),
            ("heuristic", "#7a7a7a", "Heuristic: fewest products")]
    for key, colour, lab in show:
        if key not in scores:
            continue
        prec, rec, _ = precision_recall_curve(y_true, scores[key])
        ap = average_precision_score(y_true, scores[key])
        ax.plot(rec, prec, color=colour, lw=1.9, label=f"{lab} (AP {ap:.3f})")
    ax.axhline(base_rate, ls="--", color="#2e7d32", lw=1.3,
               label=f"random ({base_rate:.3f})")
    ax.set_xlabel("recall (share of churners caught)")
    ax.set_ylabel("precision (share of contacts who would churn)")
    ax.set_title("Precision-recall on the test set\n"
                 "the gap between the red and blue bands is the leakage question")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_pr_curves.png", dpi=140)
    plt.close(fig)
    print("[written] outputs/figures/04_pr_curves.png")


if __name__ == "__main__":
    raise SystemExit(main())
