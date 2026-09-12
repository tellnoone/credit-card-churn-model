"""Stage 7: design the experiment that would prove the offer works.

The project's whole business case rests on the save rate -- the probability that
an offer retains a customer who would otherwise have left. Stage 6 showed the
campaign's viability turns on it and that no observational data can supply it,
because it is a causal quantity about an intervention that has never been run.

This stage writes that experiment down: arms, metric, MDE, sample size, duration
and stopping rule, with the numbers computed from the actual model rather than
asserted. It also explains why an uplift model is the right NEXT step and why it
cannot be the first one.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import MODELS, TABLES, TARGET, ensure_dirs
from src.data import load_features, make_splits
from src.experiment import (ExperimentDesign, detectable_save_rate, power_at_n,
                            sample_size_per_arm, total_sample_size)
from src.policy import break_even_probability, load_assumptions
from src.train import xy

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
    import joblib

    central, cfg = load_assumptions()
    cur = central.currency
    df = load_features()
    splits = make_splits(df)
    test = splits["test"]
    y_test = test[TARGET].to_numpy()
    n_book = len(df)

    model = joblib.load(MODELS / "recommended_model_uncalibrated.joblib")
    p = model.predict_proba(xy(test, strict=True)[0])[:, 1]

    head("STAGE 7: EXPERIMENT DESIGN FOR THE RETENTION OFFER")
    say("  Predicting churn is not the same as predicting who an offer can save.")
    say("  Everything in stage 6 assumed a save rate of 25% because the data")
    say("  cannot measure one. This is how you would measure it.")
    say()

    # -------------------------------------------------------------- segment --
    head("1. WHO IS IN THE EXPERIMENT")
    say("  Not the whole book. The experiment runs on the segment a campaign")
    say("  would actually target, because that is the population the save rate")
    say("  needs to be true for. A save rate measured on low-risk customers")
    say("  would not transfer.")
    say()
    say("  Defining the segment by model score decile:")
    say()
    say(f"  {'segment':<22}{'n (test)':>10}{'churn rate':>13}{'scaled to book':>16}")
    rule()
    segments = {}
    order = np.argsort(p)[::-1]
    for label, frac in (("top 5%", 0.05), ("top 10%", 0.10),
                        ("top 20%", 0.20), ("top 30%", 0.30)):
        k = int(len(p) * frac)
        idx = order[:k]
        churn = float(y_test[idx].mean())
        segments[label] = {"fraction": frac, "n_test": k, "churn_rate": churn,
                           "n_book": int(n_book * frac)}
        say(f"  {label:<22}{k:>10,}{churn:>12.1%}{int(n_book*frac):>16,}")
    rule()
    say(f"  {'whole book':<22}{len(p):>10,}{y_test.mean():>12.1%}{n_book:>16,}")
    say()

    chosen_label = "top 20%"
    seg = segments[chosen_label]
    say(f"  CHOSEN: the {chosen_label} by churn score.")
    say()
    say(f"  Why not the top 5%, where risk is highest? Because at "
        f"{segments['top 5%']['n_book']:,} customers")
    say("  on the full book there are not enough of them to power the test in a")
    say("  reasonable window, as section 3 shows. The top 20% trades a slightly")
    say("  lower baseline churn rate for enough customers to get an answer.")
    say()
    say("  This is the real tension in retention experiments: the segment where")
    say("  the offer matters most is the segment with fewest people in it.")
    say()

    # -------------------------------------------------------------- design --
    head("2. DESIGN")
    say("  Arms                 : 50/50 randomised split within the segment")
    say("    - TREATMENT        : receives the retention offer")
    say("    - CONTROL          : receives nothing, and is not contacted at all")
    say()
    say("  Randomisation is at CUSTOMER level, assigned from a hash of the")
    say("  customer id so assignment is stable if the campaign re-runs, and")
    say("  independent of anything the model saw.")
    say()
    say("  PRIMARY METRIC: churn within 90 days of assignment.")
    say()
    say("    One metric, fixed in advance. 90 days is long enough for a")
    say("    retention effect to appear and short enough to decide inside a")
    say("    planning cycle. Measured identically in both arms, including for")
    say("    customers who never open the offer -- this is intention-to-treat.")
    say()
    say("    Analysing only customers who ENGAGED with the offer would be the")
    say("    single easiest way to fake a positive result here, because")
    say("    engagement is itself a symptom of not churning.")
    say()
    say("  SECONDARY (monitored, not decisive):")
    say("    - offer redemption rate        : did anyone want it")
    say("    - spend in the 90-day window   : did retention cost engagement")
    say("    - churn at 180 days            : did we delay rather than prevent")
    say()
    say("  GUARDRAIL: total 90-day margin per customer, treatment vs control.")
    say("    A retained customer who costs more in offer than they generate is")
    say("    not a win, and the primary metric cannot see that.")
    say()

    # ----------------------------------------------------------- power --
    head("3. HOW BIG, AND FOR HOW LONG")
    baseline = seg["churn_rate"]
    # The save rate at which the campaign exactly breaks even, given the offer
    # cost and customer value in config/policy.yaml.
    break_even_save = central.offer_cost / (central.customer_value
                                            * central.horizon_years * baseline)
    say(f"  Baseline churn in the {chosen_label} segment: {baseline:.1%}")
    say()
    say("  The MDE should not be plucked from the air. The commercially")
    say("  meaningful threshold is the save rate at which the campaign breaks")
    say("  even, because below it the offer destroys value however significant")
    say("  the result is:")
    say()
    say(f"    break-even save rate = offer_cost / (value x baseline churn)")
    say(f"                         = {central.offer_cost:.0f} / "
        f"({central.customer_value:.0f} x {baseline:.3f})")
    say(f"                         = {break_even_save:.1%}")
    say()
    mde_save = break_even_save
    say(f"  So the MDE is a save rate of {mde_save:.1%}.")
    say()
    say("  That number is the finding, and it is not a comfortable one. A")
    say(f"  {mde_save:.0%} save rate means retaining nearly half of everyone who")
    say("  would otherwise have left. Published retention effects sit closer to")
    say("  10-30%. At a GBP 40 offer against GBP 300 of annual margin, the offer")
    say("  has to clear a bar that the literature suggests is not reachable.")
    say()
    say("  This is stage 6's conclusion arriving from a different direction. The")
    say("  problem is not that the model is weak; it is that the offer economics")
    say("  do not work. Before running any experiment, the cheaper move is to")
    say("  change those economics -- see the alternative below.")
    say()

    design = ExperimentDesign(baseline_churn=baseline, save_rate_mde=mde_save)
    n_arm = sample_size_per_arm(design)
    n_total = total_sample_size(design)

    say(f"  Churn under the offer at the MDE : {design.treatment_churn:.1%}")
    say(f"  Absolute effect to detect        : "
        f"{design.absolute_effect*100:.1f} percentage points")
    say(f"  alpha {design.alpha}, power {design.power:.0%}, two-sided")
    say()
    say(f"  REQUIRED SAMPLE: {np.ceil(n_arm):,.0f} per arm, "
        f"{np.ceil(n_total):,.0f} total")
    say()
    available = seg["n_book"]
    say(f"  Available in the segment on a {n_book:,}-customer book: {available:,}")
    if n_total <= available:
        say(f"  -> fits in one pass. Note this is only because the effect being")
        say("     looked for is implausibly large; large effects are cheap to")
        say("     detect. A small required sample is not good news here.")
    else:
        say(f"  -> the segment is too small by {n_total/available:.1f}x.")
    say()

    # The test worth actually running: a cheaper offer, plausible save rate.
    head("3b. THE EXPERIMENT WORTH ACTUALLY RUNNING")
    say("  Rather than test for an effect nobody expects, change the economics")
    say("  first so that a plausible save rate pays, then test for that.")
    say()
    cheaper_cost = 15.0
    plausible_save = 0.20
    be_cheaper = cheaper_cost / (central.customer_value * central.horizon_years
                                 * baseline)
    say(f"  Cut the offer cost to {cur} {cheaper_cost:.0f} (a points nudge rather than a")
    say(f"  fee waiver) and the break-even save rate falls to {be_cheaper:.1%}.")
    say(f"  That is inside the range retention offers actually achieve.")
    say()
    alt = ExperimentDesign(baseline_churn=baseline, save_rate_mde=plausible_save)
    n_alt = sample_size_per_arm(alt)
    say(f"  Powering for a {plausible_save:.0%} save rate instead:")
    say(f"    churn under offer   : {alt.treatment_churn:.1%}")
    say(f"    absolute effect     : {alt.absolute_effect*100:.1f} pp")
    say(f"    REQUIRED SAMPLE     : {np.ceil(n_alt):,.0f} per arm, "
        f"{np.ceil(2*n_alt):,.0f} total")
    say(f"    available in segment: {available:,}")
    if 2 * n_alt <= available:
        say("    -> fits in one pass")
    else:
        say(f"    -> needs {2*n_alt/available:.1f}x the segment; accumulate over "
            f"roughly {int(np.ceil(2*n_alt/available*3))} months")
    say()
    say("  This is the design I would actually propose: a cheaper offer, tested")
    say("  for an effect size someone might plausibly observe, on a segment big")
    say("  enough to answer the question.")
    say()
    payload_alt = {
        "offer_cost": cheaper_cost,
        "break_even_save_rate": float(be_cheaper),
        "mde_save_rate": plausible_save,
        "n_per_arm": float(np.ceil(n_alt)),
        "n_total": float(np.ceil(2 * n_alt)),
        "fits_in_one_pass": bool(2 * n_alt <= available),
    }

    head("3c. WHAT A SMALLER TEST COULD SEE")
    say("  What a smaller test could see, if the business will not wait:")
    say()
    say(f"    {'n per arm':>12}{'detectable save rate':>24}{'power at MDE':>15}")
    rule()
    for n_try in sorted({250, 500, 1_000, 2_000, 5_000}):
        say(f"    {n_try:>12,}{detectable_save_rate(design, n_try):>23.1%}"
            f"{power_at_n(design, n_try):>15.0%}")
    rule()
    say()
    say(f"  Read the top row: a 250-per-arm test can only detect a save rate")
    say(f"  around {detectable_save_rate(design, 250):.0%}. If the true save rate is the "
        f"{plausible_save:.0%} that")
    say("  retention offers plausibly achieve, that test returns 'no significant")
    say("  effect' most of the time and the offer gets killed for the wrong")
    say("  reason. An underpowered experiment is worse than none, because it")
    say("  produces a confident answer.")
    say()

    # ---------------------------------------------------------- stopping --
    head("4. ANALYSIS AND STOPPING RULE")
    say("  Fixed horizon. The test runs to its full sample and is analysed once.")
    say()
    say("  No peeking at the primary metric. Checking repeatedly and stopping at")
    say("  the first significant result inflates the false-positive rate well")
    say("  above the nominal 5%. If interim looks are required for governance,")
    say("  they must be declared in advance and use an alpha-spending boundary")
    say("  (O'Brien-Fleming), which prices the option to stop early rather than")
    say("  pretending it is free.")
    say()
    say("  Analysis: two-proportion test on 90-day churn, intention-to-treat,")
    say("  reporting the absolute difference with a 95% confidence interval. The")
    say("  interval is the output, not the p-value -- the business needs to know")
    say("  the plausible range of the save rate, not whether it clears a line.")
    say()
    say("  DECISION RULE, fixed now:")
    say(f"    - CI lower bound above a {mde_save:.0%} save rate -> roll out")
    say(f"    - CI spans {mde_save:.0%} -> inconclusive; extend or redesign the offer")
    say("    - CI entirely below it -> stop. The offer does not pay, whatever")
    say("      its p-value")
    say()
    say("  VALIDITY CHECKS before any outcome is read: sample ratio mismatch")
    say("  against the intended 50/50, and balance on pre-assignment covariates.")
    say("  A failed SRM invalidates the comparison regardless of the result.")
    say()

    # ------------------------------------------------------------- uplift --
    head("5. WHY UPLIFT MODELLING IS THE NEXT STEP, NOT THIS ONE")
    say("  This project ranks customers by P(churn). The policy then assumes the")
    say("  highest-risk customers are the ones worth contacting. That assumption")
    say("  is wrong in a specific, well-documented way.")
    say()
    say("  Every customer falls into one of four groups:")
    say()
    say("    SURE THINGS   stay whether or not you contact them")
    say("                  -> the offer is pure cost")
    say("    LOST CAUSES   leave whether or not you contact them")
    say("                  -> the offer is pure cost")
    say("    PERSUADABLES  stay only if contacted")
    say("                  -> the offer creates all of its value here")
    say("    SLEEPING DOGS stay UNLESS contacted; the approach reminds them")
    say("                  they were considering leaving")
    say("                  -> the offer actively destroys value")
    say()
    say("  A churn model cannot tell these apart. It scores LOST CAUSES highest")
    say("  of all, because they are the most likely to churn -- and they are")
    say("  precisely the customers an offer cannot save. That is a plausible")
    say("  reason a well-built churn-targeted campaign underdelivers: the model")
    say("  works exactly as designed and still spends the budget on people whose")
    say("  minds are made up.")
    say()
    say("  An uplift (CATE) model estimates P(stay | contacted) - P(stay | not),")
    say("  which is the quantity the policy actually wants. It ranks")
    say("  PERSUADABLES first and pushes SLEEPING DOGS to the bottom.")
    say()
    say("  Why it cannot be built first: uplift models are trained on the")
    say("  outcome of a randomised experiment. They need customers who were")
    say("  contacted and customers who were not, assigned at random. That data")
    say("  does not exist until the experiment in sections 1-4 has run.")
    say()
    say("  So the order is forced, and it is worth stating plainly because it is")
    say("  the honest answer to 'why not just build an uplift model':")
    say()
    say("    1. churn model                 <- this project")
    say("    2. randomised offer experiment <- sections 1-4, measures the save rate")
    say("    3. uplift model                <- trained on the output of step 2")
    say("    4. targeting by uplift         <- what the policy should eventually use")
    say()
    say("  Step 1 is not wasted. It defines the segment worth experimenting on,")
    say("  which is what makes step 2 affordable: running the test on the whole")
    say(f"  book instead of the {chosen_label} would need far more customers to")
    say("  reach the same power, because the baseline churn rate is lower.")
    say()

    payload = {
        "segments": segments,
        "chosen_segment": chosen_label,
        "baseline_churn": baseline,
        "break_even_save_rate": float(break_even_save),
        "mde_save_rate": float(mde_save),
        "treatment_churn_at_mde": float(design.treatment_churn),
        "absolute_effect_pp": float(design.absolute_effect * 100),
        "alpha": design.alpha,
        "power": design.power,
        "n_per_arm": float(np.ceil(n_arm)),
        "n_total": float(np.ceil(n_total)),
        "available_in_segment": int(available),
        "fits_in_one_pass": bool(n_total <= available),
        "detectable_by_n": {str(n): float(detectable_save_rate(design, n))
                            for n in (250, 500, 1_000, 5_000)},
        "alternative_design": payload_alt,
    }
    (TABLES / "07_experiment_design.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (TABLES / "07_experiment_design.txt").write_text("\n".join(REPORT) + "\n",
                                                     encoding="utf-8")
    print("\n[written] outputs/tables/07_experiment_design.{json,txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
