# Analysis Plan — Credit Card Churn Model

**Status:** Written and committed **before any modelling**.
**Author:** Aman Bhardwaj
**Date:** 2026-09-11

> Committed first so the git history shows the order decisions were made in. If
> something has to change later, it is logged in **§10 Deviations from Plan**
> with a reason and a date, rather than edited silently into the text above.

---

## 1. The business question

**Not** "can I predict churn?" — that is a modelling question.

> **Which credit-card customers should we spend retention budget on next month,
> and how much should we expect that spend to return?**

The distinction matters throughout. A model that ranks churn risk perfectly but
targets customers who would have stayed anyway, or who cannot be saved by any
offer, loses money. The deliverable is a **targeting policy with an expected
value attached**, not a leaderboard score.

### Why this framing for an SME card issuer

A churned SME card customer costs more than a churned consumer: acquisition is
slower, the relationship often carries multiple products, and interchange plus
interest on a business card is typically a larger annual contribution. So the
economics tilt toward intervening — but only where the intervention can plausibly
change the outcome.

---

## 2. Target definition

| | |
|---|---|
| **Raw column** | `Attrition_Flag` ∈ {`Existing Customer`, `Attrited Customer`} |
| **Modelled target** | `is_attrited` = 1 if `Attrited Customer`, else 0 |
| **Positive class** | Churned (the rare, expensive class) |
| **Base rate** | **16.07%** (1,627 of 10,127) |

### What this target does *not* tell us — stated up front

The dataset is a **single snapshot with no dates**. There is no observation
window, no churn date, and no "as of" timestamp. Three consequences, all of
which constrain the honest claims this project can make:

1. **No time-based split is possible.** The standard defence against leakage in
   churn modelling — train on the past, test on the future — is unavailable.
   §4 states what is done instead and why it is weaker.
2. **"Churn" is undefined in time.** We do not know whether a customer closed
   their account yesterday or a year ago, nor over what period the behavioural
   counts were accumulated relative to that event. This is the root cause of the
   leakage problem in §5.
3. **No revenue column.** Customer value cannot be measured, only assumed. §7
   makes those assumptions explicit and tests how much they can be wrong.

I would rather state these in advance than have an interviewer find them.

---

## 3. Metrics

### Primary metric: PR-AUC (average precision)

At a 16% base rate, ROC-AUC is the wrong headline. ROC-AUC rewards ranking
across the whole score range including the large, easy negative class, and looks
flattering when the positive class is rare. Precision-recall curves only care
about the positive class, which is the class the retention budget is spent on.

- **PR-AUC baseline = the base rate, 0.1607.** A random model scores that. This
  number is quoted alongside every PR-AUC so the figure cannot mislead.
- ROC-AUC is **also** reported, because it is what most stakeholders have seen
  before and refusing to show it is unhelpful. It is just not the decider.

### Calibration: Brier score + reliability curve

Ranking is not enough here. The profit calculation in §7 multiplies by
P(churn) as a **probability**, so a model that ranks well but says "0.9" when it
means "0.4" will produce a confidently wrong budget.

- **Brier score** (mean squared error of the probability). Lower is better.
- **Reliability curve** (predicted probability vs observed frequency, binned),
  plotted before and after calibration.
- If calibration is poor, fix it with **isotonic regression** (flexible, enough
  data here) or **Platt scaling** (safer on small samples), fitted **on the
  validation set only** — never on the test set, and never on the training set
  the model has already seen.

### Supporting

Precision, recall and F1 **at the chosen operating threshold** — not at the
default 0.5, which is an arbitrary artefact for an imbalanced problem. The
threshold comes from §7, not from convention.

### Uncertainty

All headline test metrics get **bootstrap 95% confidence intervals** (2,000
resamples of the test set). A point estimate of PR-AUC with no interval invites
over-reading a difference between two models that is within noise.

---

## 4. Validation scheme

**Stratified three-way split of 10,127 customers**, stratified on the target so
each part holds ≈16.07% churners:

| Split | Share | n (approx) | Purpose |
|---|---|---|---|
| **Train** | 60% | 6,076 | Fit models |
| **Validation** | 20% | 2,025 | Tune hyperparameters, pick threshold, fit calibration |
| **Test** | 20% | 2,026 | **Touched once**, at the end, to report final numbers |

- **Seed fixed** (`RANDOM_SEED = 20260911`) in one config module.
- **Hyperparameter tuning uses stratified k-fold cross-validation *within the
  training set*** (5-fold), not the validation set, so the validation set stays
  clean for threshold selection and calibration.
- **The test set is used exactly once.** No "let me just check" re-runs. If the
  test result disappoints, that is the result; tuning against it would make the
  reported number meaningless.

### The honest caveat

A random stratified split **assumes rows are exchangeable**. With no dates, that
assumption cannot be checked. A real deployment would be validated
out-of-time — train on customers as of January, test on churn observed by June —
and would very likely score **worse** than what this project reports, because a
random split cannot detect drift or period-specific effects. Any number here
should be read as an optimistic bound.

---

## 5. Leakage audit (Stage 3) — the decision that matters most

Because the snapshot has no dates, several behavioural features may have been
measured **during or after** the period in which churn occurred. If so, they are
not predictors; they are symptoms.

Every feature is classified into one of three buckets, with an argument for each:

- **SAFE** — plausibly known before the churn window (demographics, tenure,
  product holdings, credit limit).
- **SUSPECT** — 12-month behavioural aggregates that likely overlap the churn
  period (`Total_Trans_Ct`, `Total_Trans_Amt`, `Months_Inactive_12_mon`,
  `Contacts_Count_12_mon`, `Total_Revolving_Bal`, `Avg_Utilization_Ratio`).
- **LEAKED** — the two Naive Bayes columns. Dropped in staging, never modelled.
  (ROC-AUC 1.0000; see `data/PROVENANCE.md`.)

The Q4-vs-Q1 change ratios (`Total_Amt_Chng_Q4_Q1`, `Total_Ct_Chng_Q4_Q1`) are
the sharpest case: a collapse in Q4 spend relative to Q1 is close to a
*definition* of disengagement, so a model using it may be reading the churn
itself rather than predicting it.

**Plan:** train two model families —

| Version | Features |
|---|---|
| **`all`** | Everything except the leaked columns |
| **`strict`** | SAFE features only |

— and report the performance gap honestly.

**Pre-committed recommendation: the `strict` model, unless the gap is small
enough that the `all` model's extra features are clearly not doing leakage
work.** Writing this down now matters: after seeing a big PR-AUC number from the
`all` model it becomes very tempting to rationalise keeping it.

**I expect `strict` to score materially worse.** That is not a failure of the
project; it is the finding. A lower score that survives deployment is worth more
than a higher score that evaporates.

---

## 6. Baseline and models

A model is only "good" relative to something. Three reference points:

1. **Random / base rate** — PR-AUC 0.1607 by construction.
2. **Simple heuristic** — rank by a single obvious business rule (e.g. months
   inactive). This is what a competent analyst would do with no model, and is
   the bar a model must clear to justify its complexity.
3. **Logistic regression** — regularised, interpretable coefficients with odds
   ratios. Often close to the ceiling on tabular problems this size, and the
   thing a risk or compliance function can actually read.

Then **gradient boosting** (LightGBM, with XGBoost as a cross-check), tuned by
cross-validated random search within the training folds.

If boosting does not beat logistic regression by a margin that exceeds the
bootstrap CI, **the plan is to say so and prefer the simpler model.**

---

## 7. From score to decision (Stage 6)

The model outputs a probability. The business needs a list. The bridge:

```
expected_value(customer) = P(churn) × P(offer saves them) × customer_value − offer_cost
```

Target a customer when their expected value is positive; rank by it under a
budget constraint.

### The assumptions, stated because the data cannot supply them

The dataset has **no revenue**. So `customer_value` and `save_rate` are
**assumptions in `config/policy.yaml`**, not findings. Central estimates are
stated there with reasoning, and then:

- **Sensitivity analysis** sweeps both parameters across plausible ranges and
  shows how the optimal threshold moves.
- **Break-even analysis** reports how wrong each assumption can be before the
  policy stops paying — the single most useful number for a sceptical Head of
  Growth.
- **Comparison against "target everyone" and "target nobody"**, because a policy
  that cannot beat both of those trivial strategies is not worth deploying.

### The limitation I will not paper over

**P(churn) is not P(responds to offer).** Targeting the highest-risk customers
assumes risk and saveability coincide. They often do not: the highest-risk
customers may be the least saveable (already gone in spirit), and some low-risk
customers might be nudged into staying longer. The correct instrument is an
**uplift / CATE model**, which needs experimental data that does not exist here.
Stage 7 designs the experiment that would produce it.

---

## 8. Fairness (Stage 5)

`Gender`, `Customer_Age`, `Marital_Status` and `Income_Category` are all
present. For a UK financial-services retention decision this is not a
theoretical concern:

- A retention *offer* is arguably a benefit, so withholding it by protected
  characteristic is the risk, not granting it.
- Equality Act 2010 protected characteristics include sex, age and marital
  status. Using them to allocate commercial benefit invites a discrimination
  challenge even where the model is merely predictive.
- Proxies survive removal. Dropping `Gender` does not remove gender information
  if other features correlate with it, so removal will be **measured**, not
  assumed.

**Plan:** report SHAP global importance and per-group performance, then train a
variant with `Gender` and `Customer_Age` removed and **quantify the cost**. The
recommendation will weigh a small performance loss against the legal and
reputational exposure — and will say plainly if the cost is near zero, because
then the decision is easy.

---

## 9. What would change my mind

Stated in advance:

1. **If `strict` and `all` score about the same**, my leakage worry was
   overblown, and I would use the fuller feature set while saying why.
2. **If gradient boosting does not beat logistic regression** beyond the
   bootstrap CI, I ship the logistic model and treat the boosting work as a
   negative result worth reporting.
3. **If no threshold produces positive expected value** under central
   assumptions, the honest recommendation is *do not run this campaign* — and
   the project's output becomes the break-even analysis showing what would have
   to be true for it to pay.
4. **If calibration cannot be fixed**, the profit curve is unreliable and I
   would present rank-based targeting (top-N customers) instead of a
   probability-threshold policy.

---

## 10. Deviations from Plan

*Nothing below this line is written before the analysis. Every deviation is
logged with its reason and date.*

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-09-11 | 4, 6 | **Model selection moved off the single validation split onto repeated stratified CV (5-fold x 5 repeats) over train+validation.** The 60/20/20 structure and "test set touched once" rule are unchanged. | Simulated the evaluation before running it: at the test set's ~326 positives, a bootstrap 95% CI on PR-AUC is about +/-0.054 wide for a mid-strength model. Two models within ~0.1 PR-AUC of each other are therefore indistinguishable on this test set, and LightGBM vs logistic regression is very likely to fall inside that band. Averaging over 25 fits makes the *comparison* stable; the test set keeps its job of producing one unbiased final number with an honest interval. The fix is not a larger test set, which would only steal training data -- it is to stop asking the test set a question it cannot answer. |
| 2026-09-11 | 3 | Added **precision@k and lift@k** (k = 100 / 250 / 500 customers) as supporting metrics. PR-AUC remains primary. | A retention budget buys a fixed number of contacts, so precision@k is what the business feels. It is kept *secondary* deliberately: k depends on the budget, which is an assumption invented in section 7, whereas PR-AUC is threshold-free and cannot be flattered by choosing a convenient k. Assumption-free metric leads; assumption-dependent metric supports. |
| 2026-09-11 | 5 | Sharpened the rationale for preferring the `strict` model. The size of the `all`-vs-`strict` gap is reported as **the cost of caution, not as evidence about whether the suspect features are safe**. | A larger gap is equally consistent with "these features are genuinely predictive" and "these features contain the answer". The two are empirically indistinguishable, so the performance gap cannot settle the question; only reasoning about how the data was generated can, and with no dates in the snapshot that reasoning points to caution. The original wording ("unless the gap is small") implied the gap was evidence. It is not. |
| 2026-09-11 | 2 | Confirmed `"Unknown"` will be kept as its own category, with no imputation. | Measured before deciding: churn rate for `Unknown` vs known is +1.25pp (`Marital_Status`), +0.93pp (`Education_Level`), +0.84pp (`Income_Category`) against a 16.07% base. The signal is weak enough that the choice is low-stakes, which argues for the option that invents no data. |
| 2026-09-11 | 4 | Hyperparameter tuning runs on **train only** (not train+validation). | The stage-1 deviation moved model *selection* to repeated CV over train+validation, which was right for the coarse feature-set decision in stage 3. But stage 4 needs validation genuinely untouched by any fitting decision so it can host calibration and threshold choice. Tuning on train alone (6,075 rows, 5-fold) and confirming with 5x5 repeated CV keeps both properties. |
| 2026-09-11 | 3 | Calibration is now governed by an explicit **three-part acceptance rule** rather than "pick the better validation Brier": a calibrator must gain >= 0.002 validation Brier, must not reduce validation PR-AUC, and must leave >= 100 distinct scores. | The original rule was too permissive and chose isotonic everywhere. Isotonic is a step function: on validation it collapsed 2,019 LightGBM scores into **15 distinct values** to buy a 0.0015 Brier gain, and it reduced validation PR-AUC (0.2632 -> 0.2597) on the very data it was fitted to. Rule 3 is the operational one: stage 6 targets the top-k customers by score, and "the top 250" is undefined when only 15 score levels exist. Under the new rule the recommended model keeps its raw scores, and sigmoid is accepted only for the all-features LightGBM. **Disclosure on ordering:** I had already seen the first run's test metrics when I changed this rule. The justification is nonetheless independent of them and checkable on validation alone -- the 15-distinct-values collapse and the validation PR-AUC drop are both visible without the test set. I am recording that I saw test first rather than presenting the rule as if it had been designed in ignorance. |
| 2026-09-12 | 9 | **Section 9 item 3 fired.** The campaign does not pay at the pre-registered central assumptions, and the project's headline output becomes the break-even analysis, exactly as the plan said it would. | Break-even P(churn) is 0.533 and only one test customer scores above it, for GBP 2 of expected net value. More telling than the sign is the slack: the central assumptions sit 4.9%, 4.9% and 5.1% from the boundary on customer value, save rate and offer cost respectively, so the result is on the break-even line in all three directions at once. The assumptions were written into config/policy.yaml before any of this was computed, and have NOT been revised to reach a more comfortable answer. |
