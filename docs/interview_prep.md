# Interview prep

The fifteen questions most likely to come up, with answers short enough to say
out loud. The weak points are in here too — better to have an answer ready than
to be found out mid-interview.

---

## The numbers to have memorised

| | |
|---|---|
| Dataset | 10,127 customers, **16.07%** churn |
| Recommended model | LightGBM, 18 "strict" features |
| Test PR-AUC | **0.2844** [0.2445, 0.3345] vs **0.1604** base rate |
| Test ROC-AUC | 0.6675 · Brier 0.1278 · lift@100 **2.43** |
| All-features model | PR-AUC **0.9724** — and not trustworthy |
| Leakage gap | **0.7288** PR-AUC |
| Break-even churn probability | **53.3%**, model maxes at 56.1% |
| Campaign at £40 offer | 1 customer, **£2** |
| Campaign at £15 offer | 2,644 customers, **£13,543/yr** |
| Cost of the leakage audit, if skipped | **£47,461/yr** of imaginary business case |

---

## 1. Walk me through the project in two minutes.

I built a churn model for a credit-card book, but the deliverable was a
targeting decision with a pound figure, not an accuracy score.

The first model scored 0.97 PR-AUC. That was the warning, not the win. The
dataset has no dates, so for any "last 12 months" feature I couldn't show the
window closed before the churn it predicts. I split features into ones knowable
in advance and ones that might be measuring the churn itself, trained both, and
the honest model scored 0.28.

Then I costed it. At a £40 retention offer and £300 of annual customer margin,
a customer is only worth contacting above a 53% churn probability. The model
maxes out at 56%. One customer qualifies, for £2.

So the recommendation was: don't run the campaign, cut the offer price — at £15
it's worth about £13,500 a year — and run a proper experiment to measure the one
number the whole case depends on, which is how often an offer actually changes
someone's mind.

## 2. You spent all that effort and concluded "don't do it". Isn't that a failure?

The opposite. The alternative outcome was shipping the 0.97 model, which would
have projected **£47,000 a year** and delivered nothing, because it was reading
the answer. Catching that *is* the value.

And "don't do it" isn't where I stopped. The output is a break-even condition:
the campaign pays if customer value exceeds £285, or the save rate exceeds 24%,
or the offer costs under £42. Those are levers the business controls. I'd rather
hand a Head of Growth three things they can change than a profit curve built on
a number I invented.

The thing I'd actually defend hardest: a model that says "no" for the right
reasons is more useful than one that says "yes" for the wrong ones — and in a
regulated lender, considerably cheaper.

## 3. How did you know the features were leaking rather than just predictive?

I didn't, and that's the honest answer — **you can't tell from performance.** A
large gap is equally consistent with "these features are genuinely predictive"
and "these features contain the answer". They're empirically indistinguishable.

So the decision has to rest on how the data was generated, and there the dataset
is silent: no churn date, no observation window. `months_inactive_12_mon` is the
clearest case — for a churned customer, inactivity *is* the churn.

Two things made me confident anyway. `total_trans_amt` alone recovers **79%** of
the gap, taking PR-AUC from 0.24 to 0.82. One spend column doesn't do that
legitimately. And two columns in the raw file scored **ROC-AUC exactly 1.0000**
with non-overlapping ranges — someone else's model output, shipped as features.

I also wrote the recommendation down *before* running the comparison, precisely
because a 0.97 number is very persuasive after the fact.

## 4. Your assumptions look engineered to produce a negative result.

Fair challenge, and I'd make it too. Three defences.

They were committed to `config/policy.yaml` with written reasoning **before**
anything was computed — that's in the git history, not just my word. I used a
one-year horizon rather than lifetime value, which is the conservative choice
and the opposite of how retention business cases usually get inflated. And the
sensitivity grid sweeps £100–£800 and 5–50% save rates, so you can read off the
answer at whatever assumptions you prefer.

The result I'd actually defend isn't the sign, it's the **slack**: the central
case sits 4.9%, 4.9% and 5.1% from its break-even boundary on the three
parameters. It's on the line in all three directions at once. If my assumptions
were engineered, they were engineered incompetently — a genuinely rigged case
wouldn't be this marginal.

## 5. Why PR-AUC rather than ROC-AUC or accuracy?

Accuracy is useless at a 16% base rate — predict "nobody churns" and you're 84%
accurate.

ROC-AUC is better but flatters imbalanced problems, because it rewards ranking
across the large easy negative class. PR-AUC only cares about the positive
class, which is the one the budget is spent on. I always quote the base rate
(0.1604) beside it so the number can't mislead.

I report ROC-AUC too, because stakeholders expect it and refusing to show it is
unhelpful. It just doesn't decide anything.

I also chose PR-AUC over precision@k deliberately, even though precision@k is
closer to how a campaign works. PR-AUC is threshold-free, so it can't be
flattered by picking a convenient k. The assumption-free metric leads; the
assumption-dependent one supports.

## 6. Why LightGBM over logistic regression? How do you know it was worth it?

I tested it rather than assumed it. Repeated 5×5 cross-validation, paired on
identical folds: LightGBM 0.2701 vs logistic 0.2389, paired interval
**[+0.0063, +0.0641]**, winning 25 of 25 folds.

The interval excludes zero, so it's a real gain. I'd pre-committed in the plan
to shipping the logistic model if boosting hadn't cleared that bar — on a
problem this size it often doesn't, and "we used gradient boosting" is not a
justification.

I compared on cross-validation rather than the test set on purpose. With ~325
test positives, a PR-AUC difference under ~0.05 is inside the noise. The test
set can't arbitrate that; 25 folds can.

## 7. The strict model is basically one feature. Is it even useful?

Barely, and I say so in the README. `total_relationship_count` alone scores
0.2185 against the full 18-feature model's 0.2426 — **90% of it**. Strip it out
and ROC-AUC falls to 0.5709, near random.

It does beat the one-line SQL heuristic by a real margin (+0.0702 PR-AUC,
bootstrap CI [+0.0400, +0.1014]), so it earns its place. But against a low bar.

The uncomfortable part, which I flagged in the code *before* I found this: that
load-bearing feature is my weakest "safe" call. A customer winding down may
close other products before closing the card, which would make it a symptom too.

The conclusion I'd stand behind: **this dataset can't support a deployable churn
model on features demonstrably known before churn.** That's a finding about the
data. A real deployment would use behavioural features over a window that
provably closes before the prediction date — spend in months 1–6 predicting
churn in months 7–12. That's standard, it works, and it's impossible here only
because the snapshot has no dates.

## 8. Why didn't you use SMOTE or class weighting for the imbalance?

Because 16% isn't severe imbalance, and both would have damaged what I needed.

The profit calculation multiplies by P(churn) as a **probability**. Class
weighting deliberately distorts predicted probabilities away from the true base
rate — it improves separation at the cost of calibration. I'd have gained
nothing on ranking and broken the thing the business layer runs on.

SMOTE synthesises minority examples by interpolating between neighbours. On
tabular data with meaningful categoricals it invents customers who don't exist,
and it's well documented to help less than people expect once you evaluate on a
properly held-out set.

The imbalance is handled by choosing the right metric and the right threshold,
not by editing the data. The threshold comes from the profit calculation, not
from 0.5.

## 9. Explain calibration, and why you rejected isotonic.

Calibration is whether a predicted 30% actually churns 30% of the time.
Ranking isn't enough here because the expected-value formula uses the
probability itself — a model that ranks perfectly but says 0.9 when it means 0.4
produces a confidently wrong budget.

My first rule was "pick whichever calibrator has the better validation Brier".
That chose isotonic everywhere, and it was wrong. Isotonic is a step function:
it collapsed **2,019 distinct scores into 15** to buy a 0.0015 Brier
improvement, and it reduced PR-AUC on the very data it was fitted to.

The killer is operational — the policy targets the top-k customers by score, and
**"the top 250" is undefined across 15 score levels.**

So I rewrote the rule: a calibrator must gain ≥0.002 Brier, must not reduce
PR-AUC, and must leave ≥100 distinct scores. Under that, the recommended model
keeps its raw scores, which is legitimate because LightGBM without class
reweighting is already close to calibrated — the same reason I rejected class
weighting in question 8.

*(If pushed: I'd already seen test metrics when I changed that rule. It's
disclosed in the plan's deviation log. The justification is checkable on
validation alone, but the ordering is recorded rather than tidied away.)*

## 10. How would you deploy and monitor this?

The model is a fitted sklearn Pipeline, so preprocessing travels with it — no
train/serve skew from a forgotten scaler.

Monitoring, in priority order:

1. **Score distribution drift.** If the mean predicted probability moves, the
   break-even threshold targets a different number of people and the budget
   silently changes.
2. **Feature drift**, especially `total_relationship_count`, since 90% of the
   model rides on it. A change in how products are counted would break this
   model specifically.
3. **Calibration**, quarterly. Predicted vs actual churn by score decile. Drift
   here invalidates the profit calculation even if ranking holds.
4. **Targeting rate by gender and age band** — the disparity in question 11 is a
   live risk, not a build-time check.

Retraining cadence: I'd rather trigger on drift than on a calendar. And the
first thing I'd build isn't the model, it's the **date-stamped feature store**
that makes the leakage question answerable. That's the actual blocker.

## 11. You found the model discriminates. What would you actually do?

Three findings, and they don't point the same way.

Removing age and gender costs **−0.0002 PR-AUC** — nothing. So there's no
performance argument for keeping protected characteristics, and I'd drop them.

But that doesn't make the model gender-blind. **Gender is recoverable at ROC-AUC
0.9552** from the remaining features, almost entirely through `income_category`
— there isn't one woman above $60K in this dataset. Claiming fairness because
"we don't use gender" would be false and trivially falsifiable.

And the outcome disparity is real: women churn 1.17× as often as men but are
targeted **1.77×** as often. The 55+ band has the **lowest** churn of any band
and the **highest** targeting rate.

What I'd do: drop age and gender, keep income and credit limit as legitimate
economic predictors, and move the fairness control from inputs to outcomes —
report targeting rate against actual risk by group every cycle, with a tolerance
agreed in advance and escalation to a human when it breaches.

Going genuinely gender-blind means dropping income and credit limit too. That
works — recovery falls to 0.5116 — but costs 0.0329 PR-AUC, which is **40% of
the model's entire margin over random**. I'd put that trade to Compliance rather
than decide it alone, and I'd say so in the room.

## 12. What's the difference between churn modelling and uplift modelling?

A churn model ranks by P(leave). An uplift model ranks by
P(stay | contacted) − P(stay | not contacted) — the *effect of the offer*, which
is what a targeting decision actually wants.

Four customer types: **sure things** stay regardless, **lost causes** leave
regardless, **persuadables** stay only if contacted, and **sleeping dogs** stay
*unless* you contact them and remind them they were thinking of leaving.

A churn model can't separate these, and it ranks **lost causes highest of all** —
exactly the customers an offer can't save. That's a plausible reason
well-built churn campaigns underdeliver: the model works as designed and still
spends the budget on people whose minds are made up.

Why I didn't build one: uplift models train on randomised experiment data.
Contacted and not-contacted customers, assigned at random. That data doesn't
exist until the experiment runs. The order is forced — churn model, experiment,
uplift model — and the churn model isn't wasted, because it defines the segment
worth experimenting on, which is what makes the experiment affordable.

## 13. Why dbt and DuckDB for a 10,000-row CSV? Isn't that over-engineering?

For this dataset alone, yes — pandas would do it in twenty lines.

I did it because the transformation layer is where leakage gets introduced, and
SQL with tests makes that layer reviewable by people who won't read my Python.
44 dbt tests cover uniqueness, nulls and accepted values, and the leaked columns
are dropped by an explicit `select` list so they *can't* reach the feature layer
by oversight.

It also mirrors how this actually works in a fintech: the warehouse is the source
of truth, the analytics engineer owns the staging layer, and the data scientist
picks up a tested feature table. Building it that way at 10k rows is cheap
practice for building it that way at 10 million.

I'd add that I verified the tests can actually fail — I broke the row-count
expectation deliberately and confirmed it errored. A test suite nobody has seen
fail is decoration.

## 14. What's the weakest part of this analysis?

Three, in order.

**The model is close to useless on its own.** PR-AUC 0.28 against a 0.16 base
rate, resting on one feature whose classification I'm not certain about. If
`total_relationship_count` is a symptom rather than a predictor, there's almost
nothing left.

**The save rate is invented.** 25% is a literate guess, and the entire business
case turns on it. I've been explicit about that everywhere, but no amount of
sensitivity analysis substitutes for measuring it.

**The random split is optimistic.** With no dates I can't validate out-of-time,
which is the split that matters for churn. A real deployment would score worse
than anything I've reported, and I'd treat my numbers as an upper bound.

A fourth, smaller: the fairness result rests on a dataset where zero women earn
above $60K, which is not a real portfolio. The *method* transfers; that
particular finding doesn't.

## 15. What would you do differently with three more months and real data?

First month I wouldn't model at all. I'd get date-stamped features and build a
proper observation/outcome window — spend in months 1–6 predicting churn in
months 7–12. That single change makes the leakage question answerable and is
worth more than any algorithm choice.

Second, get revenue per customer from Finance. Customer value is the parameter
with the widest plausible range and it's a number the business already has.

Third, run the experiment in Stage 7 — 826 per arm at a £15 offer — to measure
the save rate. Then build the uplift model on its output, which is when the
targeting actually gets good.

What I *wouldn't* spend time on: more feature engineering or more hyperparameter
tuning. Stage 3 showed the ceiling here is the data, not the algorithm. Tuning
harder against a leaky feature set would just produce a more confident wrong
answer.

---

## Questions worth asking them

- How do you currently decide who gets a retention offer — model, rules, or
  neither?
- Do you have a feature store with point-in-time correctness, or are features
  computed at query time? *(This is the leakage question in their language.)*
- Has anyone measured the causal effect of your retention offers, or is the save
  rate an assumption there too?
- Where does Consumer Duty bite hardest on your growth models right now?
- When a model says "don't run this campaign", who has to agree before that
  sticks?

---

## If you only remember three things

1. **A 0.97 PR-AUC on a churn problem is a warning, not a win** — and I can show
   what it would have cost: £47,461 a year of business case that wasn't real.
2. **The decision was "don't run it", and that was the valuable answer** —
   delivered with the break-even conditions that would change it.
3. **Every number in the README is checked by a script** (124 of them), and the
   mistakes I made along the way are in the git history rather than tidied out.
