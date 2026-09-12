"""Stage 6: the commercial decision layer (ANALYSIS_PLAN.md section 7).

Turns churn scores into a targeting policy with a pound figure attached, and
then attacks that figure: sensitivity across the assumption grid, break-even on
each assumption individually, and comparison against the two trivial policies.

Also prices the leakage audit. The all-features model would have promised a much
larger campaign; the difference between what it promises and what the strict
model delivers is the money that would have been committed against a number that
does not survive production.
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

from src.config import (FIGURES, MODELS, RANDOM_SEED, TABLES, TARGET,
                        ensure_dirs)
from src.data import load_features, make_splits
from src.policy import (Assumptions, bootstrap_net_value,
                        break_even_probability, campaign_value,
                        load_assumptions, minimum_viable, optimal_threshold,
                        profit_curve, sensitivity_grid, target_everyone,
                        target_nobody)
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


def money(x: float, cur: str = "GBP") -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{cur} {abs(x):,.0f}"


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

    strict_model = joblib.load(MODELS / "recommended_model_uncalibrated.joblib")
    all_model = joblib.load(MODELS / "all_features_model.joblib")
    p_strict = strict_model.predict_proba(xy(test, strict=True)[0])[:, 1]
    p_all = all_model.predict_proba(xy(test, strict=False)[0])[:, 1]

    payload: dict = {"assumptions": {
        "customer_value": central.customer_value,
        "save_rate": central.save_rate,
        "offer_cost": central.offer_cost,
        "horizon_years": central.horizon_years,
        "currency": cur,
    }}

    head("STAGE 6: RETENTION TARGETING POLICY")
    say("  Every pound figure below rests on three numbers the data cannot")
    say("  supply. They are assumptions, set in config/policy.yaml:")
    say()
    say(f"    customer value : {money(central.customer_value, cur)} gross margin "
        f"per retained customer per year")
    say(f"    save rate      : {central.save_rate:.0%} of would-be churners "
        f"retained by the offer")
    say(f"    offer cost     : {money(central.offer_cost, cur)} per customer "
        f"CONTACTED (not per save)")
    say(f"    horizon        : {central.horizon_years:.0f} year "
        f"(deliberately not lifetime value)")
    say()
    say(f"  Population : the {len(test):,}-customer held-out test set "
        f"({y_test.mean():.1%} churn).")
    say(f"  Everything is out-of-sample. Scaled projections to the full")
    say(f"  {n_book:,}-customer book are labelled as such.")
    say()

    # ------------------------------------------------------------ the rule --
    head("1. THE DECISION RULE")
    say("    expected value = P(churn) x save_rate x customer_value - offer_cost")
    say()
    say("  Contact a customer when that is positive. Expected value rises with")
    say("  P(churn), so this is the same as a probability threshold, and the")
    say("  threshold has a closed form:")
    say()
    be = break_even_probability(central)
    say(f"    break-even P(churn) = offer_cost / (save_rate x value)")
    say(f"                        = {central.offer_cost:.0f} / "
        f"({central.save_rate} x {central.customer_value:.0f})")
    say(f"                        = {be:.4f}")
    say()
    say(f"  So a customer is worth contacting only if they are more than")
    say(f"  {be:.1%} likely to churn.")
    say()
    say(f"  The recommended model's scores on this population run "
        f"{p_strict.min():.3f} to {p_strict.max():.3f},")
    n_above = int((p_strict >= be).sum())
    say(f"  with {n_above} customer{'s' if n_above != 1 else ''} at or above {be:.1%}.")
    say()
    payload["break_even_probability"] = float(be)

    # ------------------------------------------------------ central result --
    head("2. THE CENTRAL RESULT")
    best = campaign_value(p_strict, central)
    everyone = target_everyone(p_strict, central)
    nobody = target_nobody(p_strict, central)
    boot = bootstrap_net_value(p_strict, central, seed=RANDOM_SEED)

    say(f"  {'policy':<28}{'contacted':>11}{'exp. saves':>12}"
        f"{'net value':>14}{'per contact':>13}")
    rule()
    for name, out in (("target nobody", nobody),
                      ("target everyone", everyone),
                      (f"optimal (p >= {be:.3f})", best)):
        say(f"  {name:<28}{out['n_targeted']:>11,}"
            f"{out['expected_saves']:>12.1f}"
            f"{money(out['net_value'], cur):>14}"
            f"{money(out['net_value_per_contact'], cur):>13}")
    rule()
    say()
    say(f"  Bootstrap 95% CI on the optimal policy's net value:")
    say(f"    {money(boot['point'], cur)}  "
        f"[{money(boot['lo'], cur)}, {money(boot['hi'], cur)}]")
    say(f"    positive in {boot['share_positive']:.0%} of resamples")
    say()
    say("  (That interval reflects only who happens to be in the population. It")
    say("  does NOT include uncertainty in the three assumptions, which is far")
    say("  larger -- see section 4.)")
    say()

    if best["net_value"] <= 0:
        say("  VERDICT UNDER CENTRAL ASSUMPTIONS: do not run this campaign.")
    elif best["n_targeted"] < 20:
        say(f"  VERDICT UNDER CENTRAL ASSUMPTIONS: technically positive, but it")
        say(f"  targets only {best['n_targeted']} customer"
            f"{'s' if best['n_targeted'] != 1 else ''} for "
            f"{money(best['net_value'], cur)}. That is not a campaign; it is a")
        say("  rounding error, and it would not survive the cost of building the")
        say("  pipeline to deliver it.")
    else:
        say(f"  VERDICT UNDER CENTRAL ASSUMPTIONS: run it, targeting "
            f"{best['n_targeted']:,} customers.")
    say()
    scaled = best["net_value"] * n_book / len(test)
    say(f"  Scaled to the full {n_book:,}-customer book: "
        f"{money(scaled, cur)} per year.")
    say("  (Straight-line scaling, which assumes the book looks like the test")
    say("  set. It does, by construction of the split -- but a real book would")
    say("  need re-scoring rather than multiplying.)")
    say()
    payload["central"] = {"optimal": best, "everyone": everyone,
                          "nobody": nobody, "bootstrap": boot,
                          "scaled_to_book": float(scaled)}

    # --------------------------------------------------- what leakage sold --
    head("3. WHAT THE LEAKY MODEL WOULD HAVE PROMISED")
    say("  The all-features model scored PR-AUC 0.97 and is almost certainly")
    say("  reading the answer. Here is what that would have looked like in a")
    say("  business case, beside what the defensible model actually delivers.")
    say()
    best_all = campaign_value(p_all, central)
    say(f"  {'model':<34}{'contacted':>11}{'net value':>14}{'scaled to book':>17}")
    rule()
    for name, out, in (("all-features (leaky, PR-AUC 0.97)", best_all),
                       ("strict (defensible, PR-AUC 0.28)", best)):
        say(f"  {name:<34}{out['n_targeted']:>11,}"
            f"{money(out['net_value'], cur):>14}"
            f"{money(out['net_value'] * n_book / len(test), cur):>17}")
    rule()
    say()
    gap = (best_all["net_value"] - best["net_value"]) * n_book / len(test)
    say(f"  The leaky model promises {money(gap, cur)} a year more than the")
    say("  defensible one. That gap is the size of the business case that would")
    say("  have been signed off on a number which does not survive production --")
    say("  and it is the clearest answer to 'why bother with a leakage audit'.")
    say()
    payload["leakage_price"] = {
        "all_features": best_all, "strict": best,
        "annual_gap_scaled": float(gap),
    }

    # ------------------------------------------------------- sensitivity --
    head("4. SENSITIVITY: HOW MUCH DOES THIS DEPEND ON THE ASSUMPTIONS?")
    say("  The honest answer is: almost entirely. The grid below shows net value")
    say("  on the test population across customer value and save rate, at the")
    say(f"  central offer cost of {money(central.offer_cost, cur)}.")
    say()
    sens = sensitivity_grid(
        p_strict, central,
        values=cfg["sensitivity"]["customer_value"],
        save_rates=cfg["sensitivity"]["save_rate"],
    )
    pivot = sens.pivot(index="save_rate", columns="customer_value", values="net_value")
    say(f"  net value ({cur}), rows = save rate, columns = customer value")
    say()
    header = "        " + "".join(f"{v:>10,.0f}" for v in pivot.columns)
    say(header)
    for s, row in pivot.iterrows():
        say(f"  {s:>5.2f} " + "".join(
            f"{v:>10,.0f}" for v in row.to_numpy()))
    say()
    n_pays = int(sens["pays"].sum())
    say(f"  The campaign pays in {n_pays} of {len(sens)} assumption combinations "
        f"({n_pays/len(sens):.0%}).")
    say()

    say("  Number of customers targeted across the same grid:")
    say()
    pivot_n = sens.pivot(index="save_rate", columns="customer_value",
                         values="n_targeted")
    say(header)
    for s, row in pivot_n.iterrows():
        say(f"  {s:>5.2f} " + "".join(f"{int(v):>10,}" for v in row.to_numpy()))
    say()
    payload["sensitivity"] = sens.to_dict(orient="records")

    # --------------------------------------------------------- break-even --
    head("5. HOW WRONG CAN THE ASSUMPTIONS BE?")
    say("  Each parameter moved on its own, with the other two held at central,")
    say("  to find where the campaign stops paying. This is the number to give a")
    say("  sceptical stakeholder: not 'trust me', but 'here is the boundary'.")
    say()
    bounds = {}
    mv = minimum_viable(p_strict, central, "customer_value", lo=1.0, hi=20_000.0)
    ms = minimum_viable(p_strict, central, "save_rate", lo=0.001, hi=1.0)
    mc = minimum_viable(p_strict, central, "offer_cost", lo=0.5, hi=5_000.0)
    bounds = {"min_customer_value": mv, "min_save_rate": ms, "max_offer_cost": mc}

    if mv is not None:
        say(f"  customer value must exceed {money(mv, cur)}   "
            f"(central {money(central.customer_value, cur)}, "
            f"{'ABOVE' if central.customer_value > mv else 'BELOW'} the line)")
    else:
        say("  customer value: no crossing found in the range tested")
    if ms is not None:
        say(f"  save rate must exceed      {ms:.1%}        "
            f"(central {central.save_rate:.0%}, "
            f"{'ABOVE' if central.save_rate > ms else 'BELOW'} the line)")
    else:
        say("  save rate: no crossing found in the range tested")
    if mc is not None:
        say(f"  offer cost must stay below {money(mc, cur)}   "
            f"(central {money(central.offer_cost, cur)}, "
            f"{'BELOW' if central.offer_cost < mc else 'ABOVE'} the line)")
    else:
        say("  offer cost: no crossing found in the range tested")
    say()
    payload["assumption_bounds"] = bounds

    # How much slack is there on each assumption? If every bound sits a few
    # percent from its central value, the "positive" result is not a business
    # case, it is noise.
    margins = {}
    if mv is not None:
        margins["customer_value"] = (central.customer_value - mv) / central.customer_value
    if ms is not None:
        margins["save_rate"] = (central.save_rate - ms) / central.save_rate
    if mc is not None:
        margins["offer_cost"] = (mc - central.offer_cost) / central.offer_cost
    if margins:
        say("  SLACK ON EACH ASSUMPTION (how far central sits from the boundary):")
        say()
        for k, v in margins.items():
            say(f"    {k:<18}{v:>+8.1%}")
        say()
        tightest = min(margins.values())
        if max(margins.values()) < 0.15:
            say(f"  Every one of those margins is under 15%, and the tightest is")
            say(f"  {tightest:.1%}. Read together they are the real finding of this")
            say("  stage: the campaign is not profitable-with-a-margin, it is")
            say("  sitting on the break-even line in all three directions at once.")
            say()
            say("  A business case that survives only while three independent")
            say("  guesses all stay within a few percent of where I happened to")
            say("  set them is not a business case. It is an argument for")
            say("  measuring the save rate before spending anything.")
            say()
        payload["assumption_margins"] = margins

    if mv is not None and central.customer_value < mv:
        say(f"  Read that top line again: the central customer-value assumption")
        say(f"  of {money(central.customer_value, cur)} sits BELOW the "
            f"{money(mv, cur)} needed to break even.")
        say("  The campaign does not pay at the assumptions I set out before")
        say("  computing anything. Rather than revise them to reach a")
        say("  comfortable answer, the finding is reported as it stands.")
        say()

    # ------------------------------------------------------------- figures --
    _plot_profit_curves(p_strict, p_all, central, cur)
    _plot_sensitivity(sens, central, cur)

    head("6. RECOMMENDATION")
    if best["net_value"] <= 0 or best["n_targeted"] < 20:
        say("  DO NOT run a blanket retention campaign on this model at these")
        say("  assumptions. Say so plainly rather than tuning the assumptions")
        say("  until the answer changes.")
        say()
        say("  What would make it viable, in order of how easily the business")
        say("  can move each lever:")
        say()
        if mc is not None:
            say(f"  1. CUT THE OFFER COST below {money(mc, cur)}. The cheapest lever")
            say("     and the one marketing controls directly. A points nudge")
            say("     rather than a fee waiver changes the arithmetic immediately.")
        if mv is not None:
            say(f"  2. TARGET HIGHER-VALUE CUSTOMERS. At {money(mv, cur)}+ of annual")
            say("     margin the campaign pays. Segment the book by value first")
            say("     and run retention only on the top tier, where the same")
            say("     offer cost buys much more.")
        say("  3. GET A BETTER MODEL, which means better DATA -- specifically")
        say("     behavioural features with a window that provably closes before")
        say("     the prediction date. Stage 3 showed the ceiling here is the")
        say("     data, not the algorithm.")
        say("  4. MEASURE THE SAVE RATE instead of assuming it. It is the most")
        say("     uncertain input and the stage 7 experiment is how to pin it.")
    else:
        say(f"  RUN the campaign, targeting the {best['n_targeted']:,} customers")
        say(f"  above P(churn) = {be:.1%}, for an expected "
            f"{money(best['net_value'], cur)} on this population.")
    say()
    say("  And the limitation that outranks all of the above:")
    say()
    say("  P(churn) IS NOT P(RESPONDS TO OFFER). This entire policy assumes the")
    say("  customers most likely to leave are the ones an offer can retain. They")
    say("  may be the least persuadable -- already decided, and now collecting a")
    say("  discount on their way out. Nothing in this dataset can test that.")
    say("  Stage 7 designs the experiment that would.")
    say()

    (TABLES / "06_policy.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (TABLES / "06_policy.txt").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print("\n[written] outputs/tables/06_policy.{json,txt}")
    return 0


def _plot_profit_curves(p_strict: np.ndarray, p_all: np.ndarray,
                        a: Assumptions, cur: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    be = break_even_probability(a)

    ax = axes[0]
    for p, colour, lab in ((p_all, "#b3452c", "all features (leaky)"),
                           (p_strict, "#1f4e79", "strict (recommended)")):
        curve = profit_curve(p, a)
        ax.plot(curve["threshold"], curve["net_value"], lw=2, color=colour, label=lab)
    ax.axhline(0, color="#333", lw=1.1)
    ax.axvline(be, ls="--", color="#2e7d32", lw=1.4,
               label=f"break-even p = {be:.3f}")
    ax.set_xlabel("targeting threshold: contact if P(churn) >= x")
    ax.set_ylabel(f"expected net value ({cur}) on 2,026 customers")
    ax.set_title("Profit curve\nwhat the leaky model promises vs what is real")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    curve = profit_curve(p_strict, a, n_points=600)
    best = campaign_value(p_strict, a)
    # Zoom on the decision region. Plotted out to all 2,026 contacts the
    # y-axis spans -56,000 and the optimum near zero becomes invisible,
    # hiding the only part of the curve anyone has to choose between.
    zoom = curve[curve['n_targeted'] <= 300]
    ax.plot(zoom['n_targeted'], zoom['net_value'], lw=2, color='#1f4e79')
    ax.axhline(0, color='#333', lw=1.1)
    ax.plot([best['n_targeted']], [best['net_value']], 'o', ms=10,
            color='#2e7d32', zorder=5,
            label=f"optimum: {best['n_targeted']} contacted, "
                  f"{money(best['net_value'], cur)}")
    full_worst = float(curve['net_value'].min())
    ax.set_xlabel('customers contacted (zoomed to the first 300)')
    ax.set_ylabel(f'expected net value ({cur})')
    ax.set_title('Strict model, decision region\n'
                 f'contacting all 2,026 would lose {money(full_worst, cur)}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("Does the campaign pay? (plan §7) — assumptions in "
                 "config/policy.yaml", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_profit_curve.png", dpi=140)
    plt.close(fig)
    print("[written] outputs/figures/06_profit_curve.png")


def _plot_sensitivity(sens: pd.DataFrame, a: Assumptions, cur: str) -> None:
    """Net value across the assumption grid.

    Note the colour scale is SEQUENTIAL, not diverging. The optimal policy can
    always fall back on targeting nobody, so its net value has a floor of zero
    by construction and no cell can ever be negative. A red-green scale would
    imply losses are possible here and they are not -- the risk in this analysis
    is a campaign that is not worth running, not one that loses money.
    """
    pivot = sens.pivot(index="save_rate", columns="customer_value",
                       values="net_value")
    pivot_n = sens.pivot(index="save_rate", columns="customer_value",
                         values="n_targeted")
    values = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    # Square-root scaling so the small-but-positive cells stay distinguishable
    # instead of being flattened by the single large corner value.
    im = ax.imshow(np.sqrt(values), cmap="YlGn", aspect="auto", origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:,.0f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel(f"customer value ({cur} annual gross margin)")
    ax.set_ylabel("save rate (probability the offer retains a would-be churner)")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            n = int(pivot_n.to_numpy()[i, j])
            label = "not worth\nrunning" if n < 20 else f"{v:,.0f}\n{n:,} sent"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="#333" if v < values.max() * 0.55 else "#fff")

    try:
        ci = list(pivot.index).index(a.save_rate)
        cj = list(pivot.columns).index(a.customer_value)
        ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False,
                                   edgecolor="#b3452c", lw=3))
        ax.text(cj, ci + 0.34, "central assumption", ha="center", fontsize=7,
                color="#b3452c", weight="bold")
    except ValueError:
        pass

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"net value ({cur}), sqrt scale")
    ax.set_title("Sensitivity: the campaign only becomes materially profitable\n"
                 "at HIGH customer value AND HIGH save rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "06_sensitivity.png", dpi=140)
    plt.close(fig)
    print("[written] outputs/figures/06_sensitivity.png")


if __name__ == "__main__":
    raise SystemExit(main())
