"""Stage 8: the single chart for the one-pager.

Written for a Head of Growth, not for a data scientist. One idea per panel, no
jargon on the axes, and the recommendation readable without the surrounding
text.

The idea it has to carry: at the offer we costed, almost nobody is worth
contacting. Make the offer cheaper and the campaign comes alive. The cliff is
between GBP 40 and GBP 20, and it is steep.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES, MODELS, TABLES, TARGET, ensure_dirs
from src.data import load_features, make_splits
from src.policy import (break_even_probability, campaign_value,
                        load_assumptions)
from src.train import xy


def main() -> int:
    ensure_dirs()
    import joblib

    central, _ = load_assumptions()
    cur = central.currency
    df = load_features()
    test = make_splits(df)["test"]
    n_book = len(df)
    scale = n_book / len(test)

    model = joblib.load(MODELS / "recommended_model_uncalibrated.joblib")
    p = model.predict_proba(xy(test, strict=True)[0])[:, 1]

    costs = [40, 30, 25, 20, 15, 10]
    rows = []
    for c in costs:
        a = central.replace(offer_cost=float(c))
        out = campaign_value(p, a)
        rows.append({
            "offer_cost": c,
            "n_targeted_book": out["n_targeted"] * scale,
            "net_value_book": out["net_value"] * scale,
            "break_even_p": break_even_probability(a),
        })

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    labels = [f"{cur} {c}" for c in costs]
    x = np.arange(len(costs))

    ax = axes[0]
    vals = [r["net_value_book"] for r in rows]
    bars = ax.bar(x, vals, color=["#b3452c" if v < 1000 else "#2e7d32" for v in vals],
                  alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{cur} {v:,.0f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("What we spend per customer we contact")
    ax.set_ylabel(f"Expected profit per year ({cur})")
    ax.set_title("A cheaper offer is worth more than a better model\n"
                 "expected annual profit across the whole customer book",
                 fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    counts = [r["n_targeted_book"] for r in rows]
    bars = ax.bar(x, counts, color=["#b3452c" if c_ < 100 else "#2e7d32"
                                    for c_ in counts], alpha=0.9)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts) * 0.02,
                f"{v:,.0f}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("What we spend per customer we contact")
    ax.set_ylabel("Customers worth contacting")
    ax.set_title("At the offer we costed, almost nobody qualifies\n"
                 f"out of {n_book:,} customers on the book", fontsize=11)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("The retention campaign only pays if the offer is cheap",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out_path = FIGURES / "08_one_pager.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    (TABLES / "08_one_pager.json").write_text(
        json.dumps({"rows": rows, "book_size": n_book,
                    "assumptions": {"customer_value": central.customer_value,
                                    "save_rate": central.save_rate}},
                   indent=2, default=float), encoding="utf-8")
    print(f"[written] {out_path.relative_to(Path(__file__).resolve().parents[1])}")
    print("[written] outputs/tables/08_one_pager.json")
    for r in rows:
        print(f"  {cur} {r['offer_cost']:>3}: "
              f"{r['n_targeted_book']:>7,.0f} customers, "
              f"{cur} {r['net_value_book']:>9,.0f}/yr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
