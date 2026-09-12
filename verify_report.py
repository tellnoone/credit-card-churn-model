"""Check every quantitative claim in README.md and docs/ against pipeline output.

The README and one-pager are written by hand, so every number in them is a
transcription that can be wrong on the day and can go stale when a script
changes. This checks each one against the JSON the pipeline actually produced.

    python verify_report.py

Exit code 1 on any mismatch. A mismatch means the prose is wrong, not the
analysis -- fix the prose.

Run it after run_all.py. It has caught real errors twice: a stale CUPED-style
calibration figure left over from an earlier run, and a rounding slip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"

Checks = list[tuple[bool, str, object, object]]
checks: Checks = []


def load(name: str) -> dict:
    path = TABLES / name
    if not path.exists():
        print(f"ERROR: {path.relative_to(ROOT)} missing. Run `python run_all.py` first.")
        raise SystemExit(2)
    return json.loads(path.read_text(encoding="utf-8"))


def chk(label: str, claimed: float, actual: float, tol: float = 5e-4) -> None:
    checks.append((abs(claimed - actual) <= tol, label, claimed, actual))


def chk_exact(label: str, claimed: object, actual: object) -> None:
    checks.append((claimed == actual, label, claimed, actual))


a3 = load("03_leakage_audit.json")
a4 = load("04_modelling.json")
a5 = load("05_explainability.json")
a6 = load("06_policy.json")
a7 = load("07_experiment_design.json")
a8 = load("08_one_pager.json")

# ---------------------------------------------------------------------------
# Key numbers: model performance table
# ---------------------------------------------------------------------------
tm = a4["test_metrics"]
chk_exact("test set size", 2026, a4["n_test"])
chk_exact("test churners", 325, a4["n_test_positive"])
chk("base rate", 0.1604, a4["base_rate_test"])

for label, key, pr, lo, hi, roc, lift in [
    ("heuristic", "heuristic", 0.2176, 0.1908, 0.2453, 0.6235, 1.56),
    ("logistic strict", "logistic_strict_raw", 0.2370, 0.2042, 0.2774, 0.6203, 1.81),
    ("lightgbm strict", "lightgbm_strict_raw", 0.2844, 0.2445, 0.3345, 0.6675, 2.43),
    ("logistic all", "logistic_all_raw", 0.7796, 0.7340, 0.8224, 0.9360, 5.49),
    ("lightgbm all", "lightgbm_all_cal", 0.9724, 0.9601, 0.9825, 0.9933, 6.23),
]:
    chk(f"{label} PR-AUC", pr, tm[key]["pr_auc"]["point"])
    chk(f"{label} PR-AUC lo", lo, tm[key]["pr_auc"]["lo"], tol=5e-3)
    chk(f"{label} PR-AUC hi", hi, tm[key]["pr_auc"]["hi"], tol=5e-3)
    chk(f"{label} ROC-AUC", roc, tm[key]["roc_auc"]["point"])
    chk(f"{label} lift@100", lift, tm[key]["lift_at_100"]["point"], tol=5e-3)

chk("heuristic Brier", 0.2713, tm["heuristic"]["brier"]["point"])
chk("lightgbm strict Brier", 0.1278, tm["lightgbm_strict_raw"]["brier"]["point"])
chk("lightgbm all Brier", 0.0202, tm["lightgbm_all_cal"]["brier"]["point"])
chk("logistic strict Brier", 0.1314, tm["logistic_strict_raw"]["brier"]["point"])
chk("logistic all Brier", 0.0644, tm["logistic_all_raw"]["brier"]["point"])

# ---------------------------------------------------------------------------
# Finding 1: leakage
# ---------------------------------------------------------------------------
chk("leakage gap (lightgbm)", 0.7288, a3["gaps"]["lightgbm"]["mean_diff"])
chk("all-features CV PR-AUC", 0.9702, a3["cv"]["lightgbm_all"]["pr_auc_mean"])
chk("strict CV PR-AUC", 0.2414, a3["cv"]["lightgbm_strict"]["pr_auc_mean"])
chk_exact("all wins all folds", 1.0, a3["gaps"]["lightgbm"]["a_wins_share"])
chk_exact("paired folds", 25, a3["gaps"]["lightgbm"]["n_folds"])

top_single = a3["single_feature"][0]
chk_exact("top gap feature", "total_trans_amt", top_single["feature"])
chk("top feature share of gap", 0.791, top_single["share_of_total_gap"], tol=5e-3)
chk("total_trans_amt PR-AUC", 0.82, top_single["pr_auc"], tol=5e-3)

chk_exact("suspect feature count", 14, a3["counts"]["suspect"])
chk_exact("safe feature count", 18, a3["counts"]["safe"])

# ---------------------------------------------------------------------------
# Finding 2: concentration
# ---------------------------------------------------------------------------
conc = a3["concentration"]
chk("strict full PR-AUC", 0.2426, conc["strict_full"]["pr_auc"])
chk("strict full ROC-AUC", 0.6201, conc["strict_full"]["roc_auc"])
chk("strict minus rel PR-AUC", 0.2015, conc["strict_minus_rel"]["pr_auc"])
chk("strict minus rel ROC-AUC", 0.5709, conc["strict_minus_rel"]["roc_auc"])
chk("rel only PR-AUC", 0.2185, conc["rel_only"]["pr_auc"])
chk("rel only ROC-AUC", 0.6124, conc["rel_only"]["roc_auc"])
chk("other 17 features add", 0.0241,
    conc["strict_full"]["pr_auc"] - conc["rel_only"]["pr_auc"], tol=5e-3)
chk("one feature share of strict", 0.90,
    conc["rel_only"]["pr_auc"] / conc["strict_full"]["pr_auc"], tol=5e-3)

# ---------------------------------------------------------------------------
# Finding 3: fairness
# ---------------------------------------------------------------------------
chk("removal cost", -0.0002, a5["removal_cost"]["mean_diff"], tol=5e-4)
chk("removal CI lo", -0.0246, a5["removal_cost"]["diff_lo"], tol=5e-3)
chk("removal CI hi", 0.0211, a5["removal_cost"]["diff_hi"], tol=5e-3)

proxy = {(p["attribute"], p["feature_set"]): p["recovery_roc_auc"]
         for p in a5["proxy_recovery"]}
chk("gender recoverable", 0.9552,
    proxy[("gender", "strict minus age+gender")], tol=5e-3)
chk("gender after stripping income", 0.5116,
    proxy[("gender", "also minus income+limit")], tol=5e-3)
chk("gender-neutral cost", 0.0329,
    a5["gender_neutral"]["pr_auc_cost"]["mean_diff"], tol=5e-3)

chk("women risk ratio", 1.17, a5["gender_disparity"]["risk_ratio_f_over_m"], tol=5e-3)
chk("women targeting ratio", 1.77,
    a5["gender_disparity"]["targeting_ratio_f_over_m"], tol=5e-3)

gp = {(g["dimension"], g["group"]): g for g in a5["disparity"]}
chk("women churn rate", 0.1726, gp[("gender", "F")]["churn_rate"])
chk("women targeted", 0.1557, gp[("gender", "F")]["targeted_share"])
chk("men churn rate", 0.1470, gp[("gender", "M")]["churn_rate"])
chk("men targeted", 0.0880, gp[("gender", "M")]["targeted_share"])
chk("women targeting per risk", 0.90,
    gp[("gender", "F")]["targeting_per_unit_risk"], tol=5e-3)
chk("men targeting per risk", 0.60,
    gp[("gender", "M")]["targeting_per_unit_risk"], tol=5e-3)

shap_top = {r["feature"]: r["share"] for r in a5["global_importance"][:5]}
chk("SHAP relationship count share", 0.304, shap_top["total_relationship_count"], tol=5e-3)
chk("SHAP credit limit share", 0.237, shap_top["credit_limit"], tol=5e-3)
chk("SHAP gender_F share", 0.090, shap_top["gender_F"], tol=5e-3)

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
chk("break-even probability", 0.533, a6["break_even_probability"], tol=5e-3)
chk_exact("customers targeted", 1, a6["central"]["optimal"]["n_targeted"])
chk("net value", 2.0, a6["central"]["optimal"]["net_value"], tol=0.5)
chk("target everyone loss", -56489, a6["central"]["everyone"]["net_value"], tol=1.0)
chk("leaky scaled value", 47472, a6["leakage_price"]["all_features"]["net_value"]
    * 10127 / 2026, tol=1.0)
chk("leakage gap", 47461, a6["leakage_price"]["annual_gap_scaled"], tol=1.0)
chk_exact("leaky contacts", 322, a6["leakage_price"]["all_features"]["n_targeted"])

chk("min customer value", 285, a6["assumption_bounds"]["min_customer_value"], tol=1.0)
chk("min save rate", 0.238, a6["assumption_bounds"]["min_save_rate"], tol=5e-3)
chk("max offer cost", 42, a6["assumption_bounds"]["max_offer_cost"], tol=1.0)
chk("value slack", 0.049, a6["assumption_margins"]["customer_value"], tol=5e-3)
chk("save rate slack", 0.049, a6["assumption_margins"]["save_rate"], tol=5e-3)
chk("offer cost slack", 0.051, a6["assumption_margins"]["offer_cost"], tol=5e-3)

chk("assumed customer value", 300.0, a6["assumptions"]["customer_value"])
chk("assumed save rate", 0.25, a6["assumptions"]["save_rate"])
chk("assumed offer cost", 40.0, a6["assumptions"]["offer_cost"])

sens = {(r["customer_value"], r["save_rate"]): r for r in a6["sensitivity"]}
chk("sensitivity 600/0.35 net", 8686, sens[(600.0, 0.35)]["net_value"], tol=1.0)
chk_exact("sensitivity 600/0.35 n", 568, sens[(600.0, 0.35)]["n_targeted"])

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
chk_exact("chosen segment", "top 20%", a7["chosen_segment"])
chk("segment baseline churn", 0.309, a7["baseline_churn"], tol=5e-3)
chk("break-even save rate", 0.432, a7["break_even_save_rate"], tol=5e-3)
chk_exact("n per arm at 40 offer", 161.0, a7["n_per_arm"])
alt = a7["alternative_design"]
chk("cheaper offer", 15.0, alt["offer_cost"])
chk("cheaper break-even save", 0.162, alt["break_even_save_rate"], tol=5e-3)
chk("alt MDE", 0.20, alt["mde_save_rate"])
chk_exact("alt n per arm", 826.0, alt["n_per_arm"])

# ---------------------------------------------------------------------------
# Boosting vs logistic
# ---------------------------------------------------------------------------
bl = a4["boosting_vs_logistic"]["strict"]
chk("boosting gain strict", 0.0313, bl["mean_diff"], tol=5e-3)
chk("boosting CI lo", 0.0063, bl["diff_lo"], tol=5e-3)
chk("boosting CI hi", 0.0641, bl["diff_hi"], tol=5e-3)

# ---------------------------------------------------------------------------
# Model vs heuristic, and score range (quoted in docs/interview_prep.md)
# ---------------------------------------------------------------------------
mvh = a4["model_vs_heuristic"]
chk("model beats heuristic by", 0.0702, mvh["mean_diff"], tol=5e-3)
chk("heuristic gap CI lo", 0.0400, mvh["diff_lo"], tol=5e-3)
chk("heuristic gap CI hi", 0.1014, mvh["diff_hi"], tol=5e-3)
chk_exact("model beats heuristic", True, mvh["beats_heuristic"])

sr = a4["score_range"]
chk("model max score", 0.561, sr["max"], tol=5e-3)
chk("model min score", 0.052, sr["min"], tol=5e-3)
chk_exact("distinct scores", 2019, sr["n_distinct"])
chk_exact("customers above 0.50", 3, sr["n_above_0_50"])

# Calibration: the isotonic collapse quoted in the interview prep
cal = a4["calibration_choice"]["lightgbm_strict"]
chk_exact("isotonic distinct scores", 15, cal["candidates"]["isotonic"]["distinct_scores"])
chk("isotonic brier gain", 0.0015, cal["candidates"]["isotonic"]["brier_gain"], tol=5e-4)
chk_exact("recommended keeps raw", True, cal["chosen"].startswith("none"))

# CV figures quoted for the boosting comparison
chk("lightgbm strict CV", 0.2701, a4["cv_confirmation"]["lightgbm_strict"]["pr_auc_mean"], tol=5e-3)
chk("logistic strict CV", 0.2389, a4["cv_confirmation"]["logistic_strict"]["pr_auc_mean"], tol=5e-3)

# ---------------------------------------------------------------------------
# One-pager (docs/one_pager.md)
# ---------------------------------------------------------------------------
op = {int(r["offer_cost"]): r for r in a8["rows"]}
for cost, n_cust, profit in [(40, 5, 10), (30, 210, 652), (25, 360, 2132),
                             (20, 940, 4920), (15, 2644, 13543), (10, 4944, 30980)]:
    chk(f"one-pager GBP{cost} customers", n_cust, op[cost]["n_targeted_book"], tol=0.6)
    chk(f"one-pager GBP{cost} profit", profit, op[cost]["net_value_book"], tol=1.0)
chk_exact("one-pager book size", 10127, a8["book_size"])

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
# The interview prep quotes how many figures this script checks. That claim can
# itself go stale, so it is checked here -- otherwise the one number in the repo
# nobody verifies would be the one about verification.
_prep = ROOT / "docs" / "interview_prep.md"
if _prep.exists():
    import re as _re
    m = _re.search(r"checked by a script\*\* \((\d+) of them\)",
                   _prep.read_text(encoding="utf-8"))
    if m:
        chk_exact("interview_prep claimed check count", int(m.group(1)), len(checks) + 1)

failed = [c for c in checks if not c[0]]
print(f"{len(checks) - len(failed)}/{len(checks)} documented figures verified")

if failed:
    print("\nMISMATCHES (the prose is wrong, not the analysis):\n")
    for _, label, claimed, actual in failed:
        print(f"  {label:<36} documented={claimed!r:>14}   actual={actual!r}")
    sys.exit(1)

print("Every figure in README.md and docs/ matches the pipeline output.")
