# Data Provenance

## The file

| | |
|---|---|
| **File** | `raw/BankChurners.csv` |
| **Rows** | 10,127 customers (plus header) |
| **Columns** | 23 as shipped; 21 after dropping two leaked columns (below) |
| **Size** | 1,510,880 bytes |
| **SHA-256** | `c91b525a2a6755a1b0b80dad1d0d008ca97ec4df34552c8f47ffa12b6184b779` |
| **Retrieved** | 2026-09-11 |

## Source

**Credit Card customers** — "Predict Churning customers", by Sakshi Goyal
(Kaggle user `sakshigoyal7`).

- Canonical page: <https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers>
- **Licence: CC0: Public Domain** — confirmed on 2026-09-11 from Kaggle's own
  metadata API (`licenseNameNullable: "CC0: Public Domain"`), not inferred from
  a third-party README.

The dataset originally circulated via a LEAPS analytics tutorial. It is widely
described as anonymised bank credit-card customer records. Treat it as a
teaching dataset: there is no published data dictionary from an actual issuer,
and no guarantee it reflects a real portfolio's behaviour.

## Retrieval and integrity check

Kaggle downloads require account credentials, so the file was retrieved from a
public GitHub mirror and **cross-checked against a second, unrelated mirror**:

| Mirror | URL |
|---|---|
| 1 (used) | `https://raw.githubusercontent.com/allmeidaapedro/Churn-Prediction-Credit-Card/main/input/BankChurners.csv` |
| 2 (check) | `https://raw.githubusercontent.com/shvuuuu/Credit_Card_Churn_Predictor/main/BankChurners.csv` |

Both are **byte-identical** (same SHA-256, same 1,510,880-byte length). Two
unrelated uploads agreeing bit-for-bit is reasonable evidence the file is the
unmodified original rather than one analyst's edited copy.

Verify at any time:

```bash
sha256sum data/raw/BankChurners.csv
# c91b525a2a6755a1b0b80dad1d0d008ca97ec4df34552c8f47ffa12b6184b779
```

## The two columns dropped immediately

These are dropped in the **staging layer** and never reach a model:

```
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1
Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
```

**Why: they are the fitted output of someone else's classifier, and they encode
the label perfectly.** Measured on the raw file:

| | Column 1 | Column 2 |
|---|---|---|
| ROC-AUC against `Attrition_Flag` | **1.0000** | **0.0000** |
| Range for *Existing* customers | 0.000008 – 0.001317 | 0.998680 – 0.999990 |
| Range for *Attrited* customers | 0.945910 – 0.999580 | 0.000420 – 0.054090 |

The two columns sum to 1.0 for every row (within float rounding), so they are
the two class posteriors of a Naive Bayes model. The highest value any
*Existing* customer scores on column 1 is **0.0013**; the lowest any *Attrited*
customer scores is **0.9459**. There is no overlap at all — a single threshold
anywhere in that gap separates the classes with 100% accuracy.

An ROC-AUC of exactly 1.0 is never a feature; it is a label wearing a different
name. Any model trained with these columns would report near-perfect metrics,
learn nothing about churn, and fail immediately in production, because at
scoring time no such column exists for a customer who has not churned yet.

The Kaggle dataset description itself advises deleting them. That advice is
followed here, and the columns are dropped in SQL at the staging boundary so it
is structurally impossible for them to reach the feature layer.

## Known data-quality issues

**"Unknown" is missingness wearing a costume.** The file has **zero** nulls by
`pandas.isna()`, which is misleading. Three categorical columns encode missing
values as the literal string `"Unknown"`:

| Column | "Unknown" rows | Share |
|---|---|---|
| `Education_Level` | 1,519 | 15.00% |
| `Income_Category` | 1,112 | 10.98% |
| `Marital_Status` | 749 | 7.40% |

This is handled explicitly rather than silently: see `ANALYSIS_PLAN.md`. The
practical point is that "no missing data" is false, and a pipeline that trusts
the null count would treat `"Unknown"` as a real category — which may in fact be
the right modelling choice, but must be a decision rather than an accident.

## Licensing

The repository's MIT `LICENSE` covers the code, the analysis plan and the
written reports. It does **not** cover this dataset, which is CC0 (public
domain) and the work of its original author, not of this repository.
