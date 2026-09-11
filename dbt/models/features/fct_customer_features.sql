-- Feature layer. One row per customer, model-ready.
--
-- Every derived feature inherits the leakage class of its inputs: a ratio built
-- from a leakage-suspect column is itself leakage-suspect. The SAFE / SUSPECT
-- classification lives in src/features.py so Python and SQL cannot disagree
-- about it, and the audit in stage 3 reads it from there.
--
-- Ordinal encodings are done HERE rather than in Python because the ordering is
-- a business fact about the data (Blue < Silver < Gold < Platinum), not a
-- modelling choice. Putting it in SQL keeps it visible to anyone reading the
-- warehouse rather than buried in a sklearn pipeline.

with base as (

    select * from {{ ref('stg_customers') }}

),

derived as (

    select
        base.*,

        ---------------------------------------------------------------------
        -- SAFE derived features (inputs known before the churn window)
        ---------------------------------------------------------------------

        -- Tenure in years reads more naturally than months in a coefficient table.
        months_on_book / 12.0                                   as tenure_years,

        -- Ordinal encodings. NULL for 'Unknown' would throw away the row's
        -- information, so Unknown gets its own flag column (already in staging)
        -- and is encoded as NULL here only where the scale is genuinely ordered.
        case card_category
            when 'Blue'     then 1
            when 'Silver'   then 2
            when 'Gold'     then 3
            when 'Platinum' then 4
        end                                                     as card_category_ord,

        case income_category
            when 'Less than $40K' then 1
            when '$40K - $60K'    then 2
            when '$60K - $80K'    then 3
            when '$80K - $120K'   then 4
            when '$120K +'        then 5
            when 'Unknown'        then null
        end                                                     as income_category_ord,

        case education_level
            when 'Uneducated'    then 1
            when 'High School'   then 2
            when 'College'       then 3
            when 'Graduate'      then 4
            when 'Post-Graduate' then 5
            when 'Doctorate'     then 6
            when 'Unknown'       then null
        end                                                     as education_level_ord,

        -- Products held per year of tenure: how fast the relationship deepened.
        -- Both inputs are SAFE.
        case
            when months_on_book > 0
                then total_relationship_count / (months_on_book / 12.0)
        end                                                     as products_per_tenure_year,

        ---------------------------------------------------------------------
        -- SUSPECT derived features (built from 12-month behavioural columns)
        ---------------------------------------------------------------------

        -- Average value of a transaction. NULLIF guards the ~0-transaction
        -- customers rather than letting a division by zero become inf.
        total_trans_amt / nullif(total_trans_ct, 0)             as avg_trans_amt,

        -- Share of the credit line actually drawn down.
        total_revolving_bal / nullif(credit_limit, 0)          as revolving_to_limit,

        -- Spend intensity relative to the line granted.
        total_trans_amt / nullif(credit_limit, 0)              as spend_to_limit,

        -- Transactions per active month. Months_Inactive is SUSPECT, so this is
        -- doubly so -- included to be audited, not because it is trusted.
        total_trans_ct / nullif(12 - months_inactive_12_mon, 0) as trans_per_active_month,

        -- A customer whose Q4 spend collapsed versus Q1. This is the sharpest
        -- leakage candidate in the dataset: it is arguably a measurement of
        -- churn rather than a predictor of it.
        case when total_ct_chng_q4_q1 < 0.5 then 1 else 0 end    as q4_activity_collapsed

    from base

)

select * from derived
