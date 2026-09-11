-- Staging: clean, rename and type the raw snapshot. No feature engineering here.
--
-- The two Naive_Bayes_Classifier_* columns are dropped at THIS boundary, by
-- being absent from the select list below. That is deliberate: an explicit
-- column list makes it structurally impossible for them to reach the feature
-- layer, whereas a `select *` with a later drop would rely on remembering.
-- They are the fitted output of someone else's model and separate the classes
-- perfectly (ROC-AUC 1.0000). See data/PROVENANCE.md.

with source as (

    select * from {{ source('raw', 'bank_churners') }}

),

renamed as (

    select
        -- Identity
        cast("CLIENTNUM" as bigint)                     as customer_id,

        -- Target. Any value outside the two known labels becomes NULL rather
        -- than silently defaulting to 0, so the not_null test would catch it.
        case "Attrition_Flag"
            when 'Attrited Customer' then 1
            when 'Existing Customer' then 0
        end                                             as is_attrited,
        "Attrition_Flag"                                as attrition_flag_raw,

        -- Demographics
        cast("Customer_Age" as integer)                 as customer_age,
        "Gender"                                        as gender,
        cast("Dependent_count" as integer)              as dependent_count,
        "Education_Level"                               as education_level,
        "Marital_Status"                                as marital_status,
        "Income_Category"                               as income_category,

        -- Relationship / product
        "Card_Category"                                 as card_category,
        cast("Months_on_book" as integer)               as months_on_book,
        cast("Total_Relationship_Count" as integer)     as total_relationship_count,

        -- Behavioural (12-month aggregates; leakage-suspect, see plan section 5)
        cast("Months_Inactive_12_mon" as integer)       as months_inactive_12_mon,
        cast("Contacts_Count_12_mon" as integer)        as contacts_count_12_mon,
        cast("Credit_Limit" as double)                  as credit_limit,
        cast("Total_Revolving_Bal" as integer)          as total_revolving_bal,
        cast("Avg_Open_To_Buy" as double)               as avg_open_to_buy,
        cast("Total_Amt_Chng_Q4_Q1" as double)          as total_amt_chng_q4_q1,
        cast("Total_Trans_Amt" as integer)              as total_trans_amt,
        cast("Total_Trans_Ct" as integer)               as total_trans_ct,
        cast("Total_Ct_Chng_Q4_Q1" as double)           as total_ct_chng_q4_q1,
        cast("Avg_Utilization_Ratio" as double)         as avg_utilization_ratio

    from source

),

flagged as (

    select
        *,
        -- "Unknown" is missingness wearing a costume: the file has zero NULLs,
        -- but three categoricals encode missing as the literal string.
        -- Flagged explicitly so downstream code treats it as a decision rather
        -- than an accident. The category itself is KEPT (plan section 2) --
        -- measured churn lift is only +0.8 to +1.3pp, so imputing would invent
        -- data for almost no gain.
        (education_level = 'Unknown')                   as education_is_unknown,
        (income_category = 'Unknown')                   as income_is_unknown,
        (marital_status  = 'Unknown')                   as marital_is_unknown
    from renamed

)

select * from flagged
