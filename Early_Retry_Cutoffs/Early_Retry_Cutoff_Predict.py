"""
Early Retry Cutoff -- batch scoring + write-back to Snowflake.

Standalone operational script. It does NOT retrain -- it reuses the artifact
saved by Early_Retry_Cutoff_Model.py (calibrated_catboost_model.joblib).

Each run it:
  1. connects to Snowflake,
  2. pulls the scoring data,
  3. loads the trained model + feature lists from the .joblib,
  4. scores every row,
  5. writes ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_PREDICTIONS.

Run with:  python Early_Retry_Cutoff_Predict.py   (inside the pinned env -- see requirements.txt)
"""
#%% Imports
import os
import joblib
from datetime import datetime

import pytz
import pandas as pd 
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ============================================================================
# Config -- the only knobs you'd change between runs
# ============================================================================
TARGET_TABLE = "ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_PREDICTIONS"    # where to write
MODEL_FILE = "calibrated_catboost_model.joblib"


CUTOFF_THRESHOLD = 0.05

# "replace" -> CREATE OR REPLACE each run; the table holds only the latest snapshot.
# "append"  -> keep history across runs; CUTOFF_DATETIME distinguishes them.
WRITE_MODE = "replace"

TARGET_DB, TARGET_SCHEMA, TARGET_NAME = TARGET_TABLE.split(".")

#%% Fetch Data
print(f"Pulling Retries ...")
ctx = snowflake.connector.connect(
    user=os.environ.get("SNOWFLAKE_USER", "BITEAM"),
    password=os.environ.get("SNOWFLAKE_PASSWORD", "B1sense@22"),
    account=os.environ.get("SNOWFLAKE_ACCOUNT", "YXBYZCG-MVA06208"),
    database="DBT_PROD.PUBLIC",
    warehouse="COMPUTE_WH",
    schema="EARLY_RETRY_CUTOFF",
)
cur = ctx.cursor()

query = f"""
WITH base_transactions AS (
    SELECT
        orders.transaction_id,
        orders.order_id,
        orders.invoice_id,
        orders.transaction_datetime,
        orders.transaction_status,
        orders.retries,
        orders.vertical_type,
        orders.super_partner_id_name,
        orders.processor_name_original,
        bin_tiers.bin_tier,
        orders.card_type,
        orders.bank,
        orders.billing_state,
        orders.offer_amount,
        orders.transaction_amount,
        orders.payment_frequency,
        (orders.offer_amount - orders.transaction_amount) AS offer_minus_payment,
        orders.response_code AS decline_code,
        orders.invoice_type,
        orders.invoice_sequence_number AS invoice_sequence
    FROM DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM AS orders
    LEFT JOIN (
        SELECT bin_tier, vertical_type_bin
        FROM DBT_PROD.ANALYTICS.stg_dbt_prod_analytics_bin_tier_matching
    ) AS bin_tiers
        ON orders.vertical_type_bin = bin_tiers.vertical_type_bin
    WHERE
        orders.retries IS NOT NULL
        AND orders.transaction_type IN ('sale', 'capture')
        AND orders.invoice_type NOT LIKE '%trial%'
        -- Rolling lower bound so this doesn't scan all history. Must be WIDER than any
        -- order lifetime you care about, so prior-invoice history is fully retained.
        AND orders.transaction_datetime >= DATEADD('month', -1, CURRENT_DATE())
        AND membership_status not in ('expired', 'cancelled', 'trial')
),

invoice_scope AS (
    -- Which invoices to SCORE: recently active, not maxed out.
    SELECT invoice_id
    FROM base_transactions
    GROUP BY invoice_id
    HAVING MAX(retries) < 30
       AND MAX(transaction_datetime) >= DATEADD('day', -30, CURRENT_DATE())
),                                             
invoice_summary AS (
    SELECT 
        order_id, 
        invoice_id,
        MIN(transaction_datetime) AS invoice_start_time,
        MAX(retries) AS invoice_max_retries,
        MAX(CASE WHEN transaction_status = 'accepted' THEN 1 ELSE 0 END) AS is_eventually_successful,
        MAX(CASE WHEN transaction_status = 'accepted' AND retries = 0 THEN 1 ELSE 0 END) AS is_success_on_retry_0
    FROM base_transactions  
    GROUP BY order_id, invoice_id
),

order_history AS (
    SELECT 
        order_id,
        invoice_id,
        invoice_start_time,
        SUM(is_eventually_successful) OVER (  
            PARTITION BY order_id 
            ORDER BY invoice_start_time 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_successful_invoices,
        LAG(invoice_max_retries) OVER (
            PARTITION BY order_id 
            ORDER BY invoice_start_time                           
        ) AS last_invoice_retry,
        SUM(invoice_max_retries) OVER (
            PARTITION BY order_id 
            ORDER BY invoice_start_time 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS sum_past_max_retries,
        COUNT(invoice_id) OVER (
            PARTITION BY order_id 
            ORDER BY invoice_start_time 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS count_past_invoices,
        COALESCE(
            SUM(CASE WHEN is_success_on_retry_0 = 0 THEN 1 ELSE 0 END) OVER (
                PARTITION BY order_id 
                ORDER BY invoice_start_time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0
        ) AS streak_group_id,
        is_success_on_retry_0
    FROM invoice_summary
),

consecutive_streaks AS (
    SELECT
        invoice_id,
        prior_successful_invoices,
        last_invoice_retry,
        sum_past_max_retries,
        count_past_invoices,
        COALESCE(
            SUM(is_success_on_retry_0) OVER (
                PARTITION BY order_id, streak_group_id 
                ORDER BY invoice_start_time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0 
        ) AS consecutive_success_retry_0
    FROM order_history
),  

transaction_sequencing AS (
    SELECT 
        b.*,
        i.is_eventually_successful,
        i.invoice_start_time,
        c.prior_successful_invoices,
        c.last_invoice_retry,
        c.consecutive_success_retry_0,
        MAX(b.retries) OVER (
            PARTITION BY b.order_id 
            ORDER BY b.transaction_datetime 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS max_retry_in_order,
        (COALESCE(c.sum_past_max_retries, 0) + b.retries) / 
        (COALESCE(c.count_past_invoices, 0) + 1.0) AS avg_retry_in_order,
        LAG(b.transaction_datetime) OVER (
            PARTITION BY b.invoice_id 
            ORDER BY b.retries 
        ) AS last_failure_datetime,
        LAG(b.decline_code) OVER (
            PARTITION BY b.invoice_id 
            ORDER BY b.retries
        ) AS last_decline_code_raw,
        COALESCE(
            SUM(CASE WHEN b.transaction_status = 'accepted' THEN b.transaction_amount ELSE 0 END) OVER (
                PARTITION BY b.order_id 
                ORDER BY b.transaction_datetime 
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0
        ) AS historical_collected_amount,
        SUM(CASE WHEN b.transaction_status = 'accepted' THEN 1 ELSE 0 END) OVER (
            PARTITION BY b.super_partner_id_name 
            ORDER BY b.transaction_datetime 
            RANGE BETWEEN INTERVAL '8 DAYS' PRECEDING AND INTERVAL '1 SECOND' PRECEDING
        ) / 
        NULLIF(COUNT(*) OVER (
            PARTITION BY b.super_partner_id_name 
            ORDER BY b.transaction_datetime 
            RANGE BETWEEN INTERVAL '8 DAYS' PRECEDING AND INTERVAL '1 SECOND' PRECEDING
        ), 0) AS super_partner_8d_success_rate, 
        SUM(CASE WHEN b.transaction_status = 'accepted' THEN 1 ELSE 0 END) OVER (
            PARTITION BY b.processor_name_original 
            ORDER BY b.transaction_datetime 
            RANGE BETWEEN INTERVAL '8 DAYS' PRECEDING AND INTERVAL '1 SECOND' PRECEDING
        ) / 
        NULLIF(COUNT(*) OVER (
            PARTITION BY b.processor_name_original 
            ORDER BY b.transaction_datetime 
            RANGE BETWEEN INTERVAL '8 DAYS' PRECEDING AND INTERVAL '1 SECOND' PRECEDING
        ), 0) AS processor_8d_success_rate,
        SUM(CASE WHEN b.transaction_status = 'accepted' THEN 1 ELSE 0 END) OVER (
            PARTITION BY b.bank 
            ORDER BY b.transaction_datetime 
            RANGE BETWEEN INTERVAL '8 DAYS' PRECEDING AND INTERVAL '1 SECOND' PRECEDING
        ) / 
        NULLIF(COUNT(*) OVER (  
            PARTITION BY b.bank 
            ORDER BY b.transaction_datetime 
            RANGE BETWEEN INTERVAL '8 DAYS' PRECEDING AND INTERVAL '1 SECOND' PRECEDING
        ), 0) AS bank_8d_success_rate
    FROM base_transactions b
    JOIN invoice_summary i ON b.invoice_id = i.invoice_id
    JOIN consecutive_streaks c ON b.invoice_id = c.invoice_id
), 

final AS (
    SELECT 
        t.order_id,
        t.invoice_id,
        t.transaction_id,
        t.transaction_datetime,  
        invoice_sequence AS invoice_number,
        t.retries,
        t.transaction_status,   
        COALESCE(t.vertical_type, 'unknown') AS vertical_type,
        COALESCE(t.super_partner_id_name, 'unknown') AS super_partner_id_name,
        case when SPLIT(processor_name_original , '-')[1] like '%TRX%' then 'NMI'  
        when SPLIT(processor_name_original , '-')[1] like '%ADYEN%' then 'ADYEN'
        else 'Others' 
        end as processor_type,
        COALESCE(t.bin_tier, 'unknown') AS bin_tier,
        COALESCE(t.card_type, 'unknown') AS card_type,
        COALESCE(t.bank, 'unknown') AS bank,   
        COALESCE(t.billing_state, 'unknown') AS billing_state,
        COALESCE(t.offer_amount, -1) AS offer_amount,
        COALESCE(t.transaction_amount, -1) AS transaction_amount, 
        COALESCE(t.payment_frequency, 'unknown') AS payment_frequency,
        COALESCE(t.offer_minus_payment, -1) AS offer_minus_payment,
        COALESCE(t.last_decline_code_raw, 'unknown') AS last_decline_code,
        COALESCE(t.invoice_type, 'unknown') AS invoice_type, 
        COALESCE(t.max_retry_in_order, -1) AS max_retry_in_order,
        COALESCE(t.avg_retry_in_order, -1) AS avg_retry_in_order,
        COALESCE(t.last_invoice_retry, -1) AS last_invoice_retry,  
        COALESCE(t.prior_successful_invoices, -1) AS prior_successful_invoice_count,
        COALESCE(t.consecutive_success_retry_0, -1) AS consecutive_success_retry_0,  
        t.historical_collected_amount,
        COALESCE(DATEDIFF('day', t.invoice_start_time, t.transaction_datetime), -1) AS days_since_initial_failure,
        COALESCE(DATEDIFF('day', t.last_failure_datetime, t.transaction_datetime), -1) AS days_since_last_failure,
        DAYOFWEEK(t.transaction_datetime) AS retry_day_of_week,
        DAY(t.transaction_datetime) AS retry_day_of_month,
        HOUR(t.transaction_datetime) AS retry_hour,
        COALESCE(t.super_partner_8d_success_rate, -1) AS super_partner_8d_success_rate,
        COALESCE(t.processor_8d_success_rate, -1) AS processor_8d_success_rate,
        COALESCE(t.bank_8d_success_rate, -1) AS bank_8d_success_rate,
        -- NOT a real label on live data: this is "accepted so far", which is 0 for
        -- every currently-declining invoice. Kept only so the schema matches training
        -- (the scoring script drops it). Do NOT run the AUC / confusion blocks against it.
        t.is_eventually_successful
    FROM transaction_sequencing t
    JOIN invoice_scope sc ON sc.invoice_id = t.invoice_id     -- CHANGED: scope restricts OUTPUT here, not feature context
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.invoice_id
        ORDER BY t.transaction_datetime DESC, t.transaction_id DESC
    ) = 1
    AND t.transaction_status <> 'accepted'    -- optional: only score invoices whose latest attempt is a decline (delete to keep all)
)

SELECT * FROM final
ORDER BY order_id, invoice_id, retries        
;
"""

cur.execute("use database dbt_prod")
cur.execute(query)
df_pred = cur.fetch_pandas_all()
print(f"  {len(df_pred):,} rows pulled")

# Columns we need to build the output table (independent of the model features).
required_out_cols = ["ORDER_ID", "INVOICE_ID", "TRANSACTION_ID",
                     "RETRIES", "HISTORICAL_COLLECTED_AMOUNT"]
missing_out = [c for c in required_out_cols if c not in df_pred.columns]
if missing_out:
    raise KeyError(f"Retires missing columns needed for the output table: {missing_out}")

#%% Load the model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, MODEL_FILE)
print(f"Loading model artifact from {model_path} ...")
artifact = joblib.load(model_path)
calibrated_model = artifact["model"]
cat_features = artifact["cat_features"]
text_features = artifact["text_features"]
drop_cols = artifact["drop_cols"]

# ============================================================================
# 4. Preprocess EXACTLY as in Early_Retry_Cutoff_Model.py's prediction cell
# ============================================================================
print("Preprocessing ...")
if "LAST_DECLINE_CODE" in df_pred.columns:
    df_pred["LAST_DECLINE_CODE"] = df_pred["LAST_DECLINE_CODE"].astype(str)
    df_pred["DECLINE_FAMILY"] = df_pred["LAST_DECLINE_CODE"].str[0]

X_new = df_pred.drop(columns=drop_cols, errors="ignore").copy()

# Categorical / text imputation
existing_cat_text = [c for c in (cat_features + text_features) if c in X_new.columns]
if existing_cat_text:
    X_new[existing_cat_text] = X_new[existing_cat_text].fillna("unknown").astype(str)

# Numeric imputation 
numeric_cols = X_new.columns.difference(cat_features + text_features)
if len(numeric_cols) > 0:
    X_new[numeric_cols] = X_new[numeric_cols].fillna(-1)

# Dig the underlying CatBoost out of CalibratedClassifierCV -> FrozenEstimator
if hasattr(calibrated_model, "calibrated_classifiers_"):
    base_wrapper = calibrated_model.calibrated_classifiers_[0].estimator
else:
    base_wrapper = calibrated_model.estimator
actual_catboost = getattr(base_wrapper, "estimator", getattr(base_wrapper, "model", base_wrapper))
feature_names = actual_catboost.feature_names_

# Fail loudly if the scoring data is missing a feature the model expects
missing_feats = [c for c in feature_names if c not in X_new.columns]
if missing_feats:
    raise KeyError(f"Scoring data is missing model features: {missing_feats}")

# Match training column order exactly
X_new = X_new[feature_names]

# ============================================================================
# 5. Predict
# ============================================================================
print("Scoring ...")
success_prob = calibrated_model.predict_proba(X_new)[:, 1]

# ============================================================================
# 6. Assemble the output table
# ============================================================================
# Pacific wall-clock time of this run (PST/PDT), formatted as an ISO STRING on purpose.
# write_pandas (snowflake-connector 4.6.0 + pandas 3.0) mis-scales real datetime columns
# -- it hands Snowflake the raw int64 as if it were epoch-seconds, producing garbage
# "year 56,000,000" timestamps (this was the "invalid date" bug). Passing a string lets
# Snowflake implicitly cast it into the TIMESTAMP_NTZ column correctly.
run_ts_pst = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M:%S.%f")

out = pd.DataFrame({
    "ORDER_ID":         df_pred["ORDER_ID"].astype(str).values,
    "INVOICE_ID":       df_pred["INVOICE_ID"].astype(str).values,
    "TRANSACTION_ID":   df_pred["TRANSACTION_ID"].astype(str).values,
    "IS_CUTOFF_EARLY":  (success_prob < CUTOFF_THRESHOLD),
    "CUTOFF_DATETIME":  run_ts_pst,
    "CUT_OFF_AT_RETRY": df_pred["RETRIES"].values,
    "CURRENT_CLTV":     df_pred["HISTORICAL_COLLECTED_AMOUNT"].values,
})

print(f"  {int(out['IS_CUTOFF_EARLY'].sum()):,} / {len(out):,} rows flagged "
      f"is_cutoff_early (P(success) < {CUTOFF_THRESHOLD})")

# ============================================================================
# 7. Write to Snowflake
# ============================================================================
DDL = f"""
CREATE {{verb}} {TARGET_TABLE} (
    ORDER_ID          VARCHAR,
    INVOICE_ID        VARCHAR,
    TRANSACTION_ID    VARCHAR,
    IS_CUTOFF_EARLY   BOOLEAN,
    CUTOFF_DATETIME   TIMESTAMP_NTZ,
    CUT_OFF_AT_RETRY  NUMBER,
    CURRENT_CLTV      FLOAT
)
"""

if WRITE_MODE == "replace":
    print(f"Replacing {TARGET_TABLE} ...")
    cur.execute(DDL.format(verb="OR REPLACE TABLE"))
else:  # append -- create only if it doesn't exist yet
    print(f"Appending to {TARGET_TABLE} ...")
    cur.execute(DDL.format(verb="TABLE IF NOT EXISTS"))

success, n_chunks, n_rows, _ = write_pandas(
    ctx, out,
    table_name=TARGET_NAME,
    database=TARGET_DB,
    schema=TARGET_SCHEMA,
)
print(f"write_pandas success={success}, rows_written={n_rows:,}")

cur.close()
ctx.close()
print("Done.")

# %%
