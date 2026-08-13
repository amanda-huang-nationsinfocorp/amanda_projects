"""
Early Retry Cutoff -- batch scoring + write-back to Snowflake.
Each run it:
  1. connects to Snowflake,
  2. pulls the scoring data,
  3. loads the trained model + feature lists from the .joblib,
  4. scores every row,
  5. writes ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_PREDICTIONS.
"""
#%% Imports
import os
import joblib
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# ============================================================================
# Config -- the only knobs you'd change between runs
# ============================================================================
TARGET_TABLE = "ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_PREDICTIONS"    # where to write
MODEL_FILE = "calibrated_catboost_model.joblib"


CUTOFF_THRESHOLD = 0.05

WRITE_MODE = "append" # "replace" or "append"

# Randomly flag exactly 10% of rows as True (reproducible via seed)
random_selection = 0.1

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
WITH
live_invoices AS (
    SELECT
        invoice_id,
        order_id
    FROM DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    WHERE retries IS NOT NULL
      AND transaction_type IN ('sale', 'capture')
      AND invoice_type NOT LIKE '%trial%'
      AND membership_status NOT IN ('cancelled', 'expired')
    GROUP BY invoice_id, order_id
    HAVING MAX(retries) < 30
       AND MAX(transaction_datetime) >= DATEADD('day', -30, CURRENT_DATE())
       
),

live_orders AS (
    SELECT DISTINCT order_id FROM live_invoices
),

base_transactions AS (
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
        AND orders.order_id IN (SELECT order_id FROM live_orders)
),

invoice_summary AS (
    -- 2. Determine the outcome and stats of each full invoice cycle
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
    -- 3. Calculate historical order stats
    -- Note: This assumes invoice cycles for a single order do not overlap in time.
    SELECT
        order_id,
        invoice_id,
        invoice_start_time,

        -- Total prior successful invoices
        SUM(is_eventually_successful) OVER (
            PARTITION BY order_id
            ORDER BY invoice_start_time
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_successful_invoices,

        -- Last invoice max retry
        LAG(invoice_max_retries) OVER (
            PARTITION BY order_id
            ORDER BY invoice_start_time
        ) AS last_invoice_retry,

        -- Prep for Dynamic Avg: Sum of past max retries & count of past invoices
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

        -- REVISED Gaps and Islands ID:
        -- Look at previous invoices. Increment group ID when a prior invoice failed on retry 0.
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
    -- 4. Correctly count the streak length WITHOUT nested window functions
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
    -- 5. Calculate point-in-time features dynamically updated row-by-row
    SELECT
        b.*,
        i.is_eventually_successful,
        i.invoice_start_time,

        c.prior_successful_invoices,
        c.last_invoice_retry,
        c.consecutive_success_retry_0,

        -- DYNAMIC MAX RETRY: all transactions for this order up to this exact row
        MAX(b.retries) OVER (
            PARTITION BY b.order_id
            ORDER BY b.transaction_datetime
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS max_retry_in_order,

        -- DYNAMIC AVG RETRY: (sum of past invoices' max retries + current row's retry) / total invoices
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

        -- Identical definition to training: non-trial accepted $ on the order,
        -- strictly before this attempt. Correct here because base_transactions
        -- carries the WHOLE order.
        COALESCE(
            SUM(CASE WHEN b.transaction_status = 'accepted' THEN b.transaction_amount ELSE 0 END) OVER (
                PARTITION BY b.order_id
                ORDER BY b.transaction_datetime
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ), 0
        ) AS historical_collected_amount,

        -- Rolling 8 Days Avg Success Rate per Super Partner
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

        -- Rolling 8 Days Avg Success Rate per Processor Original
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

        -- Rolling 8 Days Avg Success Rate per Bank
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

trial_money AS (
    SELECT
        order_id,
        SUM(CASE WHEN transaction_status = 'accepted' THEN transaction_amount ELSE 0 END) AS trial_collected_amount
    FROM DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    WHERE order_id IN (SELECT order_id FROM live_orders)
      AND invoice_type LIKE '%trial%'
    GROUP BY order_id
),

-- =====================================================================
-- Keep only LIVE invoices, one row each = the most recent attempt.
-- =====================================================================
scored AS (
    SELECT t.*
    FROM transaction_sequencing t
    JOIN live_invoices li ON li.invoice_id = t.invoice_id
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.invoice_id
        ORDER BY t.transaction_datetime DESC, t.transaction_id DESC
    ) = 1
),

-- 6. Final Selection & Imputation Handling
final AS (
    SELECT
        s.order_id,
        s.invoice_id,
        s.transaction_id,
        s.transaction_datetime,
        s.invoice_sequence AS invoice_number,
        s.retries,
        s.transaction_status,
        COALESCE(s.vertical_type, 'unknown') AS vertical_type,
        COALESCE(s.super_partner_id_name, 'unknown') AS super_partner_id_name,
        CASE WHEN SPLIT(s.processor_name_original, '-')[1] LIKE '%TRX%' THEN 'NMI'
             WHEN SPLIT(s.processor_name_original, '-')[1] LIKE '%ADYEN%' THEN 'ADYEN'
             ELSE 'Others'
        END AS processor_type,
        COALESCE(s.bin_tier, 'unknown') AS bin_tier,
        COALESCE(s.card_type, 'unknown') AS card_type,
        COALESCE(s.bank, 'unknown') AS bank,
        COALESCE(s.billing_state, 'unknown') AS billing_state,
        COALESCE(s.offer_amount, -1) AS offer_amount,
        COALESCE(s.transaction_amount, -1) AS transaction_amount,
        COALESCE(s.payment_frequency, 'unknown') AS payment_frequency,
        COALESCE(s.offer_minus_payment, -1) AS offer_minus_payment,
        COALESCE(s.last_decline_code_raw, 'unknown') AS last_decline_code,
        COALESCE(s.invoice_type, 'unknown') AS invoice_type,

        -- Historical & Dynamic Order Features
        COALESCE(s.max_retry_in_order, -1) AS max_retry_in_order,
        COALESCE(s.avg_retry_in_order, -1) AS avg_retry_in_order,
        COALESCE(s.last_invoice_retry, -1) AS last_invoice_retry,
        COALESCE(s.prior_successful_invoices, -1) AS prior_successful_invoice_count,
        COALESCE(s.consecutive_success_retry_0, -1) AS consecutive_success_retry_0,
        s.historical_collected_amount,
 
        -- Temporal Point-in-Time Features
        COALESCE(DATEDIFF('day', s.invoice_start_time, s.transaction_datetime), -1) AS days_since_initial_failure,
        COALESCE(DATEDIFF('day', s.last_failure_datetime, s.transaction_datetime), -1) AS days_since_last_failure,
        DAYOFWEEK(s.transaction_datetime) AS retry_day_of_week,
        DAY(s.transaction_datetime) AS retry_day_of_month,
        HOUR(s.transaction_datetime) AS retry_hour,

        -- Rolling Context Features
        COALESCE(s.super_partner_8d_success_rate, -1) AS super_partner_8d_success_rate,
        COALESCE(s.processor_8d_success_rate, -1) AS processor_8d_success_rate,
        COALESCE(s.bank_8d_success_rate, -1) AS bank_8d_success_rate,

        -- Eval-only (model ignores it)
        COALESCE(tm.trial_collected_amount, 0) AS trial_collected_amount,

        -- Target column kept for schema parity ONLY. On live data this is
        -- "accepted so far" = 0 for every declining invoice, NOT a usable label.
        -- Do not run the .py AUC / confusion blocks against it (single class -> error).
        s.is_eventually_successful 
 
    FROM scored s
    LEFT JOIN trial_money tm ON tm.order_id = s.order_id
    -- Only invoices currently in a declined state are cuttable. Delete this line
    -- to keep every live invoice (e.g. for monitoring).
    WHERE s.transaction_status <> 'accepted'
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

n_random = int(len(out) * random_selection)
random_mask = np.zeros(len(out), dtype=bool)
random_mask[np.random.default_rng(42).choice(len(out), size=n_random, replace=False)] = True
out["RANDOM_SELECTION"] = random_mask

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
    CURRENT_CLTV      FLOAT,
    RANDOM_SELECTION  BOOLEAN
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
