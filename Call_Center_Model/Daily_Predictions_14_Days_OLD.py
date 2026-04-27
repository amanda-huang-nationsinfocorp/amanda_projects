#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snowflake.connector
from sqlalchemy import create_engine
from sqlalchemy import text
from snowflake.sqlalchemy import URL
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")  

from sklearn.linear_model import LassoCV
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

from snowflake.connector.pandas_tools import write_pandas
from datetime import date


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


url = URL(
    user='BITEAM',
    password='B1sense@22',
    account='YXBYZCG-MVA06208',
    database="DBT_PROD.PUBLIC",
)
engine = create_engine(url) 
connection = engine.connect()


# In[4]:


holiday_function = '''
CREATE OR REPLACE FUNCTION get_holiday_name(dt DATE)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('holidays')
HANDLER = 'check_holiday'
AS
$$
import holidays

def check_holiday(dt):
    if dt is None:
        return 'None'
    # Initialize the US holiday calendar
    us_holidays = holidays.US()
    # .get() returns the holiday name if it exists, otherwise returns 'None'
    return us_holidays.get(dt, 'None')
$$;
'''
 
query = '''
with 
calls_logs as (
    select 
    *,
    SPLIT_PART(customers_tb.vertical_type_bin, '_', 1) as vertical_type
    from DBT_PROD.ANALYTICS.FCT_CSR_CALL_LOG 
    left join DBT_PROD.ANALYTICS.FCT_CUSTOMER_SALES as customers_tb using(customer_id)
    where call_start_at > $start_date and call_start_at < $end_date 
    and call_type in ('Inbound')
),

date_spine as (
    select 
    calendar_date as date,
    day_name,
    day_of_month,
    month_name,
    IFF(DAYOFMONTH(calendar_date) IN (1, 3, 15) OR DAYNAME(calendar_date) IN ('Fri', 'Sat'), 1, 0) AS is_billing_day,
    IFF(get_holiday_name(calendar_date) != 'None', 1, 0) AS is_holiday
    from DBT_PROD.ANALYTICS.DIM_CALENDAR
),

calls_agg as (
    select 
    call_date as date,
    count(*) as num_calls
    from calls_logs
    group by 1
),

transactions_agg as (
    select 
    transaction_date as date,
    count(case when cascade_type is null then transaction_id else null end) as num_core_txn,
    count(case when cascade_type is not null then transaction_id else null end) as num_casc_txn
    from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    where transaction_date > $start_date and transaction_date < $end_date and is_sale = 1 and invoice_type != 'trial_invoice'
    group by 1
),

m0_agg as (
    select 
    order_date as date,
    count(case when cascade_type is null and is_m0 = 1 then order_id else null end) as num_core_m0,
    count(case when cascade_type is not null and is_m0 = 1  then order_id else null end) as num_casc_m0
    from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    where order_date > $start_date and order_date < $end_date and is_sale = 1
    group by 1
),

date_distance as (
    select
    date,
    MAX(IFF(is_billing_day=1, date, NULL)) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_billing_date,
    MIN(IFF(is_billing_day=1, date, NULL)) OVER (ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_billing_date,
    MAX(IFF(is_holiday=1, date, NULL)) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_holiday_date,
    MIN(IFF(is_holiday=1, date, NULL)) OVER (ORDER BY date ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS next_holiday_date
    from date_spine
),

final as (
    select  
    date_spine.*,  
    calls_agg.num_calls,
    LAG(transactions_agg.num_core_txn, 14) OVER (ORDER BY date) AS num_core_txn_14d_ago,
    LAG(transactions_agg.num_casc_txn, 14) OVER (ORDER BY date) AS num_casc_txn_14d_ago,
    LAG(m0_agg.num_core_m0, 14) OVER (ORDER BY date) AS num_core_m0_14d_ago,
    LAG(m0_agg.num_casc_m0, 14) OVER (ORDER BY date) AS num_casc_m0_14d_ago,
    
    -- days since/until
    IFF(
        DATEDIFF('day', last_holiday_date, date) <= 2 OR 
        DATEDIFF('day', date, next_holiday_date) <= 2, 
        1, 0
    ) AS is_near_holiday,
    DATEDIFF('day', last_billing_date, date) AS days_since_billing_day,
    DATEDIFF('day', date, next_billing_date) AS days_until_billing_day,
    
    -- tnx and m0
    AVG(num_core_txn) OVER (ORDER BY date ROWS BETWEEN 16 PRECEDING AND 14 PRECEDING) AS rolling_avg_core_txn_3d,
    AVG(num_casc_txn) OVER (ORDER BY date ROWS BETWEEN 16 PRECEDING AND 14 PRECEDING) AS rolling_avg_casc_txn_3d,
    AVG(num_core_m0) OVER (ORDER BY date ROWS BETWEEN 16 PRECEDING AND 14 PRECEDING) AS rolling_avg_core_m0_3d,
    AVG(num_casc_m0) OVER (ORDER BY date ROWS BETWEEN 16 PRECEDING AND 14 PRECEDING) AS rolling_avg_casc_m0_3d,
    
    -- calls
    LAG(num_calls, 14) OVER (ORDER BY date) AS calls_14d_ago,
    AVG(num_calls) OVER (ORDER BY date ROWS BETWEEN 16 PRECEDING AND 14 PRECEDING) AS rolling_avg_calls_3d,
    AVG(num_calls) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING) AS rolling_avg_calls_7d,
    AVG(num_calls) OVER (ORDER BY date ROWS BETWEEN 43 PRECEDING AND 14 PRECEDING) AS rolling_avg_calls_30d,
    
    -- NEW: Historical Day of Week Ratio (Calls 14 days ago / 7-day average 14 days ago)
    LAG(num_calls, 14) OVER (ORDER BY date) / 
        NULLIF(AVG(num_calls) OVER (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 14 PRECEDING), 0) AS dow_historical_ratio_14d

    from date_spine
    left join calls_agg using(date)
    left join transactions_agg using(date)
    left join m0_agg using(date)
    left join date_distance using(date)
)

select *
from final
where date > $result_start_date and date < $result_end_date

'''


# In[5]:


end_date_retrain = (datetime.today().date() - timedelta(days=14))
end_date_retrain_str = end_date_retrain.strftime('%Y-%m-%d %H:%M:%S')


# In[6]:


with engine.connect() as con:
    con.execute(text("use database dbt_prod"))
    con.execute(text("use schema ANALYTICS"))
    con.execute(text(holiday_function))
    con.execute(text(f"set start_date = '2023-01-01';"))
    con.execute(text(f"set end_date = '{end_date_retrain_str}';"))
    con.execute(text(f"set result_start_date = '2023-01-01';"))
    con.execute(text(f"set result_end_date = '{end_date_retrain_str}';"))
    df_raw = pd.read_sql(query, con)


# # Best Version So Far

# In[8]:


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit

# -------------------------------------------------------------
# 1. DATA PREP & LEAK PREVENTION
# -------------------------------------------------------------
# Ensure chronological order
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)

# Define the anchor and target ratio using your SQL-shifted 7-day average
df_raw['anchor_7d'] = df_raw['rolling_avg_calls_7d']
df_raw['target_ratio'] = df_raw['num_calls'] / (df_raw['anchor_7d'] + 1)

# [FIX APPLIED]: Calculate the historical ratio directly on df_raw BEFORE dropping rows
df_raw['historical_ratio_14d'] = df_raw['calls_14d_ago'] / (df_raw['rolling_avg_calls_7d'].shift(14) + 1)

# [FIX APPLIED]: Do a single, clean dropna that includes all required columns (including the new ratio!)
df_clean = df_raw.dropna(subset=[
    'num_calls', 'target_ratio', 'anchor_7d', 'rolling_avg_calls_30d', 'historical_ratio_14d'
]).copy()

# Apply a slight linear weight so recent data influences the model slightly more
df_clean['sample_weight'] = np.linspace(1.0, 2.0, len(df_clean))

# CRITICAL LEAK PREVENTION: 
# Exclude the mathematical target components AND the un-shifted current-day transactions 
exclude_cols = [
    'date', 'target_ratio', 'num_calls', 'sample_weight', 'anchor_7d'
]

feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
cat_features = ['day_name', 'month_name'] # CatBoost handles these natively

# ------------------------------------------------------------------
# 2. ISOLATE THE FINAL HOLDOUT SET
# -------------------------------------------------------------------
# The last 100 days are locked away for the ultimate deployment test
df_holdout = df_clean.iloc[-90:].copy()
df_cv = df_clean.iloc[:-90].copy()

print(f"Total clean rows: {len(df_clean)}")
print(f"Features used: {len(feature_cols)} (Current-day leakage metrics dropped)")
print(f"Rows for CV (Train/Val/Test): {len(df_cv)}")
print(f"Rows for Final Holdout Test: {len(df_holdout)}\n") 

# -----------------------------------------------------------------  
# 3. 3-WAY CROSS-VALIDATION (Expanding Window) 
# ----------------------------------------------------------------- 
X_cv = df_cv[feature_cols]
y_cv = df_cv['target_ratio']
weights_cv = df_cv['sample_weight']

# Test on 60-day chunks. We use 60 days of validation for early stopping  
tscv = TimeSeriesSplit(n_splits=5, test_size=60)
val_size = 60 

fold_biases = []
best_iterations = []
all_results = [] 

# [NEW METRICS ADDED]: Lists to store observability metrics across folds
fold_maes = []
fold_avg_calls = []
fold_mae_pcts = []

print("Running 3-Way Cross Validation (Train -> Val -> Test)...\n")

for fold, (train_plus_val_idx, test_idx) in enumerate(tscv.split(X_cv)):
    
    # 3-Way Temporal Split: Train is the oldest data, Val is newer, Test is the newest
    train_idx = train_plus_val_idx[:-val_size]
    val_idx = train_plus_val_idx[-val_size:]
    
    # Extract Train Set
    X_train_cv, y_train_cv = X_cv.iloc[train_idx], y_cv.iloc[train_idx]
    w_train_cv = weights_cv.iloc[train_idx]
    
    # Extract Validation Set (Used ONLY for early stopping)
    X_val_cv, y_val_cv = X_cv.iloc[val_idx], y_cv.iloc[val_idx] 
    
    # Extract Test Set (Used ONLY for final scoring)
    X_test_cv, y_test_cv = X_cv.iloc[test_idx], y_cv.iloc[test_idx]
    
    model_cv = CatBoostRegressor( 
        iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=5, 
        cat_features=cat_features, eval_metric='MAE', 
        random_seed=42, early_stopping_rounds=75, verbose=0
    ) 
    
    # Train and Validate
    model_cv.fit(X_train_cv, y_train_cv, sample_weight=w_train_cv, 
                 eval_set=(X_val_cv, y_val_cv), use_best_model=True)
    
    best_iterations.append(model_cv.get_best_iteration())
    
    # Predict on unseen Test Set
    preds_ratio = model_cv.predict(X_test_cv)
    
    # Convert Ratio back to Volume using the anchor (which is safely 14 days old) 
    final_preds = preds_ratio * df_cv['anchor_7d'].iloc[test_idx].values
    actual_calls = df_cv['num_calls'].iloc[test_idx].values
    
    # Calculate Fold Metrics
    pct_bias = (np.sum(final_preds) - np.sum(actual_calls)) / np.sum(actual_calls)
    fold_biases.append(pct_bias) 
    
    # Calculate MAE and Average Calls
    mae = np.mean(np.abs(final_preds - actual_calls))
    avg_call_volume = np.mean(actual_calls)
    mae_pct = mae / avg_call_volume if avg_call_volume > 0 else 0
    
    fold_maes.append(mae)
    fold_avg_calls.append(avg_call_volume)
    fold_mae_pcts.append(mae_pct)
    
    start_date = df_cv['date'].iloc[test_idx[0]].strftime('%Y-%m-%d')
    end_date = df_cv['date'].iloc[test_idx[-1]].strftime('%Y-%m-%d')
    print(f"Fold {fold+1} Test ({start_date} to {end_date}): Bias = {pct_bias:+.2%} | MAE = {mae:.1f} | Avg Calls = {avg_call_volume:.0f} | MAE/Avg = {mae_pct:.2%} | Opt Trees: {model_cv.get_best_iteration()}")

    # Store results for analysis  
    fold_df = pd.DataFrame({
        'date': df_cv['date'].iloc[test_idx],
        'actual_calls': actual_calls,
        'predicted_calls': np.round(final_preds, 0),
        'error': np.round(final_preds - actual_calls, 0),
        'fold': fold + 1
    })
    all_results.append(fold_df)

avg_best_iter = int(np.mean(best_iterations))

print("\n----------------------------------------------")
print(f"Average CV Bias:    {np.mean(fold_biases):+.2%}")
print(f"Average CV MAE:     {np.mean(fold_maes):.1f}")
print(f"Average Actuals:    {np.mean(fold_avg_calls):.0f}")
print(f"Average CV MAE/Avg: {np.mean(fold_mae_pcts):.2%}")
print(f"Average Opt Trees:  {avg_best_iter}")
print("------------------------------------------------")

# -----------------------------------------------------------
# 4. DEPLOYMENT SIMULATION (Master Model on Holdout Set)
# -----------------------------------------------------------
print("\nTraining Master Model on 100% of CV Data and predicting Holdout Set...\n")

# Use the exact average optimal trees found during cross-validation
master_model = CatBoostRegressor(
    iterations=avg_best_iter, learning_rate=0.03, depth=6, l2_leaf_reg=5,
    cat_features=cat_features, eval_metric='MAE', 
    random_seed=42, verbose=0
)

# Train on all CV data (Train + Val + Test from previous steps)
master_model.fit(X_cv, y_cv, sample_weight=weights_cv)

# Predict strictly on the isolated Holdout Set
X_holdout = df_holdout[feature_cols]
holdout_preds_ratio = master_model.predict(X_holdout)

# Reconstruct Volume
holdout_final_preds = holdout_preds_ratio * df_holdout['anchor_7d'].values
holdout_actuals = df_holdout['num_calls'].values

# Compile Metrics
holdout_results = pd.DataFrame({
    'date': df_holdout['date'],
    'actual_calls': holdout_actuals,
    'predicted_calls': np.round(holdout_final_preds, 0),
    'error': np.round(holdout_final_preds - holdout_actuals, 0)
})
holdout_results['abs_error'] = np.abs(holdout_results['error'])

sum_actual_h = np.sum(holdout_actuals)
sum_pred_h = np.sum(holdout_final_preds)

holdout_bias = (sum_pred_h - sum_actual_h) / sum_actual_h
holdout_wmape = np.sum(holdout_results['abs_error']) / sum_actual_h
holdout_mae = np.mean(holdout_results['abs_error'])
holdout_avg_calls = np.mean(holdout_actuals)
holdout_mae_pct = holdout_mae / holdout_avg_calls if holdout_avg_calls > 0 else 0

print(f"HOLDOUT BIAS:    {holdout_bias:+.2%}")
print(f"HOLDOUT WMAPE:   {holdout_wmape:.2%}")
print(f"HOLDOUT MAE:     {holdout_mae:.1f}")
print(f"HOLDOUT MAE/Avg: {holdout_mae_pct:.2%}")

print("\n--- Top 5 Worst Prediction Days in Deployment Simulation ---")
print(holdout_results.sort_values('abs_error', ascending=False).head(5).to_string(index=False))


# # Predicting 

# In[10]:


start_date_recent = datetime.today().date() 
start_date_recent_str = start_date_recent.strftime('%Y-%m-%d %H:%M:%S')
end_date_recent = (datetime.today().date() + timedelta(days=14))
end_date_recent_str = end_date_recent.strftime('%Y-%m-%d %H:%M:%S')


# In[11]:


end_date_recent_str


# In[12]:


with engine.connect() as con:
    con.execute(text("use database dbt_prod"))
    con.execute(text("use schema ANALYTICS"))
    con.execute(text(holiday_function))
    con.execute(text(f"set start_date = '2026-01-01';"))
    con.execute(text(f"set end_date = '{end_date_recent_str}';"))
    con.execute(text(f"set result_start_date = '{start_date_recent_str}';"))
    con.execute(text(f"set result_end_date = '{end_date_recent_str}';"))
    df_raw_recent = pd.read_sql(query, con) 


# In[17]:


df_raw_recent.columns


# In[19]:


# ---------------------------------------------------------
# 5. PREDICT ON RECENT/NEW DATA
# -------------------------------------------------------- 

# 1. Format dates and ensure chronological order
df_raw_recent['date'] = pd.to_datetime(df_raw_recent['date'])
df_raw_recent = df_raw_recent.sort_values('date').reset_index(drop=True)

# 2. Define the anchor (SQL already gave us rolling_avg_calls_7d)
df_raw_recent['anchor_7d'] = df_raw_recent['rolling_avg_calls_7d']
df_raw_recent['historical_ratio_14d'] = df_raw_recent['calls_14d_ago'] / (df_raw_recent['rolling_avg_calls_7d'].shift(14) + 1)

# 3. Clean up NaNs (We no longer use .shift(14) here since SQL did it!)
# We just make sure SQL didn't pass us any unexpected NULLs in our core columns.
# NOTE: We DO NOT drop on 'num_calls' here, because for future dates, actual calls will be NaN!
df_recent_clean = df_raw_recent.dropna(subset=['anchor_7d', 'dow_historical_ratio_14d']).copy()

# 4. Isolate features
# Make sure X_recent has the exact same columns (and column order) as the training data
X_recent = df_recent_clean[feature_cols]

# 5. Predict the ratio using the trained master_model 
recent_preds_ratio = master_model.predict(X_recent)

# 6. Convert the predicted ratio back into total call volume
recent_predicted_calls = recent_preds_ratio * df_recent_clean['anchor_7d'].values

# 7. Compile the final output DataFrame
recent_predictions_df = pd.DataFrame({
    'date': df_recent_clean['date'],
    'predicted_calls': np.round(recent_predicted_calls, 0)
})


# In[21]:


def append(df):
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    df['pulled_at_utc'] = datetime.now(timezone.utc)
    df["pulled_at_utc"] = df["pulled_at_utc"].dt.strftime('%Y-%m-%d %H:%M:%S.%f')

    df.columns = df.columns.str.upper()
    
    # connect
    conn = snowflake.connector.connect(
    user="BITEAM",
    password="B1sense@22",
    account="YXBYZCG-MVA06208",
    warehouse="PC_DBT_WH",
    database="ANALYTICS",
    schema="CALL_VOLUME_PREDICTION_PROJECT"
    )
    
    # upload efficiently
    success, nchunks, nrows, _ = write_pandas(
        conn,
        df,
        table_name='CALL_VOLUME_PREDICTIONS_DAILY',
        database='ANALYTICS',
        schema='CALL_VOLUME_PREDICTION_PROJECT',
        overwrite=False
    )
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"------------Upload successfully at {timestamp_now}. {nrows} rows uploaded.-----------------")
 


# In[ ]:


append(recent_predictions_df)


# In[ ]:




