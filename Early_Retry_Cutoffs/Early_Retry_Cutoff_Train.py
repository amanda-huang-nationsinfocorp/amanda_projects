#%% IMPORTS
import os
import hashlib
import joblib

import numpy as np
import pandas as pd
import snowflake.connector
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

#%%  Fetch Data from Snowflake
# Start fetching data directly from Snowflake
ctx = snowflake.connector.connect(
    user='BITEAM',
    password='B1sense@22',
    account='YXBYZCG-MVA06208',
    database="DBT_PROD.PUBLIC",
    warehouse='COMPUTE_WH',
    schema='EARLY_RETRY_CUTOFF'
)

# 2. Create a cursor and execute
cur = ctx.cursor()
cur.execute(("use database dbt_prod"))
cur.execute("SELECT * FROM ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_DATA")

# 3. Fetch directly to a Pandas DataFrame using Arrow
# This is drastically faster than pd.read_sql()
df = cur.fetch_pandas_all()
print(f"Fetched {len(df):,} rows from Snowflake.")

#%% Model Training
# Model Training Starts Here
sampled_df = df

# Must match the DATEDIFF threshold in the training SQL's in-flight filter
MATURITY_DAYS = 30

# =====================================================================
# 1. Base Filtering and Sorting
# =====================================================================
sampled_df['LAST_DECLINE_CODE'] = sampled_df['LAST_DECLINE_CODE'].astype(str)
sampled_df['DECLINE_FAMILY'] = sampled_df['LAST_DECLINE_CODE'].str[0]

sampled_df['TRANSACTION_DATETIME'] = pd.to_datetime(sampled_df['TRANSACTION_DATETIME'])

# Sort ascending to guarantee Past predicts Future
sampled_df = sampled_df.sort_values('TRANSACTION_DATETIME').reset_index(drop=True)

# =====================================================================
# 1b. Invoice resolution timing
# =====================================================================

print("Computing invoice resolution times...")

invoice_times = sampled_df.groupby('INVOICE_ID', sort=False).agg(
    INVOICE_START=('TRANSACTION_DATETIME', 'min'),
    INVOICE_LAST_TXN=('TRANSACTION_DATETIME', 'max'),
    IS_EVENTUALLY_SUCCESSFUL=('IS_EVENTUALLY_SUCCESSFUL', 'first'),
)

first_accept = (
    sampled_df.loc[sampled_df['TRANSACTION_STATUS'] == 'accepted']
              .groupby('INVOICE_ID')['TRANSACTION_DATETIME'].min()
)

invoice_times['RESOLVED_AT'] = first_accept.reindex(invoice_times.index).where(
    invoice_times['IS_EVENTUALLY_SUCCESSFUL'] == 1,
    invoice_times['INVOICE_START'] + pd.Timedelta(days=MATURITY_DAYS)
)

if invoice_times['RESOLVED_AT'].isna().any():
    raise ValueError(
        f"{invoice_times['RESOLVED_AT'].isna().sum()} invoices are labelled successful "
        "but have no accepted transaction. Did the != 'accepted' filter get left in the SQL?"
    )

sampled_df = sampled_df.merge(
    invoice_times[['INVOICE_START', 'INVOICE_LAST_TXN', 'RESOLVED_AT']],
    left_on='INVOICE_ID', right_index=True, how='left'
)

# Now drop the accepted rows -- at serving time the model only ever scores declines.
#sampled_df = sampled_df[sampled_df['TRANSACTION_STATUS'] != 'accepted'].reset_index(drop=True)
print(f"  {len(sampled_df):,} rows | {sampled_df['INVOICE_ID'].nunique():,} invoices "
      f"| {sampled_df['ORDER_ID'].nunique():,} orders")

# =======================================================================
# 2. Data Processing & Pre-Split Vectorized 
print("Processing and Imputing Data...")
drop_cols = ['ORDER_ID', 'INVOICE_ID', 'TRANSACTION_ID', 'TRANSACTION_DATETIME',
             'IS_EVENTUALLY_SUCCESSFUL', 'TRANSACTION_STATUS',
             'INVOICE_START', 'INVOICE_LAST_TXN', 'RESOLVED_AT']

# Separate text features from standard categorical features for better string processing
text_features = [col for col in ['SUPER_PARTNER_ID_NAME', 'BANK'] if col in sampled_df.columns]

# Get remaining categorical features (Using errors='ignore' safely)
cat_features = sampled_df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=['object', 'category']).columns.tolist()
cat_features = [c for c in cat_features if c not in text_features]

# Vectorized Imputation BEFORE the split (Massive speedup)
sampled_df[cat_features + text_features] = sampled_df[cat_features + text_features].fillna('unknown').astype(str)

# Get purely numeric feature columns (excluding drops and target)
numeric_cols = sampled_df.columns.difference(cat_features + text_features + drop_cols)
sampled_df[numeric_cols] = sampled_df[numeric_cols].fillna(-1)

# =====================================================================
# 3. Dynamic Chronological Split (85% Past / 15% Recent OOT)
# =====================================================================
print("Splitting data (85% Past / 15% Recent OOT)...")
 
# 1. Determine the maximum date in the dataset to calculate maturity
MAX_DATE = sampled_df['TRANSACTION_DATETIME'].max()
MATURITY_CUTOFF = MAX_DATE - pd.Timedelta(days=MATURITY_DAYS)

# 2. Get invoice-level start times
invoice_elig = sampled_df.groupby('INVOICE_ID', sort=False).agg(
    INVOICE_START=('TRANSACTION_DATETIME', 'min')
)
# 3. Filter for ONLY completely finished invoices (started at least 30 days ago)  
mature_invoices = invoice_elig[invoice_elig['INVOICE_START'] <= MATURITY_CUTOFF].copy()

# 4. Find the chronological cutoff date that splits the mature invoices 85/15
split_date = mature_invoices['INVOICE_START'].quantile(0.85)
  
print(f"  Dataset max date: {MAX_DATE.date()}")
print(f"  Maturity cutoff:  {MATURITY_CUTOFF.date()}")
print(f"  85/15 Split date: {split_date.date()}")

# 5. Split the mature invoices into Past and OOT based on the split date
past_invoices = mature_invoices[mature_invoices['INVOICE_START'] < split_date].index
oot_invoices  = mature_invoices[mature_invoices['INVOICE_START'] >= split_date].index

# 6. Build the final dataframes (keeping all transactions tied to these invoices)
past_data = sampled_df[sampled_df['INVOICE_ID'].isin(past_invoices)].copy()
oot_test_df = sampled_df[sampled_df['INVOICE_ID'].isin(oot_invoices)].copy()

# Safety check: Ensure no invoice exists in both datasets
assert not (set(past_data['INVOICE_ID']) & set(oot_test_df['INVOICE_ID'])), \
    "Error: Invoices straddle the split boundary"

print(f"  past_data   : {len(past_data):>9,} rows | {past_data['ORDER_ID'].nunique():>7,} orders")
print(f"  oot_test_df : {len(oot_test_df):>9,} rows | {oot_test_df['ORDER_ID'].nunique():>7,} orders")
print(f"  orders in both (expected if customers return): "
      f"{len(set(past_data['ORDER_ID']) & set(oot_test_df['ORDER_ID'])):,}")
print(f"  dropped (immature, still in-flight): "
      f"{len(sampled_df) - len(past_data) - len(oot_test_df):,} rows")


# hashlib rather than hash(): Python salts str hashing per process, so hash() would
# reshuffle orders across splits on every run. This assignment is stable, and stays
# stable when new orders are added next month.
def order_hash_frac(order_ids, salt='retry_cutoff_v1'):
    return np.array([
        int(hashlib.md5(f'{salt}:{o}'.encode()).hexdigest()[:16], 16) / 2**64
        for o in order_ids
    ])

unique_orders = past_data['ORDER_ID'].unique()
order_frac = pd.Series(order_hash_frac(unique_orders), index=unique_orders)
row_frac = past_data['ORDER_ID'].map(order_frac)

# Same 75 / 10 / 10 / 5 boundaries as before -- but over ORDERS, not rows.
train_df       = past_data[row_frac < 0.75].copy()
stop_df        = past_data[(row_frac >= 0.75) & (row_frac < 0.85)].copy()
calib_df       = past_data[(row_frac >= 0.85) & (row_frac < 0.95)].copy()
random_test_df = past_data[row_frac >= 0.95].copy()

for _name, _part in [('train_df', train_df), ('stop_df', stop_df),
                     ('calib_df', calib_df), ('random_test_df', random_test_df)]:
    print(f"  {_name:<15}: {len(_part):>9,} rows | {_part['ORDER_ID'].nunique():>7,} orders "
          f"| base rate {_part['IS_EVENTUALLY_SUCCESSFUL'].mean():.4f}")

for _a, _b in [('train_df', train_df), ('stop_df', stop_df),
               ('calib_df', calib_df), ('random_test_df', random_test_df)]:
    for _c, _d in [('train_df', train_df), ('stop_df', stop_df),
                   ('calib_df', calib_df), ('random_test_df', random_test_df)]:
        if _a < _c:
            assert not (set(_b['ORDER_ID']) & set(_d['ORDER_ID'])), f"{_a}/{_c} share orders"
print("  [ok] no ORDER_ID spans two splits")

# 4. Separate X and y (Using errors='ignore' safely)
X_train, y_train = train_df.drop(columns=drop_cols, errors='ignore'), train_df['IS_EVENTUALLY_SUCCESSFUL']
X_stop,  y_stop  = stop_df.drop(columns=drop_cols, errors='ignore'),  stop_df['IS_EVENTUALLY_SUCCESSFUL']
X_calib, y_calib = calib_df.drop(columns=drop_cols, errors='ignore'), calib_df['IS_EVENTUALLY_SUCCESSFUL']

X_random_test, y_random_test = random_test_df.drop(columns=drop_cols, errors='ignore'), random_test_df['IS_EVENTUALLY_SUCCESSFUL']
X_oot_test,    y_oot_test    = oot_test_df.drop(columns=drop_cols, errors='ignore'),    oot_test_df['IS_EVENTUALLY_SUCCESSFUL']

# =====================================================================
# 4. Create Memory-Efficient CatBoost Pools
# =====================================================================
print("Building data pools...")
train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features)
stop_pool  = Pool(X_stop,  y_stop,  cat_features=cat_features, text_features=text_features)

# =====================================================================
# 5. Base Model Definition
# =====================================================================
print("Training Base Model...")
base_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.1,          # Updated Learning Rate
    depth=7,
    task_type="CPU",
    eval_metric='Logloss',
    custom_metric=['AUC'],
    l2_leaf_reg=6,
    thread_count=-1,
    early_stopping_rounds=50,
    verbose=50,
    random_state=42,
    cat_features=cat_features,
    text_features=text_features
)

# Train the model, strictly using the Stop pool
base_model.fit(train_pool, eval_set=stop_pool)

# =====================================================================
# 6. Probability Calibration
# =====================================================================
print("Calibrating Probabilities...")
calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method='isotonic'
)
calibrated_model.fit(X_calib, y_calib)

# =====================================================================
# 7. Final Dual-Evaluation (Random vs. Chronological)
# =====================================================================
print("\nEvaluating Model on Dual Test Sets...")

def evaluate_split(split_name, model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    loss = log_loss(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    print(f"{split_name: <15} | ROC-AUC: {auc:.4f} | Log Loss: {loss:.4f} | Brier Score: {brier:.4f}")
    return y_prob

print("--- Model Performance Summary ---")
# 1. Unseen ORDERS from the same period as training -> generalization to new customers.
random_probs = evaluate_split("Random Test", calibrated_model, X_random_test, y_random_test)

# 2. The strict future -> what the model would actually have scored after ASOF_DATE.
oot_probs    = evaluate_split("OOT Test", calibrated_model, X_oot_test, y_oot_test)

# Calculate the Degradation Penalty
auc_drop = roc_auc_score(y_random_test, random_probs) - roc_auc_score(y_oot_test, oot_probs)
print(f"\nTemporal Degradation Penalty (AUC Drop): {auc_drop:.4f}")
# CHANGED: both sides are now order-grouped and label-mature, so this gap is no longer
# contaminated by row duplication or by lifecycle-tail enrichment. What remains is
# genuine drift between the training period and the deployment window.
print("*(Both splits are order-grouped and label-mature. A large drop now means real")
print("  behavioural/processor drift between the training period and OOT window.)*")


#%% Save Model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'calibrated_catboost_model.joblib')

# 2. Package the model and the exact feature lists into one dictionary
# This ensures your prediction script applies the exact same preprocessing
model_artifact = {
    'model': calibrated_model,
    'cat_features': cat_features,
    'text_features': text_features,
    'drop_cols': drop_cols,
}

# 3. Save the artifact to the script's directory
joblib.dump(model_artifact, model_path)

print(f"Model artifact successfully saved to: {model_path}")
