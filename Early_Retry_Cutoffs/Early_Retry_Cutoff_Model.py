#%% IMPORTS
import pandas as pd
import numpy as np
import json
import re
import snowflake.connector
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
from datetime import datetime, timedelta, timezone
import pytz
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier


#%%  Fetch Data from Snowflake

# 1. Connect natively (bypassing SQLAlchemy)
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

cur.close()
ctx.close()

#%% Model Loss Score 0.3633

# 2. Data Processing
# Drop identifiers to prevent memorization
drop_cols = ['ORDER_ID', 'INVOICE_ID', 'TRANSACTION_ID', 'TRANSACTION_DATETIME']

# Force decline codes to be strings (categorical) instead of mathematical numbers
df['DECLINE_CODE'] = df['DECLINE_CODE'].astype(str)

# Separate Features (X) and Target (y)
X = df.drop(columns=drop_cols + ['IS_EVENTUALLY_SUCCESSFUL'])
y = df['IS_EVENTUALLY_SUCCESSFUL']

# Identify all categorical features for CatBoost
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Enforce Imputation Rules (Failsafe in case SQL missed any)
for col in cat_features:
    X[col] = X[col].fillna('unknown').astype(str)
for col in X.columns.difference(cat_features):
    X[col] = X[col].fillna(-1)

# 3. Grouped Train/Test Split
# Ensures all retries of the same invoice stay strictly in train OR test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['INVOICE_ID']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 4. Base Model Definition
# CatBoost is chosen for its superior handling of high-cardinality categoricals
base_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=cat_features,
    verbose=50, # Prints progress every 50 steps
    random_state=42
)

# 5. Probability Calibration Wrapper (The EV logic safeguard)
# 'isotonic' scales the raw tree outputs to true statistical probabilities
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

print("Training Calibrated Model...")
calibrated_model.fit(X_train, y_train)

# 6. Evaluation
# Extract the probability of class 1 (Successful)
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
loss = log_loss(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\n--- Model Performance ---")
print(f"ROC-AUC:     {auc:.4f}  (Measures ranking ability: 1.0 is perfect)")
print(f"Log Loss:    {loss:.4f}  (Measures overall prediction confidence: closer to 0 is better)")
print(f"Brier Score: {brier:.4f}  (Measures true probability calibration accuracy: closer to 0 is better)")

# Example: View the predicted probability of the first 5 test rows
demo_output = X_test.copy()
demo_output['PREDICTED_PROBABILITY'] = y_prob
demo_output['ACTUAL_SUCCESS'] = y_test
print("\n--- Sample Predictions ---")
print(demo_output[['RETRY_COUNT', 'DECLINE_CODE', 'PREDICTED_PROBABILITY', 'ACTUAL_SUCCESS']].head())


#%% Model V2
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier, Pool


# Force decline codes to be strings
df['DECLINE_CODE'] = df['DECLINE_CODE'].astype(str)

# 2. Chronological Split (Out-of-Time)
# Instead of random splitting, we simulate the real world: Past predicts Future.
# Assuming 35M rows: ~70% Train, ~15% Validation, ~15% Test
n = len(df)
train_end = int(n * 0.70)
valid_end = int(n * 0.85)

train_df = df.iloc[:train_end]
valid_df = df.iloc[train_end:valid_end]
test_df  = df.iloc[valid_end:]

# 3. Data Processing (Drop IDs and separate target)
drop_cols = ['ORDER_ID', 'INVOICE_ID', 'TRANSACTION_ID', 'TRANSACTION_DATETIME', 'IS_EVENTUALLY_SUCCESSFUL']
cat_features = df.drop(columns=drop_cols).select_dtypes(include=['object', 'category']).columns.tolist()

X_train, y_train = train_df.drop(columns=drop_cols), train_df['IS_EVENTUALLY_SUCCESSFUL']
X_valid, y_valid = valid_df.drop(columns=drop_cols), valid_df['IS_EVENTUALLY_SUCCESSFUL']
X_test,  y_test  = test_df.drop(columns=drop_cols),  test_df['IS_EVENTUALLY_SUCCESSFUL']

# Impute NaNs strictly
for col in cat_features:
    for dataset in [X_train, X_valid, X_test]:
        dataset[col] = dataset[col].fillna('unknown').astype(str)
for col in X_train.columns.difference(cat_features):
    for dataset in [X_train, X_valid, X_test]:
        dataset[col] = dataset[col].fillna(-1)

# 4. Create Memory-Efficient CatBoost Pools
print("Building data pools...")
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_features)

# 5. Base Model Definition (Optimized for 35M rows)
print("Training Base Model...")
base_model = CatBoostClassifier(
    iterations=2000,           # High limit, but will stop early
    learning_rate=0.05,
    depth=6,
    task_type="CPU",           
    eval_metric='Logloss',
    thread_count=-1,
    early_stopping_rounds=50,  # CRITICAL: Prevents overfitting
    verbose=50,
    random_state=42
)

# Train the model, evaluating against the validation set to trigger early stopping
base_model.fit(train_pool, eval_set=valid_pool)

# 6. Probability Calibration (Using prefit to save hours of compute)
print("Calibrating Probabilities...")
# We use 'prefit' so it doesn't retrain CatBoost, it just builds the math map using X_valid
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_valid, y_valid)

# 7. Final Evaluation on Unseen Out-of-Time Test Set
print("Evaluating Model...")
y_prob = calibrated_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
loss = log_loss(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\n--- Model Performance (On Unseen Future Data) ---")
print(f"ROC-AUC:     {auc:.4f}")
print(f"Log Loss:    {loss:.4f}")
print(f"Brier Score: {brier:.4f}")

# Sample Output
demo_output = X_test.copy()
demo_output['PREDICTED_PROBABILITY'] = y_prob
demo_output['ACTUAL_SUCCESS'] = y_test
print("\n--- Sample Predictions ---")
print(demo_output[['RETRY_COUNT', 'DECLINE_CODE', 'PREDICTED_PROBABILITY', 'ACTUAL_SUCCESS']].head())