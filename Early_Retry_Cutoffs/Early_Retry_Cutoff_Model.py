#%% IMPORTS
import pandas as pd
import numpy as np
import json
import re
import hashlib                      # CHANGED: stable order-level split assignment
import snowflake.connector
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
from datetime import datetime, timedelta, timezone
import pytz
from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier
import os
import joblib
from sklearn.frozen import FrozenEstimator

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


#%% Model Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier, Pool

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
sampled_df = sampled_df[sampled_df['TRANSACTION_STATUS'] != 'accepted'].reset_index(drop=True)
print(f"  {len(sampled_df):,} decline rows | {sampled_df['INVOICE_ID'].nunique():,} invoices "
      f"| {sampled_df['ORDER_ID'].nunique():,} orders")

# =======================================================================
# 2. Data Processing & Pre-Split Vectorized I
# =======================================================================
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
# 3. Hybrid Split (As-Of-Date OOT + Order-Grouped Random)
# =====================================================================
print("Splitting data (As-Of OOT + Order-Grouped Random)...")

ASOF_DATE = pd.Timestamp('2026-01-15')   # "retrain the model on this date"
OOT_END   = pd.Timestamp('2026-03-01')   # MUST be <= max ORDER_DATE in the extract

invoice_elig = sampled_df.groupby('INVOICE_ID', sort=False).agg(
    RESOLVED_AT=('RESOLVED_AT', 'first'),
    INVOICE_LAST_TXN=('INVOICE_LAST_TXN', 'first'),
)
eligible_invoices = invoice_elig.index[
    (invoice_elig['RESOLVED_AT'] < ASOF_DATE) &
    (invoice_elig['INVOICE_LAST_TXN'] < ASOF_DATE)
]

past_data = sampled_df[
    sampled_df['INVOICE_ID'].isin(eligible_invoices) &
    (sampled_df['TRANSACTION_DATETIME'] < ASOF_DATE)
].copy()

oot_test_df = sampled_df[
    (sampled_df['TRANSACTION_DATETIME'] >= ASOF_DATE) &
    (sampled_df['TRANSACTION_DATETIME'] < OOT_END)
].copy()

assert not (set(past_data['INVOICE_ID']) & set(oot_test_df['INVOICE_ID'])), \
    "invoices straddle the as-of boundary"

print(f"  past_data   : {len(past_data):>9,} rows | {past_data['ORDER_ID'].nunique():>7,} orders")
print(f"  oot_test_df : {len(oot_test_df):>9,} rows | {oot_test_df['ORDER_ID'].nunique():>7,} orders")
print(f"  orders in both (new invoices, expected): "
      f"{len(set(past_data['ORDER_ID']) & set(oot_test_df['ORDER_ID'])):,}")
print(f"  dropped (unresolved at ASOF / beyond OOT_END): "
      f"{len(sampled_df) - len(past_data) - len(oot_test_df):,} rows")

# --- FIX 1: order-grouped assignment --------------------------------------------
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
    depth=6,
    task_type="CPU",
    eval_metric='Logloss',
    custom_metric=['AUC'],
    l2_leaf_reg=6,
    thread_count=-1,
    early_stopping_rounds=100,
    verbose=50,
    random_state=42,
    cat_features=cat_features,
    text_features=text_features
)

# Train the model, strictly using the Stop pool
base_model.fit(train_pool, eval_set=stop_pool)

# =====================================================================
# 6. In-Flight Overfitting Diagnostic Plot
# =====================================================================
print("Plotting Learning Curves...")
history = base_model.evals_result_
iterations_range = range(len(history['learn']['Logloss']))
best_iter = base_model.get_best_iteration()

# Safely locate the validation dictionary key
val_key = 'validation' if 'validation' in history else 'validation_0'

plt.figure(figsize=(14, 5))

# Plot Log Loss
plt.subplot(1, 2, 1)
plt.plot(iterations_range, history['learn']['Logloss'], label='Train Log Loss', color='blue')
plt.plot(iterations_range, history[val_key]['Logloss'], label='Stop Log Loss', color='orange')
plt.axvline(x=best_iter, color='red', linestyle='--', label=f'Best Iter ({best_iter})')
plt.title('Log Loss (Overfitting Check)')
plt.xlabel('Trees (Iterations)')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)

# Plot ROC-AUC (Fixing the AUC KeyError by ignoring Train AUC)
plt.subplot(1, 2, 2)
if 'AUC' in history[val_key]:
    plt.plot(iterations_range, history[val_key]['AUC'], label='Stop AUC', color='orange')
    plt.axvline(x=best_iter, color='red', linestyle='--', label=f'Best Iter ({best_iter})')
    plt.title('ROC-AUC (Performance on Stop Set)')
    plt.xlabel('Trees (Iterations)')
    plt.ylabel('ROC-AUC')
    plt.legend()
    plt.grid(True)
else:
    plt.title('AUC not tracked')

plt.tight_layout()
plt.show()

# =====================================================================
# 7. Probability Calibration
# =====================================================================
print("Calibrating Probabilities...")
calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method='isotonic'
)
calibrated_model.fit(X_calib, y_calib)

# =====================================================================
# 8. Final Dual-Evaluation (Random vs. Chronological)
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
    'asof_date': ASOF_DATE,      # CHANGED: what the model knew, and when
    'oot_end': OOT_END,
}

# 3. Save the artifact to the script's directory
joblib.dump(model_artifact, model_path)

print(f"Model artifact successfully saved to: {model_path}")

#%% Predictions

# 1. Load the Model and Metadata
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'calibrated_catboost_model.joblib')

print("Loading model artifact...")
artifact = joblib.load(model_path)

# Extract components from the artifact
calibrated_model = artifact['model']
cat_features = artifact['cat_features']
text_features = artifact['text_features']
drop_cols = artifact['drop_cols']

# 2. Load and Preprocess New Data
print("Loading new data...")
# Assuming 'cur' (your database cursor) is already initialized and active in your environment
cur.execute(("use database dbt_prod"))
cur.execute("SELECT * FROM ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_SAMPLE_062025 WHERE TRANSACTION_STATUS != 'accepted' ")
df_062025 = cur.fetch_pandas_all()
df_pred = df_062025.copy()

print("Applying strict preprocessing rules...")

# 1. Format specific string columns if they exist
# CHANGED: mirrors section 1 -- DECLINE_FAMILY comes from LAST_DECLINE_CODE now.
# Without this the model's DECLINE_FAMILY column is missing and the reindex below
# raises a KeyError.
if 'LAST_DECLINE_CODE' in df_pred.columns:
    df_pred['LAST_DECLINE_CODE'] = df_pred['LAST_DECLINE_CODE'].astype(str)
    df_pred['DECLINE_FAMILY'] = df_pred['LAST_DECLINE_CODE'].str[0]

# 2. Drop core IDs and target (ignoring errors if they aren't in the new data)
X_new = df_pred.drop(columns=drop_cols, errors='ignore').copy()

# 3. Apply Imputations based on saved feature lists
# Categorical/Text Imputation
existing_cat_text = [c for c in (cat_features + text_features) if c in X_new.columns]
if existing_cat_text:
    X_new[existing_cat_text] = X_new[existing_cat_text].fillna('unknown').astype(str)

# Numeric Imputation
numeric_cols = X_new.columns.difference(cat_features + text_features)
if len(numeric_cols) > 0:
    X_new[numeric_cols] = X_new[numeric_cols].fillna(-1)

# 3. Make Predictions
print("Extracting feature names from the nested model...")

# Dig into the CalibratedClassifierCV
if hasattr(calibrated_model, 'calibrated_classifiers_'):
    # Grab the wrapper from the first fold
    base_wrapper = calibrated_model.calibrated_classifiers_[0].estimator
else:
    # Fallback if calibrated without CV somehow
    base_wrapper = calibrated_model.estimator

# Dig into your custom FrozenEstimator (Checking common attribute names)
if hasattr(base_wrapper, 'estimator'):
    actual_catboost = base_wrapper.estimator
elif hasattr(base_wrapper, 'model'):
    actual_catboost = base_wrapper.model
else:
    actual_catboost = base_wrapper

# Extract the feature names directly from the underlying CatBoost object
feature_names = actual_catboost.feature_names_

print("Generating predictions...")

# Ensure column order matches training exactly
X_new = X_new[feature_names]

# Get the probability of the positive class (IS_EVENTUALLY_SUCCESSFUL = 1)
probabilities = calibrated_model.predict_proba(X_new)[:, 1]

# Attach probabilities back to the original dataframe
df_pred['SUCCESS_PROBABILITY'] = probabilities

print("Predictions successfully generated and attached!")


#%% Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix

def plot_business_confusion_matrix(y_true, y_probs, threshold=0.05, dataset_name="Test Set"):
    """
    Evaluates business impact based on model predictions and a given threshold.
    """
    # 2. Convert probabilities to hard predictions
    y_pred = (y_probs >= threshold).astype(int)

    # 3. Calculate the Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # 4. Calculate Percentages and Create Custom Labels ugk 
    # Divide each cell by the total number of predictions to get the % of total
    cm_percentages = cm / np.sum(cm)

    # Zip the raw counts and the percentages together into a formatted string
    labels = [f"{count}\n({percentage:.1%})" for count, percentage in zip(cm.flatten(), cm_percentages.flatten())]
    # Reshape the list of strings back into a 2x2 grid to match the heatmap
    labels = np.asarray(labels).reshape(2, 2)

    # 5. Visualize the Matrix
    plt.figure(figsize=(8, 6))

    # Notice we changed annot=labels and fmt='' (empty string) so Seaborn accepts our custom text
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={'size': 14},
                xticklabels=['Predicted Fail (0)', 'Predicted Success (1)'],
                yticklabels=['Actual Fail (0)', 'Actual Success (1)'])

    plt.title(f'Confusion Matrix: {dataset_name}\n(Threshold: {threshold})', fontsize=14, pad=15)
    plt.xlabel('What the Model Predicted', fontsize=12, labelpad=10)
    plt.ylabel('What Actually Happened', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.show()

    # 6. Print the business breakdown
    tn, fp, fn, tp = cm.ravel()
    print(f"\n--- Confusion Matrix Breakdown: {dataset_name} (Threshold: {threshold}) ---")
    print(f"True Negatives (TN):  {tn:,} ({cm_percentages[0,0]:.1%}) -> Correctly cut off")
    print(f"False Positives (FP): {fp:,} ({cm_percentages[0,1]:.1%}) -> Wasted $0.10 retry")
    print(f"False Negatives (FN): {fn:,} ({cm_percentages[1,0]:.1%}) -> Missed revenue!")
    print(f"True Positives (TP):  {tp:,} ({cm_percentages[1,1]:.1%}) -> Successfully collected")

# =====================================================================
# Generate the Confusion Matrices using variables from your training script
# ====================================================================

# Check the Random Holdout Test Set
y_true_062025 = df_pred['IS_EVENTUALLY_SUCCESSFUL'].values
y_probs_062025 = df_pred['SUCCESS_PROBABILITY'].values

plot_business_confusion_matrix(y_true=y_true_062025, 
                               y_probs=y_probs_062025, 
                               threshold=0.05, 
                               dataset_name="Random Test Set (In-Sample)")



#%% Order based confusion matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix

def plot_order_level_confusion_matrix(order_ids, y_true, y_probs, threshold=0.05, dataset_name="Test Set"):
    """
    Evaluates business impact at the ORDER level.
    Rules:
    - If Actual=1: If ANY transaction is predicted 0 (cut off), order is False Negative.
                   Otherwise (ALL predicted 1), it's a True Positive.
    - If Actual=0: If ANY transaction is predicted 1 (allowed), order is False Positive.
                   Otherwise (ALL predicted 0), it's a True Negative.
    """
    # 1. Create a temporary dataframe to group predictions by order
    df = pd.DataFrame({
        'order_id': order_ids,
        'y_true': y_true,
        'y_pred': (y_probs >= threshold).astype(int)
    })

    # 2. Group by order_id and evaluate the predictions
    grouped = df.groupby('order_id').agg(
        y_true=('y_true', 'first'),                     # Actual outcome is the same for the whole order
        has_pred_0=('y_pred', lambda x: (x == 0).any()), # Did we predict 0 for ANY transaction?
        has_pred_1=('y_pred', lambda x: (x == 1).any())  # Did we predict 1 for ANY transaction?
    )

    # 3. Apply the Business Logic to categorize each order
    FN = ((grouped['y_true'] == 1) & (grouped['has_pred_0'])).sum()
    TP = ((grouped['y_true'] == 1) & (~grouped['has_pred_0'])).sum()
    FP = ((grouped['y_true'] == 0) & (grouped['has_pred_1'])).sum()
    TN = ((grouped['y_true'] == 0) & (~grouped['has_pred_1'])).sum()

    # 4. Build the Confusion Matrix array and calculate percentages
    cm = np.array([[TN, FP], 
                   [FN, TP]])
    cm_percentages = cm / np.sum(cm)

    labels = [f"{count:,}\n({percentage:.1%})" for count, percentage in zip(cm.flatten(), cm_percentages.flatten())]
    labels = np.asarray(labels).reshape(2, 2)

    # 5. Visualize the Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={'size': 14},
                xticklabels=['Predicted Fail (0)', 'Predicted Success (1)'],
                yticklabels=['Actual Fail (0)', 'Actual Success (1)'])

    plt.title(f'Order-Level Confusion Matrix: {dataset_name}\n(Threshold: {threshold})', fontsize=14, pad=15)
    plt.xlabel('Order-Level Prediction Outcome', fontsize=12, labelpad=10)
    plt.ylabel('Actual Order Outcome', fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.show()

    # 6. Print the business breakdown
    print(f"\n--- Order-Level Breakdown: {dataset_name} (Threshold: {threshold}) ---")
    print(f"Total Unique Orders:  {np.sum(cm):,}")
    print(f"True Negatives (TN):  {TN:,} ({cm_percentages[0,0]:.1%}) -> Entire bad order correctly cut off")
    print(f"False Positives (FP): {FP:,} ({cm_percentages[0,1]:.1%}) -> Allowed at least one retry on a doomed order")
    print(f"False Negatives (FN): {FN:,} ({cm_percentages[1,0]:.1%}) -> Killed a good order prematurely (Missed revenue!)")
    print(f"True Positives (TP):  {TP:,} ({cm_percentages[1,1]:.1%}) -> Allowed all retries, successfully collected")

print("\nGenerating Order-Level Confusion Matrices...")

# 2. Evaluate the Random Holdout Test Set (In-Sample)
plot_order_level_confusion_matrix(
    order_ids=random_test_df['ORDER_ID'].values, 
    y_true=y_random_test.values, 
    y_probs=random_probs, 
    threshold=0.03, 
    dataset_name="Random Test Set (In-Sample)"
)

#%% Shap
import shap
import matplotlib.pyplot as plt
from catboost import Pool

# =====================================================================
# 9. SHAP Analysis on Random Test Set (Optimized & Bug-Fixed)
# =====================================================================
print("\n--- Starting SHAP Analysis ---")

# 1. Initialize the SHAP Explainer using the base_model
explainer = shap.TreeExplainer(base_model)

# 2. Sample the data
sample_size = min(20000, len(X_random_test))
X_shap = X_random_test.sample(n=sample_size, random_state=42)

# ---------------------------------------------------------
# THE FIX: Manually create a CatBoost Pool for SHAP
# This bypasses SHAP's internal converter and ensures text_features are respected
# ---------------------------------------------------------
shap_pool = Pool(X_shap, cat_features=cat_features, text_features=text_features)

print(f"Calculating SHAP values for a random sample of {len(X_shap):,} rows...")
# Pass the Pool, not the DataFrame, to avoid the float conversion error
shap_values = explainer.shap_values(shap_pool)

# 3. Global Feature Importance (Bar Plot)
plt.figure(figsize=(10, 6))
plt.title("SHAP Global Feature Importance")
# We still pass X_shap here so the plot can pull the correct feature names and values
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# 4. Detailed Summary Plot (Beeswarm)
plt.figure(figsize=(10, 6))
plt.title("SHAP Directional Summary Plot")
shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
plt.show()

# 5. Local Explanation (Waterfall plot for a single transaction)
print("\nGenerating Local Explanation for Sample Observation 0...")

# We also need a manual Pool for the single observation
row_df = X_shap.iloc[[0]]
row_pool = Pool(row_df, cat_features=cat_features, text_features=text_features)

# Generate the Explanation object
shap_explanation = explainer(row_pool)

# Manually attach the original data and feature names to the Explanation object 
# so the waterfall plot knows how to label the graph
shap_explanation.data = row_df.values
shap_explanation.feature_names = row_df.columns.tolist()

plt.figure(figsize=(10, 6))
plt.title("SHAP Waterfall Plot: Explaining Sample Observation 0")
shap.plots.waterfall(shap_explanation[0], show=False)
plt.tight_layout()
plt.show()

#%% ECLTV based cancellation 
import pandas as pd
import numpy as np

def evaluate_cutoff_thresholds(df_pred, thresholds_to_test, ecltv_max_threshold=float('inf')):
    """
    Calculates the financial impact of cutting off retries based on a probability threshold.
    Only cuts off orders if their ECLTV_1Y is strictly less than ecltv_max_threshold.
    """
    print("Loading data and merging...")
    
    # Execute Snowflake query
    cur = ctx.cursor()
    cur.execute("""
    with ecltv as (
    SELECT ORDER_ID, CLTV_PRED_MEAN as ECLTV_1Y FROM DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER
    ),

    tb as (select
    order_id,
    div0(sum(case when is_sale = 1 and is_daydiff_interval_txn_order_0360 then transaction_amount else 0 end),
        count (distinct case when is_m0 = 1 and is_daydiff_interval_txn_order_0360 then order_id else null end)) as cltv_360,
    max(ECLTV_1Y) as ecltv_1y
    FROM DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    left join ecltv using (order_id)
    where order_date >= '2025-06-01' and order_date < '2025-07-01'
    and is_m0 = 1
    group by 1
    )

    select 
    *
    from tb
    """) 
    ecltv_df = cur.fetch_pandas_all()
    
    # 1. Ensure datetime formatting
    df_pred['TRANSACTION_DATETIME'] = pd.to_datetime(df_pred['TRANSACTION_DATETIME'])
    
    # 2. Merge predictions with the ECLTV dataframe from Snowflake
    df = pd.merge(df_pred, ecltv_df, on='ORDER_ID', how='left')
    
    # Fill missing ECLTV with 0 to prevent math errors
    df['ECLTV_1Y'] = df['ECLTV_1Y'].fillna(0)
    df['CLTV_360'] = df['CLTV_360'].fillna(0)
    
    # 3. Sort chronologically
    df = df.sort_values(by=['ORDER_ID', 'TRANSACTION_DATETIME'])
    
    # --- Calculate order-level maximums for Gain and Actual Loss ---
    print("Calculating order-level maximums...")
    order_stats = df.groupby('ORDER_ID').agg(
        MAX_RETRY_COUNT=('RETRIES', 'max')
    ).reset_index()
    
    # Merge the max stats back into the main dataframe
    df = pd.merge(df, order_stats, on='ORDER_ID', how='left')
    results = []
    
    print(f"Evaluating thresholds with an ECLTV cap of {ecltv_max_threshold}...")
    # 4. Iterate over thresholds to generate a comparative report
    for threshold in thresholds_to_test:
        
        # --- MODIFIED LOGIC: Filter by BOTH probability and ECLTV ---
        below_threshold = df[
            (df['SUCCESS_PROBABILITY'] < threshold) & 
            (df['ECLTV_1Y'] < ecltv_max_threshold)
        ]
        
        # Keep only the FIRST instance for each order
        first_cutoffs = below_threshold.drop_duplicates(subset=['ORDER_ID'], keep='first').copy()
        
        # 5. Calculate Financial Impacts
        
        # Gain: (max retry count in the order - current retry count) * 0.1
        first_cutoffs['money_gained'] = (first_cutoffs['MAX_RETRY_COUNT'] - first_cutoffs['RETRIES']) * 0.1
        
        # Loss Actual: (actual collections over 360 days - current transaction HISTORICAL_COLLECTED_AMOUNT)
        first_cutoffs['loss_actual'] = np.where(
            first_cutoffs['IS_EVENTUALLY_SUCCESSFUL'] == 1,
            first_cutoffs['CLTV_360'] - first_cutoffs['HISTORICAL_COLLECTED_AMOUNT'],
            0.0
        )
        
        # Loss Projected: (ECLTV_1Y - current transaction HISTORICAL_COLLECTED_AMOUNT)
        first_cutoffs['loss_projected'] = np.where(
            first_cutoffs['IS_EVENTUALLY_SUCCESSFUL'] == 1,
            first_cutoffs['ECLTV_1Y'] - first_cutoffs['HISTORICAL_COLLECTED_AMOUNT'],
            0.0
        )
        
        # Calculate Net Actual and Net Projected
        first_cutoffs['net_actual'] = first_cutoffs['money_gained'] - first_cutoffs['loss_actual']
        first_cutoffs['net_projected'] = first_cutoffs['money_gained'] - first_cutoffs['loss_projected']
        
        # 6. Aggregate results for this threshold
        results.append({
            'Prob Threshold': threshold,
            'ECLTV Cap': ecltv_max_threshold,
            'Orders Cut': len(first_cutoffs),
            'Total Gain': round(first_cutoffs['money_gained'].sum(), 2),
            'Loss (Actual)': round(first_cutoffs['loss_actual'].sum(), 2),
            'Net (Actual)': round(first_cutoffs['net_actual'].sum(), 2),
            'Loss (Proj)': round(first_cutoffs['loss_projected'].sum(), 2),
            'Net (Proj)': round(first_cutoffs['net_projected'].sum(), 2)
        })
        
    return pd.DataFrame(results)

# Run the evaluation
test_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15]

# Set your ECLTV limit here (e.g., only cancel orders with ECLTV < 100)
# If you don't pass this argument, it defaults to infinity (meaning no ECLTV limit).
impact_report = evaluate_cutoff_thresholds(df_pred, test_thresholds, ecltv_max_threshold=20)

print(impact_report.to_string(index=False))

#%% Top 50 orders driving Net(Actual) vs Net(Proj) divergence (threshold < 0.10)

def top_divergent_orders(df_pred, prob_threshold=0.10, ecltv_max_threshold=20, top_n=50):
    """
    Returns the orders whose CLTV_360 (actual) most differs from ECLTV_1Y (projected)
    among first-cutoffs below prob_threshold. Since money_gained and
    HISTORICAL_COLLECTED_AMOUNT cancel between net_actual and net_projected,
    the per-order divergence reduces to (ECLTV_1Y - CLTV_360), nonzero only when
    the order is eventually successful.
    """
    print("Loading data and merging...")

    # Same ECLTV / CLTV_360 pull as evaluate_cutoff_thresholds
    cur = ctx.cursor()
    cur.execute("""
    with ecltv as (
    SELECT ORDER_ID, CLTV_PRED_MEAN as ECLTV_1Y FROM DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER
    ),

    tb as (select
    order_id,
    div0(sum(case when is_sale = 1 and is_daydiff_interval_txn_order_0360 then transaction_amount else 0 end),
        count (distinct case when is_m0 = 1 and is_daydiff_interval_txn_order_0360 then order_id else null end)) as cltv_360,
    max(ECLTV_1Y) as ecltv_1y
    FROM DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    left join ecltv using (order_id)
    where order_date >= '2025-06-01' and order_date < '2025-07-01'
    and is_m0 = 1
    group by 1
    )

    select
    *
    from tb
    """)
    ecltv_df = cur.fetch_pandas_all()

    df_pred['TRANSACTION_DATETIME'] = pd.to_datetime(df_pred['TRANSACTION_DATETIME'])
    df = pd.merge(df_pred, ecltv_df, on='ORDER_ID', how='left')
    df['ECLTV_1Y'] = df['ECLTV_1Y'].fillna(0)
    df['CLTV_360'] = df['CLTV_360'].fillna(0)
    df = df.sort_values(by=['ORDER_ID', 'TRANSACTION_DATETIME'])

    # First cutoff per order below the probability threshold (ECLTV cap applied to match the report)
    below_threshold = df[
        (df['SUCCESS_PROBABILITY'] < prob_threshold) &
        (df['ECLTV_1Y'] < ecltv_max_threshold)
    ]
    first_cutoffs = below_threshold.drop_duplicates(subset=['ORDER_ID'], keep='first').copy()

    # Per-order contribution to Net(Actual) - Net(Proj); only successful orders carry loss
    first_cutoffs['net_actual_minus_proj'] = np.where(
        first_cutoffs['IS_EVENTUALLY_SUCCESSFUL'] == 1,
        first_cutoffs['ECLTV_1Y'] - first_cutoffs['CLTV_360'],
        0.0
    )
    first_cutoffs['ABS_DIFF'] = first_cutoffs['net_actual_minus_proj'].abs()

    cols = ['ORDER_ID', 'SUCCESS_PROBABILITY', 'IS_EVENTUALLY_SUCCESSFUL',
            'HISTORICAL_COLLECTED_AMOUNT', 'CLTV_360', 'ECLTV_1Y',
            'net_actual_minus_proj', 'ABS_DIFF']
    return (first_cutoffs.sort_values('ABS_DIFF', ascending=False)
                         .head(top_n)[cols]
                         .reset_index(drop=True))

top50 = top_divergent_orders(df_pred, prob_threshold=0.10, ecltv_max_threshold=20, top_n=50)
print(top50.to_string(index=False))
# %%
