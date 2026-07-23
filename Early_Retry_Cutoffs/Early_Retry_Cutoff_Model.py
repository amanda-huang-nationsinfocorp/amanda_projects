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
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool

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
}

# 3. Save the artifact to the script's directory
joblib.dump(model_artifact, model_path)

print(f"Model artifact successfully saved to: {model_path}")

#%% Predictions
import os
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score # <-- Added Import

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
# Assuming 'cur' (your database cursor) is already initialized and active
cur.execute(("use database dbt_prod"))
cur.execute("SELECT * FROM ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_SAMPLE_062025")
df_062025 = cur.fetch_pandas_all()
df_pred = df_062025.copy()

print("Applying strict preprocessing rules...")

# 1. Format specific string columns if they exist
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
    base_wrapper = calibrated_model.calibrated_classifiers_[0].estimator
else:
    base_wrapper = calibrated_model.estimator

# Dig into your custom FrozenEstimator
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

# Get the probability of the positive class
probabilities = calibrated_model.predict_proba(X_new)[:, 1]

# Attach probabilities back to the original dataframe
df_pred['SUCCESS_PROBABILITY'] = probabilities

print("Predictions successfully generated and attached!")

# =====================================================================
# 4. Evaluate Performance (NEW AUC SECTION)
# =====================================================================
print("\nEvaluating Model Performance...")

if 'IS_EVENTUALLY_SUCCESSFUL' in df_pred.columns:
    # Filter out any rows where the target might be NaN (unresolved invoices)
    valid_mask = df_pred['IS_EVENTUALLY_SUCCESSFUL'].notna()
    
    if valid_mask.sum() > 0:
        y_true = df_pred.loc[valid_mask, 'IS_EVENTUALLY_SUCCESSFUL'].astype(int)
        y_prob = df_pred.loc[valid_mask, 'SUCCESS_PROBABILITY']
        
        auc_score = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC Score on New Data: {auc_score:.4f}")
        print(f"*(Evaluated on {valid_mask.sum():,} resolved rows)*")
    else:
        print("Cannot calculate AUC: 'IS_EVENTUALLY_SUCCESSFUL' contains only null values.")
else:
    print("Cannot calculate AUC: Target column 'IS_EVENTUALLY_SUCCESSFUL' missing from data.")

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
    order_ids=df_pred['ORDER_ID'].values, 
    y_true=y_true_062025, 
    y_probs=y_probs_062025, 
    threshold=0.02, 
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

#%% ECLTV based cancellation MIGHT BE WRONG 
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
    df_pred['HISTORICAL_COLLECTED_AMOUNT'] = pd.to_datetime(df_pred['TRANSACTION_DATETIME'])
    
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


#%% ECLTV based cancellation
import pandas as pd
import numpy as np

RETRY_COST = 0.1


def load_cltv():
    cur = ctx.cursor()
    cur.execute("""
    with ecltv as (
        -- GROUP BY guards against >1 row per order fanning out the join below.
        select order_id, max(cltv_pred_mean) as ecltv_1y
        from DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER
        group by 1
    ),
    tb as (
        select
            order_id,
            div0(sum(case when is_sale = 1 and is_daydiff_interval_txn_order_0360
                          then transaction_amount else 0 end),
                 count(distinct case when is_m0 = 1 and is_daydiff_interval_txn_order_0360
                                     then order_id else null end)) as cltv_360,
            max(ecltv_1y) as ecltv_1y
        from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
        left join ecltv using (order_id)
        where order_date >= '2025-06-01' and order_date < '2025-07-01'
          and is_m0 = 1
        group by 1
    )
    select * from tb
    """)
    return cur.fetch_pandas_all()


def prepare_cutoff_frame(df_pred):
    df = df_pred.copy()
    df['HISTORICAL_COLLECTED_AMOUNT'] = df['HISTORICAL_COLLECTED_AMOUNT']+1
    df['TRANSACTION_DATETIME'] = pd.to_datetime(df['TRANSACTION_DATETIME'])
    df = df.merge(load_cltv(), on='ORDER_ID', how='left', indicator='_src')

    # tb is filtered to is_m0 = 1, so absent from it == never signed up for trial ==
    # never collected and never will. Real zeros, not missing data -- and the cheapest
    # orders to cut. Keep them.
    never_m0 = df['_src'] == 'left_only'
    df['NEVER_M0'] = never_m0            # persist for downstream diagnostics
    df.loc[never_m0, ['CLTV_360', 'ECLTV_1Y']] = 0.0
    df = df.drop(columns='_src')

    # Made M0 but no ECLTV prediction: CLTV_360 is real, the projection isn't. Left NaN
    # so `ECLTV_1Y < cap` is False and they are never cut.
    unpriceable = df['ECLTV_1Y'].isna()

    print(f"orders: {df.ORDER_ID.nunique():,} total | "
          f"{df.loc[never_m0, 'ORDER_ID'].nunique():,} never made M0 (loss=0, kept) | "
          f"{df.loc[unpriceable, 'ORDER_ID'].nunique():,} no ECLTV (never cut)")

    # Sanity: never-M0 orders should have collected nothing.
    bad = df.loc[never_m0, 'HISTORICAL_COLLECTED_AMOUNT'].max()
    if pd.notna(bad) and bad > 0:
        print(f"  WARNING: a never-M0 order shows ${bad:,.2f} collected -- is_m0 and "
              f"HISTORICAL_COLLECTED_AMOUNT disagree.")

    df = df.sort_values(['ORDER_ID', 'TRANSACTION_DATETIME'])

    # Cutting off cancels the order, so every remaining attempt across all of its
    # invoices is saved. Decision is pre-attempt, so the current row counts too.
    df['ORDER_ATTEMPTS_SAVED'] = (df.groupby('ORDER_ID')['TRANSACTION_ID'].transform('size')
                                  - df.groupby('ORDER_ID').cumcount())+5
    rev = df.iloc[::-1]
    df['FUTURE_SUCCESS'] = rev.groupby('ORDER_ID')['IS_EVENTUALLY_SUCCESSFUL'].cummax().iloc[::-1]
    return df


def _score(cut, label):
    cut = cut.copy()
    cut['money_gained'] = cut['ORDER_ATTEMPTS_SAVED'] * RETRY_COST

    # clip(0): CLTV_360 below already-collected means the two windows count different
    # things, not that cancelling earned us money.
    cut['loss_actual'] = (cut['CLTV_360'] - cut['HISTORICAL_COLLECTED_AMOUNT']).clip(lower=0)

    # ECLTV_1Y is a signup-time forecast that never learned the order died, so it stays
    # positive for true negatives and invents losses on customers who never pay. Charge
    # the projection only where an invoice at/after the cut would actually have collected.
    cut['loss_proj'] = np.where(
        cut['FUTURE_SUCCESS'] == 1,
        (cut['ECLTV_1Y'] - cut['HISTORICAL_COLLECTED_AMOUNT']).clip(lower=0),
        0.0)

    return {
        'Rule': label,
        'Orders Cut': len(cut),
        'False Negatives': int((cut['FUTURE_SUCCESS'] == 1).sum()),
        'FN Rate': f"{(cut['FUTURE_SUCCESS'] == 1).mean():.1%}",
        'Free Cuts (no CLTV)': int((cut['CLTV_360'] == 0).sum()),
        'Total Gain': round(cut['money_gained'].sum(), 2),
        'Loss (Actual)': round(cut['loss_actual'].sum(), 2),
        'Net (Actual)': round((cut['money_gained'] - cut['loss_actual']).sum(), 2),
        'Loss (Proj)': round(cut['loss_proj'].sum(), 2),
        'Net (Proj)': round((cut['money_gained'] - cut['loss_proj']).sum(), 2),
    }


def evaluate_cutoff_thresholds(df, thresholds_to_test, ecltv_max_threshold=float('inf')):
    rows = []
    for threshold in thresholds_to_test:
        cut = df[(df['SUCCESS_PROBABILITY'] < threshold) &
                 (df['ECLTV_1Y'] < ecltv_max_threshold)].drop_duplicates('ORDER_ID', keep='first')
        rows.append(_score(cut, f'model p < {threshold}'))
    return pd.DataFrame(rows)


# --- Run ---------------------------------------------------------------------
cutoff_df = prepare_cutoff_frame(df_pred)

ECLTV_CAP = 20
test_thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

impact_report = evaluate_cutoff_thresholds(cutoff_df, test_thresholds, ECLTV_CAP)
print(f"\n=== Model-driven cutoff (ECLTV cap {ECLTV_CAP}) ===")
print(impact_report.to_string(index=False))

# Does the IS_EVENTUALLY_SUCCESSFUL gate change anything? If not, the question is moot.
_c = cutoff_df[(cutoff_df['SUCCESS_PROBABILITY'] < 0.05) &
               (cutoff_df['ECLTV_1Y'] < ECLTV_CAP)].drop_duplicates('ORDER_ID', keep='first').copy()
_ungated = (_c['CLTV_360'] - _c['HISTORICAL_COLLECTED_AMOUNT']).clip(lower=0)
_gated = np.where(_c['IS_EVENTUALLY_SUCCESSFUL'] == 1, _ungated, 0.0)
_diff = _ungated.values != _gated
print(f"\nGate check @0.05: {_diff.sum():,}/{len(_c):,} orders differ, "
      f"${_ungated[_diff].sum():,.0f} at stake. Zero => a failed invoice really does end "
      f"the order, and the gate is redundant.")



#%%
#%% Ablation: which change moved the number?
# Starts from YOUR original calculation and applies my changes one at a time.
# The "Delta Net" column is the damage (or credit) attributable to that one change.

import pandas as pd
import numpy as np

RETRY_COST = 0.1
THRESHOLD = 0.05     # ablate at one fixed threshold
ECLTV_CAP = 20


def load_cltv(dedupe):
    """dedupe=False reproduces your original ecltv CTE (no GROUP BY)."""
    ecltv_cte = ("select order_id, max(cltv_pred_mean) as ecltv_1y "
                 "from DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER group by 1") if dedupe else \
                ("select order_id, cltv_pred_mean as ecltv_1y "
                 "from DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER")
    cur = ctx.cursor()
    cur.execute(f"""
    with ecltv as ({ecltv_cte}),
    tb as (
        select order_id,
               div0(sum(case when is_sale = 1 and is_daydiff_interval_txn_order_0360
                             then transaction_amount else 0 end),
                    count(distinct case when is_m0 = 1 and is_daydiff_interval_txn_order_0360
                                        then order_id else null end)) as cltv_360,
               max(ecltv_1y) as ecltv_1y
        from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
        left join ecltv using (order_id)
        where order_date >= '2025-06-01' and order_date < '2025-07-01' and is_m0 = 1
        group by 1
    )
    select * from tb
    """)
    return cur.fetch_pandas_all()


def build(df_pred, cltv_df):
    df = df_pred.copy()
    df['TRANSACTION_DATETIME'] = pd.to_datetime(df['TRANSACTION_DATETIME'])
    df = df.merge(cltv_df, on='ORDER_ID', how='left', indicator='_src')
    df['NEVER_M0'] = df['_src'] == 'left_only'      # absent from tb => never signed up
    df = df.drop(columns='_src').sort_values(['ORDER_ID', 'TRANSACTION_DATETIME'])
    df['ORDER_MAX_RETRY'] = df.groupby('ORDER_ID')['RETRIES'].transform('max')
    df['ORDER_ATTEMPTS_SAVED'] = (df.groupby('ORDER_ID')['TRANSACTION_ID'].transform('size')
                                  - df.groupby('ORDER_ID').cumcount())
    return df


def score(df, fill0, clip, gate, gain_mode, plus1):
    d = df.copy()
    if fill0:
        # your version: every missing CLTV becomes 0, including paying customers
        d['ECLTV_1Y'] = d['ECLTV_1Y'].fillna(0)
        d['CLTV_360'] = d['CLTV_360'].fillna(0)
    else:
        # mine: never-M0 are true zeros; made-M0-but-no-eCLTV stay NaN -> never cut
        d.loc[d['NEVER_M0'], ['CLTV_360', 'ECLTV_1Y']] = 0.0

    cut = (d[(d['SUCCESS_PROBABILITY'] < THRESHOLD) & (d['ECLTV_1Y'] < ECLTV_CAP)]
           .drop_duplicates('ORDER_ID', keep='first').copy())

    if gain_mode == 'attempts':
        cut['gain'] = cut['ORDER_ATTEMPTS_SAVED'] * RETRY_COST   # pre-attempt by construction
    else:
        cut['gain'] = (cut['ORDER_MAX_RETRY'] - cut['RETRIES'] + int(plus1)) * RETRY_COST

    loss = cut['CLTV_360'] - cut['HISTORICAL_COLLECTED_AMOUNT']
    if clip:
        loss = loss.clip(lower=0)
    if gate:
        loss = loss.where(cut['IS_EVENTUALLY_SUCCESSFUL'] == 1, 0.0)
    cut['loss'] = loss
    cut['net'] = cut['gain'] - cut['loss']
    return cut


# --- Pre-flight: are the ingredients even in play? ---------------------------
cur = ctx.cursor()
cur.execute("""select count(*) as dupe_orders from (
                 select order_id from DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER
                 group by 1 having count(*) > 1)""")
dupes = cur.fetch_pandas_all().iloc[0, 0]

cltv_dedup = load_cltv(dedupe=True)
cltv_raw = load_cltv(dedupe=False)
base_dedup = build(df_pred, cltv_dedup)
base_raw = build(df_pred, cltv_raw)

n_orders = base_dedup['ORDER_ID'].nunique()
never_m0 = base_dedup.loc[base_dedup['NEVER_M0'], 'ORDER_ID'].nunique()
no_ecltv = base_dedup.loc[base_dedup['ECLTV_1Y'].isna() & ~base_dedup['NEVER_M0'], 'ORDER_ID'].nunique()
multi_inv = (base_dedup.groupby('ORDER_ID')['INVOICE_ID'].nunique() > 1).mean()

print("=" * 76)
print("PRE-FLIGHT -- how much can each change possibly matter?")
print("=" * 76)
print(f"  orders in FCT_ECLTV_ORDER with >1 row : {dupes:,}   <- if 0, the SQL dedupe is a no-op")
print(f"  orders that never made M0             : {never_m0:,} / {n_orders:,} ({never_m0/n_orders:.1%})")
print(f"    -> we AGREE on these (loss truly 0)")
print(f"  orders made M0 but no eCLTV row       : {no_ecltv:,} / {n_orders:,} ({no_ecltv/n_orders:.1%})")
print(f"    -> your fillna(0) marks these free to cut; I refuse to cut them")
print(f"  orders with >1 invoice                : {multi_inv:.1%}")
print(f"    -> only these can make the GAIN formulas diverge")

# --- Cumulative ablation -----------------------------------------------------
STEPS = [
    ('0. your original',              dict(dedupe=False, fill0=True,  clip=False, gate=True,  gain_mode='order_max', plus1=False)),
    ('1. + dedupe ecltv (SQL)',       dict(dedupe=True,  fill0=True,  clip=False, gate=True,  gain_mode='order_max', plus1=False)),
    ('2. + split the two NaN cases',  dict(dedupe=True,  fill0=False, clip=False, gate=True,  gain_mode='order_max', plus1=False)),
    ('3. + clip loss at 0',           dict(dedupe=True,  fill0=False, clip=True,  gate=True,  gain_mode='order_max', plus1=False)),
    ('4. + drop success gate',        dict(dedupe=True,  fill0=False, clip=True,  gate=False, gain_mode='order_max', plus1=False)),
    ('5. + pre-attempt (+1)',         dict(dedupe=True,  fill0=False, clip=True,  gate=False, gain_mode='order_max', plus1=True)),
    ('6. + count order-wide attempts', dict(dedupe=True, fill0=False, clip=True,  gate=False, gain_mode='attempts',  plus1=True)),
]

rows, prev_net = [], None
for label, cfg in STEPS:
    base = base_dedup if cfg.pop('dedupe') else base_raw
    cut = score(base, **cfg)
    net = cut['net'].sum()
    rows.append({
        'Step': label,
        'Orders Cut': len(cut),
        'Gain': round(cut['gain'].sum(), 2),
        'Loss': round(cut['loss'].sum(), 2),
        'Net': round(net, 2),
        'Delta Net': '' if prev_net is None else round(net - prev_net, 2),
    })
    prev_net = net

ablation = pd.DataFrame(rows)
print("\n" + "=" * 76)
print(f"CUMULATIVE ABLATION  (threshold {THRESHOLD}, ECLTV cap {ECLTV_CAP})")
print("=" * 76)
print(ablation.to_string(index=False))
print("\n  Row 0 = your number. Row 6 = mine. The biggest |Delta Net| is your answer.")

# --- Where exactly does the gain diverge? ------------------------------------
final = score(base_dedup, fill0=False, clip=True, gate=False, gain_mode='attempts', plus1=True)
final['gain_yours'] = (final['ORDER_MAX_RETRY'] - final['RETRIES']) * RETRY_COST
final['n_invoices'] = final['ORDER_ID'].map(base_dedup.groupby('ORDER_ID')['INVOICE_ID'].nunique())
final['gain_diff'] = final['gain'] - final['gain_yours']

print("\n" + "=" * 76)
print("GAIN DIVERGENCE BY INVOICE COUNT")
print("=" * 76)
by_inv = final.groupby(final['n_invoices'].clip(upper=4)).agg(
    orders=('gain', 'size'), yours=('gain_yours', 'mean'),
    mine=('gain', 'mean'), total_diff=('gain_diff', 'sum'))
by_inv['ratio'] = (by_inv['mine'] / by_inv['yours']).round(2)
print(by_inv.round(2).to_string())
print("\n  Single-invoice orders should be ~1.0x (only the +1 differs).")
print("  Multi-invoice orders diverge because retry numbers restart at 0 each invoice,")
print("  so 'max retry - current retry' cannot see the invoices cancelling also kills.")

#%%
#%% False negatives + never-M0 investigation
# Add FUTURE_SUCCESS inside prepare_cutoff_frame, right after ORDER_ATTEMPTS_SAVED:
#
#     rev = df.iloc[::-1]
#     df['FUTURE_SUCCESS'] = rev.groupby('ORDER_ID')['IS_EVENTUALLY_SUCCESSFUL'].cummax().iloc[::-1]
#
# Cutting kills the whole order, so every invoice at or after this row dies with it.
# If any of them would have collected, the order was cut prematurely => false negative.
#
# And in _score, after 'Orders Cut':
#
#     'False Negatives': int((cut['FUTURE_SUCCESS'] == 1).sum()),
#     'FN Rate': f"{(cut['FUTURE_SUCCESS'] == 1).mean():.1%}",

import pandas as pd
import numpy as np

# =============================================================================
# Which never-M0 orders collected money, and why?
# =============================================================================
never = cutoff_df[cutoff_df['NEVER_M0']]
# > 1, not > 0: the +1 offset in prepare_cutoff_frame makes a true $0 order read $1.
collected = (never.groupby('ORDER_ID')['HISTORICAL_COLLECTED_AMOUNT'].max()
                  .loc[lambda s: s > 1].sort_values(ascending=False))

n_never = never['ORDER_ID'].nunique()
print("=" * 78)
print("NEVER-M0 ORDERS THAT COLLECTED MONEY")
print("=" * 78)
print(f"  never-M0 orders          : {n_never:,}")
print(f"  ...that collected > $0   : {len(collected):,} ({len(collected)/n_never:.2%})")
print(f"  total collected by them  : ${collected.sum():,.2f}")
print(f"\n  Every one of these is being cut with loss forced to $0.\n")
print(collected.head(20).to_string())

# =============================================================================
# THE DECIDER -- are they really non-M0, or did the June date filter drop them?
#
# load_cltv() filters `order_date >= '2025-06-01' and order_date < '2025-07-01'`.
# An order that DID make M0 but whose order_date falls outside June is ALSO absent
# from tb -- and my merge indicator marks it 'left_only' exactly like a true non-M0
# order. NEVER_M0 conflates the two. If is_m0=1 comes back below, that is the bug.
# =============================================================================
if len(collected):
    ids = ','.join(str(i) for i in collected.head(50).index)
    cur = ctx.cursor()
    cur.execute(f"""
    select order_id,
           max(is_m0)                                   as is_m0,
           min(order_date)                              as order_date,
           count(*)                                     as txns,
           sum(iff(transaction_status='accepted',1,0))  as accepted_txns,
           round(sum(iff(transaction_status='accepted', transaction_amount, 0)),2) as collected
    from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
    where order_id in ({ids})
    group by 1
    order by collected desc
    """)
    src = cur.fetch_pandas_all()
    print("\n" + "=" * 78)
    print("SAME ORDERS, STRAIGHT FROM THE SOURCE TABLE")
    print("=" * 78)
    print(src.to_string(index=False))
    print(f"\n  is_m0 = 1 anywhere above  -> NEVER_M0 is wrong; the June order_date filter")
    print(f"                              dropped them, not a failed trial.")
    print(f"  order_date outside June   -> confirms it.")
    print(f"  is_m0 = 0 and collected>0 -> is_m0 does not mean what you were told.")


#%% Verdict: bucket every 'never-M0-but-collected' order by the two suspects
# Cross-tabulates the source truth: did it EVER make M0, and is its order_date inside
# the June window load_cltv() filters on? If the mass lands in (is_m0=1, outside June),
# the date filter -- not is_m0 -- is what's dropping them from tb.
if len(collected):
    ids = ','.join(str(i) for i in collected.index)   # all of them, not just top 50
    cur = ctx.cursor()
    cur.execute(f"""
    select
        iff(is_m0_ever = 1, 'is_m0=1 (made M0)', 'is_m0=0 (true non-M0)') as m0_bucket,
        june_bucket,
        count(*)                    as orders,
        round(sum(collected), 2)    as collected
    from (
        select order_id,
               max(is_m0)                                                       as is_m0_ever,
               iff(min(order_date) >= '2025-06-01'
                   and max(order_date) < '2025-07-01', 'in June', 'outside June') as june_bucket,
               sum(iff(transaction_status = 'accepted', transaction_amount, 0))  as collected
        from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM
        where order_id in ({ids})
        group by order_id
    )
    group by 1, 2
    order by collected desc
    """)
    verdict = cur.fetch_pandas_all()
    print("\n" + "=" * 78)
    print("WHY THESE ORDERS ARE MISSING FROM tb  (source truth, all flagged orders)")
    print("=" * 78)
    print(verdict.to_string(index=False))
    print("\n  Rows in 'is_m0=1 (made M0)' + 'outside June' == the load_cltv date filter")
    print("  is throwing away real M0 orders. Fix: widen/remove the order_date window in")
    print("  load_cltv() to match df_pred's range, or key NEVER_M0 off is_m0, not the merge.")

# =============================================================================
# How big is the exposure? Free cuts are 76% of every cut you make.
# =============================================================================
print("\n" + "=" * 78)
print("WHAT IF THE FREE CUTS AREN'T FREE?")
print("=" * 78)
free_at_002 = 25714   # from your p<0.02 row
net_at_002 = 3359.43
print(f"  Net @ p<0.02 : ${net_at_002:,.2f} on {free_at_002:,} free cuts")
for per_order in [0.05, 0.10, 0.13, 0.20, 0.50]:
    adj = net_at_002 - free_at_002 * per_order
    flag = '  <-- Net goes NEGATIVE' if adj < 0 else ''
    print(f"  if each free cut really costs ${per_order:.2f} -> Net ${adj:>10,.2f}{flag}")
# %%
