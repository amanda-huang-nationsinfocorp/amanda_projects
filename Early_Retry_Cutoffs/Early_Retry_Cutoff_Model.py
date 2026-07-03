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
import os
import joblib

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
cur.execute("SELECT * FROM ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_DATA WHERE TRANSACTION_STATUS != 'accepted' ") 

# 3. Fetch directly to a Pandas DataFrame using Arrow
# This is drastically faster than pd.read_sql()
df = cur.fetch_pandas_all()

#%%
# =====================================================================
# 4.5 Aggressive Memory Cleanup (Prevent Kernel Crash)
# =====================================================================
print("Freeing up RAM before training...")
import gc

# Delete the massive initial dataframes that are no longer needed
del df, past_data, train_df, stop_df

# Delete the training and stopping Pandas splits because they are now safely compressed inside the CatBoost Pools!
del X_train, y_train, X_stop, y_stop

# Force the Python garbage collector to instantly dump them from RAM
gc.collect()

print("RAM cleared. Ready to train.")

#%% Model Training
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from catboost import CatBoostClassifier, Pool

sampled_df = df

# =====================================================================
# 1. Base Filtering and Sorting
# =====================================================================
# Force decline codes to be strings BEFORE attempting string slicing
sampled_df['DECLINE_CODE'] = sampled_df['DECLINE_CODE'].astype(str)
sampled_df['DECLINE_FAMILY'] = sampled_df['DECLINE_CODE'].str[0]

if 'LAST_DECLINE_CODE' in sampled_df.columns:
    sampled_df['LAST_DECLINE_CODE'] = sampled_df['LAST_DECLINE_CODE'].astype(str)

# Sort ascending to guarantee Past predicts Future
sampled_df = sampled_df.sort_values('TRANSACTION_DATETIME').reset_index(drop=True)

# =======================================================================
# 2. Data Processing & Pre-Split Vectorized I
# =======================================================================
print("Processing and Imputing Data...")
# Drop core IDs and target
drop_cols = ['ORDER_ID', 'INVOICE_ID', 'TRANSACTION_ID', 'TRANSACTION_DATETIME', 'IS_EVENTUALLY_SUCCESSFUL', 'TRANSACTION_STATUS']

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
# 3. Hybrid Split (Chronological OOT + Random In-Sample)y
# =====================================================================
print("Splitting data (Hybrid OOT + Random)...")

# 1. Slice off the strict future for the OOT Test Set (Last 10%)
n = len(sampled_df)
oot_start = int(n * 0.90)

past_data = sampled_df.iloc[:oot_start].copy()
oot_test_df = sampled_df.iloc[oot_start:].copy()

# 2. Randomly shuffle the past data to destroy time-order
print("Shuffling past data for random splits...")
past_data = past_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Split the randomized past data
m = len(past_data)
train_end = int(m * 0.75)  
stop_end  = int(m * 0.85)  
calib_end = int(m * 0.95)  

train_df       = past_data.iloc[:train_end]
stop_df        = past_data.iloc[train_end:stop_end]
calib_df       = past_data.iloc[stop_end:calib_end]
random_test_df = past_data.iloc[calib_end:]

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
    iterations=5000,           
    learning_rate=0.06,          # Updated Learning Rate
    depth=5,
    task_type="CPU",           
    eval_metric='Logloss',
    custom_metric=['AUC'],       
    l2_leaf_reg=8,
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
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
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
# 1. How well does it generalize to unseen data from the SAME time period?
random_probs = evaluate_split("Random Test", calibrated_model, X_random_test, y_random_test)

# 2. How well does it generalize to the STRICT FUTURE (Immaturity/Temporal Drift)?
oot_probs    = evaluate_split("OOT Test", calibrated_model, X_oot_test, y_oot_test)

# Calculate the Degradation Penalty
auc_drop = roc_auc_score(y_random_test, random_probs) - roc_auc_score(y_oot_test, oot_probs)
print(f"\nTemporal Degradation Penalty (AUC Drop): {auc_drop:.4f}")
print("*(If this drop is large, the recent data is highly 'immature' or behavior shifted over time)*")


#%% Save Model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'calibrated_catboost_model.joblib')

# 2. Package the model and the exact feature lists into one dictionary
# This ensures your prediction script applies the exact same preprocessing
model_artifact = {
    'model': calibrated_model,
    'cat_features': cat_features,
    'text_features': text_features,
    'drop_cols': drop_cols
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
# Replace this with your actual new dataset
# new_df = pd.read_csv('new_transactions.csv') 

# --- FOR DEMONSTRATION ONLY: Creating a dummy dataframe ---
new_df = pd.DataFrame() 
# ----------------------------------------------------------

print("Applying strict preprocessing rules...")

# 1. Format specific string columns if they exist
if 'DECLINE_CODE' in new_df.columns:
    new_df['DECLINE_CODE'] = new_df['DECLINE_CODE'].astype(str)
    new_df['DECLINE_FAMILY'] = new_df['DECLINE_CODE'].str[0]

if 'LAST_DECLINE_CODE' in new_df.columns:
    new_df['LAST_DECLINE_CODE'] = new_df['LAST_DECLINE_CODE'].astype(str)

# 2. Drop core IDs and target (ignoring errors if they aren't in the new data)
X_new = new_df.drop(columns=drop_cols, errors='ignore').copy()

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
print("Generating predictions...")

# Ensure column order matches training exactly
feature_names = calibrated_model.feature_names_in_
X_new = X_new[feature_names]

# Get the probability of the positive class (IS_EVENTUALLY_SUCCESSFUL = 1)
probabilities = calibrated_model.predict_proba(X_new)[:, 1]

# Attach probabilities back to the original dataframe
new_df['SUCCESS_PROBABILITY'] = probabilities

# Save the results
# output_path = os.path.join(script_dir, 'predictions_output.csv')
# new_df.to_csv(output_path, index=False)
# print(f"Predictions saved to: {output_path}")

print("Done.")

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
# =====================================================================

# 1. Check the Out-Of-Time (OOT) Test Set (Usually the most important for business reality)
plot_business_confusion_matrix(y_true=y_oot_test, 
                               y_probs=oot_probs, 
                               threshold=0.03, 
                               dataset_name="OOT Test Set (Strict Future)")

# 2. Check the Random Holdout Test Set
plot_business_confusion_matrix(y_true=y_random_test, 
                               y_probs=random_probs, 
                               threshold=0.03, 
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
import time

# =====================================================================
# 1. Configuration Settings (Easy to change!)
# =====================================================================
THRESHOLD = 0.03
CLTV_THRESHOLD = 15.0  

# Put the exact partner names you want to evaluate in this list. 
# If you want to evaluate EVERYONE, just leave the list empty like this: TARGET_PARTNERS = []
TARGET_PARTNERS = []

# 2. Fetch only the required columns from the CLTV table
cur = ctx.cursor()
cur.execute("SELECT ORDER_ID, CLTV_PRED_MEAN FROM DBT_PROD.ANALYTICS.FCT_ECLTV_ORDER") 
cltv_df = cur.fetch_pandas_all()
cur.close()

# 3. Bind predictions, actuals, AND CLTV directly into the DataFrame FIRST
predictions_df = random_test_df.copy() 
predictions_df['PREDICTED_PROBABILITY'] = random_probs 
predictions_df['ACTUAL_OUTCOME'] = np.array(y_random_test)
predictions_df['PREDICTED_OUTCOME'] = (random_probs >= THRESHOLD).astype(int)

# NEW: Merge the CLTV data immediately so we can use it in our decision logic
# Using a left merge ensures we don't drop rows if a CLTV happens to be missing
predictions_df = predictions_df.merge(cltv_df, on='ORDER_ID', how='left')

# Fill any completely missing CLTVs with 0 so the model still cuts them off safely
predictions_df['CLTV_PRED_MEAN'] = predictions_df['CLTV_PRED_MEAN'].fillna(0)

# =====================================================================
# *** THE NEW BUSINESS RULE ***
# A transaction is ONLY cancelled if the model predicts 0 AND the CLTV is < 5
# =====================================================================
predictions_df['IS_CANCELLED'] = ((predictions_df['PREDICTED_OUTCOME'] == 0) & 
                                  (predictions_df['CLTV_PRED_MEAN'] < CLTV_THRESHOLD)).astype(int)


# 4. Filter by Super Partner 
if TARGET_PARTNERS:
    predictions_df = predictions_df[predictions_df['SUPER_PARTNER_ID_NAME'].isin(TARGET_PARTNERS)]
    partner_label = f"Filtered ({len(TARGET_PARTNERS)} Partners)"
    print(f"Filtering down to target partners. Rows remaining: {len(predictions_df):,}")
else:
    partner_label = "ALL Partners"
    print(f"Evaluating all partners. Total rows: {len(predictions_df):,}")


# =====================================================================
# 5. Calculate Cost Savings (True Negatives)
# =====================================================================
# How many unique orders had at least one retry cancelled?
cancelled_mask = (predictions_df['IS_CANCELLED'] == 1)
total_orders_cancelled = predictions_df[cancelled_mask]['ORDER_ID'].nunique()

# A True Negative is now: We ACTUALLY cancelled it, AND the real outcome was 0
tn_mask = (predictions_df['IS_CANCELLED'] == 1) & (predictions_df['ACTUAL_OUTCOME'] == 0)
tn_count = tn_mask.sum()
cost_savings = tn_count * 0.10

# =====================================================================
# 6. Calculate True Missed Revenue (False Negatives x CLTV)
# =====================================================================
# A False Negative is now: We ACTUALLY cancelled it, BUT the real outcome was 1
fn_mask = (predictions_df['IS_CANCELLED'] == 1) & (predictions_df['ACTUAL_OUTCOME'] == 1)

# Extract unique orders so we only sum the missed CLTV once per affected order
unique_fn_orders = predictions_df[fn_mask].drop_duplicates(subset=['ORDER_ID'])

total_missed_revenue = unique_fn_orders['CLTV_PRED_MEAN'].sum()
orders_affected = unique_fn_orders['ORDER_ID'].nunique()

# Calculate final net impact
net_profit = cost_savings - total_missed_revenue

# Safe division to prevent errors if total_orders_cancelled is 0
profit_per_cancel = (net_profit / total_orders_cancelled) if total_orders_cancelled > 0 else 0

# =====================================================================
# 7. Final ROI Output
# =====================================================================
print("="*65)
print(f" 💰 FINAL ROI REPORT (RANDOM TEST SET) 💰")
print(f" Model Threshold: {THRESHOLD} | CLTV Max Limit: ${CLTV_THRESHOLD}")
print(f" Segment: {partner_label}")
print("="*65)
print(f"Total True Negatives:           {tn_count:,}")
print(f"Total Retry Cost Saved:        + ${cost_savings:,.2f}")
print(f"Total Missed Revenue (FN CLTV):- ${total_missed_revenue:,.2f}")
print("-" * 65)
if net_profit > 0:
    print(f"NET FINANCIAL IMPACT:                + ${net_profit:,.2f} (PROFITABLE)")
    print(f"NET FINANCIAL IMPACT/CANCELATION:    + ${profit_per_cancel:,.6f} (PROFITABLE)")
else:
    print(f"NET FINANCIAL IMPACT:                - ${abs(net_profit):,.2f} (LOSS)")
    print(f"NET FINANCIAL IMPACT/CANCELATION:    - ${abs(profit_per_cancel):,.6f} (LOSS)")
print("="*65)
print(f"* We cancelled {total_orders_cancelled:,} unique orders total.")
print(f"* Missed revenue came from {orders_affected:,} prematurely cancelled orders.")


#%% Grid search ecltv 
import numpy as np
import pandas as pd
import time

print("Starting ROI Grid Search Optimization...")
start_time = time.time()

# =====================================================================
# 1. Define the ranges you want to test
# =====================================================================
# Test probability thresholds from 0.01 to 0.50 in steps of 0.05
probability_thresholds = np.arange(0.01, 0.50, 0.02) 

# Test these specific CLTV limits
cltv_thresholds = [1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]

# =====================================================================
# 2. Extract raw NumPy arrays for massive speedup in the loop
# =====================================================================
probs = predictions_df['PREDICTED_PROBABILITY'].values
actuals = predictions_df['ACTUAL_OUTCOME'].values
cltvs = predictions_df['CLTV_PRED_MEAN'].values
order_ids = predictions_df['ORDER_ID'].values

results = []

# =====================================================================
# 3. Run the Grid Search
# =====================================================================
for thresh in probability_thresholds:
    # Model predicts failure (0) when probability is LESS than threshold
    pred_fail = (probs < thresh)
    
    for cltv_limit in cltv_thresholds:
        # Business rule allows cancellation (Must be predicted fail AND under CLTV limit)
        is_cancelled = pred_fail & (cltvs < cltv_limit)
        
        # Cost Savings (True Negatives)
        tn_mask = is_cancelled & (actuals == 0)
        cost_savings = np.sum(tn_mask) * 0.10
        
        # Missed Revenue (False Negatives)
        fn_mask = is_cancelled & (actuals == 1)
        
        # Grab the arrays strictly for the False Negatives
        fn_order_ids = order_ids[fn_mask]
        fn_cltvs = cltvs[fn_mask]
        
        # Put into a quick temp DataFrame to drop duplicate orders
        # (Ensures we only sum the missed CLTV once per order)
        temp_df = pd.DataFrame({'ORDER_ID': fn_order_ids, 'CLTV': fn_cltvs})
        unique_missed_orders = temp_df.drop_duplicates(subset=['ORDER_ID'])
        
        missed_revenue = unique_missed_orders['CLTV'].sum()
        orders_affected = unique_missed_orders['ORDER_ID'].nunique()
        
        net_profit = cost_savings - missed_revenue
        
        # Save the results of this combination
        results.append({
            'Model_Threshold': round(thresh, 3),
            'CLTV_Limit': cltv_limit,
            'Net_Profit': net_profit,
            'Cost_Savings': cost_savings,
            'Missed_Revenue': missed_revenue,
            'Orders_Cancelled': orders_affected
        })

# =====================================================================
# 4. Display the Best Results
# =====================================================================
# Convert results to a DataFrame and sort by the highest Net Profit
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Net_Profit', ascending=False).reset_index(drop=True)

print(f"Grid search evaluated {len(probability_thresholds) * len(cltv_thresholds)} combinations in {time.time() - start_time:.2f} seconds.\n")

print("🏆 TOP 10 MOST PROFITABLE COMBINATIONS 🏆")
print("-" * 80)
print(results_df.head(10).to_string(index=False))