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

#%% Model V2
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
# 2. Data Processing & Pre-Split Vectorized Imputation 
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
# 3. Hybrid Split (Chronological OOT + Random In-Sample)
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
    iterations=2000,           
    learning_rate=0.05,          # Updated Learning Rate
    depth=5,
    task_type="CPU",           
    eval_metric='Logloss',
    custom_metric=['AUC'],       
    l2_leaf_reg=5,
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

#%% Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Choose your Threshold
threshold = 0.05 

# 2. Convert probabilities to hard predictions
# (Assuming y_prob and y_test are already defined from your model output)
y_pred = (y_prob >= threshold).astype(int)

# 3. Calculate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# 4. Calculate Percentages and Create Custom Labels
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

plt.title(f'Confusion Matrix (Threshold: {threshold})', fontsize=14, pad=15)
plt.xlabel('What the Model Predicted', fontsize=12, labelpad=10)
plt.ylabel('What Actually Happened', fontsize=12, labelpad=10)
plt.tight_layout()
plt.show()

# 6. Print the business breakdown
tn, fp, fn, tp = cm.ravel()
print(f"\n--- Confusion Matrix Breakdown (Threshold: {threshold}) ---")
print(f"True Negatives (TN):  {tn} ({cm_percentages[0,0]:.1%}) -> Correctly cut off")
print(f"False Positives (FP): {fp} ({cm_percentages[0,1]:.1%}) -> Wasted $0.10 retry")
print(f"False Negatives (FN): {fn} ({cm_percentages[1,0]:.1%}) -> Missed revenue!")
print(f"True Positives (TP):  {tp} ({cm_percentages[1,1]:.1%}) -> Successfully collected")


#%% Cost benefit analysis 
import pandas as pd
import numpy as np
import time

print("Fetching raw data directly from Snowflake...")
start_time = time.time()

# 1. Fetch raw data using Snowflake (Assuming 'ctx' is already connected)
cur = ctx.cursor()
cur.execute("USE DATABASE dbt_prod")
cur.execute("SELECT * FROM ANALYTICS.EARLY_RETRY_CUTOFF.EARLY_RETRY_CUTOFF_DATA") 

# Fetch directly to a Pandas DataFrame using Arrow
raw_df = cur.fetch_pandas_all()

cur.close()
ctx.close() 

# Ensure the raw datetime column is formatted correctly for comparison
raw_df['TRANSACTION_DATETIME'] = pd.to_datetime(raw_df['TRANSACTION_DATETIME'])
print(f"Raw data fetched successfully in {time.time() - start_time:.2f} seconds.\n")

print("Calculating Cost Savings and Missed Revenue...")
calc_start = time.time()

# 2. Isolate predictions strictly to the unseen testing dataset (test_df)
predictions_df = test_df.copy()
predictions_df['PREDICTED_PROBABILITY'] = y_prob
predictions_df['TRANSACTION_DATETIME'] = pd.to_datetime(predictions_df['TRANSACTION_DATETIME'])

threshold = 0.02

# 3. Calculate Cost Savings (0.10 * count of True Negatives)
# A True Negative is when the model predicts < 0.30 (Fail) AND the actual outcome was 0 (Fail)
y_pred = (y_prob >= threshold).astype(int)
y_actual = predictions_df['IS_EVENTUALLY_SUCCESSFUL'].values

tn_count = ((y_pred == 0) & (y_actual == 0)).sum()
cost_savings = tn_count * 0.10

# 4. Find the EXACT Cutoff Moments strictly from the test set predictions
cutoffs = predictions_df[predictions_df['PREDICTED_PROBABILITY'] < threshold]

# Get the absolute first time the model said "Stop" for each order in the test set
first_cutoffs = cutoffs.groupby('ORDER_ID')['TRANSACTION_DATETIME'].min().reset_index()
first_cutoffs = first_cutoffs.rename(columns={'TRANSACTION_DATETIME': 'CUTOFF_TIME'})

# 5. Merge the Test Set Cutoffs with the RAW DATABASE
# This attaches the test-set cutoff timestamp to the unfiltered real-world data
merged_raw_df = raw_df.merge(first_cutoffs, on='ORDER_ID', how='inner')

# 6. Filter for True Missed Revenue (Fixing the Blind Spot)
# Look for rows that happened STRICTLY AFTER the cutoff time 
# AND where the transaction actually succeeded ('accepted')
true_missed_opportunities = merged_raw_df[
    (merged_raw_df['TRANSACTION_DATETIME'] > merged_raw_df['CUTOFF_TIME']) & 
    (merged_raw_df['TRANSACTION_STATUS'].str.lower() == 'accepted') 
]

# Deduplicate by INVOICE_ID so we only count the final successful payment amount once per invoice
unique_true_missed = true_missed_opportunities.drop_duplicates(subset=['ORDER_ID', 'INVOICE_ID'])

total_missed_revenue = unique_true_missed['TRANSACTION_AMOUNT'].sum()
orders_affected = unique_true_missed['ORDER_ID'].nunique()

net_profit = cost_savings - total_missed_revenue

print(f"Calculations completed in {time.time() - calc_start:.2f} seconds!\n")

# 7. Final ROI Output
print("="*50)
print(f" 💰 FINAL ROI REPORT (TEST SET | THRESHOLD: {threshold}) 💰")
print("="*50)
print(f"Total True Negatives:           {tn_count:,}")
print(f"Total Retry Cost Saved:        + ${cost_savings:,.2f}")
print(f"Total Missed Revenue (FN):     - ${total_missed_revenue:,.2f}")
print("-" * 50)
if net_profit > 0:
    print(f"NET FINANCIAL IMPACT:          + ${net_profit:,.2f} (PROFITABLE)")
else:
    print(f"NET FINANCIAL IMPACT:          - ${abs(net_profit):,.2f} (LOSS)")
print("="*50)
print(f"* Missed revenue came from {orders_affected} prematurely cancelled orders.")


#%% Shap
import shap
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PREPARATION: Sample Data & Extract Base Estimator
# ==============================================================================

print("Sampling data for SHAP (avoiding OOM on 35M rows)...")
# 50,000 rows gives a highly statistically significant SHAP distribution 
# without taking 3 hours to compute.
X_shap = X_test.sample(n=min(50000, len(X_test)), random_state=42)

# ==============================================================================
# 2. CALCULATE SHAP VALUES
# ==============================================================================

print("Calculating SHAP values using the base CatBoost model...")
# Initialize the TreeExplainer. CatBoost is highly optimized for this.
explainer = shap.TreeExplainer(base_model)

# Calculate SHAP values. 
# Note: For classification, this returns the log-odds impact.
shap_values = explainer.shap_values(X_shap)

# ==============================================================================
# 3. GENERATE DIAGNOSTIC PLOTS
# ==============================================================================

# A. Global Feature Importance (Bar Chart)
# This will show you exactly which features drive the most expected value.
plt.figure(figsize=(10, 8))
plt.title("Feature Importance (Mean Absolute SHAP Value)")
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.tight_layout()
plt.show()

# B. Directional Impact (Summary Dot Plot)
# Crucial for understanding HOW features impact the prediction.
# E.g., Does a high RETRY_COUNT push the probability up or down?
plt.figure(figsize=(10, 8))
plt.title("Directional Feature Impact on Eventual Success")
shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
plt.show()

# ==============================================================================
# 4. TARGETED DEPENDENCE PLOTS (To validate Data Engineering fixes)
# ==============================================================================
# Use these specific plots to validate the data engineering fixes I suggested.

# Validating the "Payday Window" fix:
# Look for non-linear spikes around the 1st, 15th, and 30th. If they exist, 
# your model is begging for an explicit IS_PAYDAY_WINDOW boolean feature.
if 'RETRY_DAY_OF_MONTH' in X_shap.columns:
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("RETRY_DAY_OF_MONTH", shap_values, X_shap, show=False)
    plt.title("SHAP Dependence: Retry Day of Month")
    plt.tight_layout()
    plt.show()

# Validating the "Payment Frequency Magnitude" fix:
# If you mapped '1w' to 7 and '1m' to 30, this plot will show if the model 
# treats the continuous magnitude differently based on the OFFER_AMOUNT.
if 'OFFER_AMOUNT' in X_shap.columns:
    plt.figure(figsize=(8, 6))
    # interaction_index='auto' usually picks the most correlated feature, 
    # but you can force it to check PAYMENT_FREQUENCY logic.
    shap.dependence_plot("OFFER_AMOUNT", shap_values, X_shap, show=False)
    plt.title("SHAP Dependence: Offer Amount")
    plt.tight_layout()
    plt.show()


#%%
