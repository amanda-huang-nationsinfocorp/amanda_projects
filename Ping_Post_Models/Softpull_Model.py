#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import json
import re
import snowflake.connector
from sqlalchemy import create_engine, text
from snowflake.sqlalchemy import URL
from datetime import datetime, timedelta, timezone
import pytz



# In[6]:


url = URL(
    user='BITEAM',
    password='B1sense@22',
    account='YXBYZCG-MVA06208',
    database="DBT_PROD.PUBLIC",
)
engine = create_engine(url)
connection = engine.connect()


#%% QUERY


query = '''
with 
pingpost as (
SELECT 
    jluvr,
    order_id,
    request_datetime,
    seller_name,
    label,
    seller_minimum_price,
    lead_user_agent,
    loan_amount,
    loan_purpose,
    finance_current_unsecured_debt_amount,
    applicant_employment_type,
    applicant_income_net_monthly_amount,
    applicant_employment_length,
    applicant_income_last_pay_day,
    applicant_income_next_pay_day,
    applicant_date_of_birth,
    applicant_current_address_state,
    applicant_current_address_residence_status,
    applicant_current_address_years,
    bank_account_bank_name,
    bank_account_years,
    --lead_creation_url,
    --campaign_id,
    consent_marketing_email
FROM 
    ANALYTICS.BLOODHOUND_PROJECT.PINGPOST_DATA_PARSED
),

soft_pull as (
select 
jluvr, 
to_date(inquiry_date) as inquiry_date, 
risk_score,
--population_rank,
YEAR(CURRENT_DATE()) - BIRTH_YEAR as age,
employer,
--negative_factor_1,
--negative_factor_2,
--negative_factor_3,
--negative_factor_4,
IFF('12' IN (negative_factor_1, negative_factor_2, negative_factor_3, negative_factor_4), 1, 0) AS has_negative_factor_12,
IFF('63' IN (negative_factor_1, negative_factor_2, negative_factor_3, negative_factor_4), 1, 0) AS has_negative_factor_63,
IFF('07' IN (negative_factor_1, negative_factor_2, negative_factor_3, negative_factor_4), 1, 0) AS has_negative_factor_07,
IFF('95' IN (negative_factor_1, negative_factor_2, negative_factor_3, negative_factor_4), 1, 0) AS has_negative_factor_95,
IFF('04' IN (negative_factor_1, negative_factor_2, negative_factor_3, negative_factor_4), 1, 0) AS has_negative_factor_04,
--positive_factor_1,
--positive_factor_2,
--positive_factor_3,
--positive_factor_4,
IFF('P05' IN (positive_factor_1, positive_factor_2, positive_factor_3, positive_factor_4), 1, 0) AS has_positive_factor_P05,
IFF('P34' IN (positive_factor_1, positive_factor_2, positive_factor_3, positive_factor_4), 1, 0) AS has_positive_factor_P34,
IFF('P08' IN (positive_factor_1, positive_factor_2, positive_factor_3, positive_factor_4), 1, 0) AS has_positive_factor_P08,
IFF('P04' IN (positive_factor_1, positive_factor_2, positive_factor_3, positive_factor_4), 1, 0) AS has_positive_factor_P04,
IFF('P95' IN (positive_factor_1, positive_factor_2, positive_factor_3, positive_factor_4), 1, 0) AS has_positive_factor_P95
from ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_SERVICE_PRODUCT_RESPONSE
),

trade_lines as (
select 
jluvr, 
count(*) as total_account_count, 
count_if(account_current_status = 'Open') as total_open_account,
count_if(account_type in ('Revolving', 'Credit Line', 'Open Account', 'Mortgage')) as revolving_account_count,
count_if(account_type = 'Installment') as installment_account_count,
count_if(account_type = 'Colletion') as collection_account_count,
sum(high_balance) as total_high_balance,
sum(current_balance) as total_current_balance,
div0(sum(current_balance), sum(high_balance)) as credit_utilization_ratio,
max(DATEDIFF('month', account_open_date, CURRENT_DATE()) / 12.0) as earliest_account_open_year,
min(DATEDIFF('month', account_open_date, CURRENT_DATE()) / 12.0) as latest_account_open_year,
--avg(DATEDIFF('month', account_open_date, account_close_date / 12.0) as avg_account_duration_year,
count_if(account_status = 'Derogatory') as derogatory_account_count,
div0(count_if(account_status = 'Derogatory'), count(*)) as derogatory_account_ratio,
div0(count_if(account_status = 'Derogatory' and account_current_status = 'Closed'),count_if(account_status = 'Derogatory')) as closed_derogatory_account_ratio,
count_if(account_status = 'Paid') as paid_account_count,
count_if(account_status in ('Refinanced', 'Transferred')) as transferred_account_count,
div0(count_if(payment_status = 'Current'),count(*)) as on_time_payment_current_ratio,
count_if(open_account_type = 'Unsecured loan') as unsecured_loan_account_count,
count_if(open_account_type = 'Secured loan') as secured_loan_account_count,
count_if(open_account_type = 'Educational') as educational_account_count,
count_if(lower(open_account_type) like 'credit card') as credit_card_account_count,
count_if(open_account_type = 'Charge account') as charge_account_count,
sum(case when account_current_status = 'Open' then monthly_payment end) as current_monthly_payment_total,
div0(sum(case when account_current_status = 'Open' then monthly_payment end), sum(current_balance)) as payment_burden_ratio,
--avg(monthly_payment) as historic_monthly_payment_avg,
avg(month_reviewed) as month_reviewed_avg,
sum(late_30_count) as late_30_total,
sum(late_60_count) as late_60_total,
sum(late_90_count) as late_90_total,
sum(amount_past_due) as past_due_amount_total,
div0(sum(amount_past_due), sum(current_balance)) as past_due_to_current_balance_ratio

from ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_TRADE_LINE
group by 1
),

inquiry as (
select 
jluvr,
count(*) as inquiry_total,
COUNT_IF(DATEDIFF('month', inquiry_date, CURRENT_DATE()) <= 6) AS inquiry_within_6_month,
div0(count_if(industry_code = 'Bank'),count(*)) as bank_inquiry_ratio,
div0(count_if(industry_code = 'Finance/Personal'),count(*)) as personal_inquiry_ratio
from ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_INQUIRY
group by 1
),

public_record as (
select 
jluvr, 
count(*) as bankcrupcy_total,
min(DATEDIFF('month', public_record_date, CURRENT_DATE()) / 12.0) as most_recent_record_year,
div0(count_if(status = 'Discharged'), count(*)) as bankcrupty_discharged_ratio
from  ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_PUBLIC_RECORD
where classification = 'Bankruptcy'
group by 1
)

select 
*
from pingpost
left join soft_pull using(jluvr)
left join trade_lines using(jluvr)
left join inquiry using(jluvr)
left join public_record using(jluvr)
where risk_score is not null
'''


# %% DATA EXTRACTION (df)
with engine.connect() as con:
    con.execute(text("use database dbt_prod"))
    df = pd.read_sql(query, con)

# %% DATA TYPE CLEANING & NULL HANDLING

# Data cleaning starts
financial_cols = [
    'loan_amount', 
    'finance_current_unsecured_debt_amount', 
    'applicant_income_net_monthly_amount'
]


for col in financial_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Convert date columns directly to datetime
date_cols = [
    'inquiry_date',
    'applicant_date_of_birth',
    'applicant_income_last_pay_day',
    'applicant_income_next_pay_day'
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# 1. Drop the order_id column
if 'order_id' in df.columns:
    df = df.drop(columns=['order_id'])

# 2. Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numeric_cols = df.select_dtypes(include=['number']).columns

# 3. Fill categorical nulls with "Unknown"
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# 4. Fill numeric nulls with 0
df[numeric_cols] = df[numeric_cols].fillna(0)


# %% DATA TRANSFORMATION (PingPost)
# Data transformation (PingPost) starts here
df['is_m0'] = (df['label'].str.strip().str.lower() == 'm0').astype(int)

def get_resident_status(applicant_current_address_residence_status: str) -> str:
    name = applicant_current_address_residence_status.lower() if applicant_current_address_residence_status else ""
    if "rent" in name:
        return "rent"
    elif "own" in name:
        return "own"
    else:
        return "others"
    
def get_employment_type(applicant_employment_type: str) -> str:
    name = applicant_employment_type.lower() if applicant_employment_type else ""
    if "full" in name:
        return "full_time"
    elif "benefit" in name:
        return "benefit"
    elif "pension" in name:
        return "benefit"
    elif "part" in name:
        return "part_time"
    else:
        return "others"
        
def get_device_type(lead_user_agent: str) -> str:
    name = lead_user_agent.lower() if lead_user_agent else ""
    if "iphone" in name:
        return "apple_mobile"
    elif "android" in name:
        return "not_apple_mobile"
    elif "macintosh" in name:
        return "apple_desktop"
    elif "windows" in name or "x11" in name:
        return "not_apple_desktop"
    elif "ipad" in name:
        return "apple_tablet"
    else:
        return 'others'
    
US_STATES = (
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
)

def get_states(code: str) -> str:
    # Handle None/NaN safety
    if not code:
        return 'others'
        
    name = code
    
    # Now 'US_STATES' refers to the global list, 'name' refers to the input 
    if name in US_STATES:
        return name
    else:
        return 'others'

df["resident_status"] = df["applicant_current_address_residence_status"].apply(get_resident_status)
df["employment_type"] = df["applicant_employment_type"].apply(get_employment_type)
df['applicant_current_address_state'] = df['applicant_current_address_state'].str[:2].str.upper()
df['state'] = df["applicant_current_address_state"].apply(get_states)

df['loan_to_age_ratio'] = df['loan_amount'] / df['age']
df['income_to_age_ratio'] = df['applicant_income_net_monthly_amount'] / df['age']


# Create a flag for Neo-banks, Fintechs, and Prepaid Cards
fintech_keywords = 'chime|cash app|gobank|netspend|varo|dave|current|revolut|monzo|sofi|ally|greendot|green dot'
df['is_fintech_bank'] = df['bank_account_bank_name'].astype(str).str.contains(fintech_keywords, case=False, na=False).astype(int)
df['is_credit_union'] = df['bank_account_bank_name'].astype(str).str.contains('credit union| fcu| cu', case=False, na=False).astype(int)

next_pay_date = pd.to_datetime(df['applicant_income_next_pay_day'], errors='coerce')
request_date = pd.to_datetime(df['request_datetime'], errors='coerce')

next_pay_date = next_pay_date.dt.tz_localize(None)
request_date = request_date.dt.tz_localize(None)

next_pay_date = next_pay_date.dt.normalize()
request_date = request_date.dt.normalize()

df['days_till_next_pay'] = (next_pay_date - request_date).dt.days

df['loan_to_income_ratio'] = df['loan_amount'] / (df['applicant_income_net_monthly_amount'] + 0.001)
df['loan_to_income_ratio'] = df['loan_to_income_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

####################################### LOCAL LEAD TIME ###########################################

# 1. The State-to-Timezone Mapping Dictionary
# (Note: Assumes 2-letter state abbreviations. If your data uses full names, let me know!)
state_tz_map = {
    # Eastern
    'CT': 'America/New_York', 'DE': 'America/New_York', 'FL': 'America/New_York', 
    'GA': 'America/New_York', 'IN': 'America/New_York', 'ME': 'America/New_York', 
    'MD': 'America/New_York', 'MA': 'America/New_York', 'MI': 'America/New_York', 
    'NH': 'America/New_York', 'NJ': 'America/New_York', 'NY': 'America/New_York', 
    'NC': 'America/New_York', 'OH': 'America/New_York', 'PA': 'America/New_York', 
    'RI': 'America/New_York', 'SC': 'America/New_York', 'VT': 'America/New_York', 
    'VA': 'America/New_York', 'WV': 'America/New_York', 'DC': 'America/New_York',
    # Central
    'AL': 'America/Chicago', 'AR': 'America/Chicago', 'IL': 'America/Chicago', 
    'IA': 'America/Chicago', 'KS': 'America/Chicago', 'KY': 'America/Chicago', 
    'LA': 'America/Chicago', 'MN': 'America/Chicago', 'MS': 'America/Chicago', 
    'MO': 'America/Chicago', 'NE': 'America/Chicago', 'ND': 'America/Chicago', 
    'OK': 'America/Chicago', 'SD': 'America/Chicago', 'TN': 'America/Chicago', 
    'TX': 'America/Chicago', 'WI': 'America/Chicago',
    # Mountain
    'AZ': 'America/Phoenix', # AZ doesn't observe Daylight Saving Time!
    'CO': 'America/Denver', 'ID': 'America/Denver', 'MT': 'America/Denver', 
    'NM': 'America/Denver', 'UT': 'America/Denver', 'WY': 'America/Denver',
    # Pacific
    'CA': 'America/Los_Angeles', 'NV': 'America/Los_Angeles', 
    'OR': 'America/Los_Angeles', 'WA': 'America/Los_Angeles',
    # Others
    'AK': 'America/Anchorage', 'HI': 'Pacific/Honolulu'
}

# 2. Map the timezone string to each row
# If a state is missing or invalid, we default to Los_Angeles (PST) to prevent errors
df['target_tz'] = df['state'].str.upper().map(state_tz_map).fillna('America/Los_Angeles')

# 3. Ensure your base datetime is actually recognized as PST (Now DST-Proof!)
# Strip timezone if already aware to avoid errors
if df['request_datetime'].dt.tz is not None:
    df['request_datetime'] = df['request_datetime'].dt.tz_localize(None)

# Create an array of 'True' to force Pandas to treat the ambiguous hour as DST
is_dst = np.ones(len(df), dtype=bool)

# Localize using our new DST rules
df['request_datetime'] = df['request_datetime'].dt.tz_localize(
    'America/Los_Angeles', 
    ambiguous=is_dst,             # Fixes the Fall Back duplicate hour
    nonexistent='shift_forward'   # Fixes the Spring Forward missing hour
)

# 4. The "Fast GroupBy" Conversion Trick
local_times = []
for tz, group in df.groupby('target_tz'):
    # Convert the chunk to the state's timezone, then immediately strip the TZ info (tz_localize(None))
    converted_chunk = group['request_datetime'].dt.tz_convert(tz).dt.tz_localize(None)
    local_times.append(converted_chunk)

# 5. Stitch it all back together and assign it to a new column
df['local_request_datetime'] = pd.concat(local_times)

# 6. Drop the temporary helper column
df = df.drop(columns=['target_tz'])

df['day_of_week'] = df['local_request_datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['local_lead_hour_of_day'] = df['local_request_datetime'].dt.hour

#  Create the macro Morning Flag (Hours 5 through 11, which covers 5:00 AM to 11:59 AM)
df['is_morning_application'] = df['local_lead_hour_of_day'].apply(lambda x: 1 if 5 <= x <= 11 else 0).astype(str)
df['seller_morning_combo'] = df['seller_name'].astype(str) + "_morning_" + df['is_morning_application']

# This guarantees our dataframe perfectly matches the order Pandas uses when grouping.
df = df.sort_values(['seller_name', 'request_datetime']).reset_index(drop=True)

############################### LOCAL LEAD TIME END ####################################


############################### Lead User Agent Parsing ####################################
df["device_type"] = df["lead_user_agent"].apply(get_device_type)

conditions_browser = [
    df['lead_user_agent'].str.contains('Edg', case=False, na=False),
    df['lead_user_agent'].str.contains('Chrome|CriOS', case=False, na=False),
    # Safari check must exclude Chrome because Chrome iOS agents contain "Safari"
    df['lead_user_agent'].str.contains('Safari', case=False, na=False) & ~df['lead_user_agent'].str.contains('Chrome|CriOS', case=False, na=False),
    df['lead_user_agent'].str.contains('Firefox|FxiOS', case=False, na=False)
]
choices_browser = ['Edge', 'Chrome', 'Safari', 'Firefox']
df['browser_family'] = np.select(conditions_browser, choices_browser, default='Other/Unknown')
# Assuming you already applied your get_device_type function to create 'device_type'
df['device_browser_combo'] = df['device_type'].astype(str) + "_" + df['browser_family']

# Extract the first 2-to-3 digit number it finds after the word "Chrome" or "Android"
df['chrome_version'] = df['lead_user_agent'].str.extract(r'Chrome/(\d{2,3})')
df['android_version'] = df['lead_user_agent'].str.extract(r'Android (\d{1,2})')

# Fill missing values with 'unknown' so CatBoost can group non-Chrome/non-Android users
df['chrome_version'] = df['chrome_version'].fillna('unknown')
df['android_version'] = df['android_version'].fillna('unknown')

# 1. Detect Social Media / In-App Browsers (HUGE signal for lead generation)
conditions_in_app = [
    df['lead_user_agent'].str.contains('FBAV|FBAN|FBIOS', case=False, na=False), # Facebook
    df['lead_user_agent'].str.contains('Instagram', case=False, na=False),       # Instagram
    df['lead_user_agent'].str.contains('Snapchat', case=False, na=False),        # Snapchat
    df['lead_user_agent'].str.contains('TikTok|Musical_ly', case=False, na=False) # TikTok
]
choices_in_app = ['Facebook_App', 'Instagram_App', 'Snapchat_App', 'TikTok_App']
df['social_media_source'] = np.select(conditions_in_app, choices_in_app, default='Browser/Organic')

# 2. Extract Specific Android Device Brands (Wealth/Income proxies)
# SM- / SAMSUNG- = Samsung
# moto = Motorola (Often budget/prepaid)
# LM- / LGL = LG
# Pixel = Google Pixel
conditions_brand = [
    df['lead_user_agent'].str.contains('SM-|SAMSUNG', case=False, na=False),
    df['lead_user_agent'].str.contains('moto', case=False, na=False),
    df['lead_user_agent'].str.contains('Pixel', case=False, na=False),
    df['lead_user_agent'].str.contains('LM-|LGL', case=False, na=False)
]
choices_brand = ['Samsung', 'Motorola', 'Google_Pixel', 'LG']
df['android_device_brand'] = np.select(conditions_brand, choices_brand, default='Other/Non-Android')

# 3. Create a powerful combo feature
df['source_device_combo'] = df['social_media_source'] + "_" + df['android_device_brand']

############################### Lead User Agent Parsing ENDS ####################################


# 3. Create a temporary dataframe with the datetime as the index for the rolling calculation
temp_df = df.set_index('request_datetime')

# 4. Calculate the rolling stats=t').agg(['mean', 'count'])
rolling_stats = temp_df.groupby('seller_name')['is_m0'].rolling('7D', closed='left').agg(['mean', 'count'])

# 5. THE FIX: Assign using `.values`
# This strips away the problematic indices and just pastes the raw numbers top-to-bottom
df['seller_7d_conversion_rate'] = rolling_stats['mean'].values
df['seller_7d_lead_volume'] = rolling_stats['count'].values

# 6. Handle Cold Starts (Vendors with 0 leads in the past 7 days)
global_cr = df['is_m0'].mean()
df['seller_7d_conversion_rate'] = df['seller_7d_conversion_rate'].fillna(global_cr)
df['seller_7d_lead_volume'] = df['seller_7d_lead_volume'].fillna(0)

rolling_3d = temp_df.groupby('seller_name')['is_m0'].rolling('3D', closed='left').agg(['mean', 'count'])
rolling_1d = temp_df.groupby('seller_name')['is_m0'].rolling('1D', closed='left').agg(['mean', 'count'])

# Map the 3-day stats back to the main dataframe
df['seller_3d_conversion_rate'] = rolling_3d['mean'].values
df['seller_3d_lead_volume'] = rolling_3d['count'].values

# Handle Cold Starts for the new 3-day columns
global_cr = df['is_m0'].mean()
df['seller_3d_conversion_rate'] = df['seller_3d_conversion_rate'].fillna(global_cr)
df['seller_3d_lead_volume'] = df['seller_3d_lead_volume'].fillna(0)

# Create the explicit Momentum Metric: 
# If this is positive, the seller is currently hotter than their weekly average.
# If this is negative, the seller's conversion rate is crashing right now.
df['cr_momentum_3d_vs_7d'] = df['seller_3d_conversion_rate'] - df['seller_7d_conversion_rate']

# Finally, sort purely by time so it's ready for the Out-Of-Time train/test split
df = df.sort_values('request_datetime').reset_index(drop=True)

# 1. KEEP: Volume Momentum (Traffic Surge)
daily_avg_7d = (df['seller_7d_lead_volume'] / 7).fillna(0)
daily_avg_3d = (df['seller_3d_lead_volume'] / 3).fillna(0)
df['volume_surge_ratio'] = daily_avg_3d / (daily_avg_7d + 1e-6)
df['volume_surge_ratio'] = df['volume_surge_ratio'].replace([np.inf, -np.inf], 1.0)
df['is_traffic_surge'] = (df['volume_surge_ratio'] > 1.5).astype(int)

# 2. KEEP: Value Index
df['seller_value_index'] = df['seller_minimum_price'] / (df['seller_7d_conversion_rate'] + 0.001)


# 4. NEW: Seller + Device Interaction
# Assuming you extracted device_type (e.g., 'apple_mobile', 'not_apple_mobile')
df['seller_device_combo'] = df['seller_name'].astype(str) + "_" + df['device_type'].astype(str)

# %% DATA TRANSFORMATION (Softpull)
# Data transformation (Softpull) starts here
# 1. Late Severity Score
df['late_severity_score'] = (df['late_30_total'] * 1) + (df['late_60_total'] * 2) + (df['late_90_total'] * 3)

# Open Accounts Ratio
# We use .replace() and .fillna() to safely handle any division by zero (which creates infinity or NaN)
df['open_accounts_ratio'] = (df['total_open_account'] / df['total_account_count'])
df['open_accounts_ratio'] = df['open_accounts_ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Convert single-digit error codes to NaN so CatBoost treats them as missing
df['risk_score'] = df['risk_score'].apply(lambda x: np.nan if x < 100 else x)

# Flag users who have absolutely no credit history
df['is_thin_file'] = (df['total_account_count'] == 0).astype(int)


# %% MODEL TRAINING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

# 1. CRITICAL: Sort chronologically to prevent time leakage
df['request_datetime'] = pd.to_datetime(df['request_datetime'])
df = df.sort_values('request_datetime').reset_index(drop=True)

# 2. Define Features to Drop (Keep raw categoricals this time!)
# Note: DO NOT drop 'employer' or 'lead_user_agent', let CatBoost handle them
cols_to_drop = [
    'is_m0',
    'jluvr', 
    'label',                             # 1-to-1 proxy for is_m0
    'lead_user_agent',                   # We extracted device/browser info, so we can drop the raw text
    'applicant_current_address_state',   # Redundant with 'state'
    'applicant_employment_type',         # Redundant with 'employment_type'
    'bank_name',                         # Redundant with 'bank_account_bank_name'
    'applicant_date_of_birth',           # We have 'age'
    'request_datetime',                  # Dropped from X, but used for splitting
    'applicant_income_last_pay_day',     # Assuming you extracted durations
    'applicant_income_next_pay_day',
    'inquiry_date'                       # CRITICAL LEAK (Happens after request)
]

# Create X and y
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df['is_m0']

# 3. Out-of-Time Split (80% Past / 20% Future)
split_index = int(len(df) * 0.9)

X_train, X_test = X.iloc[:split_index].copy(), X.iloc[split_index:].copy()
y_train, y_test = y.iloc[:split_index].copy(), y.iloc[split_index:].copy()

print(f"Training Data range: {df['request_datetime'].iloc[0]} to {df['request_datetime'].iloc[split_index-1]}")
print(f"Testing Data range:  {df['request_datetime'].iloc[split_index]} to {df['request_datetime'].iloc[-1]}")

# 4. Bulletproof Categorical Handling
cat_cols = X_train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

for col in cat_cols:
    # Convert to string and fill NaNs to prevent CatBoost errors
    X_train[col] = X_train[col].fillna('unknown').astype(str).str.strip()
    X_test[col]  = X_test[col].fillna('unknown').astype(str).str.strip()

# 5. Handle Imbalance: Scale Pos Weight
positive_count = y_train.sum()
negative_count = len(y_train) - positive_count
scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

print(f"\nClass Imbalance -> Positives: {positive_count}, Negatives: {negative_count}")
print(f"Applying scale_pos_weight: {scale_pos_weight:.2f}")

# 6. Create Pools
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_cols)

# 7. Robust Hyperparameters
params = {
    'iterations': 1500,          # High iterations, but we will use early stopping
    'learning_rate': 0.03,       # Slow learning rate for stable convergence
    'depth': 6,                  # Depth 5-7 is ideal for imbalanced tabular data
    'l2_leaf_reg': 5,            # Higher regularization prevents overfitting to messy categoricals
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'scale_pos_weight': scale_pos_weight,
    'has_time': True,            # CRITICAL: Prevents target leakage from future rows
    'random_seed': 42,
    'verbose': 100               # Print every 100 trees
}

# 8. Train Model with Early Stopping
print("\nTraining Robust Time-Aware Model...")
model = CatBoostClassifier(**params)

model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=150,   # Stops if AUC on test set doesn't improve for 150 trees
    use_best_model=True          # Rolls back to the best tree configuration
)

# 9. Evaluate Best Model
best_iteration = model.get_best_iteration()
y_pred_proba = model.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n--- Final Model Performance ---")
print(f"Best Iteration: {best_iteration}")
print(f"Out-of-Time ROC-AUC Score: {final_auc:.4f}")
print("-------------------------------\n")

# %% SHAP
import shap
import matplotlib.pyplot as plt

# 1. Initialize the SHAP Tree Explainer with your trained model
# Note: For CatBoost, passing the model directly into TreeExplainer works perfectly.
explainer = shap.TreeExplainer(model)

# 2. Calculate SHAP values
# (If X_test is massive, you can use X_sample = X_test.sample(3000, random_state=42))
print("Calculating SHAP values... (this might take a moment)")
shap_values = explainer.shap_values(X_test)

# 3. Aggregate the numerical SHAP values (Mean Absolute Impact)
# This mathematically represents how much each feature alters the prediction on average
mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

# Print the top 25 most impactful features
print("\n--- Top 25 Features by Average SHAP Impact ---")
print(shap_df.head(25).to_string(index=False))
print("----------------------------------------------\n")

# 4. Generate the SHAP Summary Plot
plt.figure(figsize=(12, 8))

# plot_type="dot" ensures you get the red/blue color scale rather than just bars
shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)

# Add a title and format
plt.title("SHAP Feature Importance (Impact on predicting is_m0=1)", fontsize=14)
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create 10 equal-sized buckets based on price
# We use qcut to ensure each bucket has the same number of leads
df['price_decile'] = pd.qcut(df['seller_minimum_price'], q=10, duplicates='drop')

# 2. Calculate the default rate (mean of is_m0) and volume for each decile
price_analysis = df.groupby('price_decile').agg(
    default_rate=('is_m0', 'mean'),
    lead_volume=('is_m0', 'count')
).reset_index()

# 3. Plot the results
plt.figure(figsize=(12, 6))
sns.barplot(x='price_decile', y='default_rate', data=price_analysis, palette='coolwarm')
plt.title('Default Rate (is_m0) across Price Tiers', fontsize=14)
plt.xlabel('Seller Minimum Price Buckets')
plt.ylabel('Default Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print the table so you can see the exact math
print(price_analysis)

# %%
