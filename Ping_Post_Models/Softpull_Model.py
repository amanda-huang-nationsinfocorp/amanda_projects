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
avg(monthly_payment) as historic_monthly_payment_avg,
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


# In[12]:


with engine.connect() as con:
    con.execute(text("use database dbt_prod"))
    df = pd.read_sql(query, con)


# In[12]:

# 2. Create the Binary Target ('m0' = 1, everything else = 0)
df['is_m0'] = (df['label'].str.strip().str.lower() == 'm0').astype(int)
conversion_rate = df['is_m0'].mean() * 100
print(f"Total Conversion Rate: {conversion_rate:.2f}%\n")

# 3. Filter for numerical columns to do correlation math 
numeric_df = df.select_dtypes(include=[np.number])

# 4. Calculate the correlation matrix
print("Calculating correlations (this might take a moment on 145 MB)...")
corr_matrix = numeric_df.corr()

# 5. Extract correlations against your target
target_corr = corr_matrix['is_m0'].drop('is_m0').sort_values(key=abs, ascending=False)
print("--- Top 15 Features Correlated with Conversion (is_m0) ---")
print(target_corr.head(15))
print("\n")

# 6. Find Heavy Multicollinearity (Correlation > 0.80 between features)
print("--- Highly Correlated Feature Pairs (|corr| > 0.8) ---")
# Flatten the matrix and drop self-correlations (1.0)
corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort", ascending=False)
corr_pairs = corr_pairs[corr_pairs != 1.0] 

# Filter for correlations strictly above 0.80 or below -0.80
high_corr = corr_pairs[abs(corr_pairs) > 0.8].dropna()
seen = set()
for index, value in high_corr.items():
    pair = frozenset(index)
    if pair not in seen:
        seen.add(pair) 
        # Exclude the target variable from this check
        if 'is_m0' not in index:
            print(f"{index[0]} and {index[1]} : {value:.3f}")




# %% DATA TYPE CLEANING & NULL HANDLING

# 1. Convert specific object columns directly to numeric
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


# %% DATA TRANSFORMATION (VALUES)

df['is_m0'] = (df['label'].str.strip().str.lower() == 'm0').astype(int)

def get_bank_name(bank_account_bank_name: str) -> str:
    name = bank_account_bank_name.lower() if bank_account_bank_name else ""
    if "bancorp" in name:
        return "bancorp"
    elif "sutton" in name:
        return "sutton"
    elif "stride" in name:
        return "stride"
    elif "wells fargo" in name:
        return "wells_fargo"
    elif "chase" in name:
        return "chase"
    elif "bank of america" in name:
        return "bank_of_america"
    else:
        return "others"

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

df["bank_name"] = df["bank_account_bank_name"].apply(get_bank_name)
df["resident_status"] = df["applicant_current_address_residence_status"].apply(get_resident_status)
df["employment_type"] = df["applicant_employment_type"].apply(get_employment_type)
df["device_type"] = df["lead_user_agent"].apply(get_device_type)
df['applicant_current_address_state'] = df['applicant_current_address_state'].str[:2].str.upper()
df['state'] = df["applicant_current_address_state"].apply(get_states)

df['loan_to_age_ratio'] = df['loan_amount'] / df['age']
df['income_to_age_ratio'] = df['applicant_income_net_monthly_amount'] / df['age']

next_pay_date = pd.to_datetime(df['applicant_income_next_pay_day'], errors='coerce')
request_date = pd.to_datetime(df['request_datetime'], errors='coerce')

next_pay_date = next_pay_date.dt.tz_localize(None)
request_date = request_date.dt.tz_localize(None)

next_pay_date = next_pay_date.dt.normalize()
request_date = request_date.dt.normalize()

df['days_till_next_pay'] = (next_pay_date - request_date).dt.days

df['loan_to_income_ratio'] = df['loan_amount'] / df['applicant_income_net_monthly_amount']
df['lead_hour_of_day'] = df['request_datetime'].dt.hour

# 1. Ensure the timestamp is a proper datetime object
df['request_datetime'] = pd.to_datetime(df['request_datetime'])

# 2. CRITICAL: Sort by Vendor FIRST, then by Time. 
# This guarantees our dataframe perfectly matches the order Pandas uses when grouping.
df = df.sort_values(['seller_name', 'request_datetime']).reset_index(drop=True)

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

# 7. Sort back purely by time so it's ready for our train/test split
df = df.sort_values('request_datetime').reset_index(drop=True)


# %%
df.dtypes
