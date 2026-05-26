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


# In[9]:


query = '''with 
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
    lead_creation_url,
    campaign_id,
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


# %%
df = df.dropna(subset=['seller_minimum_price'])

# %%
pd.set_option('display.max_rows', None)
df.isnull().sum()



#%%

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


# %%
df.dtypes
# %%
