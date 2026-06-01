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
WITH pingpost AS (
    SELECT 
        jluvr,
        order_id,
        request_datetime,
        seller_name,
        -- Mapping seller_name to super_partner_id_name
        CASE 
            WHEN seller_name = 'ZP' THEN '233 Zero Parallel LLC'
            WHEN seller_name = 'PingYo' THEN '1011 PingYo'
            WHEN seller_name = 'Bume' THEN '504 Bume Intl'
            ELSE seller_name 
        END AS mapped_seller_name,
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

soft_pull AS (
    SELECT 
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
    FROM ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_SERVICE_PRODUCT_RESPONSE
),

trade_lines AS (
    SELECT 
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
    FROM ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_TRADE_LINE
    GROUP BY 1
),

inquiry AS (
    SELECT 
        jluvr,
        count(*) as inquiry_total,
        COUNT_IF(DATEDIFF('month', inquiry_date, CURRENT_DATE()) <= 6) AS inquiry_within_6_month,
        div0(count_if(industry_code = 'Bank'),count(*)) as bank_inquiry_ratio,
        div0(count_if(industry_code = 'Finance/Personal'),count(*)) as personal_inquiry_ratio
    FROM ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_INQUIRY
    GROUP BY 1
),

public_record AS (
    SELECT 
        jluvr, 
        count(*) as bankcrupcy_total,
        min(DATEDIFF('month', public_record_date, CURRENT_DATE()) / 12.0) as most_recent_record_year,
        div0(count_if(status = 'Discharged'), count(*)) as bankcrupty_discharged_ratio
    FROM  ANALYTICS.BLOODHOUND_PROJECT.SOFTPULL_PUBLIC_RECORD
    WHERE classification = 'Bankruptcy'
    GROUP BY 1
),

-- 1. Pre-aggregate the transaction data by day and partner
daily_partner_stats AS (
    SELECT 
        super_partner_id_name,
        TO_DATE(order_date) AS order_date,
        COUNT(*) AS daily_total_count,
        COUNT_IF(is_m0 = 1) AS m0_count,
        COUNT_IF(bin_tier = 'Tier1') AS bin_tier_1_count,
        COUNT_IF(bin_tier = 'Tier2') AS bin_tier_2_count,
        COUNT_IF(bin_tier = 'Tier3') AS bin_tier_3_count,
        COUNT_IF(bin_tier = 'Tier4') AS bin_tier_4_count,
        COUNT_IF(bin_tier = 'Blacklist') AS bin_tier_Blacklist_count
    FROM DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM AS orders
    LEFT JOIN DBT_PROD.ANALYTICS.stg_dbt_prod_analytics_bin_tier_matching AS bin_tiers
        ON orders.vertical_type_bin = bin_tiers.vertical_type_bin
    WHERE super_partner_id IN (504, 1011, 233)
    GROUP BY 1, 2
),

-- 2. Calculate the rolling 3-day metrics specific to each lead's request date
-- USING THE MAPPED SELLER NAME
partner_past_3d_stats AS (
    SELECT 
        p.jluvr,
        AVG(d.m0_count) AS m0_past_3_days,
        DIV0(SUM(d.bin_tier_1_count), SUM(d.daily_total_count)) AS bin_tier_1_perc_past_3_days,
        DIV0(SUM(d.bin_tier_2_count), SUM(d.daily_total_count)) AS bin_tier_2_perc_past_3_days,
        DIV0(SUM(d.bin_tier_3_count), SUM(d.daily_total_count)) AS bin_tier_3_perc_past_3_days,
        DIV0(SUM(d.bin_tier_4_count), SUM(d.daily_total_count)) AS bin_tier_4_perc_past_3_days,
        DIV0(SUM(d.bin_tier_Blacklist_count), SUM(d.daily_total_count)) AS bin_tier_Blacklist_perc_past_3_days
    FROM pingpost p
    LEFT JOIN daily_partner_stats d
        ON p.mapped_seller_name = d.super_partner_id_name 
        AND d.order_date >= TO_DATE(p.request_datetime) - 3 
        AND d.order_date < TO_DATE(p.request_datetime)
    GROUP BY p.jluvr
),

final AS (
    SELECT 
        p.jluvr,
        p.order_id,
        p.request_datetime,
        p.seller_name,
        p.label,
        p.seller_minimum_price,
        p.lead_user_agent,
        p.loan_amount,
        p.loan_purpose,
        p.finance_current_unsecured_debt_amount,
        p.applicant_employment_type,
        p.applicant_income_net_monthly_amount,
        p.applicant_employment_length,
        p.applicant_income_last_pay_day,
        p.applicant_income_next_pay_day,
        p.applicant_date_of_birth,
        p.applicant_current_address_state,
        p.applicant_current_address_residence_status, 
        p.applicant_current_address_years,
        p.bank_account_bank_name,
        p.bank_account_years,
        p.consent_marketing_email,
        sp.inquiry_date,
        sp.risk_score,
        sp.age,
        sp.employer,
        tl.total_account_count, 
        tl.total_open_account,
        tl.revolving_account_count,
        tl.installment_account_count,
        tl.collection_account_count,
        tl.total_high_balance,
        tl.total_current_balance,
        tl.credit_utilization_ratio,
        tl.earliest_account_open_year,
        tl.latest_account_open_year,
        tl.derogatory_account_count,
        tl.derogatory_account_ratio,
        tl.closed_derogatory_account_ratio,
        tl.paid_account_count,
        tl.transferred_account_count,
        tl.on_time_payment_current_ratio,
        tl.unsecured_loan_account_count,
        tl.secured_loan_account_count,
        tl.educational_account_count,
        tl.credit_card_account_count,
        tl.charge_account_count,
        tl.current_monthly_payment_total,
        tl.payment_burden_ratio,
        tl.historic_monthly_payment_avg,
        tl.month_reviewed_avg,
        tl.late_30_total,
        tl.late_60_total,
        tl.late_90_total,
        tl.past_due_amount_total,
        tl.past_due_to_current_balance_ratio,
        iq.inquiry_total,
        iq.inquiry_within_6_month,
        iq.bank_inquiry_ratio,
        iq.personal_inquiry_ratio,
        pr.bankcrupcy_total,
        pr.most_recent_record_year,
        pr.bankcrupty_discharged_ratio,
        p3d.m0_past_3_days,
        p3d.bin_tier_1_perc_past_3_days,
        p3d.bin_tier_2_perc_past_3_days,
        p3d.bin_tier_3_perc_past_3_days,
        p3d.bin_tier_4_perc_past_3_days,
        p3d.bin_tier_Blacklist_perc_past_3_days
    FROM pingpost p
    LEFT JOIN soft_pull sp USING(jluvr)
    LEFT JOIN trade_lines tl USING(jluvr)
    LEFT JOIN inquiry iq USING(jluvr)
    LEFT JOIN public_record pr USING(jluvr)
    LEFT JOIN partner_past_3d_stats p3d USING(jluvr)
    WHERE sp.risk_score IS NOT NULL
)

SELECT *
FROM final;
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
import pandas as pd
import numpy as np

# Ensure the column is a string and handle missing values
df['lead_user_agent'] = df['lead_user_agent'].fillna('unknown').astype(str)

# ==========================================
# 1. MACRO DEVICE TYPE (Mobile vs Desktop vs Tablet)
# ==========================================
conditions_device = [
    df['lead_user_agent'].str.contains('Mobile|iPhone', case=False, regex=True),
    df['lead_user_agent'].str.contains('Tablet|iPad|Android(?!.*Mobile)', case=False, regex=True), # Android without 'Mobile' is usually a tablet
    df['lead_user_agent'].str.contains('Windows NT|Macintosh|X11', case=False, regex=True)
]
choices_device = ['Mobile', 'Tablet', 'Desktop']
df['device_type'] = np.select(conditions_device, choices_device, default='Unknown_Device')

# ==========================================
# 2. OPERATING SYSTEM (OS)
# ==========================================
conditions_os = [
    df['lead_user_agent'].str.contains('iPhone|iPad|iPod', case=False),
    df['lead_user_agent'].str.contains('Android', case=False),
    df['lead_user_agent'].str.contains('Windows', case=False),
    df['lead_user_agent'].str.contains('Mac OS X|Macintosh', case=False) & ~df['lead_user_agent'].str.contains('iPhone|iPad', case=False)
]
choices_os = ['iOS', 'Android', 'Windows', 'Mac']
df['os_family'] = np.select(conditions_os, choices_os, default='Other/Unknown')

# ==========================================
# 3. BROWSER & IN-APP TRAFFIC SOURCE
# ==========================================
# First, check if it's an In-App browser (Social Media or Generic WebViews)
conditions_source = [
    df['lead_user_agent'].str.contains('FBAV|FBAN|FBIOS', case=False),
    df['lead_user_agent'].str.contains('Instagram', case=False),
    df['lead_user_agent'].str.contains('Snapchat', case=False),
    df['lead_user_agent'].str.contains('TikTok|Musical_ly', case=False),
    df['lead_user_agent'].str.contains(' wv\)', regex=True) # Android WebViews
]
choices_source = ['Facebook_App', 'Instagram_App', 'Snapchat_App', 'TikTok_App', 'Generic_InApp']

# If it's not In-App, figure out which standard browser it is
conditions_browser = [
    df['lead_user_agent'].str.contains('Edg', case=False),
    df['lead_user_agent'].str.contains('SamsungBrowser', case=False),
    df['lead_user_agent'].str.contains('Chrome|CriOS', case=False),
    df['lead_user_agent'].str.contains('Firefox|FxiOS', case=False),
    df['lead_user_agent'].str.contains('Safari', case=False) & ~df['lead_user_agent'].str.contains('Chrome|CriOS|Edg|Samsung', case=False)
]
choices_browser = ['Edge', 'Samsung_Internet', 'Chrome', 'Firefox', 'Safari']

# Combine them: If it's In-App, label it as such; otherwise, use the standard browser name
df['traffic_source'] = np.select(conditions_source, choices_source, default=np.select(conditions_browser, choices_browser, default='Unknown_Browser'))

# ==========================================
# 4. IPHONE WEALTH & AGE PROXIES
# ==========================================
# Extract the iOS version (e.g., "18_7" becomes 18.7)
df['ios_version_str'] = df['lead_user_agent'].str.extract(r'CPU iPhone OS (\d+(?:_\d+)?)')
df['ios_version'] = df['ios_version_str'].str.replace('_', '.').astype(float, errors='ignore')

# Use OS Version to proxy device age (Assuming iOS 17+ is modern, iOS 16 and below are physically old devices)
conditions_ios_age = [
    (df['os_family'] == 'iOS') & (df['ios_version'].astype(float, errors='ignore') < 17.0),
    (df['os_family'] == 'iOS') & (df['ios_version'].astype(float, errors='ignore') >= 17.0)
]
df['ios_device_age_proxy'] = np.select(conditions_ios_age, ['Old_Legacy_iOS', 'Modern_iOS'], default='Not_iOS')

# Extract exact Apple Hardware from Facebook App traffic (e.g., FBDV/iPhone11,8)
df['exact_apple_hardware'] = df['lead_user_agent'].str.extract(r'FBDV/(iPhone\d+,\d+)')

def categorize_iphone_hardware(hw_code):
    if pd.isna(hw_code):
        return 'Not_Detected'
    try:
        # Grabs the '11' from 'iPhone11,8'
        major_version = int(hw_code.replace('iPhone', '').split(',')[0])
        
        if major_version <= 12:     return 'Older_Budget_iPhone' # iPhone X/XR/11
        elif major_version in [13, 14]: return 'Mid_Tier_iPhone' # iPhone 12/13
        elif major_version >= 15:   return 'Premium_New_iPhone'  # iPhone 14/15/16+
    except:
        return 'Not_Detected'

df['iphone_wealth_tier'] = df['exact_apple_hardware'].apply(categorize_iphone_hardware)

# ==========================================
# 5. ULTIMATE COMBO FEATURES (For CatBoost)
# ==========================================
# E.g., "Mobile_Safari" or "Desktop_Chrome"
df['device_browser_combo'] = df['device_type'] + "_" + df['traffic_source']

# E.g., "iOS_Facebook_App" or "Android_Chrome"
df['os_source_combo'] = df['os_family'] + "_" + df['traffic_source']

# Drop the temporary extraction columns to keep the dataframe clean
df = df.drop(columns=['ios_version_str', 'ios_version', 'exact_apple_hardware'])

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
df['seller_device_combo'] = df['seller_name'].astype(str) + "_" + df['device_type'].astype(str)

################################ Seller Price Starts ###############################################

# 1. The Core Price Cliff Flags
# The plot shows a steep drop right after 1.05, and a stabilization around 0.80
df['price_is_sub_0_80'] = (df['seller_minimum_price'] <= 0.80).astype(int)
df['price_is_premium_over_1_05'] = (df['seller_minimum_price'] > 1.05).astype(int)

# 2. Price Category Mapping (Giving CatBoost clean buckets to target-encode)
def assign_price_bucket(price):
    if price <= 0.80:
        return 'budget_tier'
    elif price <= 1.05:
        return 'standard_tier'
    elif price <= 1.15:
        return 'high_risk_premium'
    else:
        return 'extreme_premium'

df['price_group_tier'] = df['seller_minimum_price'].apply(assign_price_bucket)

# 3. The SHAP-Revealed Interaction: Price × Inquiries
# A high price for a borrower with low inquiries is completely different 
# from a high price for a borrower with high inquiries (credit-seeking behavior).
df['price_per_total_inquiry'] = df['seller_minimum_price'] / (df['inquiry_total'] + 1)

# Cross-interaction: Flagging high-inquiry borrowers trapped in high-risk price tiers
df['high_risk_price_heavy_seeker'] = (
    (df['price_is_premium_over_1_05'] == 1) & (df['inquiry_total'] > 4)
).astype(int)

# 4. Value-to-Risk Disconnect (Price vs Credit Bureau Baseline)
df['price_to_risk_score_ratio'] = df['seller_minimum_price'] / (df['risk_score'] + 1)
############################## Seller Price Ends ######################################



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

######################################## Employer Feature Engineering ########################################
def engineer_employer_features(df):
    # 1. Clean base string to look for clean substrings
    df['emp_clean'] = df['employer'].astype(str).str.lower().str.strip()
    
    # 2. Consolidate High-Volume Variations (Standardization)
    df['is_employer_walmart'] = df['emp_clean'].str.contains('wal.*mart|walmart', case=False, na=False).astype(int)
    df['is_employer_usps'] = df['emp_clean'].str.contains('usps|postal service', case=False, na=False).astype(int)
    df['is_employer_fedex'] = df['emp_clean'].str.contains('fed.*ex|fedex', case=False, na=False).astype(int)
    df['is_employer_doordash'] = df['emp_clean'].str.contains('door.*dash|doordash', case=False, na=False).astype(int)
    
    # Comprehensive Self-Employed / 1099 Aggregate
    self_emp_keywords = 'self|1099|freelance|independent contractor|business owner'
    df['is_self_employed_agg'] = df['emp_clean'].str.contains(self_emp_keywords, case=False, na=False).astype(int)
    
    # 3. Macro Financial Risk Groupings
    # Government Benefits & Fixed Income
    benefit_keywords = 'retired|retirement|ssa|ssi|disability|socsec|social security|benefits|va benefits'
    df['is_fixed_income_benefits'] = df['emp_clean'].str.contains(benefit_keywords, case=False, na=False).astype(int)
    
    # Public Assistance / Aid Outliers
    df['is_public_assistance_afdc'] = df['emp_clean'].str.contains('afdc|child support', case=False, na=False).astype(int)
    
    # Unproductive / Non-Working Text Entries
    df['is_non_working_text'] = df['emp_clean'].str.contains('homemaker|home maker|housewife|house wife|student|unknown|none|noemployer', case=False, na=False).astype(int)
    
    # 4. Micro Risk-Tier Flags (Based on Your Top 150 Math)
    # Toxic Outliers Tier
    toxic_employers = 'cracker barrel|taco bell|burger king|dennys|holiday inn|dunkin|waffle house|popeyes|jack in the box'
    df['is_toxic_employer_tier'] = (
        df['emp_clean'].str.contains(toxic_employers, case=False, na=False) | 
        (df['emp_clean'] == 'irs')
    ).astype(int)
    
    # Elite Safe Harbor Tier
    elite_employers = 'publix|costco|macy|meijer|cvs pharmacy|general motors|bank of america|allied universal|securitas|goodwill'
    df['is_elite_low_risk_employer'] = df['emp_clean'].str.contains(elite_employers, case=False, na=False).astype(int)
    
    # Clean up the temp processing column
    df = df.drop(columns=['emp_clean'])
    return df

# Run it on your dataset
df = engineer_employer_features(df)
############################################# Employer Feature Engineering Ends ##################################


# %% MODEL TRAINING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# ==========================================
# 1. DATA PREPARATION & HOLDOUT SPLIT
# ==========================================

# 1. CRITICAL: Sort chronologically to prevent time leakage
df['request_datetime'] = pd.to_datetime(df['request_datetime'])
df = df.sort_values('request_datetime').reset_index(drop=True)

# --- CREATE OOT HOLDOUT SET ---
# Reserve the most recent 15% of data as the true blind holdout
split_index = int(len(df) * 0.85)

# df_cv will be used for TimeSeriesSplit
df_cv = df.iloc[:split_index].copy().reset_index(drop=True)
# df_holdout is locked away until the very end
df_holdout = df.iloc[split_index:].copy().reset_index(drop=True)

# 2. Define Features to Drop 
cols_to_drop = [
    'is_m0', 'jluvr', 'label', 'lead_user_agent',                   
    'applicant_current_address_state', 'applicant_employment_type',         
    'bank_name', 'applicant_date_of_birth', 'request_datetime',                  
    'applicant_income_last_pay_day', 'applicant_income_next_pay_day',
    'inquiry_date', 'price_decile', 'employer',                   
    'bank_account_bank_name', 'seller_minimum_price', 'seller_3d_conversion_rate'
]

# NOTE: If you have toxic features from earlier evaluations, uncomment the next two lines:
# toxic_features = feature_importances[feature_importances['Importance_Score'] <= 0]['Feature'].tolist()
# final_cols_to_drop = cols_to_drop + toxic_features
final_cols_to_drop = cols_to_drop

# Create X and y for the CV set
X_cv = df_cv.drop(columns=[c for c in final_cols_to_drop if c in df_cv.columns])
y_cv = df_cv['is_m0']

# Identify Categorical Columns
cat_cols = X_cv.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

# 3. Setup Time-Series Cross-Validation (Walk-Forward Validation)
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits) 


# ==========================================
# 2. WALK-FORWARD CROSS-VALIDATION
# ==========================================

train_aucs = []
test_aucs = []
best_iterations = []

print("Starting Walk-Forward Time-Series Validation...\n")

for fold, (train_index, test_index) in enumerate(tscv.split(X_cv)):
    print(f"--- FOLD {fold + 1} ---")
    
    # Split the data chronologically for this fold
    X_train, X_test = X_cv.iloc[train_index].copy(), X_cv.iloc[test_index].copy()
    y_train, y_test = y_cv.iloc[train_index].copy(), y_cv.iloc[test_index].copy()
    
    # 4. Bulletproof Categorical Handling
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).replace('nan', 'unknown').str.strip()
        X_test[col]  = X_test[col].astype(str).replace('nan', 'unknown').str.strip()
          
    # 5. Handle Imbalance for the current training fold
    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1
    
    # 6. Create Pools
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_cols)
    test_pool = Pool(data=X_test, label=y_test, cat_features=cat_cols)
    
    # 7. "Anti-Memorization" Hyperparameters (Original)
    params = {
        'iterations': 1500,          
        'learning_rate': 0.03,       
        'depth': 6,                  # Lower depth to prevent memorizing specific rows
        'l2_leaf_reg': 10,           # High regularization to penalize complexity
        'max_ctr_complexity': 5,     # Stop CatBoost from creating complex feature crosses
        'colsample_bylevel': 0.8,    # Randomly select 80% of features for splits
        'subsample': 0.8,            # Randomly select 80% of rows for each tree
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        #'scale_pos_weight': scale_pos_weight, # Uncomment to apply dynamically
        'has_time': True,            
        'random_seed': 42,
        'verbose': 0                 # Silenced tree outputs to keep the loop readable
    }
    
    # 8. Train Model
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=150,   
        use_best_model=True          
    )
    
    # 9. Evaluate Fold Performance
    best_iteration = model.get_best_iteration()
    best_iterations.append(best_iteration)
    
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    fold_train_auc = roc_auc_score(y_train, y_train_pred_proba)
    fold_test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    train_aucs.append(fold_train_auc)
    test_aucs.append(fold_test_auc)
     
    print(f"  Train Size: {len(X_train):,} | Positives: {y_train.sum():,} | Rate: {(y_train.sum()/len(y_train))*100:.2f}%")
    print(f"  Test Size:  {len(X_test):,} | Positives: {y_test.sum():,} | Rate: {(y_test.sum()/len(y_test))*100:.2f}%")
    print(f"  Scale Pos Weight applied: {scale_pos_weight:.2f}")
    print(f"Best Iteration: {best_iteration}")
    print(f"Train AUC:      {fold_train_auc:.4f}")
    print(f"Test AUC:       {fold_test_auc:.4f}")
    print(f"Gap:            {(fold_train_auc - fold_test_auc):.4f}\n")

# 10. Final CV Evaluation
print("=== FINAL CROSS-VALIDATION RESULTS ===")  
print(f"Average Train AUC: {np.mean(train_aucs):.4f} (+/- {np.std(train_aucs):.4f})")
print(f"Average Test AUC:  {np.mean(test_aucs):.4f} (+/- {np.std(test_aucs):.4f})")
print("======================================\n") 


# ==========================================
# 3. EVALUATE FINAL MODEL ON OOT HOLDOUT
# ==========================================
print("Training final model on all CV data and evaluating on blind Holdout...")

# Prep full CV dataset (trains on 85% of total data)
X_full_cv = X_cv.copy()
for col in cat_cols:
    X_full_cv[col] = X_full_cv[col].astype(str).replace('nan', 'unknown').str.strip()

full_cv_pool = Pool(data=X_full_cv, label=y_cv, cat_features=cat_cols)

# Train using original params, but limit 'iterations' to the average best iteration 
# from CV to prevent overfitting, as we have no early_stopping evaluation set here.
optimal_iterations = int(np.mean(best_iterations))
final_params = params.copy()
final_params['iterations'] = optimal_iterations

final_model = CatBoostClassifier(**final_params)
final_model.fit(full_cv_pool)

# Prepare the Holdout set (the remaining 15% of future data)
X_holdout = df_holdout.drop(columns=[c for c in final_cols_to_drop if c in df_holdout.columns])
y_holdout = df_holdout['is_m0']

for col in cat_cols:
    X_holdout[col] = X_holdout[col].astype(str).replace('nan', 'unknown').str.strip()

# Final Blind Evaluation
holdout_pred_proba = final_model.predict_proba(X_holdout)[:, 1]
holdout_auc = roc_auc_score(y_holdout, holdout_pred_proba)

print(f"\n=== TRUE BLIND HOLDOUT PERFORMANCE ===")
print(f"Holdout Size:      {len(X_holdout):,}")
print(f"Holdout Positives: {y_holdout.sum():,} | Rate: {(y_holdout.sum()/len(y_holdout))*100:.2f}%")
print(f"Holdout AUC:       {holdout_auc:.4f}")
print("========================================")

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
# --- RUN THIS AFTER YOUR CROSS-VALIDATION LOOP ---
print("\nEvaluating True Feature Importance on Out-of-Time Data...")

# Create a clean 80/20 time-split for this evaluation
split_idx = int(len(df) * 0.8)
X_tr, X_te = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
y_tr, y_te = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

# Bulletproof Categorical Handling
for col in cat_cols: 
    X_tr[col] = X_tr[col].astype(str).replace('nan', 'unknown').str.strip()
    X_te[col]  = X_te[col].astype(str).replace('nan', 'unknown').str.strip()

# Create Pools
eval_train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
eval_test_pool = Pool(X_te, y_te, cat_features=cat_cols)

# Train a dedicated model for feature evaluation
eval_model = CatBoostClassifier(**params)
eval_model.fit(eval_train_pool, eval_set=eval_test_pool, early_stopping_rounds=150, verbose=0)

# Calculate LossFunctionChange Importance (Evaluates on the TEST pool)
importance_values = eval_model.get_feature_importance(
    data=eval_test_pool,
    type='LossFunctionChange'
)

# Create a clean DataFrame
feature_importances = pd.DataFrame({
    'Feature': eval_model.feature_names_,
    'Importance_Score': importance_values
}).sort_values(by='Importance_Score', ascending=False)

print("\n--- TOP 10 REAL FEATURES (These improve Test AUC) ---")
print(feature_importances.head(10))

print("\n--- BOTTOM 15 TOXIC FEATURES (These cause Overfitting) ---")
print(feature_importances.tail(15))

feature_importances[feature_importances['Importance_Score'] <= 0]['Feature']

# %% Confustion Matrix
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# GENERATE CONFUSION MATRIX (CONVERSION FOCUS)
# ==========================================
print("\nGenerating Confusion Matrix for Final Test Fold...")

# 1. Define your business threshold
# If your goal is to buy/approve the leads MOST likely to convert, you set a high threshold.
# Let's say you only want to buy the top 30% highest-converting leads:
threshold = np.percentile(holdout_pred_proba,30)
print(f"Custom Threshold Applied: {threshold:.4f}")

# 2. Convert raw probabilities to binary predictions
# If probability >= threshold, we predict 1 (Will Convert / Approve Lead)
# If probability < threshold, we predict 0 (Will Not Convert / Reject Lead)
y_test_pred_binary = (holdout_pred_proba >= threshold).astype(int)

# 3. Calculate the confusion matrix
cm = confusion_matrix(y_holdout, y_test_pred_binary)

# 4. Extract metrics for business context
tn, fp, fn, tp = cm.ravel()
print(f"\n--- Conversion Business Breakdown ---")
print(f"True Positives  (TP): {tp:,} (Golden Leads: We bought them, and they converted!)")
print(f"False Positives (FP): {fp:,} (Wasted Money: We bought them, but they didn't convert.)")
print(f"True Negatives  (TN): {tn:,} (Money Saved: We rejected them, and they wouldn't have converted anyway.)")
print(f"False Negatives (FN): {fn:,} (Missed Opportunity: We rejected them, but they actually converted.)")

# 5. Visualize it beautifully using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Reject (Predict 0)', 'Buy/Approve (Predict 1)'],
            yticklabels=['Did Not Convert (Actual 0)', 'Converted (Actual 1)'])

plt.title('Conversion Confusion Matrix', fontsize=15, pad=15)
plt.ylabel('Actual Outcome', fontsize=12, fontweight='bold')
plt.xlabel('Model Action', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()