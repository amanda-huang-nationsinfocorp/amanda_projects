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
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                             average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_recall_curve)
try:
    import shap                     # optional: only the SHAP beeswarm needs it
except ImportError:
    shap = None
from catboost import CatBoostClassifier
import os 
import joblib 
from sklearn.frozen import FrozenEstimator
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())          # Snowflake credentials live in the repo-root .env file

#%% Fetch data
ctx = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_TOKEN"],      # <-- Pass the token exactly like a password
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    database="DBT_PROD",           # Standardized to just the database name
    warehouse='COMPUTE_WH',
    schema='EARLY_RETRY_CUTOFF'
)
cur = ctx.cursor()


query = '''
select 
order_id,
transaction_date,
transaction_id,
processor_name_original,
payment_type,
debit_number,
retries,
transaction_amount,
bin_first,
bank,
super_partner_id_name,
TO_VARCHAR(brand_id) as brand_id,
cascade_type,
sessions_count,
billing_state,
max(case when (chargeback_case_number is not null or nof_case_num is not null) 
    and daydiff_interval_nof_transaction not in ('Greater 360','150','120','180') then 1
    else 0
end) over (partition by order_id) as is_vamp 
from DBT_PROD.ANALYTICS.FCT_TRANSACTION_INVOICE_ORDER_ITEM as orders
where orders.transaction_date >= '2025-04-01'
and orders.transaction_date <'2026-06-01'
and orders.card_type = 'Visa' 
and orders.PROCESSOR_NAME_ORIGINAL like '%ADYEN%'
and orders.is_sale = 1
and vertical_type = 'credco'
'''

cur.execute("use database dbt_prod")
cur.execute(query)
df_raw = cur.fetch_pandas_all()
print(f"  {len(df_raw):,} rows pulled")

#%% Data Preprocessing
# =====================================================================
# Turns df_raw into `df` -- ONE ROW PER ORDER -- plus explicit feature-role
# lists, ready to split and hand to CatBoost. Every step here is label-free
# (nothing looks at IS_VAMP to make a decision), so the same block can be
# replayed at prediction time -- see PREPROCESS_SPEC at the end of the cell.
# =====================================================================
TARGET         = 'IS_VAMP'
GROUP_COL      = 'ORDER_ID'   # split unit: one order must never straddle two splits
DATE_COL       = 'TRANSACTION_DATE'  # drives maturity + the out-of-time split; never a feature
MIN_CAT_COUNT  = 50           # categories rarer than this get pooled into __rare__

df = df_raw.copy()
print(f"raw: {len(df):,} rows x {df.shape[1]} cols")

# =====================================================================
# 1. Column names
#    Belt and braces: normalise casing/whitespace and unwrap any
#    unaliased SQL expression Snowflake returned literally, e.g.
#    "STRING(BRAND_ID)" -> BRAND_ID.
# =====================================================================
df.columns = (df.columns.str.strip().str.upper()
                .str.replace(r'^\w+\((.*)\)$', r'\1', regex=True)
                .str.replace(r'[^0-9A-Z_]+', '_', regex=True)
                .str.strip('_'))
assert TARGET in df.columns, f"{TARGET} missing -- got {list(df.columns)}"

# =====================================================================
# 2. Target hygiene
# =====================================================================
n_before = len(df)
df = df[df[TARGET].notna()].copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors='raise').astype('int8')
assert set(df[TARGET].unique()) <= {0, 1}, f"non-binary target: {sorted(df[TARGET].unique())}"
print(f"  target: dropped {n_before - len(df):,} null-label rows | "
      f"{df[TARGET].sum():,} positives")

# =====================================================================
# 3. De-duplicate to one row per transaction
#    FCT_TRANSACTION_INVOICE_ORDER_ITEM is at order-ITEM grain, so a
#    single transaction repeats once per line item on the order. The
#    label is per transaction, so leaving those in silently weights
#    multi-item orders 2-5x and inflates every metric.
# =====================================================================
if 'TRANSACTION_ID' in df.columns:
    conflicts = (df.groupby('TRANSACTION_ID')[TARGET].nunique() > 1).sum()
    n_before = len(df)
    df = (df.sort_values(TARGET, ascending=False)          # keep the vamp row on conflict
            .drop_duplicates(subset='TRANSACTION_ID', keep='first')
            .reset_index(drop=True))
    print(f"  dedupe: dropped {n_before - len(df):,} repeated TRANSACTION_IDs "
          f"({conflicts:,} had conflicting labels) | {df[TARGET].sum():,} positives left")

# =====================================================================
# 4. Types
# =====================================================================
id_cols = [c for c in ['ORDER_ID', 'TRANSACTION_ID'] if c in df.columns]

# BIN / BRAND_ID are digit *identifiers*, not magnitudes -- keep them as
# strings so CatBoost treats them categorically and leading zeros survive.
force_string = [c for c in ['BIN_FIRST', 'BRAND_ID'] if c in df.columns]
for c in force_string:
    df[c] = (df[c].astype('string').str.strip()
                  .str.replace(r'\.0$', '', regex=True))   # 411111.0 -> '411111'

# Timestamps: needed for the maturity filter and the chronological split.
date_cols = [c for c in [DATE_COL] if c in df.columns]
for c in date_cols:
    parsed = pd.to_datetime(df[c], errors='coerce')
    bad = parsed.isna() & df[c].notna()
    n_before = len(df)
    df[c] = parsed
    df = df[df[c].notna()].copy()
    print(f"  {c}: parsed to datetime, dropped {n_before - len(df):,} unparseable/null "
          f"({bad.sum():,} unparseable) -> {df[c].min().date()} .. {df[c].max().date()}")

# Snowflake NUMBER arrives as Decimal/object, so dtypes cannot be trusted:
# offer every non-string/id/date column to the numeric parser and let the ones
# that fail fall through to categorical. Detection rather than a hardcoded list,
# so a measure added to the SQL later does not silently become a string.
maybe_numeric = [c for c in df.columns
                 if c not in force_string + id_cols + date_cols + [TARGET]]
numeric_cols = []
for c in maybe_numeric:
    conv = pd.to_numeric(df[c], errors='coerce')
    n_valid = int(df[c].notna().sum())
    frac_num = int(conv.notna().sum()) / max(n_valid, 1)
    if frac_num < 0.70:
        if frac_num > 0:            # silent for plainly textual columns
            print(f"  [note] {c}: only {frac_num:.0%} of values parse as numbers "
                  f"-> treating as categorical")
        continue
    lost = conv.isna() & df[c].notna()
    if lost.any():
        print(f"  {c}: {lost.sum():,} unparseable values -> NaN")
    df[c] = conv.astype('float64')
    numeric_cols.append(c)

# Everything else is a string feature: strip, upper (kills 'ca' vs 'CA'
# duplicate levels), and fold the usual string-shaped nulls into real NA.
str_cols = [c for c in df.columns
            if c not in numeric_cols + id_cols + date_cols + [TARGET]]
for c in str_cols:
    df[c] = (df[c].astype('string').str.strip().str.upper()
                  .replace({'': pd.NA, 'NAN': pd.NA, 'NONE': pd.NA,
                            'NULL': pd.NA, 'N/A': pd.NA, 'UNKNOWN': pd.NA}))

# Amount sanity. No winsorising: CatBoost splits on order, not magnitude,
# so clipping the tail would only throw information away.
if 'TRANSACTION_AMOUNT' in numeric_cols:
    bad_amt = (df['TRANSACTION_AMOUNT'] <= 0)
    if bad_amt.any():
        print(f"  [warn] {bad_amt.sum():,} rows with amount <= 0 on is_sale=1 -> NaN")
        df.loc[bad_amt, 'TRANSACTION_AMOUNT'] = np.nan

# =====================================================================
# 5. Collapse to one row per order
#    IS_VAMP is now assigned per order in the SQL, so every transaction of
#    an order repeats the same answer. Left at transaction grain a 12-attempt
#    order would count 12x in training and in every metric, purely for being
#    long. Aggregating puts the rows at the label's own grain: 1 order = 1
#    observation, and the split unit and the row unit finally agree.
#
#    Trade-off to be explicit about: every feature below now describes the
#    FINISHED order, so this model scores completed orders. It is not an
#    auth-time model -- that would need attempt-so-far features instead.
# =====================================================================
ORDER_END   = 'ORDER_LAST_TXN'                          # last attempt = last chance at a dispute
NUMERIC_AGG = {'TRANSACTION_AMOUNT': ['sum', 'max']}    # order value + biggest single attempt
DEFAULT_AGG = ['max']

mixed = int((df.groupby(GROUP_COL)[TARGET].nunique() > 1).sum())
if mixed:
    print(f"  [warn] {mixed:,} orders carry mixed labels -- this is not the order-level "
          f"SQL (max(...) over (partition by order_id)); taking max() as a fallback")

# Deterministic 'first transaction' so the categorical carry-over is reproducible.
sort_keys = [GROUP_COL, DATE_COL] + (['TRANSACTION_ID'] if 'TRANSACTION_ID' in df.columns else [])
df = df.sort_values(sort_keys)

agg_spec = {
    DATE_COL:         (DATE_COL, 'min'),      # order start -> drives the time split
    ORDER_END:        (DATE_COL, 'max'),      # order end   -> drives label maturity
    'N_TRANSACTIONS': (GROUP_COL, 'size'),    # attempt count is itself a strong signal
    TARGET:           (TARGET, 'max'),
}
agg_numeric = []
for c in numeric_cols:
    for how in NUMERIC_AGG.get(c, DEFAULT_AGG):
        agg_spec[f'{c}_{how.upper()}'] = (c, how)
        agg_numeric.append(f'{c}_{how.upper()}')
for c in str_cols:                            # first non-null value, in attempt order
    agg_spec[c] = (c, 'first')

n_txn = len(df)
df = df.groupby(GROUP_COL, sort=False).agg(**agg_spec).reset_index()
numeric_cols = agg_numeric + ['N_TRANSACTIONS']
date_cols    = [DATE_COL, ORDER_END]
id_cols      = [GROUP_COL]

print(f"  {n_txn:,} transactions -> {len(df):,} orders | "
      f"{int(df[TARGET].sum()):,} vamp ({df[TARGET].mean():.4%}) | "
      f"median {df['N_TRANSACTIONS'].median():.0f} attempts/order, "
      f"max {int(df['N_TRANSACTIONS'].max())}")
assert df[GROUP_COL].is_unique, "aggregation did not produce one row per order"

# =====================================================================
# 6. Feature roles
# =====================================================================
drop_cols = id_cols + date_cols + [TARGET]
feature_cols = [c for c in df.columns if c not in drop_cols]

# Drop columns the WHERE clause already pinned to one value (e.g.
# PROCESSOR_NAME_ORIGINAL) -- they cost tree time and teach nothing.
const_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
if const_cols:
    print(f"  dropping constant columns: {const_cols}")
    drop_cols += const_cols
    feature_cols = [c for c in feature_cols if c not in const_cols]

# Free-text-ish names: bank / partner strings share tokens ('CHASE BANK USA NA'),
# which CatBoost exploits better as text than as opaque categories.
text_features = [c for c in ['SUPER_PARTNER_ID_NAME', 'BANK'] if c in feature_cols]
cat_features  = [c for c in feature_cols if c not in numeric_cols + text_features]

# =====================================================================
# 7. Pool rare categories
#    Counts only -- no label involved -- so doing this before the split
#    is not leakage. Keeps BIN_FIRST/BRAND_ID from turning into a
#    per-row identifier the model can memorise.
# =====================================================================
rare_maps = {}
for c in cat_features:
    vc = df[c].value_counts(dropna=True)
    keep = set(vc[vc >= MIN_CAT_COUNT].index)
    if len(keep) < len(vc):
        rare_maps[c] = keep
        n_rare = int((~df[c].isin(keep) & df[c].notna()).sum())
        df[c] = df[c].where(df[c].isin(keep) | df[c].isna(), '__RARE__')
        print(f"  {c}: {len(vc):,} levels -> {len(keep):,} kept "
              f"(+__RARE__ covering {n_rare:,} rows)")

# A column whose every level was rare has collapsed to one value: it now
# carries no signal, so drop it instead of feeding CatBoost a constant.
collapsed = [c for c in cat_features if df[c].nunique(dropna=False) <= 1]
if collapsed:
    print(f"  [warn] fully collapsed by pooling -> dropping {collapsed} "
          f"(lower MIN_CAT_COUNT to keep them)")
    drop_cols     += collapsed
    feature_cols   = [c for c in feature_cols if c not in collapsed]
    cat_features   = [c for c in cat_features if c not in collapsed]
    rare_maps      = {k: v for k, v in rare_maps.items() if k not in collapsed}

# =====================================================================
# 8. Impute and lock dtypes
#    -1 for numerics matches the retry-cutoff model; none of these
#    measures can legitimately be negative, so the sentinel is unambiguous.
# =====================================================================
df[cat_features + text_features] = (
    df[cat_features + text_features].fillna('unknown').astype(str)
)
df[numeric_cols] = df[numeric_cols].fillna(-1).astype('float64')

assert not df[feature_cols].isna().any().any(), "NaNs left in features"
assert df[GROUP_COL].notna().all(), f"null {GROUP_COL} -- cannot group-split safely"
assert DATE_COL in df.columns and df[DATE_COL].notna().all(), \
    f"{DATE_COL} missing/null -- no out-of-time split is possible without it"
assert DATE_COL not in feature_cols and GROUP_COL not in feature_cols

# =====================================================================
# 9. Summary
# =====================================================================
pos, n = int(df[TARGET].sum()), len(df)
print("\n--- clean dataset ---")
print(f"  orders         : {n:,}")
print(f"  transactions   : {int(df['N_TRANSACTIONS'].sum()):,} behind them")
print(f"  features       : {len(feature_cols)} "
      f"({len(numeric_cols)} numeric / {len(cat_features)} categorical / {len(text_features)} text)")
print(f"  base rate      : {pos:,} / {n:,} = {pos / n:.4%}")
print(f"  scale_pos_weight: {(n - pos) / max(pos, 1):.1f}")
print(f"  numeric        : {numeric_cols}")
print(f"  categorical    : {cat_features}")
print(f"  text           : {text_features}")
if pos / n < 0.01:
    print("  [note] heavy imbalance -- stratify the splits, judge on PR-AUC/Brier,")
    print("         and calibrate before using the scores as probabilities.")

# Replay this exact recipe in the prediction script.
PREPROCESS_SPEC = {
    'target': TARGET,
    'group_col': GROUP_COL,
    'date_col': DATE_COL,
    'order_end_col': ORDER_END,
    'numeric_agg': NUMERIC_AGG,
    'default_agg': DEFAULT_AGG,
    'drop_cols': drop_cols,
    'feature_cols': feature_cols,
    'numeric_cols': numeric_cols,
    'cat_features': cat_features,
    'text_features': text_features,
    'rare_maps': {k: sorted(v) for k, v in rare_maps.items()},
    'min_cat_count': MIN_CAT_COUNT,
}

#%% Model Training
# =====================================================================
# Chronological + order-grouped evaluation of a CatBoost VAMP classifier.
#
#   mature orders ─┬─► past (oldest 85%) ─┬─► train        75%
#                  │                      ├─► stop         10%  (early stopping)
#                  │                      ├─► calib        10%  (isotonic)
#                  │                      └─► random_test   5%  (unseen orders, same period)
#                  └─► OOT  (newest 15%)                        (unseen orders, later period)
#
# One row per order throughout, so the split unit and the row unit are the same
# thing: an order cannot straddle a boundary, and a heavily-retried order is not
# weighted up for it.
#
# Two test sets on purpose: random_test answers "does it generalise to new
# orders?", OOT answers "would it still have worked next month?". The gap
# between them is drift, and it is the number that decides retraining cadence.
# =====================================================================
# Re-imported here rather than relying on the IMPORTS cell: this cell gets re-run
# on its own after an edit, and a kernel started before an import was added would
# otherwise fail halfway through training with a bare NameError.
import os, hashlib, joblib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                             average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, precision_recall_curve)
try:
    import shap
except ImportError:
    shap = None

MATURITY_DAYS = 90    # must match the daydiff_interval_nof_transaction filter in the SQL:
                      # a dispute only counts as vamp inside that window, so a transaction
                      # younger than this has a label that is not final yet.
ORDER_RUNWAY_DAYS = 30  # how long an order can still collect retries; orders whose last
                        # attempt falls within this of the SQL window end are truncated
OOT_FRACTION  = 0.15
SALT          = 'vamp_v1'
LEAK_AUC      = 0.95  # a single feature scoring above this already knows the answer

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:                         # running as an interactive cell
    SCRIPT_DIR = os.getcwd()

# =====================================================================
# 1. Label maturity
#    Labels were observed when the data was pulled, so anything inside the
#    dispute window at pull time looks clean only because it is too young.
#    Training on those rows teaches the model that recent == safe.
#    Keyed on the order's LAST attempt: that attempt is the last one that
#    could still turn into a dispute, so it is what has to be mature.
# =====================================================================
ASOF_DATE       = pd.Timestamp.today().normalize()
MATURITY_CUTOFF = ASOF_DATE - pd.Timedelta(days=MATURITY_DAYS)

# Second, subtler cutoff, and one that only exists at order grain: the SQL window
# has a hard end, so an order still being retried when the window closed had its
# later attempts cut off. Its N_TRANSACTIONS and its max/sum aggregates are
# understated, which makes recent orders look small and safe. Require a retry
# runway between the order's last seen attempt and the end of the data.
DATA_END        = df[ORDER_END].max()
TRUNCATION_CUT  = DATA_END - pd.Timedelta(days=ORDER_RUNWAY_DAYS)

too_young     = df[ORDER_END] > MATURITY_CUTOFF     # label not final yet
truncated     = df[ORDER_END] > TRUNCATION_CUT      # order itself not finished
immature      = too_young | truncated
print(f"as-of {ASOF_DATE.date()} | data ends {DATA_END.date()}")
print(f"  label maturity cutoff  {MATURITY_CUTOFF.date()} -> {int(too_young.sum()):,} orders too young")
print(f"  order runway cutoff    {TRUNCATION_CUT.date()} -> {int(truncated.sum()):,} orders still open")
print(f"  dropping {int(immature.sum()):,} orders in total "
      f"({int(df.loc[immature, TARGET].sum()):,} of them labelled vamp)")
mature_df = df.loc[~immature].copy()
assert len(mature_df), "no order is both finished and label-mature -- widen the SQL date range"

# =====================================================================
# 2. Chronological split, cut on ORDER first-seen date
#    Cutting on the order (not the row) keeps every transaction of an order
#    on one side of the boundary.
# =====================================================================
# One row per order already, so the order's start date is simply its own.
order_start = mature_df.set_index(GROUP_COL)[DATE_COL]
split_date  = order_start.quantile(1 - OOT_FRACTION)

past_orders = order_start[order_start < split_date].index
oot_orders  = order_start[order_start >= split_date].index

past_data   = mature_df[mature_df[GROUP_COL].isin(past_orders)].copy()
oot_test_df = mature_df[mature_df[GROUP_COL].isin(oot_orders)].copy()

assert not (set(past_data[GROUP_COL]) & set(oot_test_df[GROUP_COL])), \
    "orders straddle the chronological boundary"
print(f"  split date {pd.Timestamp(split_date).date()} | "
      f"past {len(past_data):,} orders / OOT {len(oot_test_df):,} orders")

# =====================================================================
# 3. Order-level hash split of the past window
#    hashlib rather than hash(): Python salts str hashing per process, so
#    hash() would reshuffle orders between splits on every run. This is
#    stable across runs and stays stable as new orders arrive.
# =====================================================================
def order_hash_frac(order_ids, salt=SALT):
    return np.array([
        int(hashlib.md5(f'{salt}:{o}'.encode()).hexdigest()[:16], 16) / 2**64
        for o in order_ids
    ])

unique_orders = past_data[GROUP_COL].unique()
order_frac    = pd.Series(order_hash_frac(unique_orders), index=unique_orders)
row_frac      = past_data[GROUP_COL].map(order_frac)

train_df       = past_data[row_frac < 0.75].copy()
stop_df        = past_data[(row_frac >= 0.75) & (row_frac < 0.85)].copy()
calib_df       = past_data[(row_frac >= 0.85) & (row_frac < 0.95)].copy()
random_test_df = past_data[row_frac >= 0.95].copy()

splits = [('train', train_df), ('stop', stop_df), ('calib', calib_df),
          ('random_test', random_test_df), ('oot_test', oot_test_df)]
for name, part in splits:
    pos = int(part[TARGET].sum())
    print(f"  {name:<12}: {len(part):>8,} orders | "
          f"{int(part['N_TRANSACTIONS'].sum()):>9,} txns | "
          f"{pos:>6,} vamp ({part[TARGET].mean():.4%}) "
          f"| {part[DATE_COL].min().date()} .. {part[DATE_COL].max().date()}")
    if pos < 20:
        print(f"  [warn] {name} has only {pos} positives -- its metrics will be very noisy")

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_stop,  y_stop  = stop_df[feature_cols],  stop_df[TARGET]
X_calib, y_calib = calib_df[feature_cols], calib_df[TARGET]
X_random, y_random = random_test_df[feature_cols], random_test_df[TARGET]
X_oot,   y_oot   = oot_test_df[feature_cols],  oot_test_df[TARGET]

# =====================================================================
# 4. Leakage audit -- runs BEFORE training, so a red flag costs seconds
# =====================================================================
print("\n=== leakage audit ===")

# 4a. Identifiers, the split key and the label must not be inside X.
banned = set(id_cols) | set(date_cols) | {TARGET}
assert not (banned & set(feature_cols)), f"leaky columns in X: {banned & set(feature_cols)}"
print(f"  [ok] no ids/dates/target in X ({len(feature_cols)} features)")

# 4b. No order may appear in two splits (row duplication would let the model
#     memorise a sibling row instead of generalising).
for i, (na, a) in enumerate(splits):
    for nb, b in splits[i + 1:]:
        shared = set(a[GROUP_COL]) & set(b[GROUP_COL])
        assert not shared, f"{na}/{nb} share {len(shared):,} orders"
print("  [ok] no ORDER_ID spans two splits")

# 4c. Feature resolution, NOT a duplication check. The ORDER_ID assert above
#     already makes it impossible for one order to sit on both sides, so
#     identical vectors here are *different* orders that happen to look alike --
#     inevitable once traffic concentrates on a few BINs, banks and price points
#     and single-attempt orders pin the count features to constants. The overlap
#     % is therefore a property of the feature space, not evidence of leakage.
#     What does matter: vectors that appear with BOTH labels are orders no model
#     can separate, and they set a hard floor under the error rate.
sig_train = pd.util.hash_pandas_object(X_train, index=False)
distinct  = int(sig_train.nunique())
print(f"  resolution: {distinct:,} distinct vectors for {len(X_train):,} train orders "
      f"({len(X_train) / max(distinct, 1):.1f} orders per vector)")

g   = y_train.groupby(sig_train.values)
n1  = g.sum()
n   = g.size()
ambiguous = float(n[(n1 > 0) & (n1 < n)].sum()) / len(y_train)
floor     = float(np.minimum(n1, n - n1).sum()) / len(y_train)
print(f"  {ambiguous:.2%} of train orders share a vector with an opposite-label order "
      f"-> best achievable error at this resolution is {floor:.2%}")

for name, X_ in [('random_test', X_random), ('oot_test', X_oot)]:
    dup = pd.util.hash_pandas_object(X_, index=False).isin(set(sig_train)).mean()
    print(f"  {dup:>6.2%} of {name} orders match a train vector exactly "
          f"(expected at this resolution; the orders themselves are disjoint)")

# Features whose top value swallows the column are the ones costing resolution.
flat = {c: X_train[c].value_counts(normalize=True).iloc[0] for c in feature_cols}
flat = {c: v for c, v in sorted(flat.items(), key=lambda kv: -kv[1]) if v > 0.90}
if flat:
    print("  near-constant features (top value share) -- add detail here to separate orders:")
    for c, v in list(flat.items())[:6]:
        print(f"    {c:<26} {v:.1%}")

# 4d. Univariate scan: fit nothing on the eval side, then see if one column
#     alone separates the label. Numerics are scored directly; categoricals
#     get a target mean fitted on train ONLY and applied to stop.
print("  univariate AUC (a value near 1.0 means that column encodes the outcome):")
uni = {}
for c in numeric_cols:
    a = roc_auc_score(y_train, X_train[c])
    uni[c] = max(a, 1 - a)
prior = y_train.mean()
if y_stop.nunique() < 2:
    print("    [skip] the stop split has no positives -- categoricals not scanned")
else:
    for c in cat_features + text_features:
        enc = y_train.groupby(X_train[c].values).mean()
        a = roc_auc_score(y_stop, X_stop[c].map(enc).fillna(prior))
        uni[c] = max(a, 1 - a)
for c, a in sorted(uni.items(), key=lambda kv: -kv[1])[:8]:
    print(f"    {c:<26} {a:.4f}" + ("   <-- SUSPECT" if a > LEAK_AUC else ""))
leaky = [c for c, a in uni.items() if a > LEAK_AUC]
if leaky:
    print(f"  [ALERT] {leaky} single-handedly predict the label. Confirm those values "
          f"exist at scoring time before trusting any metric below.")

# 4f. Honest limit of cutting on the order's START date: an order that began
#     before the boundary but was still being retried after it contributes
#     post-boundary information to train. Small % = fine; large % = orders are
#     long-lived, so widen the gap between the train and OOT windows.
late = float((train_df[ORDER_END] >= split_date).mean())
flag = "[warn]" if late > 0.05 else "[ok]  "
print(f"  {flag} {late:.2%} of train orders were still running after the OOT split date")

# 4e. Two things the code cannot settle on its own.
print("  [review] every feature is an aggregate over the FINISHED order (max retries,")
print("           total amount, attempt count), so these metrics describe scoring a")
print("           completed order. Scoring live mid-order would need attempt-so-far")
print("           features -- do not read the numbers below as auth-time performance.")
print("  [note]   rare-category pooling used whole-sample counts (no labels), so it is")
print("           label-free; only category frequencies crossed the time boundary.")

# =====================================================================
# 5. Base model
#    Trained unweighted: CatBoost ranks rare events fine, and class weights
#    would distort the probabilities that step 6 calibrates. Use
#    scale_pos_weight only if you need raw 0.5-threshold recall.
# =====================================================================
print("\nTraining base model...")
train_pool = Pool(X_train, y_train, cat_features=cat_features, text_features=text_features)
stop_pool  = Pool(X_stop,  y_stop,  cat_features=cat_features, text_features=text_features)

base_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=6,
    task_type="CPU",
    eval_metric='AUC',           # ranking quality, not accuracy: positives are rare
    custom_metric=['Logloss', 'PRAUC'],
    early_stopping_rounds=100,
    thread_count=-1,
    random_state=42,
    verbose=100,
    cat_features=cat_features,
    text_features=text_features,
)
base_model.fit(train_pool, eval_set=stop_pool, use_best_model=True)
print(f"  stopped at iteration {base_model.get_best_iteration()} of {base_model.tree_count_}")

# =====================================================================
# 6. Calibration -- isotonic on a split the trees never saw
# =====================================================================
print("Calibrating probabilities on the calib split...")
calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method='isotonic',
)
calibrated_model.fit(X_calib, y_calib)

# =====================================================================
# 7. AUC and friends
# =====================================================================
def evaluate(name, model, X, y):
    p = model.predict_proba(X)[:, 1]
    m = {
        'ROC-AUC': roc_auc_score(y, p),
        'PR-AUC': average_precision_score(y, p),
        'LogLoss': log_loss(y, p, labels=[0, 1]),
        'Brier': brier_score_loss(y, p),
        'base_rate': y.mean(),
        'obs/exp': p.mean() / max(y.mean(), 1e-12),
    }
    print(f"  {name:<12} | ROC-AUC {m['ROC-AUC']:.4f} | PR-AUC {m['PR-AUC']:.4f} "
          f"| LogLoss {m['LogLoss']:.4f} | Brier {m['Brier']:.5f} "
          f"| base {m['base_rate']:.4%} | pred/actual {m['obs/exp']:.2f}x")
    return p, m

print("\n=== performance (calibrated) ===")
p_calib,  _      = evaluate('calib',       calibrated_model, X_calib,  y_calib)
p_random, m_rand = evaluate('random_test', calibrated_model, X_random, y_random)
p_oot,    m_oot  = evaluate('oot_test',    calibrated_model, X_oot,    y_oot)

drift = m_rand['ROC-AUC'] - m_oot['ROC-AUC']
print(f"\n  temporal degradation (random - OOT AUC): {drift:+.4f}")
if abs(drift) < 0.03:
    print("  Both splits are order-grouped and label-mature, so this gap is genuine")
    print("  drift rather than duplication -- and at this size, it is mild.")
else:
    print("  Both splits are order-grouped and label-mature, so the gap is genuine drift:")
    print("  the OOT window behaves differently from the training period. Retrain often,")
    print("  and treat the random_test numbers as the optimistic ones.")
rate_shift = m_oot['base_rate'] / max(y_train.mean(), 1e-12)
if not 0.7 <= rate_shift <= 1.4:
    print(f"  [warn] OOT vamp rate is {rate_shift:.2f}x the training rate "
          f"({y_train.mean():.4%} -> {m_oot['base_rate']:.4%}).")
    print("         Ranking may still hold, but the calibration and the threshold below")
    print("         are stale -- recalibrate on recent data before using them live.")
if m_rand['ROC-AUC'] > 0.99 or m_oot['ROC-AUC'] > 0.99:
    print("  [ALERT] AUC this high on a chargeback problem is almost always leakage.")
    print("          Re-read section 4d before believing it.")

# =====================================================================
# 8. Confusion matrices
#    Threshold is picked on calib (max F1) -- never on a test set, that is
#    how a threshold quietly overfits. 0.5 is meaningless at this base rate.
# =====================================================================
pr_c, rc_c, thr_c = precision_recall_curve(y_calib, p_calib)
f1_c = 2 * pr_c * rc_c / np.clip(pr_c + rc_c, 1e-12, None)
best = int(np.argmax(f1_c[:-1]))          # last point of the curve has no threshold
THRESHOLD, best_f1 = float(thr_c[best]), float(f1_c[best])
print(f"\n=== confusion matrices @ threshold {THRESHOLD:.5f} "
      f"(max-F1 on calib, F1={best_f1:.4f}) ===")

def show_cm(name, y, p, t=None):
    t = THRESHOLD if t is None else t
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, yhat, average='binary', zero_division=0)
    print(f"\n  {name}  (n={len(y):,}, flagged {yhat.sum():,} = {yhat.mean():.2%})")
    print(f"                 pred clean   pred vamp")
    print(f"    true clean  {tn:>11,} {fp:>11,}")
    print(f"    true vamp   {fn:>11,} {tp:>11,}")
    print(f"    precision {pr:.4f} | recall {rc:.4f} | F1 {f1:.4f} "
          f"| lift {(pr / max(y.mean(), 1e-12)):.1f}x base rate")

show_cm('random_test', y_random, p_random)
show_cm('oot_test',    y_oot,    p_oot)

# Operating points you would actually run: review the riskiest N% of volume.
print("\n  OOT capture at fixed review rates (how VAMP monitoring is really used):")
print(f"    {'review':<8} {'threshold':>10} {'precision':>10} {'recall':>8} {'lift':>7}")
seen_n = set()
for rate in [0.001, 0.005, 0.01, 0.05, 0.10]:
    t = np.quantile(p_oot, 1 - rate)
    yhat = (p_oot >= t).astype(int)
    # isotonic calibration creates probability plateaus, so two review rates can
    # land on the identical set of rows -- report each distinct set once.
    # A rate that flags a handful of orders yields precision built on 2-3 events;
    # that is noise, not an operating point.
    if yhat.sum() < 20 or yhat.sum() in seen_n:
        continue
    seen_n.add(int(yhat.sum()))
    pr, rc, _, _ = precision_recall_fscore_support(
        y_oot, yhat, average='binary', zero_division=0)
    print(f"    top {rate:>5.1%} {t:>10.5f} {pr:>10.4f} {rc:>8.4f} "
          f"{pr / max(y_oot.mean(), 1e-12):>6.1f}x")

# =====================================================================
# 9. SHAP
#    Computed on the OOT window (the distribution you actually deploy into)
#    and on the BASE model -- the isotonic wrapper is a monotone rescaling,
#    so it reorders nothing SHAP would show.
# =====================================================================
print("\n=== SHAP ===")
SHAP_N = min(20_000, len(oot_test_df))
shap_df   = oot_test_df.sample(SHAP_N, random_state=42)
shap_pool = Pool(shap_df[feature_cols], shap_df[TARGET],
                 cat_features=cat_features, text_features=text_features)

sv       = np.asarray(base_model.get_feature_importance(shap_pool, type='ShapValues'))
contrib  = sv[:, :-1]                     # one column per feature, in feature_cols order
expected = sv[0, -1]                      # log-odds of the average prediction

# Proof the attribution is exact: contributions + base value must reproduce
# the raw model output. If this fails, ignore the ranking below.
raw = base_model.predict(shap_df[feature_cols], prediction_type='RawFormulaVal')
err = np.abs(contrib.sum(axis=1) + expected - raw).max()
print(f"  reconstruction error {err:.2e} ({'exact' if err < 1e-6 else 'CHECK THIS'}) "
      f"| base log-odds {expected:.4f} | n={SHAP_N:,}")

mean_abs = pd.Series(np.abs(contrib).mean(axis=0), index=feature_cols).sort_values(ascending=False)
share    = mean_abs / mean_abs.sum()

# Direction: a strong feature has mean SHAP ~ 0 (contributions cancel out), so the
# signed mean is useless. Rank-correlating the value with its own contribution tells
# you which way the feature actually pushes -- only meaningful where values are ordered.
def shap_direction(c):
    if c not in numeric_cols:
        return 'per-level (see below)' if c in cat_features else 'per-token'
    v = shap_df[c].to_numpy(dtype=float)
    sh = contrib[:, feature_cols.index(c)]
    if np.std(v) == 0 or np.std(sh) == 0:
        return 'flat'
    r = np.corrcoef(pd.Series(v).rank(), pd.Series(sh).rank())[0, 1]
    if abs(r) < 0.2:
        return f'non-monotone (r={r:+.2f})'
    return f"higher {'raises' if r > 0 else 'lowers'} risk (r={r:+.2f})"

print(f"\n  {'feature':<26} {'mean|SHAP|':>11} {'share':>7}  direction")
for c in mean_abs.index[:15]:
    print(f"  {c:<26} {mean_abs[c]:>11.5f} {share[c]:>6.1%}  {shap_direction(c)}")

if share.iloc[0] > 0.50:
    print(f"\n  [ALERT] {mean_abs.index[0]} carries {share.iloc[0]:.0%} of all attribution.")
    print("          One feature dominating this hard is the classic leakage signature.")

# Which levels of the top categoricals drive risk -- the actionable part.
top_cats = [c for c in mean_abs.index if c in cat_features][:3]
for c in top_cats:
    s = pd.DataFrame({'level': shap_df[c].values, 'shap': contrib[:, feature_cols.index(c)]})
    agg = s.groupby('level')['shap'].agg(['mean', 'size'])
    agg = agg[agg['size'] >= 30].sort_values('mean', ascending=False)
    if agg.empty:
        continue
    show = agg if len(agg) <= 7 else pd.concat([agg.head(4), agg.tail(3)])
    print(f"\n  {c}: {'all levels' if len(agg) <= 7 else 'riskiest / safest levels'} (n>=30)")
    for lvl, r in show.iterrows():
        print(f"    {str(lvl)[:28]:<30} mean SHAP {r['mean']:>+9.5f}  (n={int(r['size']):,})")

# Bar chart: always works, cats and text included.
fig, ax = plt.subplots(figsize=(8, 0.38 * min(20, len(mean_abs)) + 1.2))
top = mean_abs.head(20)[::-1]
ax.barh(top.index, top.values, color='#2b6cb0')
ax.set_xlabel('mean |SHAP| (log-odds)')
ax.set_title(f'VAMP drivers -- OOT window, n={SHAP_N:,}')
fig.tight_layout()
bar_path = os.path.join(SCRIPT_DIR, 'vamp_shap_importance.png')
fig.savefig(bar_path, dpi=150)
plt.show()
print(f"\n  saved {bar_path}")

# Beeswarm shows direction per row, but only numerics have an orderable
# colour scale -- shap cannot rank a bank name, so cats are excluded here.
if numeric_cols and shap is not None:
    try:
        idx = [feature_cols.index(c) for c in numeric_cols]
        shap.summary_plot(contrib[:, idx], shap_df[numeric_cols].astype(float),
                          feature_names=numeric_cols, show=False)
        bee_path = os.path.join(SCRIPT_DIR, 'vamp_shap_beeswarm_numeric.png')
        plt.tight_layout()
        plt.savefig(bee_path, dpi=150)
        plt.show()
        print(f"  saved {bee_path}")
    except Exception as e:
        print(f"  [skip] beeswarm failed: {type(e).__name__}: {e}")

#%% Save Model
model_path = os.path.join(SCRIPT_DIR, 'vamp_catboost_model.joblib')

# Ship the preprocessing recipe with the model: the prediction script must
# rebuild the exact same columns, in the same order, with the same rare-level
# pooling, or the scores are quietly wrong.
model_artifact = {
    'model': calibrated_model,
    'base_model': base_model,
    'preprocess_spec': PREPROCESS_SPEC,
    'feature_cols': feature_cols,
    'cat_features': cat_features,
    'text_features': text_features,
    'numeric_cols': numeric_cols,
    'drop_cols': drop_cols,
    'threshold': float(THRESHOLD),
    'maturity_days': MATURITY_DAYS,
    'grain': 'one row per ORDER_ID',
    'trained_asof': str(ASOF_DATE.date()),
    'train_window': (str(train_df[DATE_COL].min().date()), str(train_df[DATE_COL].max().date())),
    'metrics': {'random_test': m_rand, 'oot_test': m_oot},
}
joblib.dump(model_artifact, model_path)
print(f"Model artifact saved to: {model_path}")
