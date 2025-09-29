# preprocess_forecasting.py
import pandas as pd
import numpy as np

# -----------------------------
# Config (edit as needed)
# -----------------------------
INPUT_CSV = "RFPs.csv"           # or "RFPs_with_scenarios.csv"
OUTPUT_MONTHLY = "monthly_panel.csv"
OUTPUT_TRAIN = "monthly_train.csv"
OUTPUT_VALID = "monthly_valid.csv"

GROUP_BY_SOURCE_OF_FUND = False   # False => totals per Department only
TARGET_COL = "NetTotalAmount"
USE_APPROVAL_DATE = True          # Prefer DateApproved if present
KEEP_CURRENCY = "PHP"             # Keep only PHP; set to None to convert USD
USD_TO_PHP = 56.0

VALID_MONTHS = 12                 # last N months for validation
MAKE_ROLLING = True
LAGS  = [1, 3, 12]
ROLLS = [3, 12]

# -----------------------------
# Load once, with dynamic parse_dates
# -----------------------------
cols0 = pd.read_csv(INPUT_CSV, nrows=0).columns
parse_dates = [c for c in ["DateRequested","DateNeeded","DateApproved"] if c in cols0]
df = pd.read_csv(INPUT_CSV, parse_dates=parse_dates)

# -----------------------------
# Choose date column & normalize to month-start
# -----------------------------
date_col = "DateApproved" if (USE_APPROVAL_DATE and "DateApproved" in df.columns) else "DateRequested"
df["Date"] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()

# Ensure Department exists (fallback TOTAL)
if "Department" not in df.columns:
    df["Department"] = "TOTAL"
df["Department"] = df["Department"].astype(str).str.strip().str.upper()

# -----------------------------
# Currency handling
# -----------------------------
if KEEP_CURRENCY:
    if "Currency" in df.columns:
        df = df[df["Currency"] == KEEP_CURRENCY].copy()
else:
    if "Currency" in df.columns:
        mask_usd = df["Currency"].eq("USD")
        cols_to_conv = [c for c in [TARGET_COL, "Amount", "ServiceFee", "Less:EWT"] if c in df.columns]
        df.loc[mask_usd, cols_to_conv] = df.loc[mask_usd, cols_to_conv] * USD_TO_PHP
        df["Currency"] = "PHP"

# -----------------------------
# Helpers
# -----------------------------
def to_counts(series):
    return int((series.astype(str).str.len() > 0).sum())

def mode_or_nan(s: pd.Series):
    m = s.mode()
    return m.iat[0] if not m.empty else np.nan

# -----------------------------
# Group keys
# -----------------------------
group_keys = ["Department"]
if GROUP_BY_SOURCE_OF_FUND and "SourceOfFund" in df.columns:
    df["SourceOfFund"] = df["SourceOfFund"].astype(str).str.strip()
    group_keys.append("SourceOfFund")

# -----------------------------
# Scenario-driver presence
# -----------------------------
has_enroll  = "StudentsEnrolled"    in df.columns
has_prog    = "ProgramCount"        in df.columns
has_audit   = "AccreditationAudit"  in df.columns
has_govt    = "GovtFundingChange"   in df.columns
has_tuition = "TuitionPolicy"       in df.columns

# -----------------------------
# Aggregate monthly
# -----------------------------
agg_map = {
    TARGET_COL: ("sum", "NetTotalAmount"),
    "PR": (to_counts, "Requests"),
    "PO": (to_counts, "PO_count"),
    "RR": (to_counts, "RR_count"),
}

aggs = {}
for col, (fn, outname) in agg_map.items():
    if col in df.columns:
        aggs[outname] = (col, fn)

# drivers
if has_enroll:
    aggs["StudentsEnrolled_mean"] = ("StudentsEnrolled", "mean")
    aggs["StudentsEnrolled_last"] = ("StudentsEnrolled", "last")
if has_prog:
    aggs["ProgramCount_max"] = ("ProgramCount", "max")
if has_audit:
    aggs["AccreditationAudit_any"] = ("AccreditationAudit", lambda s: int((s.fillna(0) > 0).any()))
if has_govt:
    aggs["GovtFundingChange_mean"] = ("GovtFundingChange", "mean")
if has_tuition:
    aggs["TuitionPolicy_mode"] = ("TuitionPolicy", mode_or_nan)

agg = (
    df.sort_values(["Date"])  # so "last" is well-defined
      .groupby(group_keys + ["Date"], as_index=False)
      .agg(**aggs)
)

# -----------------------------
# Ensure continuous months per series
# -----------------------------
def reindex_full_months(g):
    idx = pd.period_range(g["Date"].min(), g["Date"].max(), freq="M").to_timestamp()
    g = g.set_index("Date").reindex(idx)
    # fill numeric 0 (keep categorical NaN or ffill if you prefer)
    num_fill_cols = [c for c in [
        "NetTotalAmount","Requests","PO_count","RR_count",
        "StudentsEnrolled_mean","StudentsEnrolled_last",
        "ProgramCount_max","AccreditationAudit_any",
        "GovtFundingChange_mean"
    ] if c in g.columns]
    for c in num_fill_cols:
        g[c] = g[c].fillna(0)
    g.index.name = "Date"
    return g.reset_index()

agg = (
    agg.groupby(group_keys, group_keys=False)
       .apply(reindex_full_months)
       .reset_index(drop=True)
)

# -----------------------------
# Calendar features
# -----------------------------
agg["month"]   = agg["Date"].dt.month
agg["quarter"] = agg["Date"].dt.quarter
agg["year"]    = agg["Date"].dt.year
agg["is_fy_end"] = (agg["month"] == 12).astype(int)

# -----------------------------
# Enrollment growth features (optional)
# -----------------------------
if has_enroll:
    agg = agg.sort_values(group_keys + ["Date"])
    def add_enroll_growth(g):
        g = g.copy()
        if "StudentsEnrolled_last" in g.columns:
            g["EnrollMoM_pct"] = g["StudentsEnrolled_last"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)*100
            g["EnrollYoY_pct"] = g["StudentsEnrolled_last"].pct_change(12).replace([np.inf, -np.inf], np.nan).fillna(0)*100
        else:
            g["EnrollMoM_pct"] = 0.0
            g["EnrollYoY_pct"] = 0.0
        return g
    agg = agg.groupby(group_keys, group_keys=False).apply(add_enroll_growth).reset_index(drop=True)

# -----------------------------
# Lag / rolling on target only
# -----------------------------
agg = agg.sort_values(group_keys + ["Date"]).reset_index(drop=True)

def add_lags(s: pd.Series, lags):
    return {f"lag_{L}": s.shift(L) for L in lags}

def add_rolls(s: pd.Series, windows):
    s1 = s.shift(1)  # prevent leakage
    out = {}
    for w in windows:
        out[f"roll{w}_mean"] = s1.rolling(w, min_periods=1).mean()
        out[f"roll{w}_std"]  = s1.rolling(w, min_periods=1).std()
    return out

parts = []
for _, g in agg.groupby(group_keys, as_index=False):
    g = g.copy()
    if MAKE_ROLLING and "NetTotalAmount" in g.columns:
        for k, v in {**add_lags(g["NetTotalAmount"], LAGS),
                     **add_rolls(g["NetTotalAmount"], ROLLS)}.items():
            g[k] = v
    parts.append(g)

panel = pd.concat(parts, ignore_index=True)

# Fill NaNs from lags/rolls
num_cols = [c for c in [
    "NetTotalAmount","Requests","PO_count","RR_count",
    "StudentsEnrolled_mean","StudentsEnrolled_last","ProgramCount_max",
    "AccreditationAudit_any","GovtFundingChange_mean",
    "EnrollMoM_pct","EnrollYoY_pct",
    "lag_1","lag_3","lag_12","roll3_mean","roll3_std","roll12_mean","roll12_std"
] if c in panel.columns]
panel[num_cols] = panel[num_cols].fillna(0)

# (Optional) Drop troublesome categorical modes if you won’t use them
# if "TuitionPolicy_mode" in panel.columns:
#     panel = panel.drop(columns=["TuitionPolicy_mode"])

# -----------------------------
# Train/Validation split
# -----------------------------
panel = panel.sort_values(group_keys + ["Date"]).reset_index(drop=True)

# assert uniqueness (critical!)
dup_ct = panel.groupby(group_keys + ["Date"]).size().max()
assert dup_ct == 1, f"Panel still has duplicate rows per {group_keys}+Date (max dup count={dup_ct})."

max_date = panel["Date"].max()
cutoff = (max_date.to_period("M") - VALID_MONTHS + 1).to_timestamp()

train = panel[panel["Date"] <  cutoff].copy()
valid = panel[panel["Date"] >= cutoff].copy()

# -----------------------------
# Save outputs
# -----------------------------
panel.to_csv(OUTPUT_MONTHLY, index=False)
train.to_csv(OUTPUT_TRAIN,   index=False)
valid.to_csv(OUTPUT_VALID,   index=False)

print("✓ Saved:")
print(f" - Full monthly panel: {OUTPUT_MONTHLY}  (rows={len(panel)})")
print(f" - Train set:          {OUTPUT_TRAIN}    (rows={len(train)})")
print(f" - Valid set:          {OUTPUT_VALID}    (rows={len(valid)})")
print("\nDate range:", panel['Date'].min().date(), "→", panel['Date'].max().date(), "| Cutoff:", cutoff.date())
print("Series (groups):", panel.groupby(group_keys).ngroups, "| Group keys:", group_keys)
print("Columns:", list(panel.columns))
