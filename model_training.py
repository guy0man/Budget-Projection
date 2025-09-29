# model_training.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

TRAIN_PATH = "monthly_train.csv"
VALID_PATH = "monthly_valid.csv"
TARGET = "NetTotalAmount"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

# -------------------------
# Load
# -------------------------
train = pd.read_csv(TRAIN_PATH, parse_dates=["Date"])
valid = pd.read_csv(VALID_PATH, parse_dates=["Date"])

# Ensure group keys exist and are normalized
if "Department" not in train.columns:
    train["Department"] = "TOTAL"
if "Department" not in valid.columns:
    valid["Department"] = "TOTAL"

train["Department"] = train["Department"].astype(str).str.strip().str.upper()
valid["Department"] = valid["Department"].astype(str).str.strip().str.upper()

if "SourceOfFund" in train.columns:
    train["SourceOfFund"] = train["SourceOfFund"].astype(str).str.strip()
if "SourceOfFund" in valid.columns:
    valid["SourceOfFund"] = valid["SourceOfFund"].astype(str).str.strip()

# -------------------------
# Features
# -------------------------
cat_candidates = [
    "Department","SourceOfFund","TuitionPolicy_mode",
    "TransactionType_mode","ModeOfPayment_mode",
    "TermsOfPayment_mode","TaxRegistration_mode","TypeOfBusiness_mode"
]
cat_cols = [c for c in cat_candidates if c in train.columns]

drop_cols = ["Date", TARGET]
FEATURES = [c for c in train.columns if c not in drop_cols]

# Scenario driver columns (used by sliders in the app)
driver_cols = [c for c in [
    "StudentsEnrolled_mean","StudentsEnrolled_last",
    "EnrollMoM_pct","EnrollYoY_pct",
    "ProgramCount_max","AccreditationAudit_any",
    "GovtFundingChange_mean"
] if c in FEATURES]

# Cast categoricals
for c in cat_cols:
    train[c] = train[c].astype("category")
    valid[c] = valid[c].astype("category")

# Keys the app will filter by (department-only totals if SourceOfFund absent)
GROUP_KEYS = [c for c in ["Department","SourceOfFund"] if c in train.columns]

print("[INFO] Features:", len(FEATURES))
print("[INFO] Categorical:", cat_cols)
print("[INFO] Drivers:", driver_cols)
print("[INFO] Group keys:", GROUP_KEYS)

# -------------------------
# Models (Quantile)
# -------------------------
def q_model(alpha):
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=30,
        random_state=42,
        verbose=-1,
    )

alphas = [0.1, 0.5, 0.9]
models = {}
for a in alphas:
    m = q_model(a)
    m.fit(
        train[FEATURES],
        train[TARGET],
        categorical_feature=cat_cols if cat_cols else "auto"
    )
    models[a] = m

# -------------------------
# Validation metrics
# -------------------------
pred_p50 = models[0.5].predict(valid[FEATURES])
mse  = mean_squared_error(valid[TARGET], pred_p50)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(valid[TARGET], pred_p50)
den  = valid[TARGET].replace(0, np.nan)
mape = np.mean(np.abs((valid[TARGET] - pred_p50) / den)) * 100

print(f"[VAL] RMSE={rmse:,.2f}  MAE={mae:,.2f}  MAPE={mape:.2f}%")

# -------------------------
# Save artifacts (models + metadata)
# -------------------------
cat_levels = {c: (train[c].cat.categories.tolist() if c in train.columns else []) for c in cat_cols}

# Template features for app predictions (preserve cols & order; drop dup cols)
valid_feats = valid[["Date"] + GROUP_KEYS + FEATURES].copy()
valid_feats = valid_feats.loc[:, ~valid_feats.columns.duplicated()]

artifacts = {
    "alphas": alphas,
    "models": models,                 # dict of LGBMRegressor
    "FEATURES": FEATURES,
    "CAT_COLS": cat_cols,
    "GROUP_KEYS": GROUP_KEYS,
    "DRIVER_COLS": driver_cols,
    "CAT_LEVELS": cat_levels,
    "VALID_FEATURES_DF": valid_feats
}

out_path = ARTIFACT_DIR / "lgbm_quantile_artifacts.pkl"
joblib.dump(artifacts, out_path)
print(f"[SAVE] {out_path.resolve()}")
