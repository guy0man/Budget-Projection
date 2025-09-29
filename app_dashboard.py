# app_dashboard.py
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

ART_PATH = "artifacts/lgbm_quantile_artifacts.pkl"
VALID_PANEL_PATH = "monthly_valid.csv"   # aggregated Actuals (from preprocessing)

# ---------------------------
# Load artifacts & normalize
# ---------------------------
art = joblib.load(ART_PATH)
models      = art["models"]                  # {0.1,0.5,0.9}: LGBMRegressor
FEATURES    = art["FEATURES"]
CAT_COLS    = art["CAT_COLS"]
GROUP_KEYS  = art["GROUP_KEYS"] or ["Department"]
DRIVER_COLS = art.get("DRIVER_COLS", [])
CAT_LEVELS  = art.get("CAT_LEVELS", {})
BASE_FEATS  = art["VALID_FEATURES_DF"].copy()

# De-dup columns & normalize
BASE_FEATS = BASE_FEATS.loc[:, ~BASE_FEATS.columns.duplicated()]
BASE_FEATS["Date"] = pd.to_datetime(BASE_FEATS["Date"], errors="coerce")
if "Department" not in BASE_FEATS.columns:
    BASE_FEATS["Department"] = "TOTAL"
BASE_FEATS["Department"] = BASE_FEATS["Department"].astype(str).str.strip().str.upper()

# Ensure all categorical columns exist and are categorical with saved levels
for c in CAT_COLS:
    if c not in BASE_FEATS.columns:
        BASE_FEATS[c] = ""
    BASE_FEATS[c] = BASE_FEATS[c].astype("category")
    if c in CAT_LEVELS and len(CAT_LEVELS[c]) > 0:
        BASE_FEATS[c] = pd.Categorical(
            BASE_FEATS[c].astype(str),
            categories=[str(v) for v in CAT_LEVELS[c]]
        )

# Departments for dropdown
departments = sorted(BASE_FEATS["Department"].dropna().unique().tolist())

# ---------------------------
# Actuals (guaranteed monthly & aligned)
# ---------------------------
def load_actuals(valid_panel_path="monthly_valid.csv"):
    try:
        a = pd.read_csv(valid_panel_path, parse_dates=["Date"])
        if "Department" not in a.columns:
            a["Department"] = "TOTAL"

        # normalize labels
        a["Department"] = a["Department"].astype(str).str.strip().str.upper()

        # ensure monthly timestamps match model features (period -> timestamp @ month start)
        a["Date"] = a["Date"].dt.to_period("M").dt.to_timestamp()

        # aggregate to a single point per (Dept, Month)
        a = (
            a.groupby(["Department", "Date"], as_index=False)["NetTotalAmount"]
             .sum()
             .rename(columns={"NetTotalAmount": "Actual"})
        )

        # trim to the same date window as BASE_FEATS
        dmin, dmax = BASE_FEATS["Date"].min(), BASE_FEATS["Date"].max()
        a = a[(a["Date"] >= dmin) & (a["Date"] <= dmax)].copy()
        return a
    except Exception:
        return pd.DataFrame(columns=["Department", "Date", "Actual"])

actuals = load_actuals(VALID_PANEL_PATH)

# ---------------------------
# Helpers
# ---------------------------
def align_for_lgbm(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Match training columns, order, dtypes, and category levels (LightGBM-safe)."""
    X = feat_df.copy()

    # add missing features
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0

    # drop extras, enforce order
    X = X[FEATURES].copy()

    # enforce categoricals (and levels)
    for c in CAT_COLS:
        if c in X.columns:
            if not pd.api.types.is_categorical_dtype(X[c]):
                X[c] = X[c].astype("category")
            if c in CAT_LEVELS and len(CAT_LEVELS[c]) > 0:
                X[c] = pd.Categorical(
                    X[c].astype(str),
                    categories=[str(v) for v in CAT_LEVELS[c]]
                )
    return X

def scenario_apply(df, global_pct=0, dept_pct=0, enroll_pct=0, dept=None):
    """Apply scenario multipliers to numeric driver features (no retrain)."""
    g = df.copy()

    def mult(col, pct, mask=None):
        if col in g.columns and np.issubdtype(g[col].dtype, np.number):
            factor = 1.0 + (pct or 0) / 100.0
            if mask is None:
                g[col] = g[col] * factor
            else:
                g.loc[mask, col] = g.loc[mask, col] * factor

    # Global tweak on all drivers
    for col in DRIVER_COLS:
        mult(col, global_pct)

    # Enrollment-specific tweak
    for col in ["StudentsEnrolled_mean", "StudentsEnrolled_last"]:
        if col in DRIVER_COLS and col in g.columns:
            mult(col, enroll_pct)

    # Department-specific tweak
    if dept is not None:
        m = g["Department"].astype(str).str.upper().eq(str(dept).upper())
        for col in DRIVER_COLS:
            mult(col, dept_pct, m)

    # keep cats as category (with saved levels)
    for c in CAT_COLS:
        if c in g.columns:
            g[c] = g[c].astype("category")
            if c in CAT_LEVELS and len(CAT_LEVELS[c]) > 0:
                g[c] = pd.Categorical(
                    g[c].astype(str),
                    categories=[str(v) for v in CAT_LEVELS[c]]
                )
    return g

def predict_quantiles(feat_df):
    """Predict P10/P50/P90 for all departments in feat_df."""
    X = align_for_lgbm(feat_df)
    out = pd.DataFrame({
        "Date": feat_df["Date"].values,
        "Department": feat_df["Department"].values
    })
    out["P10"] = models[0.1].predict(X)
    out["P50"] = models[0.5].predict(X)
    out["P90"] = models[0.9].predict(X)
    return out

def fix_quantiles(df):
    """Row-wise enforce P10 <= P50 <= P90 to avoid flipped bands."""
    if df.empty or not {"P10","P50","P90"}.issubset(df.columns):
        return df
    q = df[["P10","P50","P90"]].to_numpy(dtype=float)
    q.sort(axis=1)                      # ascending row-wise
    df = df.copy()
    df["P10"], df["P50"], df["P90"] = q[:,0], q[:,1], q[:,2]
    return df

def apply_post_scaling(pred_df, dept, g_pct=0, d_pct=0, e_pct=0, enroll_elasticity=0.8):
    """
    Post-prediction multiplicative scaling when scenario drivers are NOT in FEATURES.
    This makes sliders responsive without retraining immediately.
    """
    pred = pred_df.copy()
    g_factor = 1.0 + (g_pct or 0)/100.0
    pred[["P10","P50","P90"]] = pred[["P10","P50","P90"]] * g_factor

    mask = pred["Department"].astype(str).str.upper().eq(str(dept).upper())
    d_factor = 1.0 + (d_pct or 0)/100.0
    e_factor = 1.0 + ((e_pct or 0)/100.0)*enroll_elasticity
    pred.loc[mask, ["P10","P50","P90"]] = pred.loc[mask, ["P10","P50","P90"]] * d_factor * e_factor
    return pred

def slider(id_, label, lo=-50, hi=50, val=0):
    return html.Div([
        html.Label(label, style={"fontWeight": 600}),
        dcc.Slider(
            id=id_, min=lo, max=hi, step=1, value=val,
            marks={lo: f"{lo}%", -25: "-25%", 0: "0%", 25: "+25%", hi: f"+{hi}%"},
            tooltip={"placement": "bottom"}
        )
    ])

# ---------------------------
# Dash App
# ---------------------------
app = Dash(__name__)
app.title = "Budget Forecast — Department Totals"

app.layout = html.Div([
    html.Div([
        html.Label("Department"),
        dcc.Dropdown(
            id="dept_dd",
            options=[{"label": d, "value": d} for d in departments],
            value=departments[0] if departments else None,
            clearable=False
        )
    ], style={"width": "40%", "marginBottom": "10px"}),

    html.Div([
        slider("global_adj", "Global Driver %"),
        slider("dept_adj",   "Department Driver %"),
        slider("enroll_adj", "Enrollment %"),
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginBottom": "8px"}),

    dcc.Graph(id="fig")
])

# ---------------------------
# Callback
# ---------------------------
@app.callback(
    Output("fig", "figure"),
    Input("dept_dd", "value"),
    Input("global_adj", "value"),
    Input("dept_adj", "value"),
    Input("enroll_adj", "value"),
)
def plot_forecast(dept, g_adj, d_adj, e_adj):
    # If departments list is empty, show a clear message
    if not departments:
        fig = go.Figure()
        fig.update_layout(title="No departments found. Did you run preprocessing & training?")
        return fig

    dept = (dept or "").strip().upper() if isinstance(dept, str) else str(dept).upper()

    # Predict baseline (all departments)
    base_pred = predict_quantiles(BASE_FEATS)

    # Predict scenario (driver-adjusted)
    scen_feats = scenario_apply(BASE_FEATS, g_adj, d_adj, e_adj, dept)
    scen_pred  = predict_quantiles(scen_feats)

    # If drivers aren't in the model, scale predictions so sliders have visible effect
    drivers_in_model = any(col in FEATURES for col in DRIVER_COLS)
    if not drivers_in_model:
        scen_pred = apply_post_scaling(scen_pred, dept, g_adj, d_adj, e_adj, enroll_elasticity=0.8)

    # Filter selected department
    b = base_pred[base_pred["Department"] == dept]
    s = scen_pred[scen_pred["Department"] == dept]

    # Collapse duplicate months (if any), then sort & fix quantiles
    if not b.empty:
        b = b.groupby("Date", as_index=False)[["P10","P50","P90"]].mean()
        b = fix_quantiles(b.sort_values("Date"))
    if not s.empty:
        s = s.groupby("Date", as_index=False)[["P10","P50","P90"]].mean()
        s = fix_quantiles(s.sort_values("Date"))

    # Actuals for this department
    a = actuals[actuals["Department"] == dept].sort_values("Date")

    fig = go.Figure()

    # Baseline band + line
    if not b.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([b["Date"], b["Date"][::-1]]),
            y=pd.concat([b["P90"], b["P10"][::-1]]),
            fill="toself", fillcolor="rgba(0,102,204,0.18)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
            name="Baseline 80% Interval"
        ))
        fig.add_trace(go.Scatter(
            x=b["Date"], y=b["P50"], mode="lines+markers",
            marker=dict(size=4), name="Baseline Median",
            line=dict(color="rgba(0,102,204,1.0)")
        ))

    # Scenario band + line
    if not s.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([s["Date"], s["Date"][::-1]]),
            y=pd.concat([s["P90"], s["P10"][::-1]]),
            fill="toself", fillcolor="rgba(255,102,0,0.18)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
            name="Scenario 80% Interval"
        ))
        fig.add_trace(go.Scatter(
            x=s["Date"], y=s["P50"], mode="lines",
            name="Scenario Median", line=dict(color="rgba(255,102,0,1.0)")
        ))

    # Actuals (black)
    if not a.empty:
        fig.add_trace(go.Scatter(
            x=a["Date"], y=a["Actual"], mode="lines+markers",
            name="Actual", line=dict(color="black")
        ))

    title = f"{dept} — Baseline vs Scenario"
    if b.empty and s.empty and a.empty:
        title = f"{dept}: no data available. Try another department."
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Amount",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )
    fig.update_yaxes(tickprefix="₱", separatethousands=True)
    return fig

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
