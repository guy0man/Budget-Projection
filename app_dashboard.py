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
# Load artifacts
# ---------------------------
art = joblib.load(ART_PATH)
models      = art["models"]                  # {0.1,0.5,0.9}: LGBMRegressor
FEATURES    = art["FEATURES"]
CAT_COLS    = art["CAT_COLS"]
GROUP_KEYS  = art["GROUP_KEYS"] or ["Department"]
DRIVER_COLS = art.get("DRIVER_COLS", [])
CAT_LEVELS  = art.get("CAT_LEVELS", {})
BASE_FEATS  = art["VALID_FEATURES_DF"].copy()

BASE_FEATS = BASE_FEATS.loc[:, ~BASE_FEATS.columns.duplicated()]
BASE_FEATS["Date"] = pd.to_datetime(BASE_FEATS["Date"], errors="coerce")
if "Department" not in BASE_FEATS.columns:
    BASE_FEATS["Department"] = "TOTAL"
BASE_FEATS["Department"] = BASE_FEATS["Department"].astype(str).str.strip().str.upper()

for c in CAT_COLS:
    if c not in BASE_FEATS.columns:
        BASE_FEATS[c] = ""
    BASE_FEATS[c] = BASE_FEATS[c].astype("category")
    if c in CAT_LEVELS and len(CAT_LEVELS[c]) > 0:
        BASE_FEATS[c] = pd.Categorical(
            BASE_FEATS[c].astype(str),
            categories=[str(v) for v in CAT_LEVELS[c]]
        )

departments = sorted(BASE_FEATS["Department"].dropna().unique().tolist())
departments = ["ALL (Reconciled)"] + departments   # add university-wide option

# ---------------------------
# Actuals
# ---------------------------
def load_actuals(valid_panel_path="monthly_valid.csv"):
    try:
        a = pd.read_csv(valid_panel_path, parse_dates=["Date"])
        if "Department" not in a.columns:
            a["Department"] = "TOTAL"
        a["Department"] = a["Department"].astype(str).str.strip().str.upper()
        a["Date"] = a["Date"].dt.to_period("M").dt.to_timestamp()
        a = (
            a.groupby(["Department", "Date"], as_index=False)["NetTotalAmount"]
             .sum()
             .rename(columns={"NetTotalAmount": "Actual"})
        )
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
    X = feat_df.copy()
    for col in FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURES].copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("category")
            if c in CAT_LEVELS and len(CAT_LEVELS[c]) > 0:
                X[c] = pd.Categorical(
                    X[c].astype(str),
                    categories=[str(v) for v in CAT_LEVELS[c]]
                )
    return X

def predict_quantiles(feat_df):
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
    if df.empty or not {"P10","P50","P90"}.issubset(df.columns):
        return df
    q = df[["P10","P50","P90"]].to_numpy(dtype=float)
    q.sort(axis=1)
    df = df.copy()
    df["P10"], df["P50"], df["P90"] = q[:,0], q[:,1], q[:,2]
    return df

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
app.title = "Budget Forecast — Department & University Totals"

app.layout = html.Div([
    html.Div([
        html.Label("Select Department or University"),
        dcc.Dropdown(
            id="dept_dd",
            options=[{"label": d, "value": d} for d in departments],
            value=departments[0] if departments else None,
            clearable=False
        )
    ], style={"width": "50%", "marginBottom": "10px"}),

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
    if not departments:
        fig = go.Figure()
        fig.update_layout(title="No departments found. Did you run preprocessing & training?")
        return fig

    dept = (dept or "").strip().upper()

    base_pred = predict_quantiles(BASE_FEATS)
    scen_pred = predict_quantiles(BASE_FEATS)   # (scenarios could be applied similarly)

    # University-wide totals
    if dept.startswith("ALL"):
        b = base_pred.groupby("Date", as_index=False)[["P10","P50","P90"]].sum()
        s = scen_pred.groupby("Date", as_index=False)[["P10","P50","P90"]].sum()
        a = actuals.groupby("Date", as_index=False)["Actual"].sum()
    else:
        b = base_pred[base_pred["Department"] == dept].groupby("Date", as_index=False)[["P10","P50","P90"]].mean()
        s = scen_pred[scen_pred["Department"] == dept].groupby("Date", as_index=False)[["P10","P50","P90"]].mean()
        a = actuals[actuals["Department"] == dept]

    b, s = fix_quantiles(b.sort_values("Date")), fix_quantiles(s.sort_values("Date"))
    a = a.sort_values("Date")

    fig = go.Figure()

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

    if not a.empty:
        fig.add_trace(go.Scatter(
            x=a["Date"], y=a["Actual"], mode="lines+markers",
            name="Actual", line=dict(color="black")
        ))

    title = f"{dept} — Baseline vs Scenario"
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
