# reconcile_mint.py
import numpy as np
import pandas as pd

# ---------------------------
# Utilities
# ---------------------------
def _norm_text(s):
    return s.astype(str).str.strip().str.upper()

def _ensure_cols(df, cols_defaults):
    df = df.copy()
    for c, default in cols_defaults.items():
        if c not in df.columns:
            df[c] = default
    return df

# ---------------------------
# Key builders
# ---------------------------
def build_bottom_index(df):
    df = _ensure_cols(df, {"Department": "TOTAL", "SourceOfFund": "TOTAL"})
    keys = df[["Department", "SourceOfFund"]].copy()
    keys["Department"]   = _norm_text(keys["Department"])
    keys["SourceOfFund"] = _norm_text(keys["SourceOfFund"])
    keys = keys.drop_duplicates().sort_values(["Department", "SourceOfFund"]).reset_index(drop=True)
    keys["bottom_id"] = np.arange(len(keys))
    return keys

def build_S(keys):
    depts = keys["Department"].drop_duplicates().tolist()
    n = len(keys)         # bottoms
    D = len(depts)        # department totals
    m = 1 + D + n         # total rows (grand + dept + bottoms)

    S = np.zeros((m, n), dtype=float)
    row_labels = []

    # grand total
    S[0, :] = 1.0
    row_labels.append(("TOTAL", "ALL"))

    # department totals
    for i, d in enumerate(depts, start=1):
        mask = (keys["Department"] == d).values
        S[i, mask] = 1.0
        row_labels.append(("DEPT", d))

    # bottoms (identity)
    S[1 + D : 1 + D + n, :] = np.eye(n)
    for _, r in keys.iterrows():
        row_labels.append(("BOTTOM", (r["Department"], r["SourceOfFund"])))

    return S, row_labels

# ---------------------------
# Variance estimation for WLS
# ---------------------------
def estimate_var(df, keys, qcol="P50", min_var=1.0):
    df = _ensure_cols(df, {"Department": "TOTAL", "SourceOfFund": "TOTAL"})
    if ("Actual" not in df.columns) or (qcol not in df.columns):
        return np.full(len(keys), float(min_var))

    tmp = df.copy()
    tmp["Department"]   = _norm_text(tmp["Department"])
    tmp["SourceOfFund"] = _norm_text(tmp["SourceOfFund"])

    def _var_grp(g):
        v = np.var((g["Actual"].astype(float) - g[qcol].astype(float)).values, ddof=1)
        return np.nan if g.shape[0] < 2 else v

    var_map = (
        tmp.groupby(["Department", "SourceOfFund"], dropna=False)
           .apply(_var_grp)
           .rename("var")
           .reset_index()
    )

    # Fallbacks
    med = np.nanmedian(var_map["var"].values) if np.any(np.isfinite(var_map["var"].values)) else min_var
    var_map["var"] = var_map["var"].fillna(med).clip(lower=min_var)

    v = []
    for _, r in keys.iterrows():
        row = var_map[
            (var_map["Department"] == r["Department"]) &
            (var_map["SourceOfFund"] == r["SourceOfFund"])
        ]
        v.append(float(row["var"].iloc[0]) if not row.empty else float(med))
    return np.array(v, dtype=float)

# ---------------------------
# MinT reconciliation for one date/quantile
# ---------------------------
def mint_date(S, var_b, y_b):
    y_b = np.asarray(y_b, dtype=float)
    y_b = np.nan_to_num(y_b, nan=0.0)

    # aggregate to all levels
    y_all = S @ y_b

    # covariance (diagonal from variances)
    var_b = np.asarray(var_b, dtype=float).clip(min=1e-12)
    Wb = np.diag(var_b)

    # MinT(WLS): y_b_rec = W_b S^T (S W_b S^T)^+ y_all
    SWST = S @ Wb @ S.T
    # pseudo-inverse to avoid singular errors
    SWST_pinv = np.linalg.pinv(SWST, rcond=1e-10)
    y_b_rec = Wb @ S.T @ SWST_pinv @ y_all
    return y_b_rec

# ---------------------------
# Public API
# ---------------------------
def reconcile(df_pred, quant_cols=("P10","P50","P90")):
    if df_pred.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = df_pred.copy()
    df["Date"]        = pd.to_datetime(df["Date"])
    df["Department"]  = _norm_text(df.get("Department", pd.Series(["TOTAL"] * len(df))))
    df["SourceOfFund"]= _norm_text(df.get("SourceOfFund", pd.Series(["TOTAL"] * len(df))))

    keys = build_bottom_index(df)
    S, labels = build_S(keys)
    var_b = estimate_var(df, keys, qcol="P50", min_var=1.0)

    # Prepare collectors
    per_q_frames = {q: [] for q in quant_cols}
    totals_frames = []

    for dt, g in df.groupby("Date"):
        # Right join to include bottoms that might be missing on this date
        g = keys.merge(g, on=["Department","SourceOfFund"], how="left")

        # Reconcile each quantile
        for q in quant_cols:
            if q not in g.columns:
                continue
            y_b = g[q].astype(float).fillna(0.0).values
            y_rec = mint_date(S, var_b, y_b)
            per_q_frames[q].append(pd.DataFrame({
                "Date": dt,
                "Department": keys["Department"],
                "SourceOfFund": keys["SourceOfFund"],
                q: y_rec
            }))

        # Dept / Grand totals based on reconciled P50
        if "P50" in g.columns:
            y_rec = mint_date(S, var_b, g["P50"].astype(float).fillna(0.0).values)
            y_all = S @ y_rec
            lab = pd.DataFrame(labels, columns=["Level","Name"])
            tdf = lab.copy()
            tdf["Date"] = dt
            tdf["P50"]  = y_all
            totals_frames.append(tdf)

    # Merge quantiles into one bottom table
    available_q = [q for q in quant_cols if per_q_frames[q]]
    if not available_q:
        df_bottom = pd.DataFrame(columns=["Date","Department","SourceOfFund"] + list(quant_cols))
    else:
        base = pd.concat(per_q_frames[available_q[0]], ignore_index=True)
        for q in available_q[1:]:
            qdf = pd.concat(per_q_frames[q], ignore_index=True)
            base = base.merge(qdf, on=["Date","Department","SourceOfFund"], how="outer")
        df_bottom = base.sort_values(["Date","Department","SourceOfFund"]).reset_index(drop=True)

    # Enforce non-crossing quantiles (row-wise sort)
    q_present = [q for q in quant_cols if q in df_bottom.columns]
    if len(q_present) >= 2:
        Q = df_bottom[q_present].to_numpy(dtype=float)
        Q.sort(axis=1)
        df_bottom[q_present] = Q

    # Totals (grand + dept) for P50
    if totals_frames:
        df_totals = pd.concat(totals_frames, ignore_index=True)
        df_totals = df_totals[df_totals["Level"].isin(["TOTAL","DEPT"])].copy()
        df_totals["Series"] = np.where(df_totals["Level"].eq("TOTAL"), "ALL", df_totals["Name"])
        df_totals = df_totals[["Date","Series","P50"]].sort_values(["Date","Series"])
    else:
        df_totals = pd.DataFrame(columns=["Date","Series","P50"])

    return df_bottom, df_totals
