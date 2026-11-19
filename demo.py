import gzip, pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import streamlit as st
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(page_title="DOOH Effect Measurement", layout="wide")

@st.cache_resource(show_spinner=False)
def load_snapshot(path: str = "demo_snapshot.pkl.gz") -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

snap = load_snapshot()
cfg = snap["config"]

def compute_smd(df_pool: pd.DataFrame, matches_df: pd.DataFrame, features: list) -> pd.DataFrame:
    t_df = df_pool[df_pool["store_key"].isin(matches_df["test_store_key"].unique())]
    c_weights = matches_df["control_store_key"].value_counts()
    c_df = df_pool[df_pool["store_key"].isin(c_weights.index)]
    w = c_df["store_key"].map(c_weights).to_numpy(dtype=float)
    w /= w.sum()
    rows = []
    for f in features:
        t_mean, t_var = t_df[f].mean(), t_df[f].var(ddof=1)
        c_mean = np.average(c_df[f], weights=w)
        c_var = np.average((c_df[f] - c_mean) ** 2, weights=w)
        pooled_sd = np.sqrt((t_var + c_var) / 2.0)
        smd = (t_mean - c_mean) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"Feature": f, "Test Mean": t_mean, "Ctrl Mean (Weighted)": c_mean, "SMD": smd, "Balanced (<0.1)": abs(smd) < 0.1})
    return pd.DataFrame(rows)

def run_mahalanobis_matching(df_pool: pd.DataFrame, test_ids: list, features: list, smd_thresh: float = 0.20):
    df = df_pool.dropna(subset=features).copy()
    df["store_key"] = df["store_key"].astype(int)
    stds = df[features].std(ddof=1).replace(0, 1.0)
    Z = ((df[features] - df[features].mean()) / stds).fillna(0.0).to_numpy(dtype=np.float64)

    is_test = df["store_key"].isin(test_ids).to_numpy()
    test_df, ctrl_df = df[is_test].reset_index(drop=True), df[~is_test].reset_index(drop=True)
    if test_df.empty or ctrl_df.empty:
        raise ValueError(f"Insufficient test ({len(test_df)}) or control ({len(ctrl_df)}) stores.")

    cov = np.nan_to_num(np.cov(Z, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0) + 1e-4 * np.eye(len(features))
    dists = cdist(Z[is_test], Z[~is_test], metric="mahalanobis", VI=np.linalg.inv(cov))
    c_idx = np.argmin(dists, axis=1)
    t_idx = np.arange(len(test_df))
    c_rep = ctrl_df.iloc[c_idx].reset_index(drop=True)

    all_matches = pd.DataFrame({
        "test_store_key": test_df["store_key"], "test_store_label": test_df["store_label"],
        "control_store_key": c_rep["store_key"], "control_store_label": c_rep["store_label"],
        "match_rank": 1, "distance": dists[t_idx, c_idx],
        "test_pre_sales": test_df["target_sales_pre"], "test_post_sales": test_df["target_sales_post"],
        "ctrl_pre_sales": c_rep["target_sales_pre"], "ctrl_post_sales": c_rep["target_sales_post"],
        "test_total_sales": test_df["total_sales_pre"], "ctrl_total_sales": c_rep["total_sales_pre"],
        "test_total_post_sales": test_df["total_sales_post"], "ctrl_total_post_sales": c_rep["total_sales_post"],
        "test_tx_count": test_df["tx_count_pre"], "ctrl_tx_count": c_rep["tx_count_pre"],
        "test_tx_count_post": test_df["tx_count_post"], "ctrl_tx_count_post": c_rep["tx_count_post"],
        "test_growth": test_df["total_sales_growth_pre"], "ctrl_growth": c_rep["total_sales_growth_pre"],
        "test_surface": test_df["store_surface"], "ctrl_surface": c_rep["store_surface"],
    })

    filtered = all_matches.copy()
    while len(filtered) > 1 and compute_smd(df, filtered, features)["SMD"].abs().max() > smd_thresh:
        filtered = filtered.drop(filtered["distance"].idxmax())

    all_matches["is_outlier"] = ~all_matches["test_store_key"].isin(filtered["test_store_key"])
    return filtered, all_matches, compute_smd(df, filtered, features)

def compute_did_results(matches_df: pd.DataFrame, base_w: float, camp_w: float) -> pd.DataFrame:
    agg = matches_df.groupby("test_store_key", as_index=False).agg(
        store_label=("test_store_label", "first"), avg_match_distance=("distance", "mean"),
        T_pre_raw=("test_pre_sales", "first"), T_post_raw=("test_post_sales", "first"),
        C_pre_raw=("ctrl_pre_sales", "mean"), C_post_raw=("ctrl_post_sales", "mean"),
        T_tot_pre=("test_total_sales", "first"), T_tot_post=("test_total_post_sales", "first"),
        C_tot_pre=("ctrl_total_sales", "mean"), C_tot_post=("ctrl_total_post_sales", "mean"),
        T_tx_pre=("test_tx_count", "first"), C_tx_pre=("ctrl_tx_count", "mean"),
        T_tx_post_raw=("test_tx_count_post", "first"), C_tx_post_raw=("ctrl_tx_count_post", "mean"),
    )
    for pfx, t_pre, t_post, c_pre, c_post in [("", "T_pre_raw", "T_post_raw", "C_pre_raw", "C_post_raw"), ("store_", "T_tot_pre", "T_tot_post", "C_tot_pre", "C_tot_post")]:
        agg[f"T_{pfx}pre"], agg[f"T_{pfx}post"] = agg[t_pre] / base_w, agg[t_post] / camp_w
        agg[f"C_{pfx}pre"], agg[f"C_{pfx}post"] = agg[c_pre] / base_w, agg[c_post] / camp_w
        agg[f"test_{pfx}diff"] = agg[f"T_{pfx}post"] - agg[f"T_{pfx}pre"]
        agg[f"ctrl_{pfx}diff"] = agg[f"C_{pfx}post"] - agg[f"C_{pfx}pre"]
        agg[f"did_{pfx}lift"] = agg[f"test_{pfx}diff"] - agg[f"ctrl_{pfx}diff"]
        t_g = np.where(agg[f"T_{pfx}pre"] > 0, agg[f"T_{pfx}post"] / agg[f"T_{pfx}pre"], 1.0)
        c_g = np.where(agg[f"C_{pfx}pre"] > 0, agg[f"C_{pfx}post"] / agg[f"C_{pfx}pre"], 1.0)
        agg[f"did_{pfx}index"] = (t_g / c_g) * 100.0
    return agg

def get_weeks_between(s_yw: int, e_yw: int, all_yws: list) -> int:
    try: return all_yws.index(e_yw) - all_yws.index(s_yw) + 1
    except ValueError: return max(1, e_yw - s_yw + 1)

def build_chart(df, val_col, label, title, mode, n_test, n_ctrl, camp_s, camp_e, ctrl_weights, test_ids):
    d = df.copy()
    d["is_test"] = d["store_key"].isin(test_ids)
    d["w_val"] = d[val_col] * np.where(d["is_test"], 1.0, d["store_key"].map(ctrl_weights).fillna(0.0))
    tot_w = sum(ctrl_weights.values())
    c_df = d.groupby(["is_test", "date"], as_index=False)["w_val"].sum()
    c_df["cohort"] = np.where(c_df["is_test"], "Test Stores", "Control Stores")
    div = (n_test, tot_w) if mode == "Average per Store" else (1.0, tot_w / n_test)
    c_df["val"] = np.where(c_df["is_test"], c_df["w_val"] / div[0], c_df["w_val"] / div[1])

    fig = px.line(c_df, x="date", y="val", color="cohort",
                  color_discrete_map={"Test Stores": "#1f77b4", "Control Stores": "#ff7f0e"},
                  labels={"date": "Date", "val": f"Cohort Equivalent {label}", "cohort": "Cohort"},
                  title=f"{title}: Test ({n_test} stores) vs Control ({n_ctrl} stores)")
    cs_ms, ce_ms = pd.to_datetime(camp_s).timestamp() * 1000, pd.to_datetime(camp_e).timestamp() * 1000
    fig.add_vline(x=cs_ms, line_width=2, line_dash="dash", line_color="#444", annotation_text="Campaign Start", annotation_position="top left")
    fig.add_vrect(x0=cs_ms, x1=ce_ms, fillcolor="rgba(31,119,180,0.08)", layer="below", line_width=0, annotation_text="Active Campaign Window")
    fig.update_layout(height=400, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"), margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- UI & SIDEBAR ---
st.title("Digital Out-of-Home Effect Measurement (Demo)")

with st.sidebar:
    st.title("Configuration")
    st.info("Running on pre-loaded real campaign snapshot.")
    with st.expander("Step 1: Scope and SKUs", expanded=True):
        st.text_area("Target SKU Keys", value=", ".join(map(str, cfg["target_skus"])), height=75, disabled=True)

    with st.expander("Step 2: Campaign Timing and Stores", expanded=True):
        st.text_area("Test Store IDs", value=", ".join(map(str, cfg["test_store_ids"])), height=75, disabled=True)
        c1, c2 = st.columns(2)
        camp_start = c1.number_input("Campaign Start Week", value=cfg["camp_start"], disabled=True)
        camp_end = c2.number_input("Campaign End Week", value=cfg["camp_end"], disabled=True)
        base_start, base_end = cfg["base_start"], cfg["base_end"]
        all_yws = snap["all_yws"]
        st.caption(f"Baseline: `{base_start}` to `{base_end}` ({get_weeks_between(base_start, base_end, all_yws)} wks)")

        strictness = st.selectbox("Store Similarity Preset", ["Strict", "Balanced", "Flexible"], index=1)
        smd_thresh = {"Strict": 0.10, "Balanced": 0.20, "Flexible": 0.30}[strictness]

test_store_ids = cfg["test_store_ids"]
target_skus = cfg["target_skus"]
camp_s_date, camp_e_date = snap["week_bounds"][(camp_start, camp_end)]
base_s_date, base_e_date = snap["week_bounds"][(base_start, base_end)]
base_w = float(get_weeks_between(base_start, base_end, all_yws))
camp_w = float(get_weeks_between(camp_start, camp_end, all_yws))

pool_df = snap["pool_df"]
features = ["store_surface", "total_sales_pre", "tx_count_pre", "total_sales_growth_pre", "target_sales_pre"]
matches_df, all_matches_df, smd_summary = run_mahalanobis_matching(pool_df, test_store_ids, features, smd_thresh)
matched_test_ids = tuple(sorted(matches_df["test_store_key"].unique()))
matched_ctrl_ids = tuple(sorted(matches_df["control_store_key"].unique()))
did = compute_did_results(matches_df, base_w, camp_w)
sku_meta_df = snap["sku_meta_df"]

# --- KPI CALCULATIONS & DISPLAY ---
num_test, num_ctrl, num_supp = len(did), len(matched_ctrl_ids), len(test_store_ids)
calc_idx = lambda tp, t0, cp, c0: ((tp / t0) / (cp / c0) * 100.0) if (t0 > 0 and c0 > 0) else 100.0
sku_did_idx = calc_idx(did["T_post"].mean(), did["T_pre"].mean(), did["C_post"].mean(), did["C_pre"].mean())

t_atv_pre = did["T_tot_pre"].mean() / did["T_tx_pre"].mean()
t_atv_post = did["T_store_post"].mean() / (did["T_tx_post_raw"].mean() / camp_w) if did["T_tx_post_raw"].mean() > 0 else 0.0
c_atv_pre = did["C_tot_pre"].mean() / did["C_tx_pre"].mean()
c_atv_post = did["C_store_post"].mean() / (did["C_tx_post_raw"].mean() / camp_w) if did["C_tx_post_raw"].mean() > 0 else 0.0
atv_did_idx = calc_idx(t_atv_post, t_atv_pre, c_atv_post, c_atv_pre)

t_diff, c_diff = did["test_diff"].mean(), did["ctrl_diff"].mean()
rel_lift_pct = ((t_diff - c_diff) / abs(c_diff) * 100.0) if c_diff != 0 else 0.0

kpis = [
    ("SKU Store Sales Uplift", f"{rel_lift_pct:+.1f}%", None),
    ("SKU DiD Index", f"{sku_did_idx:.1f}", None),
    ("ATV DiD Index", f"{atv_did_idx:.1f}", None),
    ("Avg Weekly Gross Sales", f"€{did['T_store_post'].mean():,.2f}", None),
    ("Matched Stores", f"{num_test:,} / {num_supp:,}", None),
]
for col, (label, val, delta) in zip(st.columns(5), kpis):
    col.metric(label, val, delta=delta)

style_metric_cards(border_left_color="#1f77b4", box_shadow=False)
st.divider()

# --- TIME-SERIES CHARTS ---
st.subheader("Overlaid Performance Trends (Test vs. Control)")
c1, c2, _ = st.columns([2, 2, 4])
is_rev = c1.segmented_control("Metric", ["Revenue (€)", "Transactions"], default="Revenue (€)", key="ts_metric_sel") == "Revenue (€)"
view_mode = c2.segmented_control("View", ["Average per Store", "Total Cohort Aggregate"], default="Average per Store", key="ts_view_mode") or "Average per Store"
y_lbl = "Sales (€)" if is_rev else "Transactions"

daily_ts = snap["daily_ts"][snap["daily_ts"]["store_key"].isin(matched_test_ids + matched_ctrl_ids)]
if not daily_ts.empty:
    ctrl_w = matches_df["control_store_key"].value_counts().to_dict()
    st.plotly_chart(build_chart(daily_ts, "store_sales" if is_rev else "store_tx", y_lbl, f"Daily Store Average {y_lbl}", view_mode, num_test, num_ctrl, camp_s_date, camp_e_date, ctrl_w, matched_test_ids), use_container_width=True)
    st.plotly_chart(build_chart(daily_ts, "sku_sales" if is_rev else "sku_tx", y_lbl, f"Daily Target SKU {y_lbl}", view_mode, num_test, num_ctrl, camp_s_date, camp_e_date, ctrl_w, matched_test_ids), use_container_width=True)

# --- DETAIL TABS ---
t1, t2, t3, t4, t5 = st.tabs(["Store-Level DiD Table", "Matched Pairs", "All Active Store Covariates", "Scoped SKUs", "Covariate Balance (SMD)"])

def render_table(tab, df, col_map, fmt_map):
    with tab:
        sub = df[list(col_map.keys())].rename(columns=col_map)
        st.dataframe(sub.style.format(fmt_map), hide_index=True, use_container_width=True, height=420)

render_table(t1, did, {
    "store_label": "Store (City)", "T_pre": "Test Base/Wk (€)", "T_post": "Test Camp/Wk (€)",
    "C_pre": "Ctrl Base/Wk (€)", "C_post": "Ctrl Camp/Wk (€)", "test_diff": "Test Diff/Wk (€)",
    "ctrl_diff": "Ctrl Diff/Wk (€)", "did_lift": "Net Lift/Store/Wk (€)", "did_index": "DiD Index", "avg_match_distance": "Avg Dist"
}, {c: "€{:,.2f}" for c in ["Test Base/Wk (€)", "Test Camp/Wk (€)", "Ctrl Base/Wk (€)", "Ctrl Camp/Wk (€)", "Test Diff/Wk (€)", "Ctrl Diff/Wk (€)", "Net Lift/Store/Wk (€)"]} | {"DiD Index": "{:.1f}", "Avg Dist": "{:.2f}"})

render_table(t2, all_matches_df.sort_values(["test_store_label", "match_rank"]), {
    "test_store_label": "Test Store (City)", "match_rank": "Match Rank", "control_store_label": "Control Store (City)",
    "distance": "Distance", "is_outlier": "Poor Match?", "test_pre_sales": "Test Target Pre (€)", "ctrl_pre_sales": "Ctrl Target Pre (€)",
    "test_total_sales": "Test Total Pre (€)", "ctrl_total_sales": "Ctrl Total Pre (€)", "test_tx_count": "Test Orders", "ctrl_tx_count": "Ctrl Orders",
    "test_growth": "Test Trend", "ctrl_growth": "Ctrl Trend", "test_surface": "Test Area (m²)", "ctrl_surface": "Ctrl Area (m²)"
}, {"Distance": "{:.2f}", "Test Target Pre (€)": "€{:,.2f}", "Ctrl Target Pre (€)": "€{:,.2f}", "Test Total Pre (€)": "€{:,.2f}", "Ctrl Total Pre (€)": "€{:,.2f}", "Test Orders": "{:,.0f}", "Ctrl Orders": "{:,.0f}", "Test Trend": "{:.2%}", "Ctrl Trend": "{:.2%}", "Test Area (m²)": "{:,.1f}", "Ctrl Area (m²)": "{:,.1f}"})

render_table(t3, pool_df, {
    "store_label": "Store (City)", "store_surface": "Area (m²)", "total_sales_pre": "Total Sales Pre (€)",
    "tx_count_pre": "Total Orders Pre", "total_sales_growth_pre": "Pre-Campaign Sales Trend",
    "target_sales_pre": "Target SKU Sales Pre (€)", "target_sales_post": "Target SKU Sales Post (€)"
}, {"Area (m²)": "{:,.1f}", "Total Sales Pre (€)": "€{:,.2f}", "Total Orders Pre": "{:,}", "Pre-Campaign Sales Trend": "{:.2%}", "Target SKU Sales Pre (€)": "€{:,.2f}", "Target SKU Sales Post (€)": "€{:,.2f}"})

with t4:
    st.caption(f"Showing {len(sku_meta_df):,} matched campaign product(s)")
    st.dataframe(sku_meta_df.rename(columns={"product_key": "Product Key", "product_name": "Product Name"}), hide_index=True, use_container_width=True)

with t5:
    st.dataframe(smd_summary.style.format({"Test Mean": "{:,.2f}", "Ctrl Mean (Weighted)": "{:,.2f}", "SMD": "{:+.3f}"}), hide_index=True, use_container_width=True)