from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# FORESIGHT — Demand & Inventory Intelligence Dashboard
# ============================================================

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT

st.set_page_config(
    page_title="FORESIGHT | Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme
# -----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

theme = st.session_state.theme
dark = theme == "Dark"

LIGHT = {
    "bg": "#f5f7fb",
    "card": "#ffffff",
    "text": "#172033",
    "muted": "#697386",
    "border": "#e4e8f0",
    "accent": "#5b4bdb",
    "accent2": "#7c6cf2",
    "plot": "plotly_white",
}
DARK = {
    "bg": "#0b1020",
    "card": "#151c2f",
    "text": "#f3f5fb",
    "muted": "#a9b2c5",
    "border": "#29334b",
    "accent": "#8b7cf6",
    "accent2": "#a99fff",
    "plot": "plotly_dark",
}
C = DARK if dark else LIGHT

st.markdown(
    f"""
<style>
    .stApp {{
        background: {C["bg"]};
        color: {C["text"]};
    }}

    [data-testid="stSidebar"] {{
        background: {"#101729" if dark else "#eef1f7"};
        border-right: 1px solid {C["border"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {C["text"]} !important;
    }}

    .hero {{
        background: linear-gradient(135deg,
            {"#1b2440" if dark else "#ffffff"} 0%,
            {"#151b31" if dark else "#f0efff"} 100%);
        border: 1px solid {C["border"]};
        border-radius: 22px;
        padding: 26px 30px;
        margin-bottom: 18px;
        box-shadow: 0 10px 30px rgba(20, 30, 60, 0.08);
    }}

    .hero-title {{
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 0;
        color: {C["text"]};
    }}

    .hero-sub {{
        margin-top: 6px;
        color: {C["muted"]};
        font-size: 15px;
    }}

    .pill {{
        display: inline-block;
        margin-top: 14px;
        padding: 6px 12px;
        border-radius: 999px;
        background: {"#272d4b" if dark else "#ebe8ff"};
        color: {C["accent"]};
        font-weight: 700;
        font-size: 12px;
    }}

    .section-title {{
        font-size: 21px;
        font-weight: 750;
        color: {C["text"]};
        margin: 8px 0 12px;
    }}

    .kpi {{
        background: {C["card"]};
        border: 1px solid {C["border"]};
        border-radius: 17px;
        padding: 18px 20px;
        min-height: 115px;
        box-shadow: 0 5px 18px rgba(20, 30, 60, 0.05);
    }}

    .kpi-label {{
        color: {C["muted"]};
        font-size: 12px;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: .5px;
    }}

    .kpi-value {{
        color: {C["text"]};
        font-size: 26px;
        font-weight: 800;
        margin-top: 8px;
    }}

    .kpi-note {{
        color: {C["muted"]};
        font-size: 11px;
        margin-top: 3px;
    }}

    .insight {{
        background: {C["card"]};
        border: 1px solid {C["border"]};
        border-left: 4px solid {C["accent"]};
        border-radius: 14px;
        padding: 14px 16px;
        margin: 7px 0;
    }}

    .footer {{
        color: {C["muted"]};
        text-align: center;
        font-size: 11px;
        padding: 18px 0 5px;
    }}

    div[data-testid="stMetric"] {{
        background: {C["card"]};
        border: 1px solid {C["border"]};
        padding: 12px 16px;
        border-radius: 15px;
    }}

    .stButton > button {{
        border-radius: 10px;
        border: 1px solid {C["border"]};
        font-weight: 650;
    }}

    .stDownloadButton > button {{
        border-radius: 10px;
        font-weight: 650;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Data
# -----------------------------
REQUIRED = ["forecast_output.csv", "risk_scores.csv", "weekly_features.csv"]
missing = [f for f in REQUIRED if not (PROCESSED / f).exists()]

if missing:
    st.error("Required model outputs are missing: " + ", ".join(missing))
    st.code(
        "python generate_data.py\n"
        "python src/pipeline.py\n"
        "python src/forecast.py\n"
        "python src/risk.py"
    )
    st.stop()


@st.cache_data
def load_data():
    forecast = pd.read_csv(
        PROCESSED / "forecast_output.csv",
        parse_dates=["week_start"],
    )
    risk = pd.read_csv(PROCESSED / "risk_scores.csv")
    weekly = pd.read_csv(
        PROCESSED / "weekly_features.csv",
        parse_dates=["week_start"],
    )
    return forecast, risk, weekly


forecast, risk, weekly = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## 📦 FORESIGHT")
st.sidebar.caption("Demand & Inventory Intelligence")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎨 Appearance")
theme_choice = st.sidebar.radio(
    "Dashboard theme",
    ["Light", "Dark"],
    index=0 if theme == "Light" else 1,
    horizontal=True,
    label_visibility="collapsed",
)

if theme_choice != theme:
    st.session_state.theme = theme_choice
    st.rerun()

st.sidebar.markdown("### 🔎 Planning filters")

categories = ["All"] + sorted(risk["category"].dropna().unique().tolist())
sel_category = st.sidebar.selectbox("Category", categories)

filtered = (
    risk
    if sel_category == "All"
    else risk[risk["category"] == sel_category]
).copy()

sku_options = ["(pick a SKU)"] + sorted(
    filtered["sku_id"].astype(str).unique().tolist()
)
sel_sku = st.sidebar.selectbox("SKU", sku_options)

statuses = ["All"] + sorted(filtered["risk_quadrant"].dropna().unique().tolist())
sel_status = st.sidebar.selectbox("Decision status", statuses)

if sel_status != "All":
    filtered = filtered[filtered["risk_quadrant"] == sel_status].copy()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model")
st.sidebar.info("LightGBM\n\n8-week demand horizon\n\nRolling-origin backtest")

# -----------------------------
# Hero
# -----------------------------
st.markdown(
    f"""
<div class="hero">
    <div class="hero-title">📦 Project FORESIGHT</div>
    <div class="hero-sub">
        AI-powered demand forecasting and inventory intelligence for SKU-level planning.
    </div>
    <div class="pill">LIGHTGBM · 8-WEEK FORECAST · ROLLING-ORIGIN BACKTEST</div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# KPI cards
# -----------------------------
total_skus = len(filtered)
sales_risk = filtered["stockout_value_at_risk"].sum()
capital_locked = filtered["overstock_capital_locked"].sum()
action_count = (filtered["risk_quadrant"] != "Healthy").sum()

kpis = [
    ("SKUs in view", f"{total_skus:,}", "Filtered planning universe"),
    ("Sales at risk", f"₹{sales_risk:,.0f}", "Potential stockout exposure"),
    ("Capital locked", f"₹{capital_locked:,.0f}", "Potential overstock exposure"),
    ("SKUs needing action", f"{action_count:,}", "Non-healthy exceptions"),
]

cols = st.columns(4)
for col, (label, value, note) in zip(cols, kpis):
    with col:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_forecast, tab_risk, tab_insights = st.tabs(
    ["📊 Overview", "📈 Forecast", "⚠️ Risk & Actions", "💡 Insights"]
)

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    st.markdown('<div class="section-title">Risk decision map</div>', unsafe_allow_html=True)

    left, right = st.columns([1.55, 1])

    with left:
        fig = px.scatter(
            filtered,
            x="overstock_risk",
            y="stockout_risk",
            size="rupee_value_at_stake",
            color="risk_quadrant",
            hover_name="sku_id",
            hover_data={
                "category": True,
                "stockout_risk": ":.2f",
                "overstock_risk": ":.2f",
                "rupee_value_at_stake": ":,.0f",
            },
            labels={
                "overstock_risk": "Overstock risk",
                "stockout_risk": "Stockout risk",
                "risk_quadrant": "Decision",
            },
            template=C["plot"],
        )
        fig.add_vline(x=0.5, line_dash="dash", opacity=0.55)
        fig.add_hline(y=0.5, line_dash="dash", opacity=0.55)
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown(
            '<div class="section-title">Priority actions</div>',
            unsafe_allow_html=True,
        )
        action_list = (
            filtered[filtered["risk_quadrant"] != "Healthy"]
            .sort_values("rupee_value_at_stake", ascending=False)
            .head(10)
        )

        if action_list.empty:
            st.success("No immediate actions in this filtered view.")
        else:
            display = action_list[
                [
                    "sku_id",
                    "category",
                    "risk_quadrant",
                    "recommended_action",
                    "rupee_value_at_stake",
                ]
            ].rename(
                columns={
                    "sku_id": "SKU",
                    "category": "Category",
                    "risk_quadrant": "Status",
                    "recommended_action": "Action",
                    "rupee_value_at_stake": "₹ at stake",
                }
            )
            st.dataframe(
                display,
                use_container_width=True,
                height=395,
                hide_index=True,
            )

    # Decision distribution
    st.markdown(
        '<div class="section-title">Decision distribution</div>',
        unsafe_allow_html=True,
    )
    counts = (
        filtered["risk_quadrant"]
        .value_counts()
        .rename_axis("Decision")
        .reset_index(name="SKUs")
    )
    fig_bar = px.bar(
        counts,
        x="Decision",
        y="SKUs",
        text="SKUs",
        template=C["plot"],
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=330,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="Number of SKUs",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# FORECAST
# ============================================================
with tab_forecast:
    st.markdown(
        '<div class="section-title">SKU demand forecast</div>',
        unsafe_allow_html=True,
    )

    if sel_sku == "(pick a SKU)":
        st.info("Select a SKU from the sidebar to inspect its 8-week forecast.")
    else:
        hist = weekly[weekly["sku_id"] == sel_sku].sort_values("week_start")
        fut = forecast[forecast["sku_id"] == sel_sku].sort_values("week_start")
        sku_matches = risk[risk["sku_id"] == sel_sku]

        if hist.empty or fut.empty or sku_matches.empty:
            st.warning("Forecast data for this SKU is unavailable.")
        else:
            sku_row = sku_matches.iloc[0]

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=hist["week_start"],
                    y=hist["units_sold"],
                    name="Actual demand",
                    mode="lines+markers",
                )
            )

            if "seasonal_naive" in hist.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=hist["week_start"],
                        y=hist["seasonal_naive"],
                        name="Seasonal-naive baseline",
                        mode="lines",
                        line=dict(dash="dot"),
                    )
                )

            if {"forecast_low_80", "forecast_high_80"}.issubset(fut.columns):
                fig2.add_trace(
                    go.Scatter(
                        x=list(fut["week_start"]) + list(fut["week_start"])[::-1],
                        y=list(fut["forecast_high_80"])
                        + list(fut["forecast_low_80"])[::-1],
                        fill="toself",
                        fillcolor="rgba(91,75,219,0.12)",
                        line=dict(width=0),
                        name="80% planning interval",
                    )
                )

            fig2.add_trace(
                go.Scatter(
                    x=fut["week_start"],
                    y=fut["forecast"],
                    name="FORESIGHT forecast",
                    mode="lines+markers",
                    line=dict(dash="dash", width=3),
                )
            )

            fig2.update_layout(
                template=C["plot"],
                height=470,
                xaxis_title="Week",
                yaxis_title="Units / week",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.05),
            )
            st.plotly_chart(fig2, use_container_width=True)

            a, b, c, d = st.columns(4)
            a.metric("8-week forecast", f"{fut['forecast'].sum():,.0f} units")
            b.metric("On-hand", f"{sku_row['on_hand_units']:,.0f} units")
            c.metric("On-order", f"{sku_row['on_order_units']:,.0f} units")
            d.metric("Decision", sku_row["risk_quadrant"])

            st.markdown(
                f"""
                <div class="insight">
                    <b>{sel_sku}</b> · {sku_row['category']}<br>
                    Recommended action: <b>{sku_row['recommended_action']}</b><br>
                    Value at stake: <b>₹{sku_row['rupee_value_at_stake']:,.0f}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# RISK & ACTIONS
# ============================================================
with tab_risk:
    st.markdown(
        '<div class="section-title">SKU-level risk & action table</div>',
        unsafe_allow_html=True,
    )

    full = (
        filtered[
            [
                "sku_id",
                "category",
                "stockout_risk",
                "overstock_risk",
                "risk_quadrant",
                "recommended_action",
                "stockout_value_at_risk",
                "overstock_capital_locked",
                "rupee_value_at_stake",
            ]
        ]
        .sort_values("rupee_value_at_stake", ascending=False)
        .rename(
            columns={
                "sku_id": "SKU",
                "category": "Category",
                "stockout_risk": "Stockout risk",
                "overstock_risk": "Overstock risk",
                "risk_quadrant": "Status",
                "recommended_action": "Recommended action",
                "stockout_value_at_risk": "₹ sales at risk",
                "overstock_capital_locked": "₹ capital locked",
                "rupee_value_at_stake": "₹ total at stake",
            }
        )
    )

    st.dataframe(
        full,
        use_container_width=True,
        height=520,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download filtered risk table",
        data=full.to_csv(index=False).encode("utf-8"),
        file_name="foresight_risk_table.csv",
        mime="text/csv",
        use_container_width=False,
    )

# ============================================================
# INSIGHTS
# ============================================================
with tab_insights:
    st.markdown(
        '<div class="section-title">Business insights</div>',
        unsafe_allow_html=True,
    )

    promo_lift_text = "not available"
    if "is_promo_week" in weekly.columns:
        promo_means = weekly.groupby("is_promo_week")["units_sold"].mean()
        if 0 in promo_means.index and 1 in promo_means.index and promo_means[0] != 0:
            lift = promo_means[1] / promo_means[0] - 1
            promo_lift_text = f"{lift:.0%}"

    insights = [
        (
            "📣 Promotion effect",
            f"Promo weeks show approximately {promo_lift_text} higher average weekly demand than non-promo weeks in the prototype data.",
        ),
        (
            "🎯 Prioritise exceptions",
            "The risk engine ranks decisions using rupee value at stake, helping operations focus on financially important SKUs first.",
        ),
        (
            "📦 Balance stock",
            "Stockout and overstock are treated as two sides of the same planning problem: protect availability while reducing excess inventory.",
        ),
        (
            "🤖 Forecast before action",
            "The dashboard combines an 8-week forecast with inventory position so recommendations are based on expected demand rather than current stock alone.",
        ),
    ]

    for title, body in insights:
        st.markdown(
            f"""
            <div class="insight">
                <b>{title}</b><br>
                <span style="color:{C["muted"]};">{body}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    x, y, z = st.columns(3)
    x.metric("Model", "LightGBM")
    y.metric("Forecast horizon", "8 weeks")
    z.metric("Validation", "Rolling-origin")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div class="footer">
        FORESIGHT · Demand & Inventory Intelligence · Prototype decision-support system<br>
        Forecast intervals are planning bands, not statistical guarantees.
    </div>
    """,
    unsafe_allow_html=True,
)
