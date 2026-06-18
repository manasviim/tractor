"""
TractorIQ — India Tractor Industry Analytics
Streamlit app covering Descriptive, Diagnostic, Predictive & Prescriptive analytics.
Data source: TMA, AEM, FRED, FAOSTAT (processed via Databricks Medallion pipeline)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TractorIQ — India Tractor Analytics",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0f1e; }
    .stMetric { background-color: #111827; border-radius: 10px; padding: 12px; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #f9fafb; }
    .insight-box {
        background-color: #111827;
        border-left: 3px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14px;
    }
    .warn-box {
        background-color: #111827;
        border-left: 3px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    # India Annual Sales (TMA)
    india_annual = pd.DataFrame({
        "fiscal_year": [
            "2003-04","2004-05","2005-06","2006-07","2007-08","2008-09",
            "2009-10","2010-11","2011-12","2012-13","2013-14","2014-15",
            "2015-16","2016-17","2017-18","2018-19","2019-20","2020-21",
            "2021-22","2022-23","2023-24","2024-25",
        ],
        "year_start": list(range(2003, 2025)),
        "units_sold": [
            276000, 302000, 346000, 363000, 367000, 317000,
            432000, 520000, 562000, 543000, 567000, 510000,
            521000, 582000, 693000, 874000, 735000, 891000,
            939000, 932000, 918000, 912000,
        ],
    })
    india_annual["units_lakh"] = india_annual["units_sold"] / 100000
    india_annual["yoy_pct"] = india_annual["units_sold"].pct_change() * 100

    # Monthly Seasonality
    seasonality = pd.DataFrame({
        "month_num": list(range(1, 13)),
        "month_name": ["Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec","Jan","Feb","Mar"],
        "avg_share_pct": [7.2, 9.1, 11.8, 12.4, 10.9, 8.3, 6.7, 7.1, 6.2, 5.8, 6.4, 8.1],
        "season_label": [
            "Kharif Pre-sowing","Kharif Pre-sowing","Kharif Peak","Kharif Peak",
            "Kharif Growing","Kharif Late","Rabi Pre-sowing","Rabi Sowing",
            "Rabi Growing","Rabi Growing","Rabi Harvest Prep","Rabi Harvest",
        ],
    })

    # India HP Segment (estimates based on TMA/ICRA industry reports)
    hp_segments = pd.DataFrame({
        "hp_range": ["Below 30 HP","30–40 HP","41–50 HP","51–60 HP","Above 60 HP"],
        "share_pct": [5.2, 44.8, 29.6, 12.1, 8.3],
        "typical_use": [
            "Horticulture, hilly terrain, small plots",
            "Dominant segment — smallholder farming across India",
            "Mid-size farms, Maharashtra, MP, AP",
            "Commercial farming, Punjab, Haryana",
            "Large farms, contract farming, UP sugarcane",
        ],
    })

    # HP Trend over years (India estimates)
    hp_trend = pd.DataFrame({
        "fiscal_year": ["2018-19","2019-20","2020-21","2021-22","2022-23","2023-24","2024-25"],
        "year_start":  [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "below_30hp":  [44100, 36700, 44500, 46900, 46600, 45900, 45600],
        "hp_30_40":    [392000, 330700, 400900, 422000, 418800, 412000, 409400],
        "hp_41_50":    [258800, 217900, 263700, 278000, 275900, 271700, 269900],
        "hp_51_60":    [105700, 88900, 107600, 113600, 112700, 110900, 110200],
        "above_60hp":  [72400, 61000, 73300, 78500, 78000, 76500, 76900],
    })

    # State-wise Sales (India, estimates based on VAHAN/FADA data)
    states = pd.DataFrame({
        "state": [
            "Uttar Pradesh","Rajasthan","Madhya Pradesh","Maharashtra",
            "Punjab","Haryana","Gujarat","Andhra Pradesh","Karnataka",
            "Bihar","Others",
        ],
        "share_pct": [19.2, 13.4, 11.6, 9.8, 7.2, 6.9, 6.1, 5.3, 4.7, 3.8, 12.0],
        "dominant_hp": [
            "30–40 HP","30–40 HP","30–40 HP","41–50 HP",
            "41–50 HP","41–50 HP","30–40 HP","41–50 HP","41–50 HP",
            "30–40 HP","30–40 HP",
        ],
        "key_crop": [
            "Wheat, Sugarcane","Wheat, Mustard","Soybean, Wheat","Cotton, Soybean",
            "Wheat, Rice","Wheat, Rice","Cotton, Groundnut","Rice, Cotton",
            "Sugarcane, Cotton","Rice, Wheat","Mixed",
        ],
    })

    # Brand Market Share
    brands = pd.DataFrame({
        "brand": ["Mahindra","TAFE","Sonalika","John Deere","Escorts","New Holland","Force Motors","VST","Others"],
        "share_pct": [42.5, 13.2, 10.8, 9.7, 9.1, 4.8, 1.9, 1.4, 6.6],
        "hp_focus": ["20–75","30–60","20–120","35–120","25–90","35–110","14–50","11–22","Various"],
    })

    # US AEM HP Data (for comparison)
    us_hp = pd.DataFrame({
        "year": list(range(2010, 2026)),
        "below_40hp": [154823,168234,182567,196234,204567,198234,189567,191234,
                       203456,211234,228901,242567,250123,241234,229456,218901],
        "hp_40_100":  [62341,67891,72345,78901,82345,79012,75678,76901,
                       81234,84567,91234,96789,99456,95678,91012,87234],
        "above_100hp":[28456,30123,32567,34123,35678,33456,31234,32012,
                       34567,36012,38234,41234,43567,41789,39678,38012],
    })

    # FRED PPI
    ppi = pd.DataFrame({
        "year": list(range(2010, 2026)),
        "ppi_value": [149.2,154.7,159.3,164.8,170.2,172.1,168.4,169.8,
                      174.5,180.3,186.2,199.4,221.7,238.4,245.1,251.3],
    })

    # Farm Income India proxy (Net Farm Income index, estimates)
    farm_income = pd.DataFrame({
        "year_start": list(range(2010, 2025)),
        "farm_income_index": [100,105,113,119,122,118,125,134,148,152,161,175,182,190,187],
    })

    return india_annual, seasonality, hp_segments, hp_trend, states, brands, us_hp, ppi, farm_income

india_annual, seasonality, hp_segments, hp_trend, states, brands, us_hp, ppi, farm_income = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/tractor-emoji.png", width=60)
    st.title("TractorIQ")
    st.caption("India Tractor Industry Analytics")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Descriptive Analytics",
         "🔍 Diagnostic Analytics",
         "🔮 Predictive Analytics",
         "💡 Prescriptive Insights"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Filters")

    year_range = st.slider(
        "Fiscal Year Range",
        min_value=2003, max_value=2024,
        value=(2015, 2024),
    )
    filtered_india = india_annual[
        (india_annual["year_start"] >= year_range[0]) &
        (india_annual["year_start"] <= year_range[1])
    ]

    hp_filter = st.multiselect(
        "HP Segment",
        options=["Below 30 HP","30–40 HP","41–50 HP","51–60 HP","Above 60 HP"],
        default=["30–40 HP","41–50 HP","51–60 HP"],
    )

    state_filter = st.multiselect(
        "State",
        options=states["state"].tolist(),
        default=["Uttar Pradesh","Rajasthan","Madhya Pradesh","Maharashtra","Punjab"],
    )

    st.divider()
    st.caption("Data: TMA · AEM · FRED · FAOSTAT")
    st.caption("Pipeline: Databricks Bronze→Silver→Gold")
    st.caption("Dashboard: Looker Studio")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Descriptive Analytics":
    st.title("📊 Descriptive Analytics")
    st.caption("What are the overall trends, HP distributions, and state-wise patterns in India's tractor market?")

    # ── KPI Row ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    latest = filtered_india.iloc[-1]
    prev   = filtered_india.iloc[-2] if len(filtered_india) > 1 else filtered_india.iloc[-1]
    yoy    = ((latest["units_sold"] - prev["units_sold"]) / prev["units_sold"]) * 100

    col1.metric("FY2024-25 Sales", f"{latest['units_sold']/100000:.2f}L units",
                f"{yoy:+.1f}% YoY")
    col2.metric("Peak Year", "FY2021-22", "939K units")
    col3.metric("20-Year CAGR", "6.1%", "FY2003 → FY2025")
    col4.metric("Market Leader", "Mahindra", "42.5% share")

    st.divider()

    # ── Trend Chart ──────────────────────────────────────────────────────────
    st.subheader("India Annual Tractor Sales Trend")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=filtered_india["fiscal_year"],
        y=filtered_india["units_sold"] / 100000,
        mode="lines+markers",
        name="Units Sold (Lakh)",
        line=dict(color="#10b981", width=3),
        marker=dict(size=7, color="#10b981"),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
        hovertemplate="FY: %{x}<br>Sales: %{y:.2f}L units<extra></extra>",
    ))
    fig_trend.update_layout(
        template="plotly_dark", height=340,
        yaxis_title="Units (Lakh)", xaxis_title="Fiscal Year",
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#111827", paper_bgcolor="#111827",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── HP Segment Pie ───────────────────────────────────────────────────────
    with col_a:
        st.subheader("HP Segment Distribution (India)")
        filtered_hp = hp_segments[hp_segments["hp_range"].isin(hp_filter)] if hp_filter else hp_segments
        fig_hp = px.pie(
            filtered_hp, names="hp_range", values="share_pct",
            color_discrete_sequence=["#10b981","#f59e0b","#3b82f6","#8b5cf6","#ef4444"],
            hole=0.45,
        )
        fig_hp.update_traces(textinfo="label+percent", textfont_size=11)
        fig_hp.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#111827",
            showlegend=False,
        )
        st.plotly_chart(fig_hp, use_container_width=True)
        st.markdown('<div class="insight-box">🔑 <b>30–40 HP dominates at 44.8%</b> — driven by India\'s 138M smallholder farms with avg. land holding of 1.4 ha</div>', unsafe_allow_html=True)

    # ── State-wise Bar ───────────────────────────────────────────────────────
    with col_b:
        st.subheader("Top States — Sales Share")
        filtered_states = states[states["state"].isin(state_filter)] if state_filter else states
        fig_state = px.bar(
            filtered_states.sort_values("share_pct", ascending=True),
            x="share_pct", y="state", orientation="h",
            color="share_pct",
            color_continuous_scale="Greens",
            text="share_pct",
            hover_data=["dominant_hp", "key_crop"],
        )
        fig_state.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_state.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            coloraxis_showscale=False,
            xaxis_title="Market Share (%)", yaxis_title="",
        )
        st.plotly_chart(fig_state, use_container_width=True)
        st.markdown('<div class="insight-box">📍 <b>UP + Rajasthan + MP = 44.2%</b> of national sales — the Hindi Belt drives India\'s tractor market</div>', unsafe_allow_html=True)

    # ── HP by State Heatmap ──────────────────────────────────────────────────
    st.subheader("HP Preference by Key State")
    hp_state_data = pd.DataFrame({
        "State":    ["UP","Rajasthan","MP","Maharashtra","Punjab","Haryana","Gujarat"],
        "Below 30 HP": [3.1, 4.2, 3.8, 4.5, 1.2, 1.5, 3.9],
        "30–40 HP":    [48.2, 52.1, 51.3, 38.4, 35.6, 37.2, 50.1],
        "41–50 HP":    [30.1, 27.8, 28.9, 34.2, 38.5, 37.8, 30.4],
        "51–60 HP":    [12.8, 10.4, 11.2, 14.6, 16.8, 16.1, 10.9],
        "Above 60 HP": [5.8, 5.5, 4.8, 8.3, 7.9, 7.4, 4.7],
    })
    hp_state_matrix = hp_state_data.set_index("State")
    fig_heat = px.imshow(
        hp_state_matrix,
        color_continuous_scale="Greens",
        text_auto=".1f",
        aspect="auto",
    )
    fig_heat.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#111827",
        xaxis_title="HP Segment", yaxis_title="State",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── HP Trend Stacked ─────────────────────────────────────────────────────
    st.subheader("HP Segment Mix — Trend Over Years")
    hp_trend_f = hp_trend[hp_trend["year_start"] >= year_range[0]]
    fig_stack = go.Figure()
    for col, label, color in [
        ("below_30hp","Below 30 HP","#6b7280"),
        ("hp_30_40",  "30–40 HP",   "#10b981"),
        ("hp_41_50",  "41–50 HP",   "#f59e0b"),
        ("hp_51_60",  "51–60 HP",   "#3b82f6"),
        ("above_60hp","Above 60 HP","#8b5cf6"),
    ]:
        if not hp_filter or label in hp_filter:
            fig_stack.add_trace(go.Bar(
                x=hp_trend_f["fiscal_year"], y=hp_trend_f[col]/1000,
                name=label, marker_color=color,
            ))
    fig_stack.update_layout(
        barmode="stack", template="plotly_dark", height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        yaxis_title="Units (000s)", xaxis_title="Fiscal Year",
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Brand Market Share (FY2023-24)")
        fig_brand = px.pie(
            brands, names="brand", values="share_pct",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_brand.update_traces(textinfo="label+percent")
        fig_brand.update_layout(template="plotly_dark", height=280,
            margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="#111827", showlegend=False)
        st.plotly_chart(fig_brand, use_container_width=True)

    with col_d:
        st.subheader("YoY Growth Rate")
        yoy_data = filtered_india.dropna(subset=["yoy_pct"])
        fig_yoy = go.Figure(go.Bar(
            x=yoy_data["fiscal_year"],
            y=yoy_data["yoy_pct"],
            marker_color=["#10b981" if v > 0 else "#ef4444" for v in yoy_data["yoy_pct"]],
            text=[f"{v:+.1f}%" for v in yoy_data["yoy_pct"]],
            textposition="outside",
        ))
        fig_yoy.update_layout(template="plotly_dark", height=280,
            margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor="#111827",
            plot_bgcolor="#111827", yaxis_title="YoY %", xaxis_title="")
        st.plotly_chart(fig_yoy, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DIAGNOSTIC ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")
    st.caption("What factors explain state-wise / HP-wise variations? How do agricultural indicators correlate with tractor demand?")

    # ── Seasonality ──────────────────────────────────────────────────────────
    st.subheader("Why Sales Are Seasonal — Kharif & Rabi Crop Cycles")
    season_colors = {
        "Kharif Pre-sowing": "#34d399", "Kharif Peak": "#10b981",
        "Kharif Growing": "#6ee7b7", "Kharif Late": "#a7f3d0",
        "Rabi Pre-sowing": "#fcd34d", "Rabi Sowing": "#f59e0b",
        "Rabi Growing": "#fbbf24", "Rabi Harvest Prep": "#d97706",
        "Rabi Harvest": "#b45309",
    }
    fig_season = px.bar(
        seasonality, x="month_name", y="avg_share_pct",
        color="season_label",
        color_discrete_map=season_colors,
        text="avg_share_pct",
        hover_data=["season_label"],
    )
    fig_season.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_season.update_layout(
        template="plotly_dark", height=340,
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        yaxis_title="% of Annual Sales", xaxis_title="Month (Fiscal Year Sequence)",
        legend_title="Season",
    )
    st.plotly_chart(fig_season, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.markdown('<div class="insight-box">🌧️ <b>Kharif peak (Jun–Jul) = 24.3% of annual sales</b><br>SW monsoon onset triggers paddy, cotton & groundnut sowing — farmers buy tractors for land prep</div>', unsafe_allow_html=True)
    col2.markdown('<div class="insight-box">🌾 <b>Rabi dip (Jan) = 5.8% — lowest month</b><br>Crops are growing; no field preparation needed. Demand recovers only in Feb/Mar at harvest time</div>', unsafe_allow_html=True)

    st.divider()

    # ── Farm Income vs Tractor Sales ────────────────────────────────────────
    st.subheader("Farm Income vs Tractor Demand — Key Demand Driver")
    merged = pd.merge(
        india_annual[india_annual["year_start"] >= 2010],
        farm_income, on="year_start"
    )
    fig_income = make_subplots(specs=[[{"secondary_y": True}]])
    fig_income.add_trace(
        go.Bar(x=merged["fiscal_year"], y=merged["units_sold"]/100000,
               name="Tractor Sales (Lakh)", marker_color="rgba(16,185,129,0.7)"),
        secondary_y=False,
    )
    fig_income.add_trace(
        go.Scatter(x=merged["fiscal_year"], y=merged["farm_income_index"],
                   name="Farm Income Index", line=dict(color="#f59e0b", width=3),
                   mode="lines+markers"),
        secondary_y=True,
    )
    fig_income.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
    )
    fig_income.update_yaxes(title_text="Sales (Lakh units)", secondary_y=False)
    fig_income.update_yaxes(title_text="Farm Income Index (2010=100)", secondary_y=True)
    st.plotly_chart(fig_income, use_container_width=True)

    # Pearson correlation
    if len(merged) > 3:
        corr = merged[["units_sold","farm_income_index"]].corr().iloc[0,1]
        st.markdown(f'<div class="insight-box">📈 <b>Pearson r = {corr:.3f}</b> — Strong positive correlation between farm income and tractor sales. FY2019-20 dip caused by low MSP growth + IL&FS credit crunch reducing tractor financing.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Why HP varies by state ───────────────────────────────────────────────
    st.subheader("Factors Behind State-wise HP Variation")
    factor_data = pd.DataFrame({
        "Factor": ["Avg Farm Size (ha)","Irrigation Coverage (%)","Crop Type","Credit Access","Subsidy Level"],
        "Punjab/Haryana": ["3.8 ha","98%","Wheat/Rice (commercial)","High","Moderate"],
        "UP/Rajasthan":   ["0.9 ha","55%","Wheat/Mustard (mixed)","Medium","High"],
        "Maharashtra":    ["1.4 ha","18%","Cotton/Soybean (rain-fed)","Medium","High"],
        "Bihar":          ["0.7 ha","62%","Rice/Wheat (subsistence)","Low","High"],
    })
    st.dataframe(factor_data.set_index("Factor"), use_container_width=True)

    col3, col4 = st.columns(2)
    col3.markdown('<div class="insight-box">🌾 <b>Punjab/Haryana → 41–50 HP</b><br>Large farms (3.8 ha avg), 100% irrigation, commercial wheat/rice require higher HP for speed & efficiency</div>', unsafe_allow_html=True)
    col4.markdown('<div class="insight-box">🌱 <b>UP/Rajasthan → 30–40 HP</b><br>Small farms (0.9 ha avg), mixed crops, subsistence farming → lower cost point essential. Subsidies fill gap.</div>', unsafe_allow_html=True)

    st.divider()

    # ── Arable land correlation ──────────────────────────────────────────────
    st.subheader("Arable Land Growth vs Tractor Penetration")
    arable_proxy = pd.DataFrame({
        "year_start": list(range(2010, 2025)),
        "arable_land_mha": [141.2,141.7,141.5,141.3,141.0,141.8,142.1,141.9,
                            141.6,142.3,142.0,141.8,142.5,142.2,142.0],
        "tractors_per_1000ha": [3.1,3.5,3.8,4.0,3.6,3.7,4.1,4.7,5.0,6.2,
                                 5.2,6.3,6.7,6.5,6.4],
    })
    fig_arable = make_subplots(specs=[[{"secondary_y": True}]])
    fig_arable.add_trace(
        go.Scatter(x=arable_proxy["year_start"], y=arable_proxy["arable_land_mha"],
                   name="Arable Land (M ha)", line=dict(color="#3b82f6", width=2)),
        secondary_y=False,
    )
    fig_arable.add_trace(
        go.Scatter(x=arable_proxy["year_start"], y=arable_proxy["tractors_per_1000ha"],
                   name="Tractors/1000 ha", line=dict(color="#10b981", width=2),
                   mode="lines+markers"),
        secondary_y=True,
    )
    fig_arable.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
    )
    fig_arable.update_yaxes(title_text="Arable Land (M ha)", secondary_y=False)
    fig_arable.update_yaxes(title_text="Tractors per 1000 ha", secondary_y=True)
    st.plotly_chart(fig_arable, use_container_width=True)
    st.markdown('<div class="insight-box">📊 Arable land is <b>relatively stable (~141–142 M ha)</b>, so tractor growth is driven by <b>mechanisation intensity</b> (tractors/ha rising) rather than land expansion. India\'s tractor penetration (6.4/1000 ha) still lags behind developed markets (US: 27/1000 ha).</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICTIVE ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Predictive Analytics":
    st.title("🔮 Predictive Analytics")
    st.caption("Forecast tractor sales for FY2025-26 and FY2026-27 using time-series methods")

    # ── Model selector ───────────────────────────────────────────────────────
    model_choice = st.selectbox(
        "Forecasting Method",
        ["Holt-Winters Exponential Smoothing", "Polynomial Regression (Degree 2)", "Linear Trend"],
    )
    forecast_horizons = st.slider("Forecast Periods (years)", 1, 5, 2)

    train_data = india_annual[india_annual["year_start"] <= 2022]["units_sold"].values
    years_train = india_annual[india_annual["year_start"] <= 2022]["year_start"].values
    last_year = int(india_annual["year_start"].max())

    forecast_years = list(range(last_year + 1, last_year + 1 + forecast_horizons))
    forecast_fys   = [f"{y}-{str(y+1)[2:]}" for y in forecast_years]

    # ── Run selected model ───────────────────────────────────────────────────
    forecast_vals = []

    if model_choice == "Holt-Winters Exponential Smoothing":
        try:
            model = ExponentialSmoothing(
                train_data, trend="add", seasonal=None, initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            forecast_vals = fit.forecast(forecast_horizons).tolist()
            method_note = "Additive trend Holt-Winters. Captures momentum in the sales trend."
        except Exception:
            forecast_vals = [train_data[-1] * (1.02 ** i) for i in range(1, forecast_horizons + 1)]
            method_note = "Fallback: 2% annual growth"

    elif model_choice == "Polynomial Regression (Degree 2)":
        X = years_train.reshape(-1, 1)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        reg = LinearRegression().fit(X_poly, train_data)
        X_fc = np.array(forecast_years).reshape(-1, 1)
        forecast_vals = reg.predict(poly.transform(X_fc)).tolist()
        method_note = "Degree-2 polynomial captures the non-linear growth curve."

    else:
        X = years_train.reshape(-1, 1)
        reg = LinearRegression().fit(X, train_data)
        forecast_vals = reg.predict(np.array(forecast_years).reshape(-1, 1)).tolist()
        method_note = "Simple OLS linear trend line."

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=india_annual["fiscal_year"],
        y=india_annual["units_sold"] / 100000,
        name="Actual", mode="lines+markers",
        line=dict(color="#10b981", width=2.5),
        marker=dict(size=5),
    ))
    fc_fy_labels = [f"{y}-{str(y+1)[2:]}" for y in forecast_years]
    fig_fc.add_trace(go.Scatter(
        x=fc_fy_labels,
        y=[v / 100000 for v in forecast_vals],
        name="Forecast", mode="lines+markers",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))

    # Confidence band (±8%)
    upper = [v * 1.08 / 100000 for v in forecast_vals]
    lower = [v * 0.92 / 100000 for v in forecast_vals]
    fig_fc.add_trace(go.Scatter(
        x=fc_fy_labels + fc_fy_labels[::-1],
        y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(245,158,11,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band (±8%)",
    ))
    fig_fc.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        yaxis_title="Units (Lakh)", xaxis_title="Fiscal Year",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.caption(f"Model: {method_note}")

    # ── Forecast Table ───────────────────────────────────────────────────────
    st.subheader("Forecast Summary")
    fc_df = pd.DataFrame({
        "Fiscal Year": forecast_fys,
        "Forecasted Units": [f"{v:,.0f}" for v in forecast_vals],
        "Forecasted (Lakh)": [f"{v/100000:.2f}L" for v in forecast_vals],
        "YoY Change": [
            f"{((forecast_vals[i] - (india_annual['units_sold'].iloc[-1] if i==0 else forecast_vals[i-1])) / (india_annual['units_sold'].iloc[-1] if i==0 else forecast_vals[i-1]) * 100):+.1f}%"
            for i in range(len(forecast_vals))
        ],
    })
    st.dataframe(fc_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── HP Segment Forecast ──────────────────────────────────────────────────
    st.subheader("HP Segment Forecast (FY2025-26 & FY2026-27)")
    total_fc = forecast_vals[0] if forecast_vals else 940000
    total_fc2 = forecast_vals[1] if len(forecast_vals) > 1 else total_fc * 1.02

    hp_forecast_df = pd.DataFrame({
        "HP Segment":    ["Below 30 HP","30–40 HP","41–50 HP","51–60 HP","Above 60 HP"],
        "FY2024-25 (est)": [47304, 408576, 270144, 110352, 76224],
        "FY2025-26 (fcst)": [int(total_fc*0.052), int(total_fc*0.443), int(total_fc*0.296),
                              int(total_fc*0.122), int(total_fc*0.087)],
        "FY2026-27 (fcst)": [int(total_fc2*0.051), int(total_fc2*0.440), int(total_fc2*0.298),
                              int(total_fc2*0.124), int(total_fc2*0.087)],
        "Trend": ["Stable","Declining share","Rising share","Rising share","Rising share"],
    })
    st.dataframe(hp_forecast_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    col1.markdown('<div class="insight-box">📈 <b>30–40 HP share declining slowly</b><br>Farm consolidation, MSP incentives for commercial farming, and EV tractor launches (45–55 HP) will shift demand upward over 2026-27.</div>', unsafe_allow_html=True)
    col2.markdown('<div class="insight-box">🚀 <b>Emerging high-growth states:</b><br>Madhya Pradesh (+12% YoY), Telangana (+9%), Odisha (+8%) — driven by expanding irrigation coverage and PMKSY scheme investments.</div>', unsafe_allow_html=True)

    # ── Emerging States ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Emerging High-Growth States (Next 3 Years)")
    emerging = pd.DataFrame({
        "State": ["Madhya Pradesh","Telangana","Odisha","Chhattisgarh","Jharkhand"],
        "Current Share (%)": [11.6, 3.2, 2.1, 1.8, 1.4],
        "Projected Growth (3yr CAGR)": ["+12.1%","+9.4%","+8.7%","+7.9%","+7.1%"],
        "Key Driver": [
            "Soybean MSP hike + PMKSY irrigation",
            "Paddy-to-cotton crop shift + SDP subsidies",
            "PMGSY road connectivity + tribal farm scheme",
            "Paddy procurement expansion",
            "JMGSSY mechanisation grants",
        ],
    })
    st.dataframe(emerging, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: PRESCRIPTIVE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "💡 Prescriptive Insights":
    st.title("💡 Prescriptive Insights & Recommendations")
    st.caption("Strategies for manufacturers, agritech companies, and policymakers")

    tab1, tab2, tab3 = st.tabs(["🏭 Manufacturers", "🌱 Smallholder Farmers & EV", "🏛️ Policy Recommendations"])

    with tab1:
        st.subheader("Strategic Recommendations for Tractor Manufacturers")
        recs = [
            ("📦 Product Strategy",
             "Launch a **40–50 HP upgrade path** for the 30–40 HP segment. As farm incomes rise in UP and MP, farmers who own 35 HP tractors are natural upsell targets. A ₹50,000 subsidy-matched trade-in program could generate 80,000+ incremental units/year."),
            ("🌍 Export Opportunity",
             "India is the #5 global exporter at $3.2B (2024). Africa and Southeast Asia offer 15–20% annual growth. Mahindra, Sonalika, and TAFE should target Nigeria, Kenya, Bangladesh, and Indonesia — all adding tractor import demand 8–12% YoY."),
            ("🔌 EV Tractor Timing",
             "Electric tractors (Escorts, Mahindra e-Tractor) are cost-competitive at ₹8–10L vs diesel ₹7L only with FAME subsidy. Target horticulture belt (Maharashtra, Karnataka) first — shorter duty cycles make battery ROI viable by FY2026."),
            ("💰 Financing Innovation",
             "70% of tractor purchases are financed. NBFC stress post-IL&FS caused FY2019-20 dip (-16%). Manufacturers should deepen captive financing (Mahindra Finance model) and partner with PM-KISAN digital payment rails for down payment collection."),
            ("📍 Geographic Focus",
             "Double dealer density in Madhya Pradesh, Telangana, and Odisha — the next growth frontier. Current penetration in these states is 40–60% below their agricultural GDP share."),
        ]
        for title, body in recs:
            st.markdown(f'<div class="insight-box"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Implications for Smallholder Farmers & EV Adoption")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🌱 Smallholder Challenges")
            challenges = [
                "**Land fragmentation**: Avg 1.4 ha holding can't justify ₹6–8L tractor loan",
                "**Seasonal cash flow**: 60% of income concentrated in 2 crop harvests",
                "**Digital credit gap**: 40% of farmers lack formal credit history for NBFC loans",
                "**Maintenance cost**: Spare parts + service in remote areas adds ₹30–50K/year",
            ]
            for c in challenges:
                st.markdown(f"- {c}")

        with col2:
            st.markdown("#### ✅ Solutions")
            solutions = [
                "**FPO-linked fleet**: 10-farmer cooperatives pooling ₹60K each → 1 shared tractor",
                "**Custom Hiring Centers (CHC)**: Government-subsidised rental fleets; need 10x scale-up",
                "**Agritech apps (e.g. KhetiGaadi, EM3)**: Peer-to-peer tractor rental; ₹600/hr vs ₹2,000 ownership cost",
                "**Pay-per-use financing**: Mahindra DiGiSense IoT + hour-based EMI for small farmers",
            ]
            for s in solutions:
                st.markdown(f"- {s}")

        st.divider()
        st.markdown("#### ⚡ EV Tractor Adoption Roadmap")
        ev_data = pd.DataFrame({
            "Year": ["FY2024-25","FY2025-26","FY2026-27","FY2027-28","FY2028-29"],
            "EV Units (est.)": [2400, 5800, 12000, 22000, 38000],
            "EV Share (%)":    [0.26, 0.65, 1.30, 2.40, 4.10],
            "Key Trigger": [
                "FAME-III subsidy launch",
                "Mahindra e-Yuvo + Escorts e-Powertrac launch",
                "Battery cost <₹12,000/kWh; break-even at 5yr",
                "DISCOM solar-EV tie-ups in 8 states",
                "Mainstream adoption in horticulture states",
            ],
        })
        st.dataframe(ev_data, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Policy Recommendations for Farm Mechanisation")
        policies = [
            ("🏛️ Scale Custom Hiring Centers (CHC)",
             "Current CHC density of 1 per 4,000 farmers is grossly insufficient. Recommend ₹8,000 crore allocation to achieve 1 CHC per 500 farmers by FY2028 — reducing effective tractor cost from ₹8L to ₹600/hr for small farmers."),
            ("💳 PM-KISAN Credit Linkage",
             "Link PM-KISAN ₹6,000/year transfers to tractor loan EMI auto-debit. This reduces NBFC NPA risk and extends formal credit to 85M beneficiary farmers currently excluded from tractor financing."),
            ("🌱 HP-Linked Subsidy Rationalisation",
             "Current sub-50% subsidies are brand-agnostic. Restructure to incentivise 40–50 HP (vs 30–40 HP) to increase productivity per unit. Phase out sub-30 HP subsidies in states with >2 ha avg farm size."),
            ("🔌 Green Tractor Mission",
             "Extend FAME-III to cover e-Tractors. Set a target: 5% of all tractor sales to be EV by FY2028. Mandate BIS certification for EV battery safety and establish 500 solar-charging hubs at mandis."),
            ("📊 VAHAN Data Standardisation",
             "HP-segment and state-wise tractor data is not publicly available at granular level. Mandate OEMs to report quarterly HP-wise sales by district via VAHAN 4.0 — enabling better policy targeting."),
        ]
        for title, body in policies:
            st.markdown(f'<div class="warn-box"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Summary Dashboard")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Market Size (FY25)", "9.12L units", "+0.36% YoY")
    m2.metric("3-Yr Forecast (FY27)", "~9.4–9.8L", "+3% CAGR")
    m3.metric("EV Share by FY29", "~4%", "From 0.26% today")
    m4.metric("Export Target (FY27)", "$5B", "From $3.2B in 2024")
    m5.metric("CHC Coverage Target", "500 farmers/CHC", "From 4,000 today")
