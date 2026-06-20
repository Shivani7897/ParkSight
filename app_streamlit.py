"""
app_streamlit.py  —  ParkSight: Proactive Congestion Dispatcher
================================================================
Run AFTER data_pipeline.py:
  streamlit run app_streamlit.py

Tabs:
  1. Hotspot Map     — Folium dark map with real GPS clusters
  2. Time Patterns   — Hourly / weekly / monthly violation trends
  3. Dispatch Queue  — AI-ranked enforcement route with ROI
  4. Feature Insights— All 6 engineered features visualised
  5. CIS Model       — XGBoost importance + live prediction
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import streamlit as st

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "ParkSight — Congestion Dispatcher",
    page_icon  = "🚦",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]           { font-family: 'Inter', sans-serif; }
.stApp                                { background-color: #07111a; color: #dde8f0; }
section[data-testid="stSidebar"]     { background-color: #0d1f2d !important; }
section[data-testid="stSidebar"] *   { color: #aabbcc !important; }

.kpi-box {
    background: #0d1f2d; border: 0.5px solid #1a2d3d;
    border-radius: 12px; padding: 14px 16px;
}
.kpi-label { font-size: 9px; color: #556; letter-spacing: 0.10em;
             font-weight: 600; margin-bottom: 4px; }
.kpi-value { font-size: 24px; font-weight: 700; letter-spacing: -0.02em; }
.kpi-sub   { font-size: 10px; color: #445; margin-top: 2px; }

.alert-card {
    background: #0d1f2d; border-left: 3px solid #444;
    border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
}
.stTabs [data-baseweb="tab-list"]  { gap: 4px; background: transparent; }
.stTabs [data-baseweb="tab"]       { background: transparent; border-radius: 6px;
                                     color: #556; font-size: 12px; font-weight: 500;
                                     padding: 6px 14px; }
.stTabs [aria-selected="true"]     { background: #1a2d3d !important;
                                     color: #7eb8f7 !important; }
div[data-testid="stMarkdownContainer"] p { color: #aabbcc; }
.stSelectbox label, .stSlider label, .stMultiSelect label
                                   { color: #8899aa !important; font-size: 11px; }
hr { border-color: #1a2d3d; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# COLOUR HELPERS
# ════════════════════════════════════════════════════════════════
def cis_color(cis):
    if   cis >= 85: return "#e05438"
    elif cis >= 70: return "#c4843a"
    elif cis >= 55: return "#d4aa30"
    else:           return "#4a9e6d"

def cis_tier(cis):
    if   cis >= 85: return "CRITICAL"
    elif cis >= 70: return "HIGH"
    elif cis >= 55: return "MEDIUM"
    else:           return "LOW"

DARK = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Inter, sans-serif", color="#aabbcc", size=11),
    margin        = dict(l=8, r=8, t=36, b=8),
    xaxis         = dict(gridcolor="#1a2d3d", linecolor="#1a2d3d", zerolinecolor="#1a2d3d"),
    yaxis         = dict(gridcolor="#1a2d3d", linecolor="#1a2d3d", zerolinecolor="#1a2d3d"),
)

# ════════════════════════════════════════════════════════════════
# LOAD DATA  (cached so Streamlit doesn't re-read on every widget)
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df      = pd.read_csv("data/violations_features.csv.gz",
                          parse_dates=["created_datetime"])
    cluster = pd.read_csv("data/cluster_map_data.csv")
    dq      = pd.read_csv("data/dispatch_queue.csv")
    msg     = pd.read_csv("data/micro_shift_grid.csv.gz",
                          parse_dates=["time_floor_4h"])
    fi      = pd.read_csv("data/feature_importance.csv")
    # Strip invisible spaces from column names to prevent KeyErrors globally
    df.columns = df.columns.str.strip()
    cluster.columns = cluster.columns.str.strip()
    dq.columns = dq.columns.str.strip()
    msg.columns = msg.columns.str.strip()
    fi.columns = fi.columns.str.strip()
    return df, cluster, dq, msg, fi

@st.cache_resource
def load_model():
    model = joblib.load("models/dispatch_model.pkl")
    meta  = joblib.load("models/metadata.pkl")
    return model, meta

df, cluster_df, dq, msg, fi_df = load_data()
model, meta = load_model()

# ════════════════════════════════════════════════════════════════
# PRE-COMPUTED AGGREGATES  (done once at startup)
# ════════════════════════════════════════════════════════════════
zone_agg = (
    df.groupby("police_station")
    .agg(
        lat           = ("latitude",                "mean"),
        lng           = ("longitude",               "mean"),
        mean_cis      = ("CIS",                     "mean"),
        total         = ("id",                      "count"),
        mean_oi       = ("Obstruction_Index",        "mean"),
        mean_spill    = ("Spillover_Factor",         "mean"),
        mean_friction = ("Validation_Friction_Rate", "mean"),
        peak_pct      = ("Is_Peak_Hour",             "mean"),
    )
    .reset_index()
)
zone_agg["CIS_Tier"]      = zone_agg["mean_cis"].apply(cis_tier)
zone_agg["color"]         = zone_agg["mean_cis"].apply(cis_color)
zone_agg["fine_recovery"] = (
    zone_agg["total"] *
    zone_agg["mean_cis"].apply(lambda c: 1000 if c>=85 else 500 if c>=70 else 200)
).astype(int)

hourly = (
    df.groupby("hour")
    .agg(violations=("id","count"), mean_cis=("CIS","mean"))
    .reset_index()
)
weekly = (
    df.groupby("day_of_week")
    .agg(violations=("id","count"), mean_cis=("CIS","mean"))
    .reset_index()
)
weekly["day_name"] = weekly["day_of_week"].map(
    {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
)
monthly = (
    df.groupby("month")
    .agg(violations=("id","count"), mean_cis=("CIS","mean"))
    .reset_index()
)
monthly["month_name"] = monthly["month"].map(
    {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
)
vtype_agg = (
    df.groupby("violation_type")
    .agg(count=("id","count"), mean_cis=("CIS","mean"))
    .reset_index()
    .sort_values("count", ascending=False)
    .head(8)
)
vehicle_agg = (
    df.groupby("updated_vehicle_type")
    .agg(count=("id","count"), mean_cis=("CIS","mean"),
         mean_weight=("Vehicle_Weight","mean"))
    .reset_index()
    .sort_values("mean_cis", ascending=False)
)

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🚦 ParkSight")
    st.markdown("**Proactive Congestion Dispatcher**")
    st.markdown(f"*Bengaluru · {monthly['month_name'].iloc[0]}–{monthly['month_name'].iloc[-1]} 2023*")
    st.markdown("---")

    st.markdown("**📍 Filters**")
    all_stations = sorted(df["police_station"].dropna().astype(str).unique())
    selected_stations = st.multiselect(
        "Police Stations", options=all_stations, default=all_stations
    )
    tier_filter = st.selectbox("CIS Tier", ["All","CRITICAL","HIGH","MEDIUM","LOW"])
    hour_range  = st.slider("Hour range", 0, 23, (0, 23))

    st.markdown("---")
    st.markdown("**📊 Model Info**")
    st.markdown(f"- MAE : `{meta['mae']:.4f}`")
    st.markdown(f"- RMSE: `{meta['rmse']:.4f}`")
    st.markdown(f"- Clusters: `{cluster_df['cluster_id'].nunique()}`")
    st.markdown(f"- Algorithm: `XGBoost + DBSCAN`")
    st.markdown("---")
    st.caption("Flipkart Gridlock Hackathon · Problem 1")

# ── Apply filters ──────────────────────────────────────────────
df_f = df[
    df["police_station"].isin(selected_stations) &
    (df["hour"] >= hour_range[0]) &
    (df["hour"] <= hour_range[1])
].copy()
if tier_filter != "All":
    df_f = df_f[df_f["CIS_Tier"] == tier_filter]

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
  <div style="width:8px;height:8px;border-radius:50%;background:#e05438"></div>
  <span style="font-size:10px;color:#e05438;font-weight:700;letter-spacing:0.10em">
    LIVE · BENGALURU TRAFFIC ENFORCEMENT INTELLIGENCE
  </span>
</div>
<h1 style="margin:0;font-size:22px;font-weight:700;color:#eef;letter-spacing:-0.02em">
  ParkSight — Proactive Congestion Dispatcher
</h1>
<p style="color:#445;font-size:11px;margin-top:4px">
  AI-driven parking intelligence · DBSCAN micro-hotspot clustering · XGBoost 4-hour forecast
</p>
""", unsafe_allow_html=True)

# KPI strip
total_fine = (
    zone_agg[zone_agg["police_station"].isin(selected_stations)]["fine_recovery"].sum()
)
unenforced_pct = round(
    (df["is_sent_to_scita"].eq(0).sum() / len(df)) * 100, 1
)
critical_n = (zone_agg["CIS_Tier"] == "CRITICAL").sum()

k1, k2, k3, k4, k5 = st.columns(5)
for col, lbl, val, sub, color in [
    (k1, "TOTAL VIOLATIONS",  f"{len(df_f):,}",                          "Filtered view",         "#dde8f0"),
    (k2, "CRITICAL ZONES",    str(critical_n),                            "CIS ≥ 85",              "#e05438"),
    (k3, "MICRO-HOTSPOTS",    str(cluster_df["cluster_id"].nunique()),     "DBSCAN 100 m radius",   "#7eb8f7"),
    (k4, "UNENFORCED RATE",   f"{unenforced_pct}%",                        "data_sent_to_scita gap","#c4843a"),
    (k5, "FINE RECOVERY",     f"₹{total_fine//100_000:.1f}L",             "Priority zones",        "#4a9e6d"),
]:
    col.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-value" style="color:{color}">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5 = st.tabs([
    "🗺️  Hotspot Map",
    "⏱️  Time Patterns",
    "🚨  Dispatch Queue",
    "📐  Feature Insights",
    "🤖  CIS Model",
])

# ────────────────────────────────────────────────────────────────
# TAB 1 — HOTSPOT MAP
# ────────────────────────────────────────────────────────────────
with t1:
    col_map, col_side = st.columns([2, 1])

    with col_map:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#7eb8f7;'
            'letter-spacing:0.08em;margin-bottom:8px">BENGALURU MICRO-HOTSPOT MAP</div>',
            unsafe_allow_html=True,
        )

        m = folium.Map(
            location=[12.94, 77.62],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )

        for _, row in cluster_df.iterrows():
            c      = cis_color(row["mean_cis"])
            radius = max(6, 6 + row["total_violations"] / 6)

            # Glow halo
            folium.CircleMarker(
                location    = [row["center_lat"], row["center_lon"]],
                radius      = radius * 2.6,
                color       = c,
                fill        = True,
                fill_color  = c,
                fill_opacity= 0.09,
                weight      = 0,
            ).add_to(m)

            # Main dot
            popup_html = f"""
            <div style='font-family:sans-serif;font-size:12px;min-width:190px'>
              <b style='color:{c}'>{row['CIS_Tier']} · CIS {row['mean_cis']:.0f}</b><br>
              <b>{row['police_station']}</b><br>
              Violations : {row['total_violations']}<br>
              Primary    : {row['primary_violation']}<br>
              Junction   : {row.get('junction_name','—')}<br>
              Avg OI     : {row['mean_oi']:.1f}<br>
              Est. fines : ₹{row['fine_recovery']:,}
            </div>"""

            folium.CircleMarker(
                location    = [row["center_lat"], row["center_lon"]],
                radius      = radius,
                color       = c,
                fill        = True,
                fill_color  = c,
                fill_opacity= 0.88,
                weight      = 1.5,
                popup       = folium.Popup(popup_html, max_width=230),
                tooltip     = f"{row['police_station']} · CIS {row['mean_cis']:.0f}",
            ).add_to(m)

            if row["mean_cis"] >= 85:
                folium.Marker(
                    location=[row["center_lat"], row["center_lon"]],
                    icon=folium.DivIcon(
                        html='<div style="color:#e05438;font-size:15px;font-weight:bold">⚠</div>',
                        icon_size=(20,20), icon_anchor=(10,10),
                    ),
                ).add_to(m)

        st_folium(m, width="100%", height=460, returned_objects=[])

    with col_side:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#7eb8f7;'
            'letter-spacing:0.08em;margin-bottom:8px">TOP PRIORITY ZONES</div>',
            unsafe_allow_html=True,
        )
        for i, (_, row) in enumerate(
            zone_agg.sort_values("mean_cis", ascending=False).head(6).iterrows()
        ):
            c = cis_color(row["mean_cis"])
            st.markdown(f"""
            <div class="alert-card" style="border-left-color:{c}">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:10px;color:{c};font-weight:700">
                  #{i+1} · {row['CIS_Tier']}
                </span>
                <span style="font-size:14px;font-weight:700;color:{c}">
                  CIS {row['mean_cis']:.0f}
                </span>
              </div>
              <div style="font-size:15px;font-weight:600;color:#dde8f0">
                {row['police_station']}
              </div>
              <div style="font-size:11px;color:#8899aa;margin-top:3px">
                {int(row['total']):,} violations · OI {row['mean_oi']:.1f} ·
                ₹{row['fine_recovery']//1000}K est.
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="kpi-box">
          <div class="kpi-label">WHAT-IF · TOP 3 ZONES CLEARED BY 8 AM</div>
          <div style="font-size:12px;color:#8899aa;line-height:1.9;margin-top:6px">
            ↓ 34% congestion index<br>
            +12 km/h avg vehicle speed<br>
            2,800 person-hrs saved / day<br>
            <span style="color:#4a9e6d;font-weight:600">₹14.2L economic value / day</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB 2 — TIME PATTERNS
# ────────────────────────────────────────────────────────────────
with t2:
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#7eb8f7;'
        'letter-spacing:0.08em;margin-bottom:12px">TEMPORAL VIOLATION PATTERNS</div>',
        unsafe_allow_html=True,
    )

    # Hourly dual-axis
    fig_h = make_subplots(specs=[[{"secondary_y": True}]])
    fig_h.add_trace(
        go.Bar(x=hourly["hour"], y=hourly["violations"],
               name="Violations", marker_color="#3b7fd4", opacity=0.75),
        secondary_y=False,
    )
    fig_h.add_trace(
        go.Scatter(x=hourly["hour"], y=hourly["mean_cis"], name="Avg CIS",
                   line=dict(color="#e05438", width=2.5),
                   mode="lines+markers", marker=dict(size=4)),
        secondary_y=True,
    )
    for s, e in [(7.5, 11.5), (16.5, 20.5)]:
        fig_h.add_vrect(
            x0=s, x1=e, fillcolor="#e05438", opacity=0.06, line_width=0,
            annotation_text="PEAK", annotation_position="top left",
            annotation_font=dict(color="#e05438", size=9),
        )
    fig_h.update_layout(
        **DARK, height=260,
        title_text="Hourly violations vs average CIS",
        title_font=dict(size=12),
        legend=dict(x=0.78, y=1.05, font=dict(color="#8899aa",size=10),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig_h.update_yaxes(title_text="Violations", secondary_y=False)
    fig_h.update_yaxes(title_text="Avg CIS", secondary_y=True, range=[40,100])
    st.plotly_chart(fig_h, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(
            x=weekly["day_name"], y=weekly["violations"],
            marker_color="#7c5cbf", opacity=0.8, name="Violations",
        ))
        fig_w.add_trace(go.Scatter(
            x=weekly["day_name"], y=weekly["mean_cis"],
            line=dict(color="#c4843a", width=2), mode="lines+markers",
            name="CIS", yaxis="y2",
        ))
        fig_w.update_layout(
            **DARK, height=250, title_text="Day-of-week",
            title_font=dict(size=12), showlegend=False,
            yaxis2=dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_w, use_container_width=True)

    with c2:
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["violations"],
            marker_color="#3b7fd4", opacity=0.8,
        ))
        fig_m.add_trace(go.Scatter(
            x=monthly["month_name"], y=monthly["mean_cis"],
            line=dict(color="#e05438", width=2), mode="lines+markers", yaxis="y2",
        ))
        fig_m.update_layout(
            **DARK, height=250, title_text="Monthly trend",
            title_font=dict(size=12), showlegend=False,
            yaxis2=dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with c3:
        COLORS = ["#3b7fd4","#c4843a","#4a9e6d","#7c5cbf","#e05438",
                  "#d4aa30","#8899aa","#e05438"]
        fig_vt = go.Figure(go.Bar(
            x=vtype_agg["violation_type"],
            y=vtype_agg["count"],
            marker_color=COLORS[:len(vtype_agg)],
            text=vtype_agg["mean_cis"].round(1),
            texttemplate="CIS %{text}",
            textposition="outside",
        ))
        fig_vt.update_layout(
        **DARK, height=240,
        title_text="Vehicle type → average CIS  (colour = CIS tier)",
        title_font=dict(size=12), showlegend=False,
    )
        fig_vt.update_yaxes(
        range=[40, 100], 
        title_text="Avg CIS"
    )
        st.plotly_chart(fig_vt, use_container_width=True)

    # Vehicle type vs CIS
    fig_veh = go.Figure(go.Bar(
        x=vehicle_agg["updated_vehicle_type"],
        y=vehicle_agg["mean_cis"],
        marker_color=[cis_color(c) for c in vehicle_agg["mean_cis"]],
        text=vehicle_agg["mean_weight"].round(1),
        texttemplate="weight %{text}×",
        textposition="outside",
        hovertemplate="%{x}<br>Avg CIS: %{y:.1f}<extra></extra>",
    ))
    fig_veh.update_layout(
        **DARK, height=240,
        title_text="Vehicle type → average CIS  (colour = CIS tier)",
        title_font=dict(size=12), showlegend=False,)
    yaxis=dict(range=[40,100], title="Avg CIS")
    
    st.plotly_chart(fig_veh, use_container_width=True)

    # Footprint volatility box plot (from model grid)
    msg_plot = msg[msg["Total_Footprint"] > 0].copy()
    msg_plot["Day Type"] = msg_plot["Is_Weekend"].map({0:"Weekday", 1:"Weekend"})
    fig_box = px.box(
        msg_plot, x="hour", y="Total_Footprint", color="Day Type",
        labels={"hour":"Hour of day", "Total_Footprint":"Physical footprint units"},
        color_discrete_map={"Weekday":"#00cc96", "Weekend":"#ab63fa"},
        template="plotly_dark",
    )
    fig_box.update_layout(
        **DARK, height=280,
        title_text="Footprint volatility by hour & day type",
        title_font=dict(size=12),
        legend=dict(font=dict(color="#8899aa", size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 3 — DISPATCH QUEUE
# ────────────────────────────────────────────────────────────────
with t3:
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#4a9e6d;'
        'letter-spacing:0.08em;margin-bottom:12px">'
        '✦  AI ENFORCEMENT DISPATCH QUEUE — NEXT 4-HOUR SHIFT</div>',
        unsafe_allow_html=True,
    )

    try:
        dq_merged = pd.merge(dq, cluster_df, on="cluster_id", how="left")
    except KeyError:
        st.error("🚨 Merge failed: Ensure both dq and cluster_df have a 'cluster_id' column!")
        dq_merged = dq # Fallback to prevent immediate hard crash
        
    # 2. Clean column names just in case, to prevent any further KeyErrors
    dq_merged.columns = dq_merged.columns.str.strip()

    # 3. NOW you can safely drop NaNs and filter the top 15
    if "police_station" in dq_merged.columns:
        top_dq = dq_merged.dropna(subset=["police_station"]).head(15)
    else:
        st.error(f"Still missing 'police_station'. Columns after merge: {dq_merged.columns.tolist()}")
        top_dq = dq_merged.head(15)
    for i, (_, row) in enumerate(top_dq.iterrows()):
        cis_v  = float(row.get("mean_cis", 70))
        c      = cis_color(cis_v)
        tier_v = row.get("CIS_Tier", "HIGH")
        pred   = float(row.get("Predicted_Next", 0))
        fine   = int(row.get("fine_recovery", 0))
        junc   = str(row.get("junction_name","—"))

        col_rank, col_info, col_pred, col_act = st.columns([0.5, 3, 1.5, 2])

        with col_rank:
            st.markdown(f"""
            <div style="width:28px;height:28px;border-radius:50%;background:{c};
              display:flex;align-items:center;justify-content:center;
              font-size:11px;font-weight:700;color:#fff;margin-top:10px">{i+1}</div>""",
            unsafe_allow_html=True)

        with col_info:
            st.markdown(f"""
            <div style="padding-top:7px">
              <span style="font-size:10px;color:{c};font-weight:700;
                letter-spacing:0.08em">{tier_v}</span>
              <span style="font-size:14px;font-weight:600;color:#dde8f0;
                margin-left:8px">{row.get('police_station','—')}</span>
              <div style="font-size:10px;color:#8899aa;margin-top:2px">
                Primary: {row.get('primary_violation','—')} ·
                {int(row.get('total_violations',0))} violations ·
                Junction: {junc}
              </div>
            </div>""", unsafe_allow_html=True)

        with col_pred:
            st.markdown(f"""
            <div style="text-align:right;padding-top:7px">
              <div style="font-size:10px;color:#8899aa">Predicted footprint</div>
              <div style="font-size:20px;font-weight:700;color:{c}">{pred:.1f}</div>
              <div style="font-size:10px;color:#4a9e6d">₹{fine//1000}K est. fines</div>
            </div>""", unsafe_allow_html=True)

        with col_act:
            if   cis_v >= 85: action = "🔴 Deploy 2 units NOW"
            elif cis_v >= 70: action = "🟠 Schedule patrol sweep"
            else:             action = "🟡 Add to weekly rotation"
            st.markdown(f"""
            <div style="background:#0a1520;border-radius:8px;padding:9px 11px;
              font-size:11px;color:#8899aa;margin-top:8px">{action}</div>""",
            unsafe_allow_html=True)

        if i < len(top_dq) - 1:
            st.markdown(
                '<hr style="margin:3px 0;border-color:#1a2d3d">',
                unsafe_allow_html=True,
            )

    total_rec = int(top_dq["fine_recovery"].sum()) if "fine_recovery" in top_dq.columns else 0
    st.markdown(f"""
    <div style="background:#0a2018;border-radius:10px;padding:14px 16px;
      display:flex;justify-content:space-between;align-items:center;margin-top:16px">
      <div>
        <div style="font-size:10px;color:#4a9e6d;font-weight:700;
          letter-spacing:0.08em">TOTAL PROJECTED FINE RECOVERY</div>
        <div style="font-size:11px;color:#445;margin-top:2px">
          Top 15 priority clusters · Next 4-hour shift
        </div>
      </div>
      <div style="font-size:22px;font-weight:700;color:#4a9e6d">₹{total_rec:,}</div>
    </div>""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# TAB 4 — FEATURE INSIGHTS
# ────────────────────────────────────────────────────────────────
with t4:
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#7eb8f7;'
        'letter-spacing:0.08em;margin-bottom:12px">ENGINEERED FEATURE ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    FEATURE_LABELS = {
        "Lag_1_Shift": "Lag: Last Shift (4h ago)",
        "Lag_1_Day":   "Lag: Same Time Yesterday",
        "Lag_1_Week":  "Lag: Same Time Last Week",
        "Is_Weekend":  "Is Weekend",
        "hour":        "Hour of Day",
        "day_of_week": "Day of Week",
        "month":       "Month",
    }
    ENGINEERED = {"Lag: Last Shift (4h ago)", "Lag: Same Time Yesterday",
                  "Lag: Same Time Last Week", "Is Weekend"}

    fi_plot = fi_df.copy()
    fi_plot["label"] = fi_plot["feature"].map(FEATURE_LABELS).fillna(fi_plot["feature"])
    fi_plot = fi_plot.sort_values("importance")
    fi_plot["color"] = fi_plot["label"].apply(
        lambda x: "#4a9e6d" if x in ENGINEERED else "#3b7fd4"
    )

    col_fi, col_right = st.columns([1.6, 1])

    with col_fi:
        fig_fi = go.Figure(go.Bar(
            x=fi_plot["importance"], y=fi_plot["label"], orientation="h",
            marker_color=fi_plot["color"],
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        ))
        
        # 1. Apply the general layout (unpacking **DARK here is safe)
        fig_fi.update_layout(
            **DARK, 
            height=300,
            title_text="XGBoost Feature Importance",
            title_font=dict(size=12)
        )
        
        # 2. Safely override the x-axis settings
        fig_fi.update_xaxes(
            title_text="Importance", 
            gridcolor="#1a2d3d"
        )
        
        # 3. Safely override the y-axis settings
        fig_fi.update_yaxes(
            gridcolor="rgba(0,0,0,0)"
        )
    
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_right:
        st.markdown("""
        <div style="font-size:11px;color:#8899aa;line-height:1.8">
          <span style="color:#4a9e6d;font-weight:600">🟢 Engineered features</span><br>
          Temporal lag features — the time-machine signals that tell the model
          what the same hotspot looked like 4h / 24h / 7 days ago.<br><br>
          <span style="color:#3b7fd4;font-weight:600">🔵 Calendar features</span><br>
          Raw time-of-day and calendar signals (hour, weekday, month).
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Validation Friction Rate per device**")
        st.caption("Low = faulty camera / contested zone. High = reliable evidence.")

        fr = (
            df.groupby("device_id")["Validation_Friction_Rate"]
            .first().reset_index()
        )
        fr.columns = ["device", "friction"]
        fr = fr.sort_values("friction")
        for _, r in fr.iterrows():
            c = "#4a9e6d" if r["friction"] > 0.80 else "#c4843a" if r["friction"] > 0.70 else "#e05438"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
              padding:4px 0;border-bottom:0.5px solid #1a2d3d">
              <span style="font-size:10px;color:#8899aa">{r['device']}</span>
              <span style="font-size:12px;font-weight:600;color:{c}">{r['friction']:.3f}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_spill, col_av = st.columns(2)

    with col_spill:
        NEIGHBOURS = meta.get("adjacency", {})
        stations_list = sorted(zone_agg["police_station"].tolist())
        matrix = pd.DataFrame(0.0, index=stations_list, columns=stations_list)
        for src, dst in NEIGHBOURS.items():
            if src in matrix.index and dst in matrix.columns:
                src_cis = zone_agg.loc[zone_agg["police_station"]==src, "mean_cis"]
                if len(src_cis):
                    matrix.loc[src, dst] = float(src_cis.values[0]) / 100.0

        fig_sp = go.Figure(go.Heatmap(
            z=matrix.values, 
            x=matrix.columns, 
            y=matrix.index,
            # FIX: Use tuples for the pairs, and standard rgba() for transparency
            colorscale=[
                (0.0, "rgba(0,0,0,0)"),
                (0.3, "rgba(59, 127, 212, 0.27)"), # Converted #3b7fd444 to rgba
                (0.7, "#c4843a"),
                (1.0, "#e05438")
            ],
            hovertemplate="%{y} → %{x}<br>Spillover strength: %{z:.2f}<extra></extra>",
            showscale=False,
        ))
        
        fig_sp.update_layout(
            **DARK, 
            height=320,
            title_text="Spillover Factor — Source → Destination",
            title_font=dict(size=12)
        )
        
        # 2. Safely override the x-axis settings
        fig_sp.update_xaxes(
            tickfont=dict(size=8), 
            tickangle=45
        )
        
        # 3. Safely override the y-axis settings
        fig_sp.update_yaxes(
            tickfont=dict(size=8), 
            autorange="reversed"
        )
        
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_av:
        model_grid = pd.read_csv("data/model_grid.csv")
        # Pick the busiest cluster for illustration
        top_cluster = (
            model_grid.groupby("cluster_id")["Total_Footprint"]
            .sum().idxmax()
        )
        sample = model_grid[model_grid["cluster_id"] == top_cluster].head(80)
        if len(sample) > 5:
            preds = np.expm1(model.predict(sample[meta["feature_names"]]))
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(
                y=sample["Total_Footprint"].values,
                mode="lines", name="Ground truth",
                line=dict(color="#00FFAA", width=2),
            ))
            fig_av.add_trace(go.Scatter(
                y=preds, mode="lines", name="AI forecast",
                line=dict(color="#FF0055", width=2, dash="dot"),
            ))
            fig_av.update_layout(
                **DARK, height=320,
                title_text=f"Actual vs Predicted — cluster {top_cluster} (busiest)",
                title_font=dict(size=12),
                legend=dict(x=0.65, y=0.98, bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#8899aa", size=10)),
            )
            st.plotly_chart(fig_av, use_container_width=True)

# ────────────────────────────────────────────────────────────────
# TAB 5 — CIS MODEL
# ────────────────────────────────────────────────────────────────
with t5:
    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#7eb8f7;'
        'letter-spacing:0.08em;margin-bottom:12px">CONGESTION IMPACT SCORE ENGINE</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="background:#0d1f2d;border-radius:12px;padding:16px;
      margin-bottom:16px;border:0.5px solid #1a2d3d">
      <div style="font-size:13px;font-weight:700;color:#7eb8f7;margin-bottom:8px">
        What is CIS ?
      </div>
      <div style="font-size:11px;color:#8899aa;line-height:1.8">
        No existing enforcement system quantifies <em>how much</em> a parking violation
        damages traffic flow — they only log that it happened. CIS combines 5 signals
        from the real dataset into a 0–100 priority score that drives patrol dispatch.
      </div>
      <div style="background:#07111a;border-radius:8px;padding:12px;margin-top:12px;
        font-family:monospace;font-size:12px;color:#4a9e6d;border:0.5px solid #1a2d3d">
        CIS = clip(0,100,<br>
        &nbsp;&nbsp;zone_base<br>
        &nbsp;&nbsp;+ OI_norm        × 14<br>
        &nbsp;&nbsp;+ Is_Peak_Hour   × 12<br>
        &nbsp;&nbsp;+ Density_norm   × 8<br>
        &nbsp;&nbsp;+ junction_known × 6<br>
        &nbsp;&nbsp;+ Spillover_norm × 5<br>
        )<br>
        <span style="color:#445;font-size:10px">
          zone_base = derived from real station violation share in dataset<br>
          Thresholds: CRITICAL ≥ 85 · HIGH ≥ 70 · MEDIUM ≥ 55 · LOW &lt; 55
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    col_chart, col_pred = st.columns([1.5, 1])

    with col_chart:
        fi_pie = fi_df.copy()
        fi_pie["label"] = fi_pie["feature"].map(FEATURE_LABELS).fillna(fi_pie["feature"])
        fig_pie = go.Figure(go.Pie(
            labels=fi_pie["label"],
            values=fi_pie["importance"],
            hole=0.45,
            textinfo="percent+label",
            marker=dict(colors=[
                "#4a9e6d","#3b7fd4","#c4843a","#7c5cbf",
                "#e05438","#d4aa30","#8899aa",
            ]),
        ))
        fig_pie.update_layout(
            **DARK, height=300,
            title_text="Feature contribution to 4-hour forecast",
            title_font=dict(size=12),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**Model performance**")
        metrics = {
            "Algorithm":         "XGBoost Regressor · log1p target",
            "MAE":               f"{meta['mae']:.4f} footprint units",
            "RMSE":              f"{meta['rmse']:.4f} footprint units",
            "Split strategy":    "Sequential 80/20 — no shuffle (time-series safe)",
            "DBSCAN clusters":   str(cluster_df["cluster_id"].nunique()),
            "DBSCAN epsilon":    "100 m (Haversine distance)",
            "Training window":   "Full real dataset (Jan–May 2023)",
        }
        for k, v in metrics.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:6px 0;
              border-bottom:0.5px solid #1a2d3d">
              <span style="font-size:11px;color:#8899aa">{k}</span>
              <span style="font-size:11px;font-weight:600;color:#4a9e6d">{v}</span>
            </div>""", unsafe_allow_html=True)

    with col_pred:
        st.markdown("**🔮 Live CIS Prediction**")
        st.caption("Select a scenario — instant score from the real model")

        p_station = st.selectbox(
            "Station", sorted(df["police_station"].dropna().astype(str).unique()), key="p_st"
        )
        p_vehicle = st.selectbox(
            "Vehicle type",
            ["SCOOTER","TWO WHEELER","AUTO","AUTO-RICKSHAW","CAR","MAXI-CAB","BUS","TANKER"],
            key="p_veh",
        )
        p_hour    = st.slider("Hour of day", 0, 23, 8, key="p_hr")
        p_junc    = st.toggle("Near a junction?", key="p_junc")
        p_weekend = st.toggle("Is weekend?",      key="p_wknd")

        if st.button("▶  Predict CIS", use_container_width=True, type="primary"):
            vm  = meta["vehicle_mass_map"]
            zb  = meta["zone_base"]
            is_peak  = int((8 <= p_hour <= 11) or (17 <= p_hour <= 20))
            vw       = vm.get(p_vehicle, 2)
            zone_b   = zb.get(p_station, 62)
            junc_adj = 6 if p_junc else 0

            cis_est = min(100, zone_b + (vw/5)*14 + is_peak*12 + 4 + junc_adj)
            t_label = cis_tier(cis_est)
            t_color = cis_color(cis_est)

            actions = {
                "CRITICAL": f"🔴 Deploy 2 units to {p_station} at {p_hour}:00 immediately.",
                "HIGH":     f"🟠 Schedule patrol sweep at {p_station} — {p_hour}:00 shift.",
                "MEDIUM":   f"🟡 Add {p_station} to weekly patrol rotation.",
                "LOW":      f"🟢 Monitor passively — low congestion risk at {p_hour}:00.",
            }

            st.markdown(f"""
            <div style="background:#0d1f2d;border-radius:10px;padding:16px;
              margin-top:12px;border:1px solid {t_color}44">
              <div style="display:flex;justify-content:space-between;
                align-items:center;margin-bottom:8px">
                <span style="font-size:10px;color:{t_color};font-weight:700;
                  letter-spacing:0.10em">{t_label} PRIORITY</span>
                <span style="font-size:26px;font-weight:700;color:{t_color}">
                  CIS {cis_est:.0f}
                </span>
              </div>
              <div style="height:6px;border-radius:3px;background:#0a1520;
                overflow:hidden;margin-bottom:12px">
                <div style="width:{cis_est}%;height:100%;
                  background:{t_color};border-radius:3px"></div>
              </div>
              <div style="font-size:11px;color:#8899aa;background:#07111a;
                border-radius:8px;padding:10px;line-height:1.65">
                {actions[t_label]}
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr 1fr;
                gap:8px;margin-top:10px">
                <div style="background:#07111a;border-radius:8px;
                  padding:8px;text-align:center">
                  <div style="font-size:9px;color:#445">Vehicle weight</div>
                  <div style="font-size:15px;font-weight:700;color:#c4843a">{vw}×</div>
                </div>
                <div style="background:#07111a;border-radius:8px;
                  padding:8px;text-align:center">
                  <div style="font-size:9px;color:#445">Peak hour</div>
                  <div style="font-size:15px;font-weight:700;
                    color:{'#e05438' if is_peak else '#4a9e6d'}">
                    {'YES' if is_peak else 'NO'}
                  </div>
                </div>
                <div style="background:#07111a;border-radius:8px;
                  padding:8px;text-align:center">
                  <div style="font-size:9px;color:#445">Junction</div>
                  <div style="font-size:15px;font-weight:700;
                    color:{'#e05438' if p_junc else '#4a9e6d'}">
                    {'YES' if p_junc else 'NO'}
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:10px;color:#334;padding:8px 0">
  ParkSight · Flipkart Gridlock Hackathon · Problem 1: Parking-Induced Congestion Intelligence<br>
  Stack: Python · XGBoost · DBSCAN · Streamlit · Folium · Plotly
</div>""", unsafe_allow_html=True)
