"""
Siemens Asset Digital Twin & Predictive Maintenance (PdM) Interactive Application
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Self-contained path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.pdm_engine import classify_iso_vibration, calculate_health_index, load_models, FEATURE_COLS

st.set_page_config(
    page_title="Siemens Asset Digital Twin & PdM",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Contrast Fully Responsive Industrial SCADA Theme
st.markdown("""
<style>
    /* Global Container */
    .stApp, [data-testid="stAppViewContainer"], body {
        background-color: #0b1120 !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    p, span, label, div {
        color: #e2e8f0;
    }
    
    /* Top KPI Cards */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    .kpi-title {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        color: #ffffff;
        font-size: clamp(1.2rem, 1.6vw, 1.7rem);
        font-weight: 800;
        margin-top: 3px;
        white-space: nowrap;
    }
    .kpi-sub {
        color: #38bdf8;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Industrial SCADA Telemetry Cards */
    .scada-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
    }
    .scada-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .scada-name {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .scada-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .scada-val-row {
        display: flex;
        align-items: baseline;
        gap: 6px;
        margin-bottom: 8px;
    }
    .scada-val {
        color: #ffffff;
        font-size: clamp(1.5rem, 2vw, 2.1rem);
        font-weight: 800;
        line-height: 1;
    }
    .scada-unit {
        color: #38bdf8;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .meter-track {
        width: 100%;
        height: 10px;
        background-color: #0f172a;
        border-radius: 5px;
        position: relative;
        overflow: hidden;
        border: 1px solid #334155;
    }
    .meter-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }
    .meter-scale {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Section Titles */
    .section-header {
        color: #38bdf8 !important;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Sidebar High-Contrast */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚙️ Siemens Asset Digital Twin & Predictive Maintenance (PdM)")
st.markdown("<p style='color: #94a3b8; font-size: 1.05rem; margin-bottom: 18px;'>Real-Time Machine Prognostics, ISO 10816-3 Vibration Severity & AI-Driven Fault Diagnostics</p>", unsafe_allow_html=True)

# Load cached data
@st.cache_data
def get_dataset():
    return load_data()

df_all = get_dataset()

# Sidebar: Controls
st.sidebar.markdown("### 🕹️ Asset Controls")
machine_list = sorted(df_all['Machine_ID'].unique())
selected_machine = st.sidebar.selectbox("Select Machine ID", options=machine_list, index=0)

line_list = ["All"] + sorted(list(df_all['Production_Line_ID'].unique()))
selected_line = st.sidebar.selectbox("Filter Production Line", options=line_list, index=0)

df_mach = df_all[df_all['Machine_ID'] == selected_machine]
if selected_line != "All":
    df_mach = df_mach[df_mach['Production_Line_ID'] == selected_line]

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Replay Timeline (Time Steps)")
st.sidebar.caption("Scrub through time to simulate live telemetry and observe how AI prognostics update as machine conditions change.")

total_records = len(df_mach)
max_idx = max(0, total_records - 1)

# Step slider with clear label
record_idx = st.sidebar.slider(
    "Timeline Step (Minute)",
    min_value=0,
    max_value=max_idx,
    value=0,
    help=f"Step 0 to {max_idx} represents {total_records:,} continuous minutes of sensor records for this machine."
)

current_record = df_mach.iloc[record_idx]
current_time_str = str(current_record.get('Timestamp', 'N/A'))

st.sidebar.markdown(f"""
<div style="background: #1e293b; padding: 10px 12px; border-radius: 8px; border: 1px solid #334155; margin-top: 6px;">
    <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">Selected Timestamp</div>
    <div style="color: #38bdf8; font-size: 0.95rem; font-weight: 700; margin-top: 2px;">📅 {current_time_str}</div>
    <div style="color: #cbd5e1; font-size: 0.8rem; margin-top: 4px;">Minute <b>{record_idx + 1:,}</b> of <b>{total_records:,}</b></div>
</div>
""", unsafe_allow_html=True)

# Load models
rul_model, ft_model, comp_model = load_models()

# Top KPIs Calculation
health_score = calculate_health_index(current_record)
iso_status, iso_color = classify_iso_vibration(current_record['RMS_Vibration'])

# Top KPIs Row (Crisp, High Contrast HTML + Fluid Typography)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Machine State</div>
        <div class="kpi-value">{current_record['Machine_Type']}</div>
        <div class="kpi-sub">{current_record['Machine_ID']} • Line {current_record['Production_Line_ID']}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Asset Health Index</div>
        <div class="kpi-value" style="color: {'#10b981' if health_score > 60 else '#f59e0b' if health_score > 30 else '#ef4444'};">{health_score:.1f}%</div>
        <div class="kpi-sub">Target: &ge; 75.0%</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">ISO 10816 Vibration</div>
        <div class="kpi-value" style="color: {iso_color};">{current_record['RMS_Vibration']:.2f} <span style="font-size: 0.9rem; color: #94a3b8;">mm/s</span></div>
        <div class="kpi-sub">{iso_status.split(':')[0]}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Predicted RUL</div>
        <div class="kpi-value" style="color: #38bdf8;">{current_record['RUL']:.1f} <span style="font-size: 0.9rem; color: #94a3b8;">hrs</span></div>
        <div class="kpi-sub">TTF: {current_record['TTF']:.1f} hrs</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    fail_prob = float(current_record.get('Failure_Probability', 0.15))
    fail_color = "#10b981" if fail_prob < 0.25 else "#f59e0b" if fail_prob < 0.5 else "#ef4444"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Failure Risk</div>
        <div class="kpi-value" style="color: {fail_color};">{fail_prob*100:.1f}%</div>
        <div class="kpi-sub">{'Normal' if fail_prob < 0.25 else '⚠️ Elevated Risk'}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main Section: Digital Twin Telemetry vs. Prognostics Radar
left_col, right_col = st.columns([1, 1])

def render_scada_meter(title, value, unit, min_val, max_val, warn_thresh, crit_thresh, is_inverse=False):
    """
    Renders an industrial SCADA telemetry card with dedicated bold number and horizontal calibrated meter bar.
    Guaranteed 100% visible at ANY screen resolution without text-gauge overlap.
    """
    pct = min(100.0, max(0.0, ((value - min_val) / max(0.01, max_val - min_val)) * 100.0))
    
    if not is_inverse:
        if value < warn_thresh:
            status_text = "NORMAL"
            badge_bg = "rgba(16, 185, 129, 0.15)"
            badge_border = "#10b981"
            badge_color = "#10b981"
            bar_color = "linear-gradient(90deg, #10b981, #34d399)"
        elif value < crit_thresh:
            status_text = "ELEVATED"
            badge_bg = "rgba(245, 158, 11, 0.15)"
            badge_border = "#f59e0b"
            badge_color = "#f59e0b"
            bar_color = "linear-gradient(90deg, #f59e0b, #fbbf24)"
        else:
            status_text = "CRITICAL ALARM"
            badge_bg = "rgba(239, 68, 68, 0.15)"
            badge_border = "#ef4444"
            badge_color = "#ef4444"
            bar_color = "linear-gradient(90deg, #ef4444, #f87171)"
    else:
        status_text = "NORMAL"
        badge_bg = "rgba(16, 185, 129, 0.15)"
        badge_border = "#10b981"
        badge_color = "#10b981"
        bar_color = "linear-gradient(90deg, #38bdf8, #0284c7)"

    return f"""
    <div class="scada-card">
        <div class="scada-header">
            <span class="scada-name">{title}</span>
            <span class="scada-badge" style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border};">{status_text}</span>
        </div>
        <div class="scada-val-row">
            <span class="scada-val">{value:.2f}</span>
            <span class="scada-unit">{unit}</span>
        </div>
        <div class="meter-track">
            <div class="meter-fill" style="width: {pct:.1f}%; background: {bar_color};"></div>
        </div>
        <div class="meter-scale">
            <span>{min_val} {unit}</span>
            <span>Warn: {warn_thresh} {unit}</span>
            <span>Max: {max_val} {unit}</span>
        </div>
    </div>
    """

with left_col:
    st.markdown("<div class='section-header'>🔬 Mechanical & Thermal Physical Telemetry</div>", unsafe_allow_html=True)
    
    # 1. Bearing Temperature
    st.markdown(
        render_scada_meter(
            "Bearing Temperature",
            current_record['Bearing_Temperature'],
            "°C", 0, 150, 70, 100
        ),
        unsafe_allow_html=True
    )
    
    # 2. RMS Vibration
    st.markdown(
        render_scada_meter(
            "RMS Vibration (ISO 10816-3)",
            current_record['RMS_Vibration'],
            "mm/s", 0, 10, 1.8, 4.5
        ),
        unsafe_allow_html=True
    )
    
    # 3. Motor Temperature
    st.markdown(
        render_scada_meter(
            "Motor Temperature",
            current_record['Motor_Temperature'],
            "°C", 0, 180, 90, 130
        ),
        unsafe_allow_html=True
    )
    
    # 4. Oil Particle Count
    st.markdown(
        render_scada_meter(
            "Oil Particle Contamination",
            current_record['Oil_Particle_Count'],
            "ppm", 0, 1500, 600, 1000
        ),
        unsafe_allow_html=True
    )

with right_col:
    st.markdown("<div class='section-header'>🤖 AI Prognostics & Fault Diagnosis</div>", unsafe_allow_html=True)
    
    feat_dict = {col: float(current_record.get(col, 0.0) if pd.notnull(current_record.get(col)) else 0.0) for col in FEATURE_COLS}
    feat_input = pd.DataFrame([feat_dict])
    pred_rul = rul_model.predict(feat_input)[0] if rul_model else current_record['RUL']
    
    st.markdown(f"""
    <div style='background: #1e293b; padding: 14px 18px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 12px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);'>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">AI Model Forecast</div>
                <div style="color: #38bdf8; font-size: clamp(1.4rem, 1.8vw, 1.8rem); font-weight: 800; line-height: 1.1;">{pred_rul:.1f} <span style="font-size: 0.9rem; color: #94a3b8;">Operating Hours</span></div>
            </div>
            <div style="text-align: right;">
                <div style="color: #94a3b8; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Actual Benchmark</div>
                <div style="color: #f8fafc; font-size: clamp(1.4rem, 1.8vw, 1.8rem); font-weight: 800; line-height: 1.1;">{current_record['RUL']:.1f} <span style="font-size: 0.9rem; color: #94a3b8;">Hours</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if ft_model and comp_model:
        pred_fault_type = ft_model.predict(feat_input)[0]
        fault_type_probs = ft_model.predict_proba(feat_input)[0]
        fault_classes = ft_model.classes_
        
        pred_comp = comp_model.predict(feat_input)[0]
        comp_probs = comp_model.predict_proba(feat_input)[0]
        comp_classes = comp_model.classes_
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=fault_type_probs,
            theta=fault_classes,
            fill='toself',
            name='Failure Mode',
            line=dict(color='#38bdf8', width=2),
            fillcolor='rgba(56, 189, 248, 0.3)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=comp_probs,
            theta=comp_classes,
            fill='toself',
            name='Faulty Component',
            line=dict(color='#f43f5e', width=2),
            fillcolor='rgba(244, 63, 94, 0.3)'
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1e293b",
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#cbd5e1", size=10), gridcolor="#334155"),
                angularaxis=dict(tickfont=dict(color="#ffffff", size=11, family="sans-serif"), gridcolor="#334155")
            ),
            height=320,
            margin=dict(l=20, r=20, t=25, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            legend=dict(font=dict(color="#ffffff", size=11), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown(f"""
        <div style="background-color: #1e293b; border-left: 4px solid #38bdf8; padding: 12px 16px; border-radius: 8px; font-size: 0.95rem; color: #f8fafc; border: 1px solid #334155;">
            💡 <b>Diagnosed Failure Mode:</b> <span style="color: #38bdf8; font-weight: bold;">{pred_fault_type}</span> &nbsp;|&nbsp; 
            <b>Vulnerable Component:</b> <span style="color: #f43f5e; font-weight: bold;">{pred_comp}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='section-header'>📈 Multi-Sensor Telemetry Rolling History</div>", unsafe_allow_html=True)

history_window = df_mach.iloc[max(0, record_idx - 100):record_idx + 100]
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=history_window['Timestamp'],
    y=history_window['Bearing_Temperature'],
    name="Bearing Temp (°C)",
    line=dict(color="#38bdf8", width=2)
))
fig_trend.add_trace(go.Scatter(
    x=history_window['Timestamp'],
    y=history_window['RMS_Vibration'] * 20,
    name="RMS Vibration (x20 scaled)",
    line=dict(color="#f59e0b", width=2)
))
fig_trend.add_trace(go.Scatter(
    x=history_window['Timestamp'],
    y=history_window['Power_Consumption'] / 10,
    name="Power (kW / 10 scaled)",
    line=dict(color="#10b981", width=2)
))
fig_trend.update_layout(
    height=340,
    paper_bgcolor="#1e293b",
    plot_bgcolor="#0f172a",
    font=dict(color="#ffffff"),
    xaxis=dict(gridcolor="#334155", title="Timestamp", tickfont=dict(color="#cbd5e1"), titlefont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#334155", title="Sensor Values", tickfont=dict(color="#cbd5e1"), titlefont=dict(color="#94a3b8")),
    legend=dict(font=dict(color="#ffffff"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=15, r=15, t=30, b=15)
)
st.plotly_chart(fig_trend, use_container_width=True)

# Dataset attribution in sidebar & footer
st.sidebar.markdown("---")
st.sidebar.caption("📊 **Dataset Source:** [Kaggle.com](https://www.kaggle.com/) — Siemens Industrial Telemetry (3-Year Continuous Stream)")
st.caption("Siemens Asset Digital Twin & PdM Platform • Process Engineering & ML Prognostics • Dataset sourced from Kaggle.com")
