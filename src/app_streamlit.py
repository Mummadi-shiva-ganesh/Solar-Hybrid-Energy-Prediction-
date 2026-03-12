import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime
import time

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Sopanel - Solar Energy Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Background & Typography */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }

    /* Custom Card Styling */
    .st-emotion-cache-12w0qpk { /* This targets the default container/card in some versions */
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    
    /* Custom Card Helper Classes */
    .card {
        background-color: #1e293b;
        color: #ffffff;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .big-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #ffffff;
    }

    .text-muted {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* Solar Visual Animation */
    .solar-visual {
        height: 120px;
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02));
        border-radius: 12px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .solar-panel-grid {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        width: 180px;
    }

    .solar-cell {
        background: rgba(255, 255, 255, 0.05);
        border: 1.5px solid rgba(16, 185, 129, 0.2);
        border-radius: 6px;
        height: 30px;
        position: relative;
        overflow: hidden;
    }

    .solar-cell.active {
        background: rgba(16, 185, 129, 0.2);
        border-color: rgba(16, 185, 129, 0.6);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }

    .solar-cell::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(234, 179, 8, 0.3), transparent);
        animation: shine 2.5s infinite;
    }

    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Logo Styling */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
        padding: 10px 0;
    }

    .logo-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #10b981, #eab308);
        border-radius: 8px;
    }

    .logo-text {
        font-weight: 700;
        font-size: 1.4rem;
        color: #ffffff;
    }

    /* Input Styling Override */
    .stNumberInput input {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-color: #334155 !important;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# API Configuration
API_BASE = "http://localhost:5000/api"
API_KEY = "solar-yield-secret-2026"

# Init Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = 0.0

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div class="logo-container">
            <div class="logo-icon"></div>
            <div class="logo-text">Sopanel</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Dashboard")
    st.button("🏠 Overview", type="secondary", disabled=False)
    
    st.divider()
    
    st.markdown("### Insights")
    st.caption("Detailed Performance Reports")
    st.button("📊 Analytics", disabled=True)
    st.button("⚙️ Settings", disabled=True)
    
    st.markdown("### User")
    st.markdown("👤 **Jyoth** (Admin)")
    if st.button("Log Out"):
        st.info("Log out feature not implemented.")

# --- Header ---
st.markdown('<h1 style="margin-bottom: 30px;">Overview</h1>', unsafe_allow_html=True)

# (Top Stats Row removed as requested)

st.write("") # Spacer

# --- Main Interaction Section ---
main_col1, main_col2 = st.columns([1, 1], gap="large")

with main_col1:
    with st.container(border=True):
        st.markdown('<p class="card-title">Performance Monitoring</p>', unsafe_allow_html=True)
        
        # Input Layout
        ic1, ic2 = st.columns(2)
        with ic1:
            ambient_temp = st.number_input("Ambient Temp (°C)", value=25.5, step=0.1)
            module_temp = st.number_input("Module Temp (°C)", value=35.2, step=0.1)
        with ic2:
            irradiation = st.number_input("Irradiation (0-1.0)", value=0.8, step=0.01, min_value=0.0, max_value=1.0)
            st.write("") # Spacer
            predict_btn = st.button("Generate Prediction", use_container_width=True)
        
        if predict_btn:
            try:
                payload = {"features": {"AMBIENT_TEMPERATURE": ambient_temp, "MODULE_TEMPERATURE": module_temp, "IRRADIATION": irradiation}}
                r = requests.post(f"{API_BASE}/predict", json=payload, headers={"X-API-KEY": API_KEY}, timeout=5)
                data = r.json()
                if data.get("status") == "success":
                    res_kw = data.get("prediction", 0.0)
                    st.session_state.last_prediction = res_kw
                    st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "val": res_kw})
                    if len(st.session_state.history) > 10: st.session_state.history.pop()
                else:
                    st.error(f"Error: {data.get('error')}")
            except Exception as e:
                st.error("API Offline. Using fallback mode.")
                st.session_state.last_prediction = (irradiation * 400) + (ambient_temp * 0.5)

        # Output Display
        last_val = st.session_state.last_prediction
        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown(f'<p class="text-muted">Predicted Output</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-value">{last_val:.2f} <span style="font-size: 1rem;">kWh</span></p>', unsafe_allow_html=True)
        with oc2:
            st.markdown(f'<p class="text-muted">System Load</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-value">6.8 <span style="font-size: 1rem;">kWh</span></p>', unsafe_allow_html=True)

        # Animated Grid
        active_cells = min(6, max(0, int((last_val / 30) * 6))) if last_val > 0 else 0
        cells_html = "".join([f'<div class="solar-cell {"active" if i < active_cells else ""}"></div>' for i in range(6)])
        st.markdown(f'<div class="solar-visual"><div class="solar-panel-grid">{cells_html}</div></div>', unsafe_allow_html=True)
        
        st.markdown("📊 **Model Accuracy: 82.5%** · ")


with main_col2:
    with st.container(border=True):
        st.markdown('<p class="card-title">Real-time History</p>', unsafe_allow_html=True)
        if not st.session_state.history:
             st.markdown('<p class="text-muted">No records found.</p>', unsafe_allow_html=True)
        else:
            for item in st.session_state.history[:5]:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #334155;">
                    <span>🕒 {item['time']}</span>
                    <span style="color: #10b981; font-weight: 700;">{item['val']:.2f} kWh</span>
                </div>
                """, unsafe_allow_html=True)

if predict_btn:
    st.rerun()
