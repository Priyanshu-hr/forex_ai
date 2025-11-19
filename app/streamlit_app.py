"""
ForexAI - Professional Forex Prediction Dashboard
Optimized version with smooth UI/UX and improved performance
"""

import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
import sys
import os
import numpy as np
import joblib
from functools import lru_cache
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from prediction_engine import PredictionEngine

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="ForexAI - Forex Prediction System",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== UNIFIED STYLES ====================

st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    /* Root Colors */
    :root {
        --bg-dark: #0f172a;
        --bg-darker: #0a0e27;
        --primary: #3b82f6;
        --primary-dark: #1e40af;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --gray-light: #f1f5f9;
        --gray-text: #94a3b8;
        --card-bg: #1a1f3a;
        --border: #30363d;
    }
    
    /* Base Styles */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, var(--bg-dark) 0%, var(--bg-darker) 100%);
        color: var(--gray-light);
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    /* Typography */
    h1, h2, h3 { color: var(--gray-light); font-weight: 800; }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(90deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.6) 50%, rgba(26, 31, 58, 0.8) 100%);
        border-bottom: 2px solid var(--primary);
        padding: 24px 40px;
        margin-bottom: 30px;
        border-radius: 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .logo-text {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary), var(--primary-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stat-box {
        padding: 12px 16px;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        background: rgba(59, 130, 246, 0.12);
        border-color: rgba(59, 130, 246, 0.25);
    }
    
    /* Containers */
    .container-main {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.7) 0%, rgba(20, 24, 41, 0.7) 100%);
        border: 1px solid rgba(59, 130, 246, 0.15);
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .container-main:hover {
        border-color: rgba(59, 130, 246, 0.25);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.1);
    }
    
    /* Ticker */
    .ticker-container {
        background: linear-gradient(135deg, var(--card-bg) 0%, rgba(20, 24, 41, 0.9) 100%);
        border: 2px solid var(--primary);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
    }
    
    .pair-name-large {
        font-size: 2rem;
        font-weight: 900;
        color: var(--gray-light);
        letter-spacing: -1px;
    }
    
    .price-display {
        font-size: 3rem;
        font-weight: 900;
        color: var(--primary);
        font-family: 'Monaco', 'Courier New', monospace;
        letter-spacing: -2px;
    }
    
    /* Signal Container */
    .signal-container {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.9) 0%, rgba(20, 24, 41, 0.9) 100%);
        border-radius: 12px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 2px solid;
        position: relative;
        overflow: hidden;
    }
    
    .signal-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0.5;
    }
    
    .signal-buy {
        border-color: rgba(16, 185, 129, 0.5);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(16, 185, 129, 0.02)) !important;
    }
    
    .signal-sell {
        border-color: rgba(239, 68, 68, 0.5);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.02)) !important;
    }
    
    .signal-title {
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    
    .signal-confidence {
        font-size: 4rem;
        font-weight: 900;
        font-family: 'Monaco', monospace;
        letter-spacing: -3px;
        margin-top: 16px;
    }
    
    /* Technical Indicators Grid */
    .indicator-card {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.8), rgba(20, 24, 41, 0.8));
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .indicator-card:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.15);
    }
    
    .indicator-label {
        font-size: 0.8rem;
        color: var(--gray-text);
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    
    .indicator-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--primary);
        font-family: 'Monaco', monospace;
        margin-bottom: 8px;
    }
    
    .indicator-status {
        font-size: 0.75rem;
        color: var(--gray-text);
    }
    
    /* Model Cards */
    .model-card {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.8), rgba(20, 24, 41, 0.8));
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: var(--primary);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.1);
        transform: translateY(-2px);
    }
    
    .model-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    
    .model-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--gray-light);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .model-icon {
        font-size: 1.8rem;
    }
    
    .decision-badge {
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .decision-buy {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .decision-sell {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .decision-na {
        background: rgba(255, 255, 255, 0.05);
        color: var(--gray-text);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .perf-metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 12px;
    }
    
    .perf-metric-item {
        background: rgba(0, 0, 0, 0.2);
        padding: 12px;
        border-radius: 8px;
        border-left: 3px solid var(--primary);
        text-align: center;
    }
    
    .perf-metric-label {
        font-size: 0.7rem;
        color: var(--gray-text);
        text-transform: uppercase;
        margin-bottom: 4px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    .perf-metric-value {
        font-size: 1.2rem;
        font-weight: 800;
        color: var(--primary);
        font-family: 'Monaco', monospace;
    }
    
    /* Table Styles */
    [data-testid="dataframe"] {
        background: linear-gradient(135deg, rgba(26, 31, 58, 0.6), rgba(20, 24, 41, 0.6)) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(26, 31, 58, 0.5);
        border-radius: 8px;
        color: var(--gray-text);
        padding: 12px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    
    /* Dividers */
    hr {
        border: 1px solid var(--border) !important;
        margin: 24px 0 !important;
    }
    
    /* Training Stats Grid */
    .training-stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin: 24px 0;
    }
    
    .training-stat-box {
        background: rgba(0, 0, 0, 0.3);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid var(--border);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .training-stat-box:hover {
        border-color: var(--primary);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.1);
    }
    
    .training-stat-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--primary);
        font-family: 'Monaco', monospace;
        margin-bottom: 8px;
    }
    
    .training-stat-label {
        font-size: 0.75rem;
        color: var(--gray-text);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Utility */
    .text-muted { color: var(--gray-text); }
    .text-primary { color: var(--primary); }
    .text-success { color: var(--success); }
    .text-danger { color: var(--danger); }
    
    /* Responsive */
    @media (max-width: 1200px) {
        .training-stats-grid { grid-template-columns: repeat(2, 1fr); }
        .perf-metrics-grid { grid-template-columns: 1fr; }
        .price-display { font-size: 2.2rem; }
        .signal-confidence { font-size: 2.8rem; }
    }
    
    @media (max-width: 768px) {
        .header-container { padding: 16px 20px; }
        .container-main { padding: 16px; }
        .ticker-container { padding: 16px; }
        .pair-name-large { font-size: 1.4rem; }
        .price-display { font-size: 1.8rem; }
        .training-stats-grid { grid-template-columns: 1fr; }
    }
    
    /* Hide default elements */
    #MainMenu, footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS & CONFIG ====================

PAIR_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X"
}

PAIR_DESCRIPTIONS = {
    "EUR/USD": "Euro vs US Dollar - Most traded pair globally",
    "GBP/USD": "British Pound vs US Dollar - High volatility",
    "USD/JPY": "US Dollar vs Japanese Yen - Safe haven pair",
    "USD/CAD": "US Dollar vs Canadian Dollar - Oil correlated",
    "AUD/USD": "Australian Dollar vs US Dollar - Commodity linked"
}

MODELS_CONFIG = {
    'rf': {'name': 'Random Forest', 'icon': '🌲', 'accuracy': 0.575, 'precision': 0.562},
    'gb': {'name': 'Gradient Boosting', 'icon': '📈', 'accuracy': 0.581, 'precision': 0.578},
    'lr': {'name': 'Logistic Regression', 'icon': '📊', 'accuracy': 0.554, 'precision': 0.541}
}

# ==================== CACHE & UTILITIES ====================

@st.cache_resource
def load_engine():
    """Load prediction engine once"""
    return PredictionEngine()

@lru_cache(maxsize=1)
def get_time():
    """Get current UTC and IST time"""
    utc_now = datetime.now(pytz.UTC)
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = utc_now.astimezone(ist)
    return utc_now, ist_now

def load_models_and_meta():
    """Load trained models and metadata"""
    base = 'data/models'
    models = {}
    meta = {}
    try:
        model_files = {
            'rf': 'EURUSD_random_forest.pkl',
            'gb': 'EURUSD_gradient_boosting.pkl',
            'lr': 'EURUSD_logistic_regression.pkl'
        }
        
        for key, filename in model_files.items():
            path = os.path.join(base, filename)
            if os.path.exists(path):
                models[key] = joblib.load(path)
        
        scaler_path = os.path.join(base, 'scaler.pkl')
        feat_path = os.path.join(base, 'feature_columns.pkl')
        
        if os.path.exists(scaler_path):
            meta['scaler'] = joblib.load(scaler_path)
        if os.path.exists(feat_path):
            meta['feature_columns'] = joblib.load(feat_path)
            
    except Exception as e:
        st.warning(f"Error loading models: {str(e)}")
    
    return models, meta

def build_feature_vector(feature_columns, inds):
    """Build feature vector from indicators"""
    row = []
    lower_inds = {k.lower(): v for k, v in inds.items()}
    
    for feat in feature_columns:
        key = feat.lower()
        val = 0.0
        
        if key in lower_inds:
            try:
                val = float(lower_inds[key])
            except:
                val = 0.0
        else:
            # Handle special cases
            if 'sma' in key:
                for alt in ['sma_50', 'sma_20', 'sma_10', 'sma_5']:
                    if alt in lower_inds:
                        try:
                            val = float(lower_inds[alt])
                        except:
                            pass
                        break
            elif 'bb_width' in key:
                try:
                    up = float(lower_inds.get('bb_upper', 0))
                    lo = float(lower_inds.get('bb_lower', 0))
                    mid = float(lower_inds.get('bb_middle', 1))
                    val = ((up - lo) / mid) * 100 if mid != 0 else 0.0
                except:
                    val = 0.0
        
        try:
            row.append(float(val))
        except:
            row.append(0.0)
    
    return np.array(row).reshape(1, -1)

def get_model_decisions(models, meta, inds):
    """Get BUY/SELL decisions from individual models"""
    decisions = {'rf': 'N/A', 'gb': 'N/A', 'lr': 'N/A'}
    
    if not models or 'feature_columns' not in meta or 'scaler' not in meta:
        return decisions
    
    try:
        X_row = build_feature_vector(meta['feature_columns'], inds)
        X_scaled = meta['scaler'].transform(X_row)
        
        for key in decisions:
            if key in models:
                try:
                    pred = models[key].predict(X_scaled)[0]
                    decisions[key] = 'BUY' if int(pred) == 1 else 'SELL'
                except:
                    decisions[key] = 'N/A'
    except Exception as e:
        st.debug(f"Error in model decisions: {str(e)}")
    
    return decisions

# ==================== HEADER ====================

def render_header():
    """Render page header"""
    utc_time, ist_time = get_time()
    
    st.markdown(f"""
    <div class="header-container">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:20px;">
            <div>
                <div class="logo-text">ForexAI</div>
                <div style="color:var(--gray-text); font-size:0.95rem; margin-top:4px;">Enterprise Forex Prediction System</div>
            </div>
            <div style="display:flex; gap:12px; flex-wrap:wrap;">
                <div class="stat-box"><div style="font-size:0.75rem;color:var(--gray-text)">Status</div><div style="font-weight:700;color:var(--primary)"><span style="display:inline-block;width:8px;height:8px;background:var(--success);border-radius:50%;margin-right:6px;"></span>LIVE</div></div>
                <div class="stat-box"><div style="font-size:0.75rem;color:var(--gray-text)">Source</div><div style="font-weight:700;color:var(--primary)">YFINANCE</div></div>
                <div class="stat-box"><div style="font-size:0.75rem;color:var(--gray-text)">IST Time</div><div style="font-weight:700;color:var(--primary);">{ist_time.strftime('%H:%M:%S')}</div></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== MAIN CONTENT ====================

def render_ticker(pair_name, price, ist_time, utc_time):
    """Render ticker section"""
    st.markdown(f"""
    <div class="ticker-container">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:20px;">
            <div>
                <div class="pair-name-large">{pair_name}</div>
                <div style="color:var(--gray-text); font-size:0.95rem; margin-top:8px;">{PAIR_DESCRIPTIONS[pair_name]}</div>
            </div>
            <div style="text-align:right;">
                <div class="price-display">${price:.5f}</div>
                <div style="color:var(--gray-text); font-size:0.85rem; margin-top:8px;">UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_signal(direction, confidence):
    """Render main signal section"""
    is_buy = direction == 'UP'
    signal_text = "BUY SIGNAL - BULLISH CONSENSUS" if is_buy else "SELL SIGNAL - BEARISH CONSENSUS"
    signal_class = "signal-buy" if is_buy else "signal-sell"
    signal_color = "var(--success)" if is_buy else "var(--danger)"
    
    st.markdown(f"""
    <div class="signal-container {signal_class}">
        <div style="font-size:1.3rem; font-weight:800; color:{signal_color};">{signal_text}</div>
        <div style="margin-top:8px; font-size:0.9rem; color:var(--gray-text);">Ensemble Prediction Confidence (rule-based)</div>
        <div class="signal-confidence" style="color:{signal_color};">{confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

def render_technical_indicators(inds):
    """Render technical indicators grid"""
    st.markdown("### 📊 Technical Indicators")
    
    cols = st.columns(4)
    
    indicators = [
        ("RSI (14)", f"{inds['rsi']:.1f}", "Momentum"),
        ("MACD", f"{inds['macd']:.6f}", "BULLISH" if inds['macd'] > inds['macd_signal'] else "BEARISH"),
        ("SMA 20", f"${inds['sma_20']:.5f}", "ABOVE" if inds['price'] > inds['sma_20'] else "BELOW"),
        ("Volatility", f"{inds['volatility']:.4f}%", "20-day Avg"),
    ]
    
    for col, (label, value, status) in zip(cols, indicators):
        with col:
            st.markdown(f"""
            <div class="indicator-card">
                <div class="indicator-label">{label}</div>
                <div class="indicator-value">{value}</div>
                <div class="indicator-status">{status}</div>
            </div>
            """, unsafe_allow_html=True)

def render_model_card(model_key, decision):
    """Render individual model card"""
    model_info = MODELS_CONFIG[model_key]
    badge_class = f"decision-{decision.lower()}" if decision in ['BUY', 'SELL'] else "decision-na"
    
    st.markdown(f"""
    <div class="model-card">
        <div class="model-header">
            <div class="model-title">
                <span class="model-icon">{model_info['icon']}</span>
                {model_info['name']}
            </div>
            <div class="decision-badge {badge_class}">{decision}</div>
        </div>
        <div class="perf-metrics-grid">
            <div class="perf-metric-item">
                <div class="perf-metric-label">Accuracy</div>
                <div class="perf-metric-value">{model_info['accuracy']:.1%}</div>
            </div>
            <div class="perf-metric-item">
                <div class="perf-metric-label">Precision</div>
                <div class="perf-metric-value">{model_info['precision']:.1%}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== SESSION STATE ====================

if 'show_all' not in st.session_state:
    st.session_state.show_all = False
if 'show_about' not in st.session_state:
    st.session_state.show_about = False

# ==================== MAIN APP ====================

render_header()

# Control Panel
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    pair_name = st.selectbox(
        "Select Currency Pair",
        list(PAIR_MAP.keys()),
        label_visibility="collapsed",
        key="pair_select"
    )

with col2:
    if st.button("🔄 Refresh", use_container_width=True, key="refresh_btn"):
        st.cache_resource.clear()
        st.rerun()

with col3:
    if st.button("📊 All Pairs", use_container_width=True, key="all_pairs_btn"):
        st.session_state.show_all = True
        st.rerun()

with col4:
    if st.button("ℹ️ About", use_container_width=True, key="about_btn"):
        st.session_state.show_about = True
        st.rerun()

# ==================== PAGE ROUTING ====================

if st.session_state.show_all:
    st.markdown("## 📈 All Currency Pairs - Live Predictions")
    cols = st.columns(5)
    
    engine = load_engine()
    for idx, pair in enumerate(PAIR_MAP.keys()):
        with cols[idx]:
            with st.spinner(f"Loading {pair}..."):
                result = engine.get_prediction(PAIR_MAP[pair])
                if result:
                    p = result['prediction']
                    pr = result['indicators']['price']
                    signal = "BUY" if p['direction'] == 'UP' else "SELL"
                    color = "var(--success)" if signal == "BUY" else "var(--danger)"
                    
                    st.markdown(f"""
                    <div class="ticker-container" style="border-color: {color};">
                        <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 10px;">{pair}</div>
                        <div style="font-size: 1.3rem; color: var(--primary); font-family: Monaco; font-weight: 700; margin-bottom: 8px;">${pr:.5f}</div>
                        <div style="font-size: 0.9rem; font-weight: 700; color: {color};">{signal} - {p['confidence']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.divider()
    if st.button("← Back to Main", use_container_width=True, key="back_btn"):
        st.session_state.show_all = False
        st.rerun()

elif st.session_state.show_about:
    st.markdown("## ℹ️ About ForexAI")
    st.markdown("""
    ### 🎓 Final Year University Project
    
    **Technology Stack:**
    - **Frontend:** Streamlit
    - **Backend:** Python (Pandas, NumPy, scikit-learn)
    - **Data Source:** yfinance
    - **Deployment:** Streamlit Cloud
    
    **Project Overview:**
    ForexAI is an enterprise-grade Forex prediction system that predicts next-day market direction 
    (BUY/SELL) using three machine learning models working in ensemble.
    
    **Features:**
    - Real-time currency pair analysis
    - Multi-model ensemble predictions
    - Technical indicator analysis
    - Individual model performance metrics
    - Live market data integration
    
    **Disclaimer:**
    ⚠️ *For educational purposes only. Not financial advice. Always conduct your own analysis and consult with financial advisors.*
    """)
    
    st.divider()
    if st.button("← Back", use_container_width=True, key="about_back_btn"):
        st.session_state.show_about = False
        st.rerun()

else:
    # ==================== MAIN DASHBOARD ====================
    
    engine = load_engine()
    models, meta = load_models_and_meta()
    
    with st.spinner("📡 Loading live market data..."):
        result = engine.get_prediction(PAIR_MAP[pair_name])
    
    if result is None:
        st.error("❌ Failed to fetch market data. Please try again.")
        st.stop()
    
    pred = result['prediction']
    inds = result['indicators']
    price = inds['price']
    utc_time, ist_time = get_time()
    
    # Get model decisions
    model_decisions = get_model_decisions(models, meta, inds)
    
    # Render sections
    render_ticker(pair_name, price, ist_time, utc_time)
    render_signal(pred['direction'], pred['confidence'])
    render_technical_indicators(inds)
    
    # Analysis Tabs
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Moving Averages", "📈 Momentum", "🎯 Technical Details", "🤖 Model Details"])
    
    with tab1:
        st.markdown("#### Moving Average Analysis")
        df_ma = pd.DataFrame({
            "MA": ["SMA 5", "SMA 10", "SMA 20", "SMA 50"],
            "Price": [f"${inds['sma_5']:.5f}", f"${inds['sma_10']:.5f}", f"${inds['sma_20']:.5f}", f"${inds['sma_50']:.5f}"],
            "Status": [
                "✓ ABOVE" if price > inds['sma_5'] else "✗ BELOW",
                "✓ ABOVE" if price > inds['sma_10'] else "✗ BELOW",
                "✓ ABOVE" if price > inds['sma_20'] else "✗ BELOW",
                "✓ ABOVE" if price > inds['sma_50'] else "✗ BELOW",
            ]
        })
        st.dataframe(df_ma, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### Momentum Indicators")
        df_mom = pd.DataFrame({
            "Indicator": ["RSI", "MACD", "Signal", "Histogram"],
            "Value": [f"{inds['rsi']:.1f}", f"{inds['macd']:.6f}", f"{inds['macd_signal']:.6f}", f"{inds['macd_hist']:.6f}"],
        })
        st.dataframe(df_mom, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("#### Technical Details")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ATR (14)", f"{inds['atr']:.6f}")
            st.metric("BB Upper", f"${inds['bb_upper']:.5f}")
            st.metric("Daily Return", f"{inds['daily_return']:.4f}%")
        with col2:
            st.metric("BB Middle", f"${inds['bb_middle']:.5f}")
            st.metric("BB Lower", f"${inds['bb_lower']:.5f}")
            st.metric("Intraday Range", f"{inds['intraday_range']:.4f}%")
    
    with tab4:
        st.markdown("#### 🤖 Model Performance (Three Models)")
        st.markdown("##### Individual Model Results (with real-time decision)")
        
        for model_key in ['rf', 'gb', 'lr']:
            decision = model_decisions.get(model_key, 'N/A')
            render_model_card(model_key, decision)
        
        st.divider()
        
        st.markdown("##### Performance Comparison")
        perf_data = {
            'Model': [MODELS_CONFIG[k]['name'] for k in ['rf', 'gb', 'lr']],
            'Accuracy': [f"{MODELS_CONFIG[k]['accuracy']:.1%}" for k in ['rf', 'gb', 'lr']],
            'Precision': [f"{MODELS_CONFIG[k]['precision']:.1%}" for k in ['rf', 'gb', 'lr']]
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.info("""
        **💡 About Model Decisions:**
        - Real-time BUY/SELL decisions shown for each trained model
        - Decisions based on model predictions if model files exist in `data/models/`
        - Models are evaluated on Accuracy & Precision metrics
        """)
        
        st.divider()
        st.markdown("##### Training Dataset Statistics")
        st.markdown("""
        <div class="training-stats-grid">
            <div class="training-stat-box">
                <div class="training-stat-value">960</div>
                <div class="training-stat-label">Training Records</div>
            </div>
            <div class="training-stat-box">
                <div class="training-stat-value">240</div>
                <div class="training-stat-label">Testing Records</div>
            </div>
            <div class="training-stat-box">
                <div class="training-stat-value">18</div>
                <div class="training-stat-label">Input Features</div>
            </div>
            <div class="training-stat-box">
                <div class="training-stat-value">5 Years</div>
                <div class="training-stat-label">Historical Data</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================

st.divider()
st.markdown("""
<div style="text-align:center;padding:24px 20px;color:var(--gray-text);">
    <div style="font-weight:700;color:var(--gray-light);margin-bottom:8px;">ForexAI - Final Year University Project</div>
    <small>AI/ML Powered Forex Prediction System | Live Market Intelligence</small>
    <br><br>
    <small style="opacity:0.7;">⚠️ For educational purposes only. Not financial advice. Always conduct your own analysis.</small>
</div>
""", unsafe_allow_html=True)
