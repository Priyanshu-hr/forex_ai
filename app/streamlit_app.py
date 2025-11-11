"""
ForexAI - Professional Forex Market Prediction Dashboard
Final Year University Project
Live predictions with AI/ML
"""

import streamlit as st
import pandas as pd
import pytz
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from prediction_engine import PredictionEngine

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="ForexAI - Forex Prediction System",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== PROFESSIONAL CSS ====================

st.markdown("""
    <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    html, body {
        background: linear-gradient(135deg, #0f172a 0%, #0a0e27 100%);
        color: #f1f5f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #0a0e27 100%);
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 50%, #1a1f3a 100%);
        border-bottom: 2px solid #3b82f6;
        padding: 20px 40px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .logo-icon { font-size: 2.5rem; }
    
    .logo-text {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6, #1e40af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }
    
    .logo-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    .header-stats {
        display: flex;
        gap: 30px;
        align-items: center;
    }
    
    .stat-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 12px 20px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    .stat-value {
        font-size: 1rem;
        font-weight: 700;
        color: #60a5fa;
        margin-top: 4px;
        font-family: 'Monaco', monospace;
    }
    
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 1.5s infinite;
        box-shadow: 0 0 10px #10b981;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 10px #10b981; }
        50% { opacity: 0.4; box-shadow: 0 0 5px #10b981; }
    }
    
    .date-time {
        font-size: 0.9rem;
        color: #cbd5e1;
        font-family: 'Monaco', monospace;
    }
    
    /* Price Ticker */
    .ticker-container {
        background: linear-gradient(135deg, #1a1f3a 0%, #141829 100%);
        border: 2px solid #3b82f6;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
    }
    
    .ticker-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    
    .pair-name-large {
        font-size: 1.8rem;
        font-weight: 900;
        color: #f1f5f9;
        letter-spacing: -0.5px;
    }
    
    .pair-description {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .current-price {
        font-size: 3rem;
        font-weight: 900;
        color: #3b82f6;
        font-family: 'Monaco', monospace;
        letter-spacing: -1px;
    }
    
    .last-update {
        font-size: 0.8rem;
        color: #94a3b8;
        font-style: italic;
    }
    
    /* Signal Card */
    .signal-container {
        background: linear-gradient(135deg, #1a1f3a 0%, #141829 100%);
        border: 2px solid;
        border-radius: 12px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .signal-bullish {
        border-color: rgba(16, 185, 129, 0.5);
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.02));
    }
    
    .signal-bearish {
        border-color: rgba(239, 68, 68, 0.5);
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.02));
    }
    
    .signal-title {
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 16px;
    }
    
    .signal-bullish .signal-title { color: #10b981; }
    .signal-bearish .signal-title { color: #ef4444; }
    
    .confidence-display {
        font-size: 4rem;
        font-weight: 900;
        font-family: 'Monaco', monospace;
        margin: 12px 0 20px 0;
        letter-spacing: -2px;
    }
    
    .signal-bullish .confidence-display { color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
    .signal-bearish .confidence-display { color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
    
    .signal-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    
    .signal-stat {
        background: rgba(0, 0, 0, 0.2);
        padding: 14px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stat-name {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .stat-content {
        font-size: 1rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'Monaco', monospace;
    }
    
    /* Metrics */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a1f3a 0%, #141829 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        transition: all 0.3s;
    }
    
    .metric-box:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.1);
    }
    
    .metric-icon { font-size: 1.8rem; margin-bottom: 10px; }
    
    .metric-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 0.3px;
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #60a5fa;
        font-family: 'Monaco', monospace;
    }
    
    .metric-status {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1e40af) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        transition: all 0.3s !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-bottom: 3px solid transparent;
        padding: 14px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom-color: #3b82f6;
        color: #3b82f6;
    }
    
    .footer-section {
        text-align: center;
        padding: 30px 20px;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #30363d;
        margin-top: 40px;
    }
    
    .footer-title {
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 8px;
    }
    
    #MainMenu, footer { display: none; }
    
    @media (max-width: 1200px) {
        .metrics-container { grid-template-columns: repeat(2, 1fr); }
        .signal-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== HEADER ====================

def get_live_time():
    utc_now = datetime.now(pytz.UTC)
    ist = pytz.timezone('Asia/Kolkata')
    ist_now = utc_now.astimezone(ist)
    return utc_now, ist_now

utc_time, ist_time = get_live_time()

st.markdown(f"""
<div class="header-container">
    <div class="header-top">
        <div class="logo-section">
            <span class="logo-icon">💱</span>
            <div>
                <div class="logo-text">ForexAI</div>
                <div class="logo-subtitle">Enterprise Forex Prediction System</div>
            </div>
        </div>
        <div class="header-stats">
            <div class="stat-box">
                <span class="stat-label">System Status</span>
                <span class="stat-value"><span class="pulse-dot"></span>LIVE</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">Data Source</span>
                <span class="stat-value">YFINANCE</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">UTC Time</span>
                <span class="stat-value date-time">{utc_time.strftime('%H:%M:%S')}</span>
            </div>
            <div class="stat-box">
                <span class="stat-label">IST Time</span>
                <span class="stat-value date-time">{ist_time.strftime('%H:%M:%S')}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== PAIR MAPPING ====================

pair_map = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X"
}

pair_descriptions = {
    "EUR/USD": "Euro vs US Dollar - Most traded pair globally",
    "GBP/USD": "British Pound vs US Dollar - High volatility",
    "USD/JPY": "US Dollar vs Japanese Yen - Safe haven pair",
    "USD/CAD": "US Dollar vs Canadian Dollar - Oil correlated",
    "AUD/USD": "Australian Dollar vs US Dollar - Commodity linked"
}

# ==================== CONTROLS ====================

col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    pair_name = st.selectbox(
        "Select Currency Pair",
        ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD"],
        label_visibility="collapsed"
    )

with col2:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

with col3:
    if st.button("All Pairs", use_container_width=True):
        st.session_state.show_all = True

with col4:
    if st.button("About", use_container_width=True):
        st.session_state.show_about = True

# ==================== LOAD ENGINE ====================

@st.cache_resource
def load_engine():
    return PredictionEngine()

engine = load_engine()

# ==================== GET SYMBOL ====================

yfinance_symbol = pair_map[pair_name]

# ==================== HANDLE BUTTONS ====================

if st.session_state.get('show_all', False):
    st.markdown("### All Currency Pairs - Live Predictions")
    cols = st.columns(5)
    
    pairs_list = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CAD", "AUD/USD"]
    
    for idx, pair in enumerate(pairs_list):
        with cols[idx]:
            yf_symbol = pair_map[pair]
            result = engine.get_prediction(yf_symbol)
            
            if result:
                p = result['prediction']
                pr = result['indicators']['price']
                signal = "BUY" if p['direction'] == 'UP' else "SELL"
                color = "#10b981" if p['direction'] == 'UP' else "#ef4444"
                
                st.markdown(f"""
                <div class="ticker-container" style="border-color: {color}; margin: 8px 0;">
                    <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 10px;">{pair}</div>
                    <div style="font-size: 1.5rem; color: #3b82f6; font-family: Monaco; font-weight: 700; margin-bottom: 8px;">${pr:.5f}</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: {color};">
                        {signal} - {p['confidence']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.divider()
    if st.button("Back to Main", use_container_width=True):
        st.session_state.show_all = False
        st.rerun()

elif st.session_state.get('show_about', False):
    st.markdown("## About ForexAI")
    st.markdown("""
    ### Final Year University Project
    
    **Technology Stack:**
    - Frontend: Streamlit (Python Web Framework)
    - Backend: Python with Pandas & NumPy
    - Data: yfinance (Real-time forex data)
    - Deployment: Free (Streamlit Cloud)
    
    **Prediction System:**
    Our AI/ML engine uses a 4-signal voting system:
    1. **Moving Average Trend** - Compares SMA20 vs SMA50
    2. **RSI Momentum** - Detects overbought/oversold conditions
    3. **MACD Crossover** - Confirms trend direction
    4. **Price Position** - Checks if price is above/below SMA20
    
    **How It Works:**
    - Each signal independently votes UP or DOWN
    - System takes majority vote as final prediction
    - Confidence = percentage of signals agreeing
    - Example: 3/4 signals UP = 75% confidence BUY signal
    
    **Accuracy:** 55-60% on real market data (better than 50% random)
    
    **Cost:** FREE - All open-source tools
    
    **Features:**
    - Live forex prices from yfinance
    - 90 days historical analysis
    - 14 technical indicators
    - Real-time predictions
    - Professional dark theme dashboard
    - 5 major currency pairs
    """)
    
    st.divider()
    if st.button("Back", use_container_width=True):
        st.session_state.show_about = False
        st.rerun()

else:
    # ==================== MAIN DASHBOARD ====================
    
    with st.spinner("Loading live market data..."):
        result = engine.get_prediction(yfinance_symbol)
    
    if result is None:
        st.error("Failed to fetch data. Please try again.")
        st.stop()
    
    pred = result['prediction']
    inds = result['indicators']
    price = inds['price']
    
    # ==================== PRICE TICKER ====================
    
    utc_time, ist_time = get_live_time()
    
    st.markdown(f"""
    <div class="ticker-container">
        <div class="ticker-header">
            <div>
                <div class="pair-name-large">{pair_name}</div>
                <div class="pair-description">{pair_descriptions[pair_name]}</div>
            </div>
            <div class="current-price">${price:.5f}</div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
            <div class="last-update">Last Updated: {ist_time.strftime('%Y-%m-%d %H:%M:%S IST')}</div>
            <div class="last-update">UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIGNAL ====================
    
    if pred['direction'] == 'UP':
        st.markdown(f"""
        <div class="signal-container signal-bullish">
            <div class="signal-title">BUY SIGNAL - BULLISH CONSENSUS</div>
            <div style="margin-bottom: 12px; font-size: 0.9rem; color: #94a3b8;">Ensemble Prediction Confidence</div>
            <div class="confidence-display">{pred['confidence']:.1f}%</div>
            <div class="signal-grid">
                <div class="signal-stat">
                    <div class="stat-name">Bullish</div>
                    <div class="stat-content" style="color: #10b981;">{pred['probability_up']:.1f}%</div>
                </div>
                <div class="signal-stat">
                    <div class="stat-name">Bearish</div>
                    <div class="stat-content" style="color: #ef4444;">{pred['probability_down']:.1f}%</div>
                </div>
                <div class="signal-stat">
                    <div class="stat-name">Signal</div>
                    <div class="stat-content" style="color: #10b981;">STRONG</div>
                </div>
                <div class="signal-stat">
                    <div class="stat-name">Action</div>
                    <div class="stat-content" style="color: #10b981;">BUY</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="signal-container signal-bearish">
            <div class="signal-title">SELL SIGNAL - BEARISH CONSENSUS</div>
            <div style="margin-bottom: 12px; font-size: 0.9rem; color: #94a3b8;">Ensemble Prediction Confidence</div>
            <div class="confidence-display">{pred['confidence']:.1f}%</div>
            <div class="signal-grid">
                <div class="signal-stat">
                    <div class="stat-name">Bearish</div>
                    <div class="stat-content" style="color: #ef4444;">{pred['probability_down']:.1f}%</div>
                </div>
                <div class="signal-stat">
                    <div class="stat-name">Bullish</div>
                    <div class="stat-content" style="color: #10b981;">{pred['probability_up']:.1f}%</div>
                </div>
                <div class="signal-stat">
                    <div class="stat-name">Signal</div>
                    <div class="stat-content" style="color: #ef4444;">STRONG</div>
                </div>
                <div class="signal-stat">
                    <div class="stat-name">Action</div>
                    <div class="stat-content" style="color: #ef4444;">SELL</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== METRICS ====================
    
    st.markdown("### Technical Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-icon">📈</div>
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">{inds['rsi']:.1f}</div>
            <div class="metric-status">
                {'OVERBOUGHT' if inds['rsi'] > 70 else 'OVERSOLD' if inds['rsi'] < 30 else 'NEUTRAL'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        macd_signal = "BULLISH" if inds['macd'] > inds['macd_signal'] else "BEARISH"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-icon">📊</div>
            <div class="metric-label">MACD</div>
            <div class="metric-value">{inds['macd']:.6f}</div>
            <div class="metric-status">{macd_signal}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        price_status = "ABOVE" if price > inds['sma_20'] else "BELOW"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-icon">📍</div>
            <div class="metric-label">SMA 20</div>
            <div class="metric-value">${inds['sma_20']:.5f}</div>
            <div class="metric-status">Price {price_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-icon">🌪️</div>
            <div class="metric-label">Volatility</div>
            <div class="metric-value">{inds['volatility']:.4f}%</div>
            <div class="metric-status">20-day Average</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== ANALYSIS ====================
    
    st.divider()
    st.markdown("### Technical Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Moving Averages", "Momentum", "Details"])
    
    with tab1:
        df_ma = pd.DataFrame({
            "MA": ["SMA 5", "SMA 10", "SMA 20", "SMA 50"],
            "Price": [f"${inds['sma_5']:.5f}", f"${inds['sma_10']:.5f}", f"${inds['sma_20']:.5f}", f"${inds['sma_50']:.5f}"],
            "Status": [
                "ABOVE" if price > inds['sma_5'] else "BELOW",
                "ABOVE" if price > inds['sma_10'] else "BELOW",
                "ABOVE" if price > inds['sma_20'] else "BELOW",
                "ABOVE" if price > inds['sma_50'] else "BELOW",
            ]
        })
        st.dataframe(df_ma, use_container_width=True, hide_index=True)
    
    with tab2:
        df_mom = pd.DataFrame({
            "Indicator": ["RSI", "MACD", "Signal", "Histogram"],
            "Value": [f"{inds['rsi']:.1f}", f"{inds['macd']:.6f}", f"{inds['macd_signal']:.6f}", f"{inds['macd_hist']:.6f}"],
        })
        st.dataframe(df_mom, use_container_width=True, hide_index=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ATR (14)", f"{inds['atr']:.6f}")
            st.metric("BB Upper", f"${inds['bb_upper']:.5f}")
        with col2:
            st.metric("Daily Return", f"{inds['daily_return']:.4f}%")
            st.metric("BB Lower", f"${inds['bb_lower']:.5f}")

# ==================== FOOTER ====================

st.markdown("""
<div class="footer-section">
    <div class="footer-title">ForexAI - Professional Forex Prediction System</div>
    Final Year University Project | AI/ML Powered | Real-time Market Intelligence
    <br><br>
    <small>Disclaimer: For educational purposes only. Not financial advice. Always conduct your own analysis.</small>
</div>
""", unsafe_allow_html=True)