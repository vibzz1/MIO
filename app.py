import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import concurrent.futures
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nselib import capital_market
from functools import partial

warnings.filterwarnings("ignore")

# --- UI Setup ---
st.set_page_config(page_title="MIO Champ Screener", layout="wide")
st.title("📈 MIO Champion Setup Screener")
st.markdown("Automated scan for high-probability momentum setups.")

# --- Data Fetching & Industry Mapping ---
@st.cache_data(ttl=3600)
def get_nifty_500():
    try:
        n50 = capital_market.nifty50_equity_list()
        nn50 = capital_market.niftynext50_equity_list()
        mid150 = capital_market.niftymidcap150_equity_list()
        sml250 = capital_market.niftysmallcap250_equity_list()
        df = pd.concat([n50, nn50, mid150, sml250], ignore_index=True)
        
        tickers = list(set(df['Symbol'].tolist()))
        # Build an official NSE industry dictionary so we don't rely on Yahoo Finance
        ind_map = dict(zip(df['Symbol'], df['Industry']))
        
        return tickers, ind_map
    except: return [], {}

@st.cache_data(ttl=3600)
def get_all_nse():
    try:
        df = capital_market.equity_list()
        df.columns = df.columns.str.upper() 
        raw = df['SYMBOL'].tolist()
        tickers = list(set([t for t in raw if isinstance(t, str) and "DUMMY" not in t]))
        
        # Borrow the industry map from the top 500 as a baseline 
        _, ind_map = get_nifty_500()
        
        return tickers, ind_map
    except: return [], {}

# --- Screener Logic (ORIGINAL — UNTOUCHED) ---
def check_stock(ticker, ind_map):
    symbol = f"{ticker}.NS"
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        if len(df) < 70: return None 

        df.dropna(inplace=True)
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR_20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        df['ATR_1']  = ta.true_range(df['High'], df['Low'], df['Close'])
        df['ADVOL_20'] = df['Volume'].rolling(20).mean()
        df['ADVOL_50'] = df['Volume'].rolling(50).mean()

        df.dropna(inplace=True)
        if len(df) < 22: return None

        sma50_trend_dn_20 = df['SMA_50'].iloc[-1] < df['SMA_50'].iloc[-21]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        c1 = latest['ADVOL_20'] > 50000
        c2 = latest['ADVOL_50'] > 50000
        c3 = (df['SMA_20'].iloc[-5:] >= df['SMA_50'].iloc[-5:]).all()
        c4 = not (latest['Close'] < latest['SMA_50'] and sma50_trend_dn_20)
        c5 = latest['Close'] > latest['SMA_10']
        c6 = latest['Close'] > latest['SMA_20']
        c7 = latest['SMA_10'] > latest['SMA_20']
        c8 = latest['Close'] > prev['Close']
        c9 = latest['ATR_1'] > (latest['ATR_20'] * 0.6)
        c10 = latest['Close'] > (latest['Low'] + ((latest['High'] - latest['Low']) * 0.4))

        if all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]):
            
            # Check our local NSE map first to bypass YF rate limits
            industry = ind_map.get(ticker)
            
            # If it's an obscure micro-cap not in the 500, fallback to YF
            if not industry or pd.isna(industry):
                try: industry = stock.info.get('industry', 'N/A')
                except: industry = 'N/A'
                
            return {"Ticker": ticker, "Industry": industry, "chart_data": df}
            
    except: pass
    return None


# =============================================================================
# LAYER 2: SETUP QUALITY SCORING (auto-runs on MIO results)
# =============================================================================

def _enrich_df(df):
    """Add extra indicators needed for scoring without touching original scan data."""
    if 'SMA_150' not in df.columns:
        df['SMA_150'] = ta.sma(df['Close'], length=150)
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = ta.sma(df['Close'], length=200)
    if 'ATR_5' not in df.columns:
        df['ATR_5'] = ta.atr(df['High'], df['Low'], df['Close'], length=5)
    if 'ATR_50' not in df.columns:
        df['ATR_50'] = ta.atr(df['High'], df['Low'], df['Close'], length=50)
    return df


def score_base_quality(df, latest):
    """Base Quality (0-33): depth from peak, ATR contraction, tight closes."""
    high_50 = df['High'].rolling(50).max().iloc[-1]
    dist_from_peak = (high_50 - latest['Close']) / high_50 * 100

    atr_5 = latest.get('ATR_5', 0) if not pd.isna(latest.get('ATR_5', np.nan)) else 0
    atr_50 = latest.get('ATR_50', atr_5) if not pd.isna(latest.get('ATR_50', np.nan)) else atr_5
    atr_contraction = (atr_50 - atr_5) / atr_50 * 100 if atr_50 > 0 else 0

    if 3 <= dist_from_peak <= 15:
        depth_pts = 13
    elif 1 <= dist_from_peak < 3:
        depth_pts = 9
    elif 15 < dist_from_peak <= 25:
        depth_pts = 6
    elif dist_from_peak < 1:
        depth_pts = 4
    else:
        depth_pts = 1

    if atr_contraction >= 30:
        tight_pts = 13
    elif atr_contraction >= 15:
        tight_pts = 9
    elif atr_contraction >= 0:
        tight_pts = 5
    else:
        tight_pts = 2

    tight_days = sum(
        abs(df['Close'].iloc[i] - df['Close'].iloc[i - 1]) / df['Close'].iloc[i - 1] < 0.01
        for i in range(-10, 0)
    )
    tight_bonus = min(tight_days, 7)

    score = min(depth_pts + tight_pts + tight_bonus, 33)
    details = {
        'Base Depth': f"{dist_from_peak:.1f}%",
        'ATR Contr': f"{atr_contraction:.0f}%",
        'Tight Days': tight_days
    }
    return score, details


def score_stage(df, latest):
    """Stage (0-33): MA stacking, 200DMA trend, 52W positioning."""
    sma150 = latest.get('SMA_150', np.nan)
    sma200 = latest.get('SMA_200', np.nan)

    sma_stack_perfect = False
    sma_stack_good = latest['Close'] > latest['SMA_20'] > latest['SMA_50']

    if not pd.isna(sma150):
        sma_stack_perfect = (
            latest['Close'] > latest['SMA_10'] > latest['SMA_20'] >
            latest['SMA_50'] > sma150
        )

    sma200_up = False
    if not pd.isna(sma200) and len(df) >= 21:
        sma200_col = df['SMA_200'].dropna()
        if len(sma200_col) >= 21:
            sma200_up = sma200_col.iloc[-1] > sma200_col.iloc[-21]

    high_252 = df['High'].max()
    low_252 = df['Low'].min()
    pct_above_52w_low = (latest['Close'] - low_252) / low_252 * 100
    pct_below_52w_high = (high_252 - latest['Close']) / high_252 * 100

    score = 0
    if sma_stack_perfect:
        score += 13
    elif sma_stack_good:
        score += 8
    else:
        score += 3

    if sma200_up:
        score += 7

    if pct_above_52w_low >= 30:
        score += 7
    elif pct_above_52w_low >= 15:
        score += 4

    if pct_below_52w_high <= 25:
        score += 6
    elif pct_below_52w_high <= 40:
        score += 3

    score = min(score, 33)
    stack_label = "Perfect" if sma_stack_perfect else ("Good" if sma_stack_good else "Weak")
    details = {
        'MA Stack': stack_label,
        '200DMA': "↑" if sma200_up else "↓",
        '> 52WL': f"{pct_above_52w_low:.0f}%",
        '< 52WH': f"{pct_below_52w_high:.0f}%"
    }
    return score, details


def score_timing(df, latest):
    """Timing (0-34): distance from 20DMA, volume dry-up, candle quality."""
    pct_from_20dma = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
    abs_dist = abs(pct_from_20dma)

    if abs_dist <= 1:
        ma_pts = 13
    elif abs_dist <= 3:
        ma_pts = 10
    elif abs_dist <= 5:
        ma_pts = 6
    else:
        ma_pts = 2

    vol_5 = df['Volume'].iloc[-5:].mean()
    vol_20 = latest['ADVOL_20']
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1

    if vol_ratio < 0.7:
        vol_pts = 11
    elif vol_ratio < 0.9:
        vol_pts = 7
    elif vol_ratio < 1.2:
        vol_pts = 4
    else:
        vol_pts = 1

    candle_range = latest['High'] - latest['Low']
    candle_body = abs(latest['Close'] - latest['Open'])
    body_ratio = candle_body / candle_range if candle_range > 0 else 0

    if body_ratio > 0.6 and latest['Close'] > latest['Open']:
        candle_pts = 10
    elif latest['Close'] > latest['Open']:
        candle_pts = 5
    else:
        candle_pts = 1

    score = min(ma_pts + vol_pts + candle_pts, 34)
    details = {
        'Dist 20DMA': f"{pct_from_20dma:+.1f}%",
        'Vol Ratio': f"{vol_ratio:.2f}",
        'Body': f"{body_ratio:.0%}"
    }
    return score, details


def score_setup(res):
    """Run all 3 scoring dimensions on a single MIO result."""
    df = _enrich_df(res['chart_data'].copy())
    latest = df.iloc[-1]

    base_score, base_det = score_base_quality(df, latest)
    stage_score, stage_det = score_stage(df, latest)
    timing_score, timing_det = score_timing(df, latest)

    total = base_score + stage_score + timing_score

    if total >= 75:
        grade = "A+"
    elif total >= 60:
        grade = "A"
    elif total >= 45:
        grade = "B"
    else:
        grade = "C"

    # SL = below base low (lowest low in last 50 bars with small buffer)
    base_low = df['Low'].iloc[-50:].min()
    entry = latest['Close']
    sl = round(base_low * 0.995, 2)
    risk = round((entry - sl) / entry * 100, 1)

    return {
        'Grade': grade,
        'Total': total,
        'Base': base_score,
        'Stage': stage_score,
        'Timing': timing_score,
        'Entry': round(entry, 2),
        'SL': sl,
        'Risk%': risk,
        'base_det': base_det,
        'stage_det': stage_det,
        'timing_det': timing_det
    }


# =============================================================================
# DASHBOARD
# =============================================================================

scan_mode = st.radio("Select Market Universe:", ["Nifty 500 (Fast)", "All NSE Stocks (~2,200 Stocks, Slower)"])

if st.button("🚀 Run Market Scan", type="primary"):
    
    if "Nifty 500" in scan_mode:
        tickers, ind_map = get_nifty_500()
    else:
        tickers, ind_map = get_all_nse()
    
    if not tickers:
        st.error("Failed to pull market data. The NSE server might be busy.")
    else:
        st.info(f"Crunching {len(tickers)} stocks... Please wait 60-90 seconds.")
        
        passed_results = []
        progress_bar = st.progress(0)
        
        check_func = partial(check_stock, ind_map=ind_map)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = executor.map(check_func, tickers)
            for i, result in enumerate(results):
                if result: passed_results.append(result)
                progress_bar.progress(min((i + 1) / len(tickers), 1.0))
                
        progress_bar.empty()

        if passed_results:
            st.success(f"🔥 Found {len(passed_results)} setups!")
            
            # =============================================================
            # AUTO-SCORE ALL RESULTS
            # =============================================================
            st.info("🧠 Scoring setups: Base Quality · Stage · Timing...")
            scored_results = []
            for res in passed_results:
                try:
                    scores = score_setup(res)
                    scored_results.append({**res, **scores})
                except:
                    scored_results.append({
                        **res,
                        'Grade': '?', 'Total': 0,
                        'Base': 0, 'Stage': 0, 'Timing': 0,
                        'Entry': 0, 'SL': 0, 'Risk%': 0,
                        'base_det': {}, 'stage_det': {}, 'timing_det': {}
                    })

            # Sort by total score descending
            scored_results.sort(key=lambda x: x.get('Total', 0), reverse=True)

            # --- 1. RANKED SCORECARD ---
            st.subheader("📋 Ranked Scorecard")
            table_data = []
            for s in scored_results:
                row = {
                    'Ticker': s['Ticker'],
                    'Industry': s['Industry'],
                    'Grade': s.get('Grade', '?'),
                    'Score': s.get('Total', 0),
                    'Base /33': s.get('Base', 0),
                    'Stage /33': s.get('Stage', 0),
                    'Timing /34': s.get('Timing', 0),
                    'CMP': s.get('Entry', 0),
                    'SL': s.get('SL', 0),
                    'Risk%': s.get('Risk%', 0),
                }
                for det_key in ['base_det', 'stage_det', 'timing_det']:
                    if s.get(det_key):
                        row.update(s[det_key])
                table_data.append(row)

            df_scored = pd.DataFrame(table_data)
            df_scored.index = df_scored.index + 1

            def color_grade(val):
                return {
                    "A+": "background-color:#00b060;color:white;font-weight:bold",
                    "A": "background-color:#4ade80;color:black;font-weight:bold",
                    "B": "background-color:#fbbf24;color:black",
                    "C": "background-color:#ff333a;color:white"
                }.get(val, "")

            def color_score(val):
                try:
                    v = int(val)
                    if v >= 75: return "background-color:#00b060;color:white"
                    elif v >= 60: return "background-color:#4ade80;color:black"
                    elif v >= 45: return "background-color:#fbbf24;color:black"
                    else: return "background-color:#ff333a;color:white"
                except: return ""

            styled = df_scored.style.map(color_grade, subset=['Grade']).map(color_score, subset=['Score'])
            st.dataframe(styled, use_container_width=True, height=500)

            st.divider()

            # --- 2. CHARTS + SCORE PANEL (side by side) ---
            st.subheader("📊 Charts + Setup Scores (Ranked by Quality)")

            for res in scored_results:
                grade = res.get('Grade', '?')
                total = res.get('Total', 0)
                grade_emoji = {"A+": "🟢", "A": "🟡", "B": "🟠", "C": "🔴"}.get(grade, "⚪")

                col_chart, col_score = st.columns([3, 1])

                with col_chart:
                    st.markdown(f"### {grade_emoji} {grade} | **{res['Ticker']}** — {total}/100 | {res['Industry']}")
                    df_chart = res['chart_data']
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

                    fig.add_trace(go.Candlestick(x=df_chart.index,
                                    open=df_chart['Open'], high=df_chart['High'],
                                    low=df_chart['Low'], close=df_chart['Close'],
                                    name='Price',
                                    increasing_line_color='#00b060', increasing_fillcolor='#00b060',
                                    decreasing_line_color='#ff333a', decreasing_fillcolor='#ff333a'), 
                                    row=1, col=1)

                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], 
                                             line=dict(color='orange', width=1.5), 
                                             name='20 DMA'), row=1, col=1)

                    colors = ['#00b060' if row['Close'] >= row['Open'] else '#ff333a' for index, row in df_chart.iterrows()]
                    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], 
                                         marker_color=colors, name='Volume'), row=2, col=1)

                    fig.update_layout(height=600, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                    fig.update_xaxes(rangeslider_visible=False)
                    
                    st.plotly_chart(fig, use_container_width=True)

                with col_score:
                    st.markdown("#### 🧠 Setup Score")

                    # Grade badge
                    g_color = {"A+": "#00b060", "A": "#4ade80", "B": "#fbbf24", "C": "#ff333a"}.get(grade, "#888")
                    st.markdown(
                        f'<div style="background:{g_color};color:white;padding:12px;'
                        f'border-radius:8px;text-align:center;font-size:28px;font-weight:bold;'
                        f'margin-bottom:10px;">{grade} · {total}/100</div>',
                        unsafe_allow_html=True
                    )

                    # Score bars
                    for label, val, mx in [('🧱 Base', res.get('Base', 0), 33),
                                           ('📶 Stage', res.get('Stage', 0), 33),
                                           ('⏱️ Timing', res.get('Timing', 0), 34)]:
                        pct = int(val / mx * 100)
                        bar_color = "#00b060" if pct >= 70 else "#fbbf24" if pct >= 50 else "#ff333a"
                        st.markdown(f"**{label}**: {val}/{mx}")
                        st.markdown(
                            f'<div style="background:#333;border-radius:4px;height:14px;margin-bottom:8px;">'
                            f'<div style="background:{bar_color};width:{pct}%;height:100%;border-radius:4px;"></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    st.divider()

                    # Detail metrics
                    base_det = res.get('base_det', {})
                    stage_det = res.get('stage_det', {})
                    timing_det = res.get('timing_det', {})

                    detail_lines = []
                    if base_det:
                        detail_lines.append(f"📐 Depth: {base_det.get('Base Depth', '-')}")
                        detail_lines.append(f"🔧 ATR Contr: {base_det.get('ATR Contr', '-')}")
                        detail_lines.append(f"📏 Tight Days: {base_det.get('Tight Days', '-')}")
                    if stage_det:
                        detail_lines.append(f"📊 MA Stack: {stage_det.get('MA Stack', '-')}")
                        detail_lines.append(f"📈 200DMA: {stage_det.get('200DMA', '-')}")
                        detail_lines.append(f"⬆ >52WL: {stage_det.get('> 52WL', '-')}")
                        detail_lines.append(f"⬇ <52WH: {stage_det.get('< 52WH', '-')}")
                    if timing_det:
                        detail_lines.append(f"🎯 20DMA: {timing_det.get('Dist 20DMA', '-')}")
                        detail_lines.append(f"📉 Vol: {timing_det.get('Vol Ratio', '-')}")

                    for line in detail_lines:
                        st.markdown(f"<small>{line}</small>", unsafe_allow_html=True)

                    st.divider()

                    # Entry / SL
                    st.metric("Entry", f"₹{res.get('Entry', 0)}")
                    st.metric("SL (Base Low)", f"₹{res.get('SL', 0)}")
                    st.metric("Risk", f"{res.get('Risk%', 0)}%")

                st.markdown("---")
        else:
            st.warning("No stocks matched the criteria today.")
