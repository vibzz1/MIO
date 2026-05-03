import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import concurrent.futures
import warnings
from nselib import capital_market
from functools import partial
from streamlit_lightweight_charts import renderLightweightCharts

warnings.filterwarnings("ignore")

# =============================================================================
# UI SETUP + CUSTOM STYLING
# =============================================================================
st.set_page_config(page_title="MIO Champ Screener", layout="wide")

st.markdown("""
<style>
    /* Dark modern theme overrides */
    .stApp { background-color: #0e1117; }
    
    /* Grade badges */
    .grade-badge {
        display: inline-block; padding: 8px 18px; border-radius: 8px;
        font-size: 26px; font-weight: 800; text-align: center;
        color: white; margin-bottom: 8px; width: 100%;
    }
    .grade-aplus { background: linear-gradient(135deg, #00b060, #00d47e); }
    .grade-a { background: linear-gradient(135deg, #22c55e, #4ade80); color: #000; }
    .grade-b { background: linear-gradient(135deg, #eab308, #fbbf24); color: #000; }
    .grade-c { background: linear-gradient(135deg, #dc2626, #ef4444); }
    
    /* Score bars */
    .score-bar-bg {
        background: #1e222d; border-radius: 6px; height: 18px;
        margin-bottom: 6px; overflow: hidden;
    }
    .score-bar-fill {
        height: 100%; border-radius: 6px; transition: width 0.5s ease;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700; color: white;
    }
    .bar-green { background: linear-gradient(90deg, #00b060, #00d47e); }
    .bar-yellow { background: linear-gradient(90deg, #eab308, #fbbf24); }
    .bar-red { background: linear-gradient(90deg, #dc2626, #ef4444); }
    
    /* Detail metrics */
    .detail-row {
        display: flex; justify-content: space-between; padding: 3px 0;
        font-size: 13px; border-bottom: 1px solid #1e222d;
    }
    .detail-label { color: #9ca3af; }
    .detail-value { color: #e5e7eb; font-weight: 600; }
    
    /* Section headers */
    .section-divider {
        background: linear-gradient(90deg, #00b060, transparent);
        height: 3px; border-radius: 2px; margin: 20px 0 10px 0;
    }
    
    /* Trade box */
    .trade-box {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 12px; margin-top: 8px;
    }
    .trade-entry { color: #4ade80; font-size: 18px; font-weight: 700; }
    .trade-sl { color: #ef4444; font-size: 18px; font-weight: 700; }
    .trade-risk { color: #fbbf24; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 MIO Champion Setup Screener")
st.markdown("*Automated scan for high-probability momentum setups with quality scoring*")

# =============================================================================
# DATA FETCHING (ORIGINAL — UNTOUCHED)
# =============================================================================
@st.cache_data(ttl=3600)
def get_nifty_500():
    try:
        n50 = capital_market.nifty50_equity_list()
        nn50 = capital_market.niftynext50_equity_list()
        mid150 = capital_market.niftymidcap150_equity_list()
        sml250 = capital_market.niftysmallcap250_equity_list()
        df = pd.concat([n50, nn50, mid150, sml250], ignore_index=True)
        tickers = list(set(df['Symbol'].tolist()))
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
        _, ind_map = get_nifty_500()
        return tickers, ind_map
    except: return [], {}

# =============================================================================
# SCREENER LOGIC (ORIGINAL — UNTOUCHED)
# =============================================================================
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
            industry = ind_map.get(ticker)
            if not industry or pd.isna(industry):
                try: industry = stock.info.get('industry', 'N/A')
                except: industry = 'N/A'
            return {"Ticker": ticker, "Industry": industry, "chart_data": df}

    except: pass
    return None

# =============================================================================
# LAYER 2: SCORING ENGINE
# =============================================================================

def _enrich_df(df):
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
    high_50 = df['High'].rolling(50).max().iloc[-1]
    dist_from_peak = (high_50 - latest['Close']) / high_50 * 100
    atr_5 = latest.get('ATR_5', 0) if not pd.isna(latest.get('ATR_5', np.nan)) else 0
    atr_50 = latest.get('ATR_50', atr_5) if not pd.isna(latest.get('ATR_50', np.nan)) else atr_5
    atr_contraction = (atr_50 - atr_5) / atr_50 * 100 if atr_50 > 0 else 0

    if 3 <= dist_from_peak <= 15: depth_pts = 13
    elif 1 <= dist_from_peak < 3: depth_pts = 9
    elif 15 < dist_from_peak <= 25: depth_pts = 6
    elif dist_from_peak < 1: depth_pts = 4
    else: depth_pts = 1

    if atr_contraction >= 30: tight_pts = 13
    elif atr_contraction >= 15: tight_pts = 9
    elif atr_contraction >= 0: tight_pts = 5
    else: tight_pts = 2

    tight_days = sum(abs(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.01 for i in range(-10, 0))
    tight_bonus = min(tight_days, 7)
    score = min(depth_pts + tight_pts + tight_bonus, 33)
    return score, {'Base Depth': f"{dist_from_peak:.1f}%", 'ATR Contr': f"{atr_contraction:.0f}%", 'Tight Days': tight_days}


def score_stage(df, latest):
    sma150 = latest.get('SMA_150', np.nan)
    sma200 = latest.get('SMA_200', np.nan)
    sma_stack_perfect = False
    sma_stack_good = latest['Close'] > latest['SMA_20'] > latest['SMA_50']
    if not pd.isna(sma150):
        sma_stack_perfect = latest['Close'] > latest['SMA_10'] > latest['SMA_20'] > latest['SMA_50'] > sma150

    sma200_up = False
    if not pd.isna(sma200) and len(df) >= 21:
        sma200_col = df['SMA_200'].dropna()
        if len(sma200_col) >= 21:
            sma200_up = sma200_col.iloc[-1] > sma200_col.iloc[-21]

    high_252, low_252 = df['High'].max(), df['Low'].min()
    pct_above_low = (latest['Close'] - low_252) / low_252 * 100
    pct_below_high = (high_252 - latest['Close']) / high_252 * 100

    score = 13 if sma_stack_perfect else (8 if sma_stack_good else 3)
    if sma200_up: score += 7
    score += 7 if pct_above_low >= 30 else (4 if pct_above_low >= 15 else 0)
    score += 6 if pct_below_high <= 25 else (3 if pct_below_high <= 40 else 0)
    score = min(score, 33)

    stack_label = "Perfect" if sma_stack_perfect else ("Good" if sma_stack_good else "Weak")
    return score, {'MA Stack': stack_label, '200DMA': "↑" if sma200_up else "↓",
                   '>52WL': f"{pct_above_low:.0f}%", '<52WH': f"{pct_below_high:.0f}%"}


def score_timing(df, latest):
    pct_20dma = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
    abs_dist = abs(pct_20dma)
    if abs_dist <= 1: ma_pts = 13
    elif abs_dist <= 3: ma_pts = 10
    elif abs_dist <= 5: ma_pts = 6
    else: ma_pts = 2

    vol_5 = df['Volume'].iloc[-5:].mean()
    vol_20 = latest['ADVOL_20']
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1
    if vol_ratio < 0.7: vol_pts = 11
    elif vol_ratio < 0.9: vol_pts = 7
    elif vol_ratio < 1.2: vol_pts = 4
    else: vol_pts = 1

    cr = latest['High'] - latest['Low']
    body = abs(latest['Close'] - latest['Open'])
    br = body / cr if cr > 0 else 0
    if br > 0.6 and latest['Close'] > latest['Open']: candle_pts = 10
    elif latest['Close'] > latest['Open']: candle_pts = 5
    else: candle_pts = 1

    score = min(ma_pts + vol_pts + candle_pts, 34)
    return score, {'Dist 20DMA': f"{pct_20dma:+.1f}%", 'Vol Ratio': f"{vol_ratio:.2f}", 'Body': f"{br:.0%}"}


def score_setup(res):
    df = _enrich_df(res['chart_data'].copy())
    latest = df.iloc[-1]
    base_s, base_d = score_base_quality(df, latest)
    stage_s, stage_d = score_stage(df, latest)
    timing_s, timing_d = score_timing(df, latest)
    total = base_s + stage_s + timing_s

    if total >= 75: grade = "A+"
    elif total >= 60: grade = "A"
    elif total >= 45: grade = "B"
    else: grade = "C"

    base_low = df['Low'].iloc[-50:].min()
    entry = latest['Close']
    sl = round(base_low * 0.995, 2)
    risk = round((entry - sl) / entry * 100, 1)

    return {'Grade': grade, 'Total': total, 'Base': base_s, 'Stage': stage_s,
            'Timing': timing_s, 'Entry': round(entry, 2), 'SL': sl, 'Risk%': risk,
            'base_det': base_d, 'stage_det': stage_d, 'timing_det': timing_d}


# =============================================================================
# TV-STYLE CHART RENDERER
# =============================================================================

def render_tv_chart(df, ticker, grade, total):
    candle_data = [{"time": idx.strftime("%Y-%m-%d"), "open": round(r['Open'],2),
                    "high": round(r['High'],2), "low": round(r['Low'],2),
                    "close": round(r['Close'],2)} for idx, r in df.iterrows()]

    volume_data = [{"time": idx.strftime("%Y-%m-%d"), "value": int(r['Volume']),
                    "color": "#26a69a80" if r['Close'] >= r['Open'] else "#ef535080"}
                   for idx, r in df.iterrows()]

    sma20_data = [{"time": idx.strftime("%Y-%m-%d"), "value": round(r['SMA_20'],2)}
                  for idx, r in df.iterrows() if not pd.isna(r.get('SMA_20', np.nan))]

    sma50_data = [{"time": idx.strftime("%Y-%m-%d"), "value": round(r['SMA_50'],2)}
                  for idx, r in df.iterrows() if not pd.isna(r.get('SMA_50', np.nan))]

    chart_opts = [
        {
            "height": 400,
            "layout": {"background": {"type": "solid", "color": "#131722"},
                       "textColor": "#d1d4dc", "fontSize": 12},
            "grid": {"vertLines": {"color": "#1e222d"}, "horzLines": {"color": "#1e222d"}},
            "crosshair": {"mode": 0},
            "priceScale": {"borderColor": "#2d3748"},
            "timeScale": {"borderColor": "#2d3748", "timeVisible": False},
            "watermark": {"visible": True, "fontSize": 32, "horzAlign": "center",
                          "vertAlign": "center", "color": "rgba(255,255,255,0.03)",
                          "text": f"{ticker} | {grade} ({total})"}
        },
        {
            "height": 120,
            "layout": {"background": {"type": "solid", "color": "#131722"},
                       "textColor": "#d1d4dc", "fontSize": 10},
            "grid": {"vertLines": {"color": "#1e222d"}, "horzLines": {"color": "#1e222d"}},
            "timeScale": {"borderColor": "#2d3748", "timeVisible": False}
        }
    ]

    series_price = [
        {"type": "Candlestick", "data": candle_data,
         "options": {"upColor": "#26a69a", "downColor": "#ef5350",
                     "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
                     "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"}},
        {"type": "Line", "data": sma20_data,
         "options": {"color": "#ff9800", "lineWidth": 2, "title": "20 DMA"}},
        {"type": "Line", "data": sma50_data,
         "options": {"color": "#ab47bc", "lineWidth": 2, "title": "50 DMA"}}
    ]

    series_vol = [
        {"type": "Histogram", "data": volume_data,
         "options": {"priceFormat": {"type": "volume"}, "priceScaleId": "vol"}}
    ]

    renderLightweightCharts([
        {"chart": chart_opts[0], "series": series_price},
        {"chart": chart_opts[1], "series": series_vol}
    ], key=f"tv_{ticker}")


# =============================================================================
# SCORE PANEL HTML RENDERER
# =============================================================================

def render_score_panel(res):
    grade = res.get('Grade', '?')
    total = res.get('Total', 0)
    grade_class = {'A+': 'grade-aplus', 'A': 'grade-a', 'B': 'grade-b', 'C': 'grade-c'}.get(grade, 'grade-c')

    st.markdown(f'<div class="grade-badge {grade_class}">{grade} · {total}/100</div>', unsafe_allow_html=True)

    dims = [('🧱 Base', res.get('Base', 0), 33),
            ('📶 Stage', res.get('Stage', 0), 33),
            ('⏱️ Timing', res.get('Timing', 0), 34)]

    for label, val, mx in dims:
        pct = int(val / mx * 100)
        bar_class = "bar-green" if pct >= 70 else "bar-yellow" if pct >= 50 else "bar-red"
        st.markdown(f"**{label}**")
        st.markdown(
            f'<div class="score-bar-bg">'
            f'<div class="score-bar-fill {bar_class}" style="width:{pct}%">{val}/{mx}</div>'
            f'</div>', unsafe_allow_html=True)

    # Details
    st.markdown('<div style="margin-top:8px">', unsafe_allow_html=True)
    all_details = {}
    for k in ['base_det', 'stage_det', 'timing_det']:
        all_details.update(res.get(k, {}))
    for label, value in all_details.items():
        st.markdown(f'<div class="detail-row"><span class="detail-label">{label}</span>'
                    f'<span class="detail-value">{value}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Trade box
    st.markdown(
        f'<div class="trade-box">'
        f'<div class="trade-entry">Entry ₹{res.get("Entry", 0)}</div>'
        f'<div class="trade-sl">SL ₹{res.get("SL", 0)}</div>'
        f'<div class="trade-risk">Risk {res.get("Risk%", 0)}%</div>'
        f'</div>', unsafe_allow_html=True)


# =============================================================================
# DASHBOARD
# =============================================================================

scan_mode = st.radio("Select Market Universe:", ["Nifty 500 (Fast)", "All NSE Stocks (~2,200 Stocks, Slower)"], horizontal=True)

if st.button("🚀 Run Market Scan", type="primary"):

    if "Nifty 500" in scan_mode:
        tickers, ind_map = get_nifty_500()
    else:
        tickers, ind_map = get_all_nse()

    if not tickers:
        st.error("Failed to pull market data. The NSE server might be busy.")
    else:
        st.info(f"⚡ Scanning {len(tickers)} stocks...")

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

            # --- AUTO-SCORE ---
            st.info("🧠 Scoring: Base Quality · Stage · Timing...")
            scored_results = []
            for res in passed_results:
                try:
                    scores = score_setup(res)
                    scored_results.append({**res, **scores})
                except:
                    scored_results.append({**res, 'Grade': '?', 'Total': 0, 'Base': 0,
                        'Stage': 0, 'Timing': 0, 'Entry': 0, 'SL': 0, 'Risk%': 0,
                        'base_det': {}, 'stage_det': {}, 'timing_det': {}})

            scored_results.sort(key=lambda x: x.get('Total', 0), reverse=True)

            # --- RANKED SCORECARD TABLE ---
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📋 Ranked Scorecard")

            table_data = []
            for s in scored_results:
                row = {'Ticker': s['Ticker'], 'Industry': s['Industry'],
                       'Grade': s.get('Grade','?'), 'Score': s.get('Total',0),
                       'Base': s.get('Base',0), 'Stage': s.get('Stage',0),
                       'Timing': s.get('Timing',0), 'CMP': s.get('Entry',0),
                       'SL': s.get('SL',0), 'Risk%': s.get('Risk%',0)}
                for dk in ['base_det', 'stage_det', 'timing_det']:
                    if s.get(dk): row.update(s[dk])
                table_data.append(row)

            df_scored = pd.DataFrame(table_data)
            df_scored.index = df_scored.index + 1

            def color_grade(val):
                return {"A+": "background-color:#00b060;color:white;font-weight:bold",
                        "A": "background-color:#4ade80;color:black;font-weight:bold",
                        "B": "background-color:#fbbf24;color:black",
                        "C": "background-color:#ff333a;color:white"}.get(val, "")

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

            # --- TV CHARTS + SCORE PANELS ---
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📈 Charts + Setup Scores (Ranked)")

            for res in scored_results:
                grade = res.get('Grade', '?')
                total = res.get('Total', 0)
                grade_emoji = {"A+": "🟢", "A": "🟡", "B": "🟠", "C": "🔴"}.get(grade, "⚪")

                col_chart, col_score = st.columns([3, 1])

                with col_chart:
                    st.markdown(f"### {grade_emoji} **{res['Ticker']}** | {res['Industry']}")
                    render_tv_chart(res['chart_data'], res['Ticker'], grade, total)

                with col_score:
                    render_score_panel(res)

                st.markdown("---")
        else:
            st.warning("No stocks matched the criteria today.")
