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
    .stApp { background-color: #0e1117; }
    .grade-badge {
        display: inline-block; padding: 8px 18px; border-radius: 8px;
        font-size: 26px; font-weight: 800; text-align: center;
        color: white; margin-bottom: 8px; width: 100%;
    }
    .grade-aplus { background: linear-gradient(135deg, #00b060, #00d47e); }
    .grade-a { background: linear-gradient(135deg, #22c55e, #4ade80); color: #000; }
    .grade-b { background: linear-gradient(135deg, #eab308, #fbbf24); color: #000; }
    .grade-c { background: linear-gradient(135deg, #dc2626, #ef4444); }
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
    .detail-row {
        display: flex; justify-content: space-between; padding: 3px 0;
        font-size: 13px; border-bottom: 1px solid #1e222d;
    }
    .detail-label { color: #9ca3af; }
    .detail-value { color: #e5e7eb; font-weight: 600; }
    .section-divider {
        background: linear-gradient(90deg, #00b060, transparent);
        height: 3px; border-radius: 2px; margin: 20px 0 10px 0;
    }
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
st.markdown("*Scan → Score → Rank · Calibrated against 41 real setups*")

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
# LAYER 2: SCORING ENGINE v2
# Calibrated against 41 real charts from Afzal's Champions Club
#
# What every A+ setup has (studied from CAT, TSM, MU, Delong, TRT,
# TDPOWERSYS, DATAPATTNS, INDIANB, HINDCOPPER, POCL, FEDFINA, GRMOVER,
# THANGAMAYL, SENORES, VEDL, MOTHERSON, NMDC, BANKBARODA, ONGC, AVANTI,
# CHENNPETRO, ANUPAMRAS, UPL, Luxshare, Resonac, Daewon Cable, etc.):
#
# 1. STRONG PRIOR MOVE: 30-200% clean staircase before base
# 2. VOLUME 3-ACT STORY: Expansion (move) → Contraction (base) → Expansion (trigger)
# 3. BASE AT RESISTANCE: Consolidation near prior highs, not in middle of nowhere
# 4. CLEAN STRUCTURE: Not choppy — smooth candles, clear direction
# 5. SETUP CANDLE: Green, strong body, confirming end of base
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


# ------------------------------------------------------------------
# DIMENSION 1: BASE QUALITY (0-33)
# What it checks: Is the consolidation well-formed?
# A+ bases: tight, shallow (3-15% from peak), volume drying up,
# ATR contracting, forming at/near resistance
# ------------------------------------------------------------------
def score_base_quality(df, latest):
    # --- Base depth from recent high ---
    high_50 = df['High'].rolling(50).max().iloc[-1]
    dist_from_peak = (high_50 - latest['Close']) / high_50 * 100

    # --- ATR contraction (current vs historical) ---
    atr_5 = latest.get('ATR_5', 0) if not pd.isna(latest.get('ATR_5', np.nan)) else 0
    atr_50 = latest.get('ATR_50', atr_5) if not pd.isna(latest.get('ATR_50', np.nan)) else atr_5
    atr_contraction = (atr_50 - atr_5) / atr_50 * 100 if atr_50 > 0 else 0

    # --- VOLUME 3-ACT STORY (NEW) ---
    # Compare volume during base (last 15 bars) vs prior move (bars -50 to -15)
    # Every A+ setup shows volume drying up in the base
    vol_base = df['Volume'].iloc[-15:].mean()
    vol_prior = df['Volume'].iloc[-50:-15].mean() if len(df) >= 50 else vol_base
    vol_contraction = (vol_prior - vol_base) / vol_prior * 100 if vol_prior > 0 else 0

    # --- BASE AT RESISTANCE (NEW) ---
    # Is price near the highest high of last 100 bars? (forming at resistance)
    # A+ setups form bases AT or NEAR prior highs, not in the middle
    high_100 = df['High'].iloc[-100:].max() if len(df) >= 100 else df['High'].max()
    pct_from_resistance = (high_100 - latest['Close']) / high_100 * 100

    # --- Tight closes ---
    tight_days = sum(
        abs(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.015
        for i in range(-10, 0)
    )

    # --- SCORING ---
    score = 0

    # Base depth (0-8): 3-15% from peak is ideal
    if 3 <= dist_from_peak <= 15:
        score += 8
    elif 1 <= dist_from_peak < 3:
        score += 6
    elif 15 < dist_from_peak <= 25:
        score += 4
    elif dist_from_peak < 1:
        score += 3  # at highs, no real base formed
    else:
        score += 1

    # ATR contraction (0-8): volatility drying up in base
    if atr_contraction >= 30:
        score += 8
    elif atr_contraction >= 15:
        score += 6
    elif atr_contraction >= 0:
        score += 3
    else:
        score += 1  # volatility expanding = bad

    # Volume contraction in base (0-8): THE 3-act story
    if vol_contraction >= 40:
        score += 8  # major dry-up — textbook
    elif vol_contraction >= 20:
        score += 6
    elif vol_contraction >= 5:
        score += 4
    else:
        score += 1  # no contraction — not a real base

    # Base at resistance (0-5): near prior highs = better
    if pct_from_resistance <= 5:
        score += 5  # at resistance — breakout territory
    elif pct_from_resistance <= 15:
        score += 3  # building toward resistance
    else:
        score += 1  # far from resistance

    # Tight closes bonus (0-4)
    score += min(tight_days, 4)

    score = min(score, 33)

    return score, {
        'Base Depth': f"{dist_from_peak:.1f}%",
        'ATR Contr': f"{atr_contraction:.0f}%",
        'Vol Contr': f"{vol_contraction:.0f}%",
        'Near Res': f"{pct_from_resistance:.0f}%",
        'Tight Days': tight_days
    }


# ------------------------------------------------------------------
# DIMENSION 2: STAGE + PRIOR MOVE (0-33)
# What it checks: Is the stock in the RIGHT part of the cycle?
# AND was the move INTO the base clean and strong?
#
# A+ setups: Strong 30-200% clean staircase → then base
# 200DMA rising, early S2 (1st-2nd base), MA perfectly stacked
# BAD: Choppy recovery (Berger), late S2 (3rd+ base), 200DMA down
# ------------------------------------------------------------------
def score_stage(df, latest):
    sma150 = latest.get('SMA_150', np.nan)
    sma200 = latest.get('SMA_200', np.nan)
    sma50 = latest['SMA_50']
    sma20 = latest['SMA_20']

    # --- MA Stacking ---
    sma_stack_perfect = False
    sma_stack_good = latest['Close'] > sma20 > sma50

    if not pd.isna(sma150) and not pd.isna(sma200):
        sma_stack_perfect = (
            latest['Close'] > latest['SMA_10'] > sma20 > sma50 > sma150 > sma200
        )

    # --- 200 DMA Trend ---
    sma200_up = False
    sma200_flat = False
    sma200_val = np.nan
    has_200dma = False

    sma200_col = df['SMA_200'].dropna() if 'SMA_200' in df.columns else pd.Series(dtype=float)
    if len(sma200_col) >= 2:
        has_200dma = True
        sma200_val = sma200_col.iloc[-1]
        lookback = min(len(sma200_col) - 1, 21)
        pct_change_200 = (sma200_col.iloc[-1] / sma200_col.iloc[-1 - lookback] - 1) * 100
        sma200_up = pct_change_200 > 0.5
        sma200_flat = -0.5 <= pct_change_200 <= 0.5

    if not has_200dma and 'SMA_150' in df.columns:
        sma150_col = df['SMA_150'].dropna()
        if len(sma150_col) >= 10:
            lookback = min(len(sma150_col) - 1, 21)
            pct_change_150 = (sma150_col.iloc[-1] / sma150_col.iloc[-1 - lookback] - 1) * 100
            sma200_up = pct_change_150 > 0.5
            sma200_flat = -0.5 <= pct_change_150 <= 0.5
            sma200_val = sma150_col.iloc[-1]
            has_200dma = True

    price_above_200 = latest['Close'] > sma200_val if has_200dma else False
    sma50_above_200 = sma50 > sma200_val if has_200dma else False
    dist_from_200 = ((latest['Close'] - sma200_val) / sma200_val * 100) if has_200dma and sma200_val > 0 else 0

    # --- BASE COUNT (FIXED) ---
    # A real base needs a meaningful pullback (≥5% from local high)
    # Not every 20DMA wiggle. This was overcounting before.
    base_count = 0
    lookback_bars = min(len(df), 120)
    local_high = df['Close'].iloc[-lookback_bars]
    in_pullback = False

    for i in range(-lookback_bars, 0):
        close_i = df['Close'].iloc[i]
        # Track running high
        if close_i > local_high:
            local_high = close_i
        # Pullback starts when price drops 5%+ from local high
        pullback_pct = (local_high - close_i) / local_high * 100
        if pullback_pct >= 5 and not in_pullback:
            in_pullback = True
        # Pullback ends when price recovers above local high
        elif in_pullback and close_i >= local_high * 0.97:
            base_count += 1
            in_pullback = False
            local_high = close_i

    # --- PRIOR MOVE QUALITY (NEW — the biggest missing piece) ---
    # Every A+ setup has a strong, clean move BEFORE the base
    # Measure: price gain from lowest low in last 200 bars to recent high
    low_200 = df['Low'].iloc[-min(len(df), 200):].min()
    high_recent = df['High'].iloc[-50:].max()
    prior_move_pct = (high_recent / low_200 - 1) * 100 if low_200 > 0 else 0

    # Move cleanliness: ratio of up-closes to total in the move period
    move_period = df.iloc[-min(len(df), 120):-15]
    if len(move_period) > 10:
        up_days = (move_period['Close'] > move_period['Open']).sum()
        move_cleanliness = up_days / len(move_period) * 100
    else:
        move_cleanliness = 50

    # --- STAGE IDENTIFICATION ---
    is_early_s2 = sma200_up and price_above_200 and sma50_above_200 and base_count <= 1
    is_mid_s2 = sma200_up and price_above_200 and sma50_above_200 and base_count == 2
    is_late_s2 = sma200_up and price_above_200 and base_count >= 3
    is_s1b = (sma200_flat or (not sma200_up and sma50_above_200)) and price_above_200
    is_s1_early = not sma200_up and not sma200_flat and price_above_200
    is_s4_s1 = not sma200_up and not price_above_200

    # --- SCORING ---
    score = 0

    # MA stacking (0-6)
    if sma_stack_perfect:
        score += 6
    elif sma_stack_good:
        score += 4
    else:
        score += 1

    # 200DMA health (0-6)
    if sma200_up:
        score += 6
    elif sma200_flat:
        score += 3
    else:
        score += 0

    # Stage position (0-8)
    if is_early_s2:
        score += 8
    elif is_s1b:
        score += 6
    elif is_mid_s2:
        score += 5
    elif is_late_s2:
        score += 2
    elif is_s1_early:
        score += 1
    else:
        score += 0

    # Prior move quality (0-8) — THE KEY DIFFERENTIATOR
    # CAT: 60%+ move. HINDCOPPER: 100%+. Delong: 200%+. All A+.
    # Berger: barely recovering from decline. That's the difference.
    if prior_move_pct >= 80:
        score += 8  # Monster move — TDPOWERSYS, HINDCOPPER, Delong territory
    elif prior_move_pct >= 50:
        score += 6  # Strong move — CAT, TSM, INDIANB territory
    elif prior_move_pct >= 30:
        score += 4  # Decent move
    elif prior_move_pct >= 15:
        score += 2  # Weak move
    else:
        score += 0  # No real prior move — not a setup

    # Move cleanliness bonus (0-5): clean staircase > choppy
    if move_cleanliness >= 55:
        score += 5  # More up days than down — clean trend
    elif move_cleanliness >= 50:
        score += 3
    else:
        score += 1  # Choppy — Afzal explicitly dislikes this (Sagility comment)

    # --- HARD PENALTIES ---
    if not sma200_up and not sma200_flat:
        score = min(score, 12)
    if not price_above_200:
        score -= 4
    if not sma50_above_200:
        score -= 3
    if dist_from_200 > 35 and sma200_up:
        score -= 2  # Overextended

    score = max(min(score, 33), 0)

    # --- Labels ---
    if is_early_s2: stg_label = "S2·1st"
    elif is_s1b: stg_label = "S1b"
    elif is_mid_s2: stg_label = "S2·2nd"
    elif is_late_s2: stg_label = f"S2·{base_count}th"
    elif is_s1_early: stg_label = "S1"
    elif is_s4_s1: stg_label = "S4/1"
    else: stg_label = "?"

    stack_label = "Perfect" if sma_stack_perfect else ("Good" if sma_stack_good else "Weak")

    return score, {
        'MA Stack': stack_label,
        '200DMA': "↑" if sma200_up else ("→" if sma200_flat else "↓"),
        'Stg': stg_label,
        'Bases': base_count,
        'Prior Move': f"{prior_move_pct:.0f}%",
        'Trend Clean': f"{move_cleanliness:.0f}%"
    }


# ------------------------------------------------------------------
# DIMENSION 3: TIMING (0-34)
# What it checks: Is THIS the right moment to enter?
# A+ timing: near 20DMA, volume dried up then expanding on trigger,
# strong green candle with body in upper part, near breakout level
# ------------------------------------------------------------------
def score_timing(df, latest):
    # --- Distance from 20DMA ---
    pct_20dma = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
    abs_dist = abs(pct_20dma)

    if abs_dist <= 2:
        ma_pts = 10  # Right at the MA — ideal pullback entry
    elif abs_dist <= 4:
        ma_pts = 7
    elif abs_dist <= 6:
        ma_pts = 4
    else:
        ma_pts = 1  # Extended — not ideal timing

    # --- Volume on trigger day vs base average ---
    vol_today = latest['Volume']
    vol_base_avg = df['Volume'].iloc[-15:-1].mean()  # exclude today
    vol_expansion = (vol_today / vol_base_avg) if vol_base_avg > 0 else 1

    if vol_expansion >= 2.0:
        vol_pts = 8  # 2x+ volume on trigger — textbook (ANUPAMRAS had this)
    elif vol_expansion >= 1.5:
        vol_pts = 6
    elif vol_expansion >= 1.0:
        vol_pts = 3
    else:
        vol_pts = 1  # Below average volume — weak trigger

    # --- Candle quality ---
    cr = latest['High'] - latest['Low']
    body = abs(latest['Close'] - latest['Open'])
    br = body / cr if cr > 0 else 0
    close_position = (latest['Close'] - latest['Low']) / cr if cr > 0 else 0.5

    if br > 0.6 and latest['Close'] > latest['Open'] and close_position > 0.7:
        candle_pts = 8  # Strong green, closed near high — textbook trigger
    elif br > 0.4 and latest['Close'] > latest['Open']:
        candle_pts = 5  # Decent green
    elif latest['Close'] > latest['Open']:
        candle_pts = 3  # Weak green
    else:
        candle_pts = 0  # Red candle — not a trigger

    # --- Breakout proximity (NEW) ---
    # How close is price to breaking above recent resistance?
    # Every Afzal trade has a horizontal line drawn at resistance
    high_20 = df['High'].iloc[-20:].max()
    high_50 = df['High'].iloc[-50:].max()
    pct_from_breakout = (high_50 - latest['Close']) / high_50 * 100

    if latest['Close'] >= high_20:
        bkout_pts = 8  # Breaking out TODAY — CAT, TSM, Delong moment
    elif pct_from_breakout <= 3:
        bkout_pts = 6  # Very close to breakout
    elif pct_from_breakout <= 8:
        bkout_pts = 4  # Approaching
    else:
        bkout_pts = 1  # Far from breakout

    score = min(ma_pts + vol_pts + candle_pts + bkout_pts, 34)

    return score, {
        'Dist 20DMA': f"{pct_20dma:+.1f}%",
        'Vol Exp': f"{vol_expansion:.1f}x",
        'Body': f"{br:.0%}",
        'Close Pos': f"{close_position:.0%}",
        'Near Bkout': f"{pct_from_breakout:.1f}%"
    }


# ------------------------------------------------------------------
# FINAL SCORE
# ------------------------------------------------------------------
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

    # SL = below base low (lowest low in last 50 bars)
    base_low = df['Low'].iloc[-50:].min()
    entry = latest['Close']
    sl = round(base_low * 0.995, 2)
    risk = round((entry - sl) / entry * 100, 1)

    return {'Grade': grade, 'Total': total, 'Base': base_s, 'Stage': stage_s,
            'Timing': timing_s, 'Entry': round(entry, 2), 'SL': sl, 'Risk%': risk,
            'base_det': base_d, 'stage_det': stage_d, 'timing_det': timing_d}


# =============================================================================
# TV-STYLE CHART RENDERER (20DMA only, no label)
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
         "options": {"color": "#ff9800", "lineWidth": 2, "title": ""}}
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
# SCORE PANEL RENDERER
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

            st.info("🧠 Scoring: Base · Stage · Timing...")
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

            # --- RANKED SCORECARD ---
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
