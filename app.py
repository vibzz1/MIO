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
st.markdown("*Scan → Score → Rank · Calibrated against real setups*")

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
import threading
_diag_lock = threading.Lock()
_diag = {"data_fail": 0, "low_data": 0, "c1_vol20": 0, "c2_vol50": 0, "c3_sma_cross": 0,
         "c4_below50_dn": 0, "c5_below10": 0, "c6_below20": 0, "c7_sma10_20": 0,
         "c8_red_day": 0, "c9_low_atr": 0, "c10_close_pos": 0, "passed": 0, "checked": 0}

def _diag_reset():
    for k in _diag: _diag[k] = 0

def check_stock(ticker, ind_map):
    symbol = f"{ticker}.NS"
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y", auto_adjust=False)
        if len(df) < 70:
            with _diag_lock: _diag["low_data"] += 1
            return None

        df.dropna(inplace=True)
        # Use unadjusted Close to match MIO's raw NSE data
        # yfinance auto_adjust=False gives raw OHLC + separate 'Adj Close'
        if 'Adj Close' in df.columns:
            df.drop(columns=['Adj Close'], inplace=True)
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR_20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        df['ATR_1']  = ta.true_range(df['High'], df['Low'], df['Close'])
        df['ADVOL_20'] = df['Volume'].rolling(20).mean()
        df['ADVOL_50'] = df['Volume'].rolling(50).mean()
        # Dollar volume = price * shares traded (MIO's advol)
        df['DVOL_20'] = (df['Close'] * df['Volume']).rolling(20).mean()
        df['DVOL_50'] = (df['Close'] * df['Volume']).rolling(50).mean()

        df.dropna(inplace=True)
        if len(df) < 22:
            with _diag_lock: _diag["low_data"] += 1
            return None

        with _diag_lock: _diag["checked"] += 1

        sma50_trend_dn_20 = df['SMA_50'].iloc[-1] < df['SMA_50'].iloc[-21]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # MIO: advol(20) > 100 AND advol(50) > 100
        # Problem: VCP-pattern stocks have contracted 20d volume during base
        # Fix: pass if EITHER 20d or 50d avg volume > 100K (50d captures pre-base liquidity)
        c1 = (latest['ADVOL_20'] > 100000) or (latest['ADVOL_50'] > 100000)
        c2 = c1  # single liquidity gate
        # MIO: !(sma(20)<sma(50))@{0..20}
        # @{0..20} = condition at ALL bars. ! negates the result.
        # = NOT(sma20 < sma50 at ALL 21 bars) = sma20 >= sma50 at SOME bar in 21
        c3 = not (df['SMA_20'].iloc[-21:] < df['SMA_50'].iloc[-21:]).all()
        c4 = not (latest['Close'] < latest['SMA_50'] and sma50_trend_dn_20)
        c5 = latest['Close'] > latest['SMA_10']
        c6 = latest['Close'] > latest['SMA_20']
        c7 = latest['SMA_10'] > latest['SMA_20']
        c8 = latest['Close'] > prev['Close']
        c9 = latest['ATR_1'] > (latest['ATR_20'] * 0.6)
        c10 = latest['Close'] > (latest['Low'] + ((latest['High'] - latest['Low']) * 0.4))

        # Track which condition fails first
        if not c1:
            with _diag_lock: _diag["c1_vol20"] += 1
        elif not c2:
            with _diag_lock: _diag["c2_vol50"] += 1
        elif not c3:
            with _diag_lock: _diag["c3_sma_cross"] += 1
        elif not c4:
            with _diag_lock: _diag["c4_below50_dn"] += 1
        elif not c5:
            with _diag_lock: _diag["c5_below10"] += 1
        elif not c6:
            with _diag_lock: _diag["c6_below20"] += 1
        elif not c7:
            with _diag_lock: _diag["c7_sma10_20"] += 1
        elif not c8:
            with _diag_lock: _diag["c8_red_day"] += 1
        elif not c9:
            with _diag_lock: _diag["c9_low_atr"] += 1
        elif not c10:
            with _diag_lock: _diag["c10_close_pos"] += 1

        if all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]):
            with _diag_lock: _diag["passed"] += 1
            industry = ind_map.get(ticker)
            if not industry or pd.isna(industry):
                try: industry = stock.info.get('industry', 'N/A')
                except: industry = 'N/A'
            return {"Ticker": ticker, "Industry": industry, "chart_data": df}

    except:
        with _diag_lock: _diag["data_fail"] += 1
    return None

# =============================================================================
# LAYER 2: SCORING ENGINE v3 — FIXED FOR BREAKOUT-DAY DISTORTION
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
# FIX v3: Measure base metrics BEFORE trigger candle (iloc[-2])
#          so a +14% breakout day doesn't distort ATR/volume readings.
#          Also: normalize base depth against prior rally magnitude.
# ------------------------------------------------------------------
def score_base_quality(df, latest):
    # Use pre-trigger bar for base quality measurement
    pre_trigger = df.iloc[-2]
    
    # --- Base depth from recent high (use pre-trigger) ---
    high_50 = df['High'].iloc[-51:-1].max()  # 50-bar high BEFORE trigger
    dist_from_peak = (high_50 - pre_trigger['Close']) / high_50 * 100

    # --- ATR contraction (BEFORE trigger candle) ---
    # Recompute ATR_5 excluding today's candle
    if len(df) >= 7:
        atr_5_pre = ta.atr(df['High'].iloc[-7:-1], df['Low'].iloc[-7:-1], 
                           df['Close'].iloc[-7:-1], length=5)
        atr_5_val = atr_5_pre.iloc[-1] if atr_5_pre is not None and len(atr_5_pre) > 0 else 0
    else:
        atr_5_val = 0
    atr_50_val = pre_trigger.get('ATR_50', atr_5_val)
    if pd.isna(atr_5_val): atr_5_val = 0
    if pd.isna(atr_50_val): atr_50_val = atr_5_val
    atr_contraction = (atr_50_val - atr_5_val) / atr_50_val * 100 if atr_50_val > 0 else 0

    # --- VOLUME 3-ACT STORY (BEFORE trigger) ---
    vol_base = df['Volume'].iloc[-16:-1].mean()  # base period excl today
    vol_prior = df['Volume'].iloc[-50:-15].mean() if len(df) >= 50 else vol_base
    vol_contraction = (vol_prior - vol_base) / vol_prior * 100 if vol_prior > 0 else 0

    # --- BASE AT RESISTANCE ---
    high_100 = df['High'].iloc[-100:].max() if len(df) >= 100 else df['High'].max()
    pct_from_resistance = (high_100 - pre_trigger['Close']) / high_100 * 100

    # --- DEPTH vs PRIOR RALLY (NEW) ---
    # A 50% pullback after 200% run is HEALTHY. A 15% pullback after 20% run is noise.
    low_200 = df['Low'].iloc[-min(len(df), 200):].min()
    prior_rally = (high_100 / low_200 - 1) * 100 if low_200 > 0 else 0
    depth_ratio = dist_from_peak / prior_rally if prior_rally > 0 else 1
    # depth_ratio < 0.5 means correction is less than half the rally — healthy

    # --- Tight closes (BEFORE trigger) ---
    tight_days = sum(
        abs(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.015
        for i in range(-11, -1)  # exclude trigger day
    )

    # --- SCORING ---
    score = 0

    # Base depth (0-8): 3-15% from peak is ideal, BUT adjust for rally size
    if depth_ratio <= 0.3 and dist_from_peak >= 3:
        score += 8  # Shallow relative to rally — KNOWLEDG, HINDCOPPER territory
    elif 3 <= dist_from_peak <= 15:
        score += 8
    elif 15 < dist_from_peak <= 25:
        score += 5 if depth_ratio <= 0.4 else 4  # Deeper but healthy if rally was big
    elif 1 <= dist_from_peak < 3:
        score += 3  # barely pulled back, questionable base
    elif dist_from_peak < 1:
        score += 1  # at highs, definitely no base
    elif 25 < dist_from_peak <= 50 and depth_ratio <= 0.3:
        score += 5  # Deep in absolute terms but small relative to a monster rally
    else:
        score += 1

    # ATR contraction (0-8)
    if atr_contraction >= 30:
        score += 8
    elif atr_contraction >= 15:
        score += 6
    elif atr_contraction >= 0:
        score += 3
    else:
        score += 1

    # Volume contraction in base (0-8)
    if vol_contraction >= 40:
        score += 8
    elif vol_contraction >= 20:
        score += 6
    elif vol_contraction >= 5:
        score += 4
    elif vol_contraction >= -10:
        score += 2
    else:
        score += 0  # volume expanding in "base" = not a base

    # Base at resistance (0-5)
    if pct_from_resistance <= 5:
        score += 5
    elif pct_from_resistance <= 15:
        score += 3
    elif pct_from_resistance <= 30 and depth_ratio <= 0.3:
        score += 3
    else:
        score += 1

    # Tight closes bonus (0-4)
    score += min(tight_days, 4)

    # ===== CONSOLIDATION EXISTENCE CHECK =====
    # A real base = stock PAUSES in a tight range. No pause = no setup.
    atr_20_pre = pre_trigger.get('ATR_20', 0)
    if pd.isna(atr_20_pre) or atr_20_pre == 0:
        atr_20_pre = df['ATR_20'].iloc[-2] if 'ATR_20' in df.columns else 1

    ranging_days = 0
    for i in range(-21, -1):
        if abs(i) > len(df): continue
        cc_change = abs(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] * 100
        day_range = (df['High'].iloc[i] - df['Low'].iloc[i])
        if cc_change < 2.0 and day_range < atr_20_pre * 1.5:
            ranging_days += 1

    base_high = df['High'].iloc[-21:-1].max()
    base_low_20 = df['Low'].iloc[-21:-1].min()
    base_width_pct = (base_high - base_low_20) / base_high * 100 if base_high > 0 else 0

    has_real_base = ranging_days >= 10 and 3 <= base_width_pct <= 20
    no_base = ranging_days < 7

    # HARD CAPS: NO BASE = NO SCORE
    if no_base:
        score = min(score, 5)
    elif not has_real_base and dist_from_peak < 3:
        score = min(score, 8)
    if vol_contraction < -20:
        score = min(score, 6)

    score = min(score, 33)

    return score, {
        'Base Depth': f"{dist_from_peak:.1f}%",
        'ATR Contr': f"{atr_contraction:.0f}%",
        'Vol Contr': f"{vol_contraction:.0f}%",
        'Near Res': f"{pct_from_resistance:.0f}%",
        'Depth/Rally': f"{depth_ratio:.2f}",
        'Tight Days': tight_days,
        'Range Days': f"{ranging_days}/20",
        'Base Width': f"{base_width_pct:.1f}%",
    }


# ------------------------------------------------------------------
# DIMENSION 2: STAGE + PRIOR MOVE (0-33)
# FIX v3: S1b with strong prior move + price reclaiming 200DMA
#          gets a higher cap (20 instead of 12). The hard cap only
#          crushes genuine Stage 4 declines with no prior move.
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
    sma200_dn = False
    sma200_val = np.nan
    has_200dma = False
    pct_change_200 = 0

    sma200_col = df['SMA_200'].dropna() if 'SMA_200' in df.columns else pd.Series(dtype=float)
    if len(sma200_col) >= 2:
        has_200dma = True
        sma200_val = sma200_col.iloc[-1]
        lookback = min(len(sma200_col) - 1, 21)
        pct_change_200 = (sma200_col.iloc[-1] / sma200_col.iloc[-1 - lookback] - 1) * 100
        sma200_up = pct_change_200 > 0.5
        sma200_flat = -0.5 <= pct_change_200 <= 0.5
        sma200_dn = pct_change_200 < -0.5

    if not has_200dma and 'SMA_150' in df.columns:
        sma150_col = df['SMA_150'].dropna()
        if len(sma150_col) >= 10:
            lookback = min(len(sma150_col) - 1, 21)
            pct_change_150 = (sma150_col.iloc[-1] / sma150_col.iloc[-1 - lookback] - 1) * 100
            sma200_up = pct_change_150 > 0.5
            sma200_flat = -0.5 <= pct_change_150 <= 0.5
            sma200_dn = pct_change_150 < -0.5
            sma200_val = sma150_col.iloc[-1]
            has_200dma = True
            pct_change_200 = pct_change_150

    price_above_200 = latest['Close'] > sma200_val if has_200dma else False
    sma50_above_200 = sma50 > sma200_val if has_200dma else False
    dist_from_200 = ((latest['Close'] - sma200_val) / sma200_val * 100) if has_200dma and sma200_val > 0 else 0

    # --- 200DMA RATE OF DECLINE (NEW) ---
    # Distinguish "200DMA declining fast" (genuine S4) from "200DMA barely declining / flattening out" (S1b transition)
    sma200_flattening = has_200dma and sma200_dn and pct_change_200 > -2.0  # declining but slowly

    # --- BASE COUNT ---
    base_count = 0
    lookback_bars = min(len(df), 120)
    local_high = df['Close'].iloc[-lookback_bars]
    in_pullback = False

    for i in range(-lookback_bars, 0):
        close_i = df['Close'].iloc[i]
        if close_i > local_high:
            local_high = close_i
        pullback_pct = (local_high - close_i) / local_high * 100
        if pullback_pct >= 5 and not in_pullback:
            in_pullback = True
        elif in_pullback and close_i >= local_high * 0.97:
            base_count += 1
            in_pullback = False
            local_high = close_i

    # --- PRIOR MOVE QUALITY ---
    low_200 = df['Low'].iloc[-min(len(df), 200):].min()
    high_recent = df['High'].iloc[-50:].max()
    prior_move_pct = (high_recent / low_200 - 1) * 100 if low_200 > 0 else 0

    # Move cleanliness
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
    # S1b: 200DMA declining or flat, but price has reclaimed it, showing transition
    is_s1b = (sma200_flat or sma200_flattening or (sma200_dn and price_above_200 and sma50_above_200)) and price_above_200
    is_s1_early = not sma200_up and not sma200_flat and price_above_200 and not is_s1b
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
        score += 4
    elif sma200_flattening:
        score += 3  # Declining but flattening — transition zone
    else:
        score += 0

    # Stage position (0-8)
    if is_early_s2:
        score += 8
    elif is_s1b:
        score += 6  # S1b is buyable per rules
    elif is_mid_s2:
        score += 5
    elif is_late_s2:
        score += 2
    elif is_s1_early:
        score += 1
    else:
        score += 0

    # Prior move quality (0-8)
    if prior_move_pct >= 80:
        score += 8
    elif prior_move_pct >= 50:
        score += 6
    elif prior_move_pct >= 30:
        score += 4
    elif prior_move_pct >= 15:
        score += 2
    else:
        score += 0

    # Move cleanliness bonus (0-5)
    if move_cleanliness >= 55:
        score += 5
    elif move_cleanliness >= 50:
        score += 3
    else:
        score += 1

    # --- HARD PENALTIES (REVISED) ---
    # v3: S1b with strong prior move gets a HIGHER cap
    # The old blanket cap of 12 killed valid S1b setups like KNOWLEDG
    if not sma200_up and not sma200_flat:
        if is_s1b and prior_move_pct >= 50:
            # S1b transition with strong prior move — allow up to 22
            score = min(score, 22)
        elif sma200_flattening and prior_move_pct >= 30:
            # 200DMA almost flat + decent prior move — allow up to 18
            score = min(score, 18)
        else:
            # Genuine decline, no prior move — hard cap
            score = min(score, 12)
    
    if not price_above_200 and has_200dma:
        score -= 4
    if not sma50_above_200 and has_200dma:
        score -= 3
    if dist_from_200 > 35 and sma200_up:
        score -= 2

    # LATE S2 HARD CAP: 3rd+ base = momentum exhausting, avoid per system rules
    if is_late_s2:
        score = min(score, 12)

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
        '200DMA': f"{'↑' if sma200_up else ('→' if sma200_flat else ('↗' if sma200_flattening else '↓'))} ({pct_change_200:+.1f}%)",
        'Stg': stg_label,
        'Bases': base_count,
        'Prior Move': f"{prior_move_pct:.0f}%",
        'Trend Clean': f"{move_cleanliness:.0f}%"
    }


# ------------------------------------------------------------------
# DIMENSION 3: TIMING (0-34)
# FIX v3: Fresh breakouts with volume are EXEMPT from extension
#          penalty. Extension penalty only applies to stale moves
#          (stock drifting up for days without a base).
# ------------------------------------------------------------------
def score_timing(df, latest):
    # --- Distance from 20DMA ---
    pct_20dma = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
    abs_dist = abs(pct_20dma)

    if abs_dist <= 2:
        ma_pts = 10
    elif abs_dist <= 4:
        ma_pts = 7
    elif abs_dist <= 6:
        ma_pts = 4
    else:
        ma_pts = 1

    # --- Volume on trigger day vs base average ---
    vol_today = latest['Volume']
    vol_base_avg = df['Volume'].iloc[-15:-1].mean()
    vol_expansion = (vol_today / vol_base_avg) if vol_base_avg > 0 else 1

    if vol_expansion >= 2.0:
        vol_pts = 8
    elif vol_expansion >= 1.5:
        vol_pts = 6
    elif vol_expansion >= 1.0:
        vol_pts = 3
    else:
        vol_pts = 1

    # --- Candle quality ---
    cr = latest['High'] - latest['Low']
    body = abs(latest['Close'] - latest['Open'])
    br = body / cr if cr > 0 else 0
    close_position = (latest['Close'] - latest['Low']) / cr if cr > 0 else 0.5

    if br > 0.6 and latest['Close'] > latest['Open'] and close_position > 0.7:
        candle_pts = 8
    elif br > 0.4 and latest['Close'] > latest['Open'] and close_position > 0.5:
        candle_pts = 5
    elif br > 0.3 and latest['Close'] > latest['Open']:
        candle_pts = 2
    elif latest['Close'] > latest['Open']:
        candle_pts = 1  # Weak green / doji
    else:
        candle_pts = 0

    # Weak candle flag
    is_weak_candle = br < 0.35 or (latest['Close'] <= latest['Open']) or close_position < 0.4

    # --- Breakout freshness ---
    high_50 = df['High'].iloc[-50:].max()
    pct_from_breakout = (high_50 - latest['Close']) / high_50 * 100

    days_above_resistance = 0
    for i in range(-10, 0):
        prior_high = df['High'].iloc[:len(df)+i-1].iloc[-50:].max()
        if df['Close'].iloc[i] > prior_high * 0.98:
            days_above_resistance += 1

    is_fresh_breakout = days_above_resistance <= 3 and latest['Close'] >= high_50 * 0.98
    is_stale_breakout = days_above_resistance > 5

    if is_fresh_breakout:
        bkout_pts = 8
    elif pct_from_breakout <= 3 and not is_stale_breakout:
        bkout_pts = 6
    elif pct_from_breakout <= 8:
        bkout_pts = 4
    elif is_stale_breakout:
        bkout_pts = 0
    else:
        bkout_pts = 1

    # --- EXTENSION PENALTY (REVISED v3) ---
    # KEY FIX: Fresh breakouts with volume are EXEMPT.
    # A +14% breakout candle with 3x volume IS the setup — not "extended".
    # Extension penalty only applies to stocks that drifted up gradually
    # without a clear base/breakout pattern.
    extension_penalty = 0
    is_volume_breakout = is_fresh_breakout and vol_expansion >= 1.5

    if not is_volume_breakout:
        # Only penalize if NOT a fresh volume breakout
        if pct_20dma > 15:
            extension_penalty = 12
        elif pct_20dma > 10:
            extension_penalty = 8
        elif pct_20dma > 8:
            extension_penalty = 4
    else:
        # Fresh breakout with volume: mild penalty only if extremely extended
        if pct_20dma > 25:
            extension_penalty = 6  # Even breakouts shouldn't be +25% above MA
        elif pct_20dma > 20:
            extension_penalty = 3

    # --- BREAKOUT DAY BONUS (NEW) ---
    # If this is a fresh volume breakout, give bonus points for MA distance
    # because the extension IS the signal
    breakout_bonus = 0
    if is_volume_breakout and pct_20dma > 5:
        breakout_bonus = min(int(pct_20dma / 3), 6)  # Up to 6 bonus pts for powerful breakouts

    raw_score = ma_pts + vol_pts + candle_pts + bkout_pts + breakout_bonus
    score = max(min(raw_score - extension_penalty, 34), 0)

    # HARD CAPS: Weak trigger = not a setup
    if is_weak_candle:
        score = min(score, 10)
    if vol_expansion < 0.8:
        score = min(score, 8)

    return score, {
        'Dist 20DMA': f"{pct_20dma:+.1f}%",
        'Vol Exp': f"{vol_expansion:.1f}x",
        'Body': f"{br:.0%}",
        'Close Pos': f"{close_position:.0%}",
        'Near Bkout': f"{pct_from_breakout:.1f}%",
        'Bkout Age': f"{days_above_resistance}d",
        'Fresh BO': "✅" if is_volume_breakout else "❌",
        'Candle': "Strong" if not is_weak_candle else "Weak"
    }


# ------------------------------------------------------------------
# VOLUME PROFILE (NEW — 25 pts)
# ------------------------------------------------------------------
def score_volume_profile(df, latest):
    """
    3 components:
      1. Up/Down Volume Ratio (10) — accumulation signal
      2. Base Volume Contraction (10) — supply absorbed during base
      3. Today's Volume vs avg (5) — entry day confirmation
    """
    score = 0
    detail = {}

    # 1. Up/Down Vol Ratio over last 20 bars
    last20 = df.iloc[-20:]
    up_mask = last20['Close'] >= last20['Open']
    up_vol = last20.loc[up_mask, 'Volume'].sum()
    dn_vol = last20.loc[~up_mask, 'Volume'].sum()
    ud_ratio = (up_vol / dn_vol) if dn_vol > 0 else 999.0

    if ud_ratio >= 1.5:   ud_pts = 10
    elif ud_ratio >= 1.2: ud_pts = 7
    elif ud_ratio >= 1.0: ud_pts = 4
    else:                 ud_pts = 0
    score += ud_pts
    detail['ud_ratio'] = round(min(ud_ratio, 99.0), 2)
    detail['ud_pts'] = ud_pts

    # 2. Volume Contraction: base (last 20) vs prior trend (50 bars before)
    base_vol = df['Volume'].iloc[-20:].mean()
    if len(df) >= 70:
        trend_vol = df['Volume'].iloc[-70:-20].mean()
    else:
        trend_vol = df['Volume'].iloc[:-20].mean() if len(df) > 20 else base_vol
    contraction = (base_vol / trend_vol) if trend_vol > 0 else 1.0

    if contraction <= 0.70:   vc_pts = 10
    elif contraction <= 0.90: vc_pts = 7
    elif contraction <= 1.10: vc_pts = 4
    else:                     vc_pts = 0
    score += vc_pts
    detail['contraction'] = round(contraction, 2)
    detail['vc_pts'] = vc_pts

    # 3. Today's Volume vs 50-day avg
    today_vol = latest['Volume']
    avg_vol = df['Volume'].iloc[-50:].mean()
    vol_ratio = (today_vol / avg_vol) if avg_vol > 0 else 1.0

    if vol_ratio >= 1.5:   tv_pts = 5
    elif vol_ratio >= 1.0: tv_pts = 3
    elif vol_ratio >= 0.7: tv_pts = 1
    else:                  tv_pts = 0
    score += tv_pts
    detail['vol_ratio'] = round(vol_ratio, 2)
    detail['tv_pts'] = tv_pts

    score = max(min(score, 25), 0)
    detail['total'] = score
    return score, detail


# ------------------------------------------------------------------
# FINAL SCORE
# ------------------------------------------------------------------
def score_setup(res):
    df = _enrich_df(res['chart_data'].copy())
    latest = df.iloc[-1]

    base_s_raw, base_d = score_base_quality(df, latest)      # 0-33
    stage_s_raw, stage_d = score_stage(df, latest)           # 0-33
    timing_s_raw, timing_d = score_timing(df, latest)        # 0-34
    volume_s, volume_d = score_volume_profile(df, latest)    # 0-25

    # Rescale legacy scores to 25 each (preserves all existing tuning logic)
    base_s = round(base_s_raw * 25 / 33)
    stage_s = round(stage_s_raw * 25 / 33)
    timing_s = round(timing_s_raw * 25 / 34)

    total = base_s + stage_s + timing_s + volume_s  # 0-100

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
            'Timing': timing_s, 'Volume': volume_s,
            'Entry': round(entry, 2), 'SL': sl, 'Risk%': risk,
            'base_det': base_d, 'stage_det': stage_d,
            'timing_det': timing_d, 'volume_det': volume_d}


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

    dims = [('🧱 Base', res.get('Base', 0), 25),
            ('📶 Stage', res.get('Stage', 0), 25),
            ('⏱️ Timing', res.get('Timing', 0), 25),
            ('📊 Volume', res.get('Volume', 0), 25)]

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
    for k in ['base_det', 'stage_det', 'timing_det', 'volume_det']:
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
        _diag_reset()

        passed_results = []
        progress_bar = st.progress(0)
        check_func = partial(check_stock, ind_map=ind_map)

        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = executor.map(check_func, tickers)
            for i, result in enumerate(results):
                if result: passed_results.append(result)
                progress_bar.progress(min((i + 1) / len(tickers), 1.0))
        progress_bar.empty()

        # --- DIAGNOSTIC BREAKDOWN ---
        with st.expander("🔍 Scan Diagnostics — Where stocks drop off", expanded=True):
            d = _diag.copy()
            st.markdown(f"""
| Stage | Count | Note |
|---|---|---|
| 📥 Tickers sent | **{len(tickers)}** | Universe |
| ❌ Data fail (yfinance error) | **{d['data_fail']}** | API timeout / no data |
| ❌ Low data (<70 bars) | **{d['low_data']}** | Insufficient history |
| ✅ Actually checked | **{d['checked']}** | Had valid data |
| ❌ c1: Vol20/50 < 100K | **{d['c1_vol20']}** | Low volume |
| ❌ c2: (merged) | **{d['c2_vol50']}** | — |
| ❌ c3: SMA20 < SMA50 | **{d['c3_sma_cross']}** | Not in uptrend |
| ❌ c4: Below SMA50 + dn | **{d['c4_below50_dn']}** | Downtrend |
| ❌ c5: Below SMA10 | **{d['c5_below10']}** | Below short MA |
| ❌ c6: Below SMA20 | **{d['c6_below20']}** | Below 20DMA |
| ❌ c7: SMA10 < SMA20 | **{d['c7_sma10_20']}** | MAs not stacked |
| ❌ c8: Red day | **{d['c8_red_day']}** | Closed lower |
| ❌ c9: Low ATR | **{d['c9_low_atr']}** | Weak range |
| ❌ c10: Close position | **{d['c10_close_pos']}** | Closed in lower 40% |
| ✅ **PASSED** | **{d['passed']}** | Setups found |
""")

        if passed_results:
            st.success(f"🔥 Found {len(passed_results)} setups!")

            st.info("🧠 Scoring: Base · Stage · Timing · Volume...")
            scored_results = []
            for res in passed_results:
                try:
                    scores = score_setup(res)
                    scored_results.append({**res, **scores})
                except:
                    scored_results.append({**res, 'Grade': '?', 'Total': 0, 'Base': 0,
                        'Stage': 0, 'Timing': 0, 'Volume': 0, 'Entry': 0, 'SL': 0, 'Risk%': 0,
                        'base_det': {}, 'stage_det': {}, 'timing_det': {}, 'volume_det': {}})

            scored_results.sort(key=lambda x: x.get('Total', 0), reverse=True)

            # --- RANKED SCORECARD ---
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📋 Ranked Scorecard")

            table_data = []
            for s in scored_results:
                row = {'Ticker': s['Ticker'], 'Industry': s['Industry'],
                       'Grade': s.get('Grade','?'), 'Score': s.get('Total',0),
                       'Base': s.get('Base',0), 'Stage': s.get('Stage',0),
                       'Timing': s.get('Timing',0), 'Volume': s.get('Volume',0),
                       'CMP': s.get('Entry',0),
                       'SL': s.get('SL',0), 'Risk%': s.get('Risk%',0)}
                for dk in ['base_det', 'stage_det', 'timing_det', 'volume_det']:
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
