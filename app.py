import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
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

# --- Screener Logic ---
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

# --- Dashboard Controls ---
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
        
        # Package the check_stock function with our industry map so threads can use it safely
        check_func = partial(check_stock, ind_map=ind_map)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = executor.map(check_func, tickers)
            for i, result in enumerate(results):
                if result: passed_results.append(result)
                progress_bar.progress(min((i + 1) / len(tickers), 1.0))
                
        progress_bar.empty()

        if passed_results:
            st.success(f"🔥 Found {len(passed_results)} setups!")
            
            # --- 1. LIST VIEW ---
            st.subheader("📋 List View")
            df_results = pd.DataFrame([{k: v for k, v in res.items() if k != 'chart_data'} for res in passed_results])
            df_results.index = df_results.index + 1
            st.dataframe(df_results, use_container_width=True)

            st.divider()

            # --- 2. INTERACTIVE CHART VIEW ---
            st.subheader("📊 Interactive Chart View (Scroll to Zoom, Click & Drag to Pan)")
            for res in passed_results:
                st.markdown(f"### **{res['Ticker']}** | {res['Industry']}")
                df_chart = res['chart_data']
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # 1. Vibrant, Solid Candlesticks
                fig.add_trace(go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'], high=df_chart['High'],
                                low=df_chart['Low'], close=df_chart['Close'],
                                name='Price',
                                increasing_line_color='#00b060', increasing_fillcolor='#00b060',
                                decreasing_line_color='#ff333a', decreasing_fillcolor='#ff333a'), 
                                row=1, col=1)

                # 2. Add 20 DMA (Orange Line)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], 
                                         line=dict(color='orange', width=1.5), 
                                         name='20 DMA'), row=1, col=1)

                # 3. Volume Bar Chart
                colors = ['#00b060' if row['Close'] >= row['Open'] else '#ff333a' for index, row in df_chart.iterrows()]
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], 
                                     marker_color=colors, name='Volume'), row=2, col=1)

                fig.update_layout(height=600, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                fig.update_xaxes(rangeslider_visible=False)
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        else:
            st.warning("No stocks matched the criteria today.")
