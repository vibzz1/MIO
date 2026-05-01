import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nselib import capital_market

warnings.filterwarnings("ignore")

# --- UI Setup ---
st.set_page_config(page_title="MIO Champ Screener", layout="wide")
st.title("📈 MIO Champion Setup Screener")
st.markdown("Automated scan for high-probability momentum setups.")

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def get_nifty_500():
    try:
        n50 = capital_market.nifty50_equity_list()
        nn50 = capital_market.niftynext50_equity_list()
        mid150 = capital_market.niftymidcap150_equity_list()
        sml250 = capital_market.niftysmallcap250_equity_list()
        df = pd.concat([n50, nn50, mid150, sml250], ignore_index=True)
        return list(set(df['Symbol'].tolist()))
    except: return []

@st.cache_data(ttl=3600)
def get_all_nse():
    try:
        df = capital_market.equity_list()
        df.columns = df.columns.str.upper() 
        raw = df['SYMBOL'].tolist()
        return list(set([t for t in raw if isinstance(t, str) and "DUMMY" not in t]))
    except: return []

# --- Screener Logic ---
def check_stock(ticker):
    symbol = f"{ticker}.NS"
    try:
        stock = yf.Ticker(symbol)
        # INCREASED TO 1 YEAR: Gives you more data to zoom out and look at the macro trend
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
            try: industry = stock.info.get('industry', 'N/A')
            except: industry = 'N/A'
            # Now returning the full 1-year dataframe instead of cutting it off at 100 days
            return {"Ticker": ticker, "Industry": industry, "chart_data": df}
            
    except: pass
    return None

# --- Dashboard Controls ---
scan_mode = st.radio("Select Market Universe:", ["Nifty 500 (Fast)", "All NSE Stocks (~2,200 Stocks, Slower)"])

if st.button("🚀 Run Market Scan", type="primary"):
    tickers = get_nifty_500() if "Nifty 500" in scan_mode else get_all_nse()
    
    if not tickers:
        st.error("Failed to pull market data. The NSE server might be busy.")
    else:
        st.info(f"Crunching {len(tickers)} stocks... Please wait 60-90 seconds.")
        
        passed_results = []
        progress_bar = st.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = executor.map(check_stock, tickers)
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
                
                # Create a 2-row layout: Top for Price, Bottom for Volume
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # 1. Candlestick Chart
                fig.add_trace(go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'], high=df_chart['High'],
                                low=df_chart['Low'], close=df_chart['Close'],
                                name='Price'), row=1, col=1)

                # 2. Add 20 DMA (Orange Line)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA_20'], 
                                         line=dict(color='orange', width=1.5), 
                                         name='20 DMA'), row=1, col=1)

                # 3. Volume Bar Chart (Green for Up days, Red for Down days)
                colors = ['#00b060' if row['Close'] >= row['Open'] else '#ff333a' for index, row in df_chart.iterrows()]
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], 
                                     marker_color=colors, name='Volume'), row=2, col=1)

                # Clean up the layout
                fig.update_layout(height=600, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                fig.update_xaxes(rangeslider_visible=False) # Hides the bulky default slider so you can just use mouse scroll
                
                # Render the interactive chart
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        else:
            st.warning("No stocks matched the criteria today.")
