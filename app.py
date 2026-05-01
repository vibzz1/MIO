import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import warnings
import mplfinance as mpf
from nselib import capital_market

warnings.filterwarnings("ignore")

# --- UI Setup ---
st.set_page_config(page_title="MIO Champ Screener", layout="wide")
st.title("📈 MIO Champion Setup Screener")
st.markdown("Automated scan for high-probability momentum setups.")

# --- Data Fetching (Cached for speed) ---
@st.cache_data(ttl=3600) # Caches the list for 1 hour so it doesn't re-download every click
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
        df = stock.history(period="6mo")
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

        # Goldilocks Logic
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
            return {"Ticker": ticker, "Industry": industry, "chart_data": df.tail(100)}
            
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
        # Progress bar for the web UI
        progress_bar = st.progress(0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            results = executor.map(check_stock, tickers)
            for i, result in enumerate(results):
                if result: passed_results.append(result)
                # Update progress bar
                progress_bar.progress(min((i + 1) / len(tickers), 1.0))
                
        progress_bar.empty()

        if passed_results:
            st.success(f"🔥 Found {len(passed_results)} setups!")
            
            # --- 1. LIST VIEW (Interactive Table) ---
            st.subheader("📋 List View")
            # Create a clean dataframe for the web table
            df_results = pd.DataFrame([{k: v for k, v in res.items() if k != 'chart_data'} for res in passed_results])
            df_results.index = df_results.index + 1
            st.dataframe(df_results, use_container_width=True)

            st.divider()

            # --- 2. CHART VIEW ---
            st.subheader("📊 Chart View")
            for res in passed_results:
                st.markdown(f"### **{res['Ticker']}** | {res['Industry']}")
                
                df_chart = res['chart_data']
                apdict = [
                    mpf.make_addplot(df_chart['SMA_10'], color='blue', width=1.2),
                    mpf.make_addplot(df_chart['SMA_20'], color='orange', width=1.2),
                    mpf.make_addplot(df_chart['SMA_50'], color='red', width=1.2)
                ]
                
                # returnfig=True is required to pass the chart to Streamlit
                fig, axlist = mpf.plot(
                    df_chart, type='candle', addplot=apdict, style='yahoo', 
                    volume=True, figsize=(12, 5), returnfig=True
                )
                st.pyplot(fig)
                st.markdown("---") # Adds a clean visual break between charts
        else:
            st.warning("No stocks matched the criteria today.")