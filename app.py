import streamlit as st
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import warnings
from streamlit_lightweight_charts import renderLightweightCharts
from nselib import capital_market
from functools import partial
import datetime
from dhanhq import dhanhq
import requests
import io
import time

warnings.filterwarnings("ignore")

# --- UI Setup ---
st.set_page_config(page_title="MIO Champ Screener", layout="wide")
st.title("📈 MIO Champion Setup Screener")
st.markdown("Automated scan for high-probability momentum setups using Dhan API.")

# --- API Credentials (Input via Streamlit Sidebar) ---
st.sidebar.header("🔑 Dhan API Settings")
client_id = st.sidebar.text_input("Client ID", type="password")
access_token = st.sidebar.text_input("Access Token", type="password")

if client_id and access_token:
    dhan = dhanhq(client_id, access_token)
else:
    st.sidebar.warning("Please enter your Dhan API credentials to run the scan.")

# --- Data Fetching & Security Mapping ---
@st.cache_data(ttl=86400) 
def get_dhan_security_map():
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    # Added User-Agent header to bypass Dhan/Cloudflare bot protection
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Force an error if the download fails
        df = pd.read_csv(io.StringIO(response.text), low_memory=False)
        nse_eq = df[(df['EXCH_ID'] == 'NSE') & (df['INSTRUMENT'] == 'EQUITY')]
        return dict(zip(nse_eq['SEM_CUSTOM_SYMBOL'], nse_eq['SEM_SMST_SECURITY_ID']))
    except Exception as e:
        st.error(f"🚨 Dhan Security Master Error: {e}")
        return {}

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
    except Exception as e:
        st.error(f"🚨 NSE Server Error (Nifty 500): {e}. The NSE website might be down for weekend maintenance.")
        return [], {}

@st.cache_data(ttl=3600)
def get_all_nse():
    try:
        df = capital_market.equity_list()
        df.columns = df.columns.str.upper() 
        raw = df['SYMBOL'].tolist()
        tickers = list(set([t for t in raw if isinstance(t, str) and "DUMMY" not in t]))
        _, ind_map = get_nifty_500()
        return tickers, ind_map
    except Exception as e:
        st.error(f"🚨 NSE Server Error (All NSE): {e}. The NSE website might be down for weekend maintenance.")
        return [], {}
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

# --- The Engine (Powered by Dhan) ---
def check_stock(ticker, ind_map, target_date, sec_map, dhan_client):
    sec_id = sec_map.get(ticker)
    if not sec_id: return None

    try:
        # Dhan Historical Data API Call
        # We fetch 250 days of data ending on the target date
        target_dt = pd.to_datetime(target_date)
        from_date = target_dt - datetime.timedelta(days=250)
        
        req = dhan_client.get_historical_prices(
            symbol=ticker,
            exchange_segment='NSE_EQ',
            instrument_type='EQUITY',
            expiry_code=0,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=target_dt.strftime('%Y-%m-%d')
        )
        
        if req.get('status') != 'success' or not req.get('data'):
            return None
            
        data = req['data']
        df = pd.DataFrame({
            'Open': data['open'],
            'High': data['high'],
            'Low': data['low'],
            'Close': data['close'],
            'Volume': data['volume']
        }, index=pd.to_datetime(data['start_Time']))
        
        if len(df) < 70: return None

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
        # Strict 20-day original formula
        c3 = (df['SMA_20'].iloc[-20:] >= df['SMA_50'].iloc[-20:]).all() 
        c4 = not (latest['Close'] < latest['SMA_50'] and sma50_trend_dn_20)
        c5 = latest['Close'] > latest['SMA_10']
        c6 = latest['Close'] > latest['SMA_20']
        c7 = latest['SMA_10'] > latest['SMA_20']
        c8 = latest['Close'] > prev['Close']
        c9 = latest['ATR_1'] > (latest['ATR_20'] * 0.6)
        c10 = latest['Close'] > (latest['Low'] + ((latest['High'] - latest['Low']) * 0.4))

        if all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]):
            industry = ind_map.get(ticker, 'N/A')
            return {"Ticker": ticker, "Industry": industry, "chart_data": df}
            
    except Exception:
        pass
    
    # Respect API rate limits
    time.sleep(0.1) 
    return None

# --- Dashboard Controls ---
col1, col2 = st.columns(2)

with col1:
    scan_mode = st.radio("Select Market Universe:", ["Nifty 500 (Fast)", "All NSE Stocks (~2,200 Stocks, Slower)"])

with col2:
    target_date = st.date_input("📅 Target Scan Date (HIST Function)", value=datetime.date.today())

if st.button("🚀 Run Market Scan", type="primary"):
    if not client_id or not access_token:
        st.error("Please enter your Dhan API credentials in the sidebar first.")
    else:
        if "Nifty 500" in scan_mode:
            tickers, ind_map = get_nifty_500()
        else:
            tickers, ind_map = get_all_nse()
        
        sec_map = get_dhan_security_map()
        
        if not tickers or not sec_map:
            st.error("Failed to pull market or security data.")
        else:
            st.info(f"Crunching {len(tickers)} stocks via Dhan API... Please wait.")
            
            passed_results = []
            progress_bar = st.progress(0)
            
            check_func = partial(check_stock, ind_map=ind_map, target_date=target_date, sec_map=sec_map, dhan_client=dhan)
            
            # Reduced max_workers to 5 to respect broker API rate limits
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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

                # --- 2. TRADINGVIEW INTERACTIVE CHART VIEW ---
                st.subheader(f"📊 Chart View (Data up to {target_date.strftime('%Y-%m-%d')})")
                for res in passed_results:
                    st.markdown(f"### **{res['Ticker']}** | {res['Industry']}")
                    
                    df_plot = res['chart_data'].tail(150).copy()
                    df_plot['time'] = df_plot.index.strftime('%Y-%m-%d')
                    
                    candles = df_plot[['time', 'Open', 'High', 'Low', 'Close']].rename(
                        columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
                    ).to_dict('records')
                    
                    sma20 = df_plot[['time', 'SMA_20']].dropna().rename(
                        columns={'SMA_20': 'value'}
                    ).to_dict('records')

                    volume = df_plot[['time', 'Volume', 'Close', 'Open']].copy()
                    volume['color'] = volume.apply(
                        lambda row: 'rgba(0, 176, 96, 0.5)' if row['Close'] >= row['Open'] else 'rgba(255, 51, 58, 0.5)', axis=1
                    )
                    volume = volume[['time', 'Volume', 'color']].rename(columns={'Volume': 'value'}).to_dict('records')

                    chartOptions = {
                        "layout": { "textColor": '#d1d4dc', "background": { "type": 'solid', "color": '#131722' } },
                        "grid": { "vertLines": { "color": '#363c4e' }, "horzLines": { "color": '#363c4e' } },
                        "crosshair": { "mode": 1 },
                        "priceScale": { "borderColor": '#485c7b' },
                        "timeScale": { "borderColor": '#485c7b', "timeVisible": True },
                        "height": 500
                    }

                    series_list = [
                        {
                            "type": 'Candlestick',
                            "data": candles,
                            "options": {
                                "upColor": '#00b060', "downColor": '#ff333a', 
                                "borderVisible": False, 
                                "wickUpColor": '#00b060', "wickDownColor": '#ff333a'
                            }
                        },
                        {
                            "type": 'Line',
                            "data": sma20,
                            "options": {
                                "color": '#ffa726',
                                "lineWidth": 2,
                                "title": '20 DMA'
                            }
                        },
                        {
                            "type": 'Histogram',
                            "data": volume,
                            "options": {
                                "priceFormat": {"type": 'volume'},
                                "priceScaleId": "",
                                "scaleMargins": {"top": 0.8, "bottom": 0}
                            }
                        }
                    ]

                    renderLightweightCharts([{"chart": chartOptions, "series": series_list}], 'chart_' + res['Ticker'])
                    st.markdown("---")
            else:
                st.warning(f"No stocks matched the criteria on {target_date.strftime('%B %d, %Y')}.")
