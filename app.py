import streamlit as st
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import warnings
from streamlit_lightweight_charts import renderLightweightCharts
from nselib import capital_market
from functools import partial
import datetime
import requests
import time

warnings.filterwarnings("ignore")

# --- UI Setup ---
st.set_page_config(page_title="MIO Champ Screener", layout="wide")
st.title("📈 MIO Champion Setup Screener")
st.markdown("Automated scan for high-probability momentum setups using a Direct Native Dhan API Tunnel.")

# --- API Credentials ---
st.sidebar.header("🔑 Dhan API Settings")
client_id = st.sidebar.text_input("Client ID", type="password")
access_token = st.sidebar.text_input("Access Token", type="password")

if not (client_id and access_token):
    st.sidebar.warning("Please enter your Dhan API credentials to run the scan.")

# --- Data Fetching (CSV mapping completely removed for speed) ---
@st.cache_data(ttl=3600)
def get_nifty_500():
    try:
        n50 = capital_market.nifty50_equity_list()
        nn50 = capital_market.niftynext50_equity_list()
        mid150 = capital_market.niftymidcap150_equity_list()
        sml250 = capital_market.niftysmallcap250_equity_list()
        df = pd.concat([n50, nn50, mid150, sml250], ignore_index=True)
        return list(set(df['Symbol'].tolist())), dict(zip(df['Symbol'], df['Industry']))
    except Exception as e:
        st.error(f"🚨 NSE Server Error: {e}")
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
        st.error(f"🚨 NSE Server Error: {e}")
        return [], {}

# --- The Engine (Direct REST API Tunnel) ---
def check_stock(ticker, ind_map, target_date, cid, token):
    try:
        target_dt = pd.to_datetime(target_date)
        from_date = target_dt - datetime.timedelta(days=250)
        
        url = "https://api.dhan.co/charts/historical"
        headers = {
            "access-token": token,
            "client-id": cid,
            "Content-Type": "application/json"
        }
        # Direct payload sending the exact text symbol, bypassing the library bug
        payload = {
            "symbol": ticker,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "expiryCode": 0,
            "fromDate": from_date.strftime('%Y-%m-%d'),
            "toDate": target_dt.strftime('%Y-%m-%d')
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        # Rigorous Error Catching for exact API responses
        if response.status_code == 429:
            return {"error": f"Rate Limit (429) hit on {ticker}. Dhan is throttling us."}
        elif response.status_code != 200:
            return {"error": f"HTTP {response.status_code} on {ticker}: {response.text}"}
            
        req = response.json()
        
        if req.get('status') != 'success' or not req.get('data') or len(req['data'].get('close', [])) == 0:
            return {"error": f"Dhan API returned empty data for {ticker}."}
            
        data = req['data']
        time_keys = data.get('start_Time') or data.get('timestamp')
        
        if not time_keys:
            return {"error": f"Missing timestamp data for {ticker}"}
        
        # Foolproof epoch/string date parser
        try:
            if isinstance(time_keys[0], str) and "-" in time_keys[0]:
                dt_index = pd.to_datetime(time_keys)
            elif time_keys[0] > 1e11:
                dt_index = pd.to_datetime(time_keys, unit='ms')
            else:
                dt_index = pd.to_datetime(time_keys, unit='s')
        except:
            dt_index = pd.to_datetime(time_keys)
        
        df = pd.DataFrame({
            'Open': data['open'],
            'High': data['high'],
            'Low': data['low'],
            'Close': data['close'],
            'Volume': data['volume']
        }, index=dt_index)
        
        df = df.apply(pd.to_numeric)
        df.sort_index(inplace=True) 
        
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
        c3 = (df['SMA_20'].iloc[-5:] >= df['SMA_50'].iloc[-5:]).all() 
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
            
    except Exception as e:
        return {"error": f"Crash on {ticker}: {str(e)}"}
    
    finally:
        # 🚨 STRICT THROTTLE: Placed in a 'finally' block so it mathematically 
        # guarantees a 0.5s delay even if the code crashes, preventing Dhan IP bans.
        time.sleep(0.5) 
        
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
        
        if not tickers:
            st.error("Failed to pull market data from NSE.")
        else:
            st.info(f"Crunching stocks via Native API... Processing safely to respect institutional limits.")
            
            passed_results = []
            api_errors = []
            progress_bar = st.progress(0)
            
            check_func = partial(check_stock, ind_map=ind_map, target_date=target_date, cid=client_id, token=access_token)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                results = executor.map(check_func, tickers)
                for i, result in enumerate(results):
                    if result:
                        if "error" in result:
                            api_errors.append(result["error"])
                        else:
                            passed_results.append(result)
                    progress_bar.progress(min((i + 1) / len(tickers), 1.0))
                    
            progress_bar.empty()

            if api_errors:
                with st.expander("⚠️ View API Errors (Rate limits or missing data)"):
                    for err in api_errors[:20]: 
                        st.write(err)

            if passed_results:
                st.success(f"🔥 Found {len(passed_results)} setups on {target_date.strftime('%B %d, %Y')}!")
                
                st.subheader("📋 List View")
                df_results = pd.DataFrame([{k: v for k, v in res.items() if k != 'chart_data'} for res in passed_results])
                df_results.index = df_results.index + 1
                st.dataframe(df_results, use_container_width=True)

                st.divider()

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
                            "options": {"color": '#ffa726', "lineWidth": 2, "title": '20 DMA'}
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
