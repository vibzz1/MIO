import streamlit as st
import pandas as pd
import pandas_ta as ta
import concurrent.futures
import warnings
from streamlit_lightweight_charts import renderLightweightCharts
from nselib import capital_market
from functools import partial
import datetime
import yfinance as yf

warnings.filterwarnings("ignore")

# --- UI Setup ---
st.set_page_config(page_title="MIO Champ Screener", layout="wide")
st.title("📈 MIO Champion Setup Screener")
st.markdown("Automated scan for high-probability momentum setups using split-adjusted data and Native TradingView charts.")

# --- Data Fetching ---
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
        return [], {}

# --- The Engine (Adjusted yfinance Data) ---
def check_stock(ticker, ind_map, effective_date):
    try:
        start_date = effective_date - datetime.timedelta(days=250)
        # yfinance end date is exclusive, so we explicitly add 1 day to ensure the target date is captured
        end_date = effective_date + datetime.timedelta(days=1) 
        
        # This natively handles all split/dividend adjustments to keep MAs perfectly smooth
        ticker_obj = yf.Ticker(f"{ticker}.NS")
        df = ticker_obj.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty or len(df) < 70:
            return {"error": f"Insufficient data for {ticker}"}
            
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Clean timezone issues
        df.index = df.index.tz_localize(None) 
        df.sort_index(inplace=True) 
        
        # Rigorous Data Sanitizer
        df = df[df['Volume'] > 0]
        
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
        
    return None

# --- Dashboard Controls ---
col1, col2 = st.columns(2)

with col1:
    scan_mode = st.radio("Select Market Universe:", ["Nifty 500 (Fast)", "All NSE Stocks (~2,200 Stocks, Slower)"])

with col2:
    raw_target_date = st.date_input("📅 Target Scan Date", value=datetime.date.today())

# AUTO-DATE SNAPPER: Intelligently forces weekend requests back to Friday
effective_target = pd.to_datetime(raw_target_date)
while effective_target.weekday() >= 5: 
    effective_target -= datetime.timedelta(days=1)

if st.button("🚀 Run Market Scan", type="primary"):
    if "Nifty 500" in scan_mode:
        tickers, ind_map = get_nifty_500()
    else:
        tickers, ind_map = get_all_nse()
        
    if not tickers:
        st.error("Failed to pull market data.")
    else:
        if raw_target_date != effective_target.date():
            st.warning(f"Weekend detected. Auto-snapping scan date back to last trading day: **{effective_target.strftime('%A, %B %d, %Y')}**")
            
        st.info(f"Crunching stocks via High-Speed Adjusted Data Pipeline... Processing at full speed.")
        
        passed_results = []
        progress_bar = st.progress(0)
        
        check_func = partial(check_stock, ind_map=ind_map, effective_date=effective_target)
        
        # Max workers massively increased because we are free from API rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            results = executor.map(check_func, tickers)
            for i, result in enumerate(results):
                if result and "error" not in result:
                    passed_results.append(result)
                progress_bar.progress(min((i + 1) / len(tickers), 1.0))
                
        progress_bar.empty()

        if passed_results:
            st.success(f"🔥 Found {len(passed_results)} setups for {effective_target.strftime('%B %d, %Y')}!")
            
            st.subheader("📋 List View")
            df_results = pd.DataFrame([{k: v for k, v in res.items() if k != 'chart_data'} for res in passed_results])
            df_results.index = df_results.index + 1
            st.dataframe(df_results, use_container_width=True)

            st.divider()

            st.subheader(f"📊 Chart View (Data up to {effective_target.strftime('%Y-%m-%d')})")
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

                series_list = []
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
                                "priceScale": {  # <--- We must nest the margins inside this specific dictionary
                                    "scaleMargins": {
                                        "top": 0.8, # This forces volume to stay in the bottom 20%
                                        "bottom": 0
                                    }
                                }
                            }
                        }
    

                renderLightweightCharts([{"chart": chartOptions, "series": series_list}], 'chart_' + res['Ticker'])
                st.markdown("---")
        else:
            st.warning(f"No stocks matched the criteria on {effective_target.strftime('%B %d, %Y')}.")
