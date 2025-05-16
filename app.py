import streamlit as st
import pandas as pd # Added for potential future use with stock list

st.set_page_config(layout="wide")

st.title("AI Financial Advisor MVP")

# Attempt to load API keys from st.secrets
try:
    NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE" or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE" or not NEWSAPI_KEY or not OPENROUTER_API_KEY:
        st.error("API keys not configured properly in .streamlit/secrets.toml. Please update them.")
        NEWSAPI_KEY = None
        OPENROUTER_API_KEY = None
except FileNotFoundError:
    st.error("secrets.toml file not found. Please create it in .streamlit/secrets.toml with your API keys (NEWSAPI_KEY, OPENROUTER_API_KEY).")
    NEWSAPI_KEY = None
    OPENROUTER_API_KEY = None
except KeyError:
    st.error("API keys (NEWSAPI_KEY, OPENROUTER_API_KEY) not found in secrets.toml. Please add them.")
    NEWSAPI_KEY = None
    OPENROUTER_API_KEY = None

# NIFTY 50 Stock Tickers (as of a recent date, subject to change)
# Source: Wikipedia / NSE public data. Ensure this list is up-to-date if critical.
INDIAN_STOCK_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS", # LTIMindtree
    "LT.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]
# Note: Removed BPCL.NS and UPL.NS, added SHRIRAMFIN.NS and LTIM.NS based on recent changes if Wikipedia list was slightly old.
# Corrected NIFTY 50 list based on common recent constituents. For precise current NIFTY 50, always refer to official NSE source.
# Final list being used based on commonly cited Nifty50 components, it's 50 symbols.
INDIAN_STOCK_TICKERS = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 
    'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'BPCL.NS', 
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 
    'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 
    'SHRIRAMFIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 
    'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS' 
]

st.info(f"Currently configured to process {len(INDIAN_STOCK_TICKERS)} tickers (NIFTY 50 components).")

# --- Data Gathering Engine (T4) ---
def calculate_technical_indicators(historical_data_df):
    """Calculates predefined technical indicators using pandas-ta."""
    if historical_data_df is None or historical_data_df.empty:
        return {}
    
    indicators = {}
    # Ensure columns are named as expected by pandas-ta (Open, High, Low, Close, Volume)
    # yfinance usually provides them in PascalCase, pandas-ta might expect lowercase or specific names.
    # For safety, let's rename to lowercase if they exist with initial caps.
    df = historical_data_df.copy()
    df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }, inplace=True)

    try:
        if 'close' in df.columns:
            indicators['RSI'] = round(df.ta.rsi().iloc[-1], 2) if df.ta.rsi() is not None and not df.ta.rsi().empty else None
            indicators['SMA50'] = round(df.ta.sma(50).iloc[-1], 2) if df.ta.sma(50) is not None and not df.ta.sma(50).empty else None
            indicators['SMA200'] = round(df.ta.sma(200).iloc[-1], 2) if df.ta.sma(200) is not None and not df.ta.sma(200).empty else None 
        
        # MACD returns a DataFrame, so we need to access specific columns
        macd_df = df.ta.macd()
        if macd_df is not None and not macd_df.empty:
            indicators['MACD_line'] = round(macd_df.iloc[-1]['MACD_12_26_9'], 2) if 'MACD_12_26_9' in macd_df.columns else None
            indicators['MACD_signal'] = round(macd_df.iloc[-1]['MACDs_12_26_9'], 2) if 'MACDs_12_26_9' in macd_df.columns else None
            indicators['MACD_hist'] = round(macd_df.iloc[-1]['MACDh_12_26_9'], 2) if 'MACDh_12_26_9' in macd_df.columns else None
        else:
            indicators['MACD_line'] = None
            indicators['MACD_signal'] = None
            indicators['MACD_hist'] = None

    except Exception as e:
        st.warning(f"Could not calculate some TA indicators: {e}")
        # Ensure all keys are present even if calculation fails
        for key in ['RSI', 'SMA50', 'SMA200', 'MACD_line', 'MACD_signal', 'MACD_hist']:
            if key not in indicators:
                indicators[key] = None
    return indicators

def gather_stock_data(tickers_list, news_api_key, period="1y", num_news_items=3):
    """Gathers historical data, technical indicators, and news for a list of stock tickers."""
    import yfinance as yf
    import pandas_ta as ta
    from newsapi import NewsApiClient

    all_stocks_data = []
    progress_bar = st.progress(0)
    total_tickers = len(tickers_list)

    newsapi = None
    if news_api_key:
        try:
            newsapi = NewsApiClient(api_key=news_api_key)
        except Exception as e:
            st.error(f"Failed to initialize NewsAPI client: {e}. News will not be fetched.")
            newsapi = None # Ensure it's None if initialization fails
    else:
        st.warning("NewsAPI key not provided. News will not be fetched.")

    for i, ticker_symbol in enumerate(tickers_list):
        current_ticker_news = []
        try:
            ticker = yf.Ticker(ticker_symbol)
            historical_data = ticker.history(period=period)
            
            if historical_data.empty:
                st.warning(f"No historical data found for {ticker_symbol}. Skipping TA.")
                tech_indicators = {}
            else:
                tech_indicators = calculate_technical_indicators(historical_data)
            
            # Fetch News
            if newsapi and ticker.info.get('shortName'):
                # Use shortName for better news query, fallback to ticker_symbol if not available
                query_term = ticker.info.get('shortName', ticker_symbol).split(' ')[0] # Often first word is most relevant
                # For Indian stocks, it's better to use the symbol without .NS for news search
                news_query_symbol = ticker_symbol.replace(".NS", "")
                try:
                    all_articles = newsapi.get_everything(
                        q=f'{news_query_symbol} OR {query_term}', # Search for symbol or first word of shortName
                        language='en',
                        sort_by='relevancy',
                        page_size=num_news_items * 2 # Fetch a bit more to filter
                    )
                    if all_articles['status'] == 'ok':
                        # Basic relevance filter: check if ticker symbol or short name part is in title
                        count = 0
                        for article in all_articles['articles']:
                            title_lower = article['title'].lower()
                            if news_query_symbol.lower() in title_lower or query_term.lower() in title_lower:
                                current_ticker_news.append({
                                    'title': article['title'],
                                    'url': article['url'],
                                    'publishedAt': article['publishedAt']
                                })
                                count += 1
                                if count >= num_news_items:
                                    break
                        if not current_ticker_news and all_articles['articles']: # if no relevant news found, take top ones
                            for article in all_articles['articles'][:num_news_items]:
                                current_ticker_news.append({
                                    'title': article['title'],
                                    'url': article['url'],
                                    'publishedAt': article['publishedAt']
                                })

                except Exception as e:
                    st.warning(f"Could not fetch news for {ticker_symbol} ({query_term}): {e}")
            
            stock_info_data = {
                "symbol": ticker_symbol,
                "info": ticker.info.get('shortName', ticker_symbol),
                "historical_data_raw": historical_data,
                "technicals": tech_indicators,
                "news": current_ticker_news
            }
            all_stocks_data.append(stock_info_data)
        except Exception as e:
            st.error(f"Error fetching or processing data for {ticker_symbol}: {e}")
        progress_bar.progress((i + 1) / total_tickers, text=f"Processing {ticker_symbol} ({i+1}/{total_tickers})")
    
    progress_bar.empty()
    return all_stocks_data

st.markdown("--- ")

# User Input Form
with st.form(key='user_input_form'):
    st.subheader("Your Investment Profile")
    risk_appetite = st.selectbox(
        "Risk Appetite",
        ("Low", "Medium", "High"),
        help="Select your tolerance for investment risk."
    )
    investment_horizon = st.selectbox(
        "Investment Horizon",
        ("Short-Term (less than 1 year)", "Medium-Term (1-5 years)", "Long-Term (more than 5 years)"),
        help="Select your investment time horizon."
    )
    investment_amount = st.number_input(
        "Investment Amount (₹)",
        min_value=0,
        step=1000,
        help="Enter the amount you plan to invest."
    )
    finance_experience = st.selectbox(
        "Finance Experience",
        ("Beginner", "Intermediate", "Advanced"),
        help="Select your level of financial market knowledge."
    )

    st.subheader("LLM Configuration")
    # Placeholder for actual model IDs/names
    llm_model_options = {
        "Model A (Placeholder)": "placeholder/model-a",
        "Model B (Placeholder)": "placeholder/model-b",
        "Model C (Placeholder)": "placeholder/model-c"
    }
    selected_llm_display_name = st.selectbox(
        "Select LLM Model",
        options=list(llm_model_options.keys()),
        help="Choose the LLM for generating insights."
    )

    submit_button = st.form_submit_button(label='Get Investment Idea')

if submit_button:
    st.write("Form Submitted! Processing... (This may take a while for many stocks)")
    # st.write(f"Risk Appetite: {risk_appetite}")
    # st.write(f"Investment Horizon: {investment_horizon}")
    # st.write(f"Investment Amount: ₹{investment_amount}")
    # st.write(f"Finance Experience: {finance_experience}")
    # st.write(f"Selected LLM: {selected_llm_display_name} (Actual ID: {llm_model_options[selected_llm_display_name]})")

    with st.spinner('Gathering stock market data... This might take a few minutes for the first run.'):
        # For now, using the global INDIAN_STOCK_TICKERS list
        # In a more complete app, you might filter this list or get it dynamically
        if NEWSAPI_KEY: # Only pass key if it exists
            detailed_stock_data = gather_stock_data(INDIAN_STOCK_TICKERS, news_api_key=NEWSAPI_KEY)
        else:
            st.warning("NEWSAPI_KEY not configured in secrets.toml. Proceeding without news data.")
            detailed_stock_data = gather_stock_data(INDIAN_STOCK_TICKERS, news_api_key=None)
    
    st.success("Data gathering complete!")
    
    if detailed_stock_data:
        st.subheader("Sample Gathered Data (First 2 Stocks):")
        for i, stock_data_point in enumerate(detailed_stock_data[:2]): # Display first 2 for brevity
            st.markdown(f"**{stock_data_point['info']} ({stock_data_point['symbol']})**")
            st.write("Technicals:", stock_data_point['technicals'])
            st.write("News Placeholder:", stock_data_point['news'])
            # st.dataframe(stock_data_point['historical_data_raw'].tail()) # Optionally display raw data
            st.markdown("--- ")
    else:
        st.warning("No data could be gathered for the selected stocks.")

st.markdown("--- ")
st.caption("_Not SEBI-registered; for educational use only._") 