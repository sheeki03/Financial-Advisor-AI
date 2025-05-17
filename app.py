import streamlit as st
import pandas as pd # Added for potential future use with stock list
import datetime # For date calculations
import requests # For OpenRouter API calls
import json # For OpenRouter API calls
from newsapi import NewsApiClient # Ensure NewsApiClient is imported at the top
import pandas_ta as ta # Ensure pandas_ta is imported at the top
import os

st.set_page_config(layout="wide")

def read_prompt_file():
    """Read the prompt file and return its contents."""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts', 'personal_finance_advisor_v1.md')
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error reading prompt file: {e}")
        return None

st.title("AI Financial Advisor MVP")

# Attempt to load API keys from st.secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY_HERE":
    st.error("OpenRouter API Key is not configured or is a placeholder. LLM features will not work. Please set it in .streamlit/secrets.toml")
    OPENROUTER_API_KEY = None # Explicitly set to None if invalid

if not NEWSAPI_KEY or NEWSAPI_KEY == "YOUR_NEWSAPI_KEY_HERE":
    st.warning("NewsAPI Key is not configured or is a placeholder. News data will not be fetched. Please set it in .streamlit/secrets.toml")
    NEWSAPI_KEY = None # Explicitly set to None if invalid

# NIFTY 50 Stock Tickers (as of a recent date, subject to change)
# Source: Wikipedia / NSE public data. Ensure this list is up-to-date if critical.
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

# --- Helper Function for Date Offset ---
def get_date_n_months_ago(n_months):
    """Return the date *n* months ago as a pandas.Timestamp."""
    return pd.Timestamp.today() - pd.DateOffset(months=n_months)

# --- Technical Analysis Helper Functions ---
def _passes_ta_filter(technicals, risk_profile):
    """Checks if a stock's technicals meet criteria for a given risk profile."""
    rsi = technicals.get('RSI')
    sma50 = technicals.get('SMA50')
    sma200 = technicals.get('SMA200')
    macd_line = technicals.get('MACD_line')
    macd_signal = technicals.get('MACD_signal')
    # macd_hist = technicals.get('MACD_hist') # Calculated from line and signal if needed

    if risk_profile == "Low":
        # Low Risk: Prefers non-overbought, not in a death cross, and MACD not bearish.
        # Conditions pass if data is missing, making it more inclusive.
        cond1_rsi_ok = (rsi is None) or (rsi < 65) 
        cond2_sma_ok = (sma50 is None or sma200 is None) or (sma50 >= sma200)
        cond3_macd_ok = (macd_line is None or macd_signal is None) or (macd_line >= macd_signal)
        return cond1_rsi_ok and cond2_sma_ok and cond3_macd_ok

    elif risk_profile == "High":
        # High Risk: Requires strong bullish signals, data must be present.
        cond1_rsi_ok = (rsi is not None and rsi > 55)
        cond2_sma_ok = (sma50 is not None and sma200 is not None and sma50 > sma200) # Golden Cross
        cond3_macd_ok = (macd_line is not None and macd_signal is not None and macd_line > macd_signal) # Bullish MACD
        return cond1_rsi_ok and cond2_sma_ok and cond3_macd_ok

    elif risk_profile == "Medium":
        # Medium Risk: RSI in a healthy range, and MACD not overtly bearish.
        # RSI must be present for this profile.
        cond1_rsi_ok = (rsi is not None and 35 < rsi < 70)
        cond2_macd_ok = (macd_line is None or macd_signal is None) or (macd_line >= macd_signal) # MACD not bearish
        return cond1_rsi_ok and cond2_macd_ok
    
    return True # Default to pass if risk_profile is unknown (should not happen)

# --- Data Gathering Engine (T4) ---
def calculate_technical_indicators(historical_data_df):
    """Calculates predefined technical indicators using pandas-ta."""
    if historical_data_df is None or historical_data_df.empty:
        return {}
    
    indicators = {}
    df = historical_data_df.copy()
    df.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }, inplace=True)

    try:
        if 'close' in df.columns and len(df['close']) > 1:
            indicators['RSI'] = round(df.ta.rsi().iloc[-1], 2) if df.ta.rsi() is not None and not df.ta.rsi().empty else None
            indicators['SMA50'] = round(df.ta.sma(50).iloc[-1], 2) if df.ta.sma(50) is not None and not df.ta.sma(50).empty and len(df) >= 50 else None
            indicators['SMA200'] = round(df.ta.sma(200).iloc[-1], 2) if df.ta.sma(200) is not None and not df.ta.sma(200).empty and len(df) >= 200 else None
        
        macd_df = df.ta.macd()
        if macd_df is not None and not macd_df.empty:
            indicators['MACD_line'] = round(macd_df.iloc[-1]['MACD_12_26_9'], 2) if 'MACD_12_26_9' in macd_df.columns else None
            indicators['MACD_signal'] = round(macd_df.iloc[-1]['MACDs_12_26_9'], 2) if 'MACDs_12_26_9' in macd_df.columns else None
            indicators['MACD_hist'] = round(macd_df.iloc[-1]['MACDh_12_26_9'], 2) if 'MACDh_12_26_9' in macd_df.columns else None # MACD_hist is calculated
        else:
            indicators['MACD_line'] = None
            indicators['MACD_signal'] = None
            indicators['MACD_hist'] = None
    except Exception as e:
        st.warning(f"Could not calculate some TA indicators for a stock: {e}")
        for key in ['RSI', 'SMA50', 'SMA200', 'MACD_line', 'MACD_signal', 'MACD_hist']:
            if key not in indicators:
                indicators[key] = None
    return indicators

@st.cache_data(ttl="12h")
def _fetch_news_for_ticker_cached(api_key_for_cache, news_query_symbol, query_term, num_news_items_for_cache):
    if not api_key_for_cache:
        return {"status": "error", "message": "API key not provided.", "articles": []}

    try:
        local_newsapi_client = NewsApiClient(api_key=api_key_for_cache)
    except Exception as e:
        return {"status": "error", "message": f"Failed to initialize NewsAPI client: {e}", "articles": []}

    processed_news_list = []
    try:
        all_articles_response = local_newsapi_client.get_everything(
            q=f'{news_query_symbol} OR {query_term}',
            language='en',
            sort_by='relevancy',
            page_size=max(num_news_items_for_cache * 2, 20) # Fetch more to allow filtering, ensure reasonable page size
        )
        if all_articles_response['status'] == 'ok':
            count = 0
            for article in all_articles_response['articles']:
                title_lower = article['title'].lower()
                if news_query_symbol.lower() in title_lower or \
                   (query_term.lower() != news_query_symbol.lower() and query_term.lower() in title_lower):
                    processed_news_list.append({
                        'title': article['title'],
                        'url': article['url'],
                        'publishedAt': article['publishedAt']
                    })
                    count += 1
                    if count >= num_news_items_for_cache:
                        break
            
            if len(processed_news_list) < num_news_items_for_cache and all_articles_response['articles']:
                existing_urls = {news['url'] for news in processed_news_list}
                for article in all_articles_response['articles']:
                    if len(processed_news_list) >= num_news_items_for_cache:
                        break
                    if article['url'] not in existing_urls:
                        processed_news_list.append({
                            'title': article['title'],
                            'url': article['url'],
                            'publishedAt': article['publishedAt']
                        })
                        existing_urls.add(article['url'])
            
            return {"status": "ok", "message": "News fetched successfully", "articles": processed_news_list}
        else:
            error_message = all_articles_response.get('message', 'Unknown error from NewsAPI')
            return {"status": "error", "message": f"NewsAPI error: {error_message}", "articles": []}
    except Exception as e:
        return {"status": "error", "message": f"Exception during news fetch: {e}", "articles": []}

def gather_stock_data(tickers_list, news_api_key_local, period="1y", num_news_items=3):
    """Gathers historical data, technical indicators, and news for a list of stock tickers."""
    import yfinance as yf
    import pandas_ta as ta

    all_stocks_data = []
    progress_bar = st.progress(0)
    total_tickers = len(tickers_list)

    newsapi = None
    if news_api_key_local:
        from newsapi import NewsApiClient
        try:
            newsapi = NewsApiClient(api_key=news_api_key_local)
        except Exception as e:
            st.error(f"Failed to initialize NewsAPI client: {e}. News will not be fetched.")
            newsapi = None
    # No warning if key is None from initial check, already handled

    date_6_months_ago = get_date_n_months_ago(6)

    for i, ticker_symbol in enumerate(tickers_list):
        current_ticker_news = []
        six_month_change_val = None
        tech_indicators = {} # Initialize here
        try:
            ticker = yf.Ticker(ticker_symbol)
            historical_data = ticker.history(period=period)
            
            if historical_data.empty:
                st.warning(f"No historical data found for {ticker_symbol}. Skipping TA and 6m change.")
                # tech_indicators remains empty
            else:
                tech_indicators = calculate_technical_indicators(historical_data) # Populated here
                try:
                    price_today = historical_data['Close'].iloc[-1]
                    
                    # Corrected logic for timezone-aware/naive comparison
                    # date_6_months_ago is already a naive pd.Timestamp from get_date_n_months_ago(6)
                    if historical_data.index.tz is not None:
                        # If index is timezone-aware, convert it to naive for comparison against naive date_6_months_ago
                        price_6_months_ago_series = historical_data[historical_data.index.tz_localize(None) <= date_6_months_ago]['Close']
                    else:
                        # If index is already naive, compare directly
                        price_6_months_ago_series = historical_data[historical_data.index <= date_6_months_ago]['Close']

                    if not price_6_months_ago_series.empty:
                        price_6_months_ago = price_6_months_ago_series.iloc[-1]
                        if price_6_months_ago != 0: 
                            six_month_change_val = round(((price_today - price_6_months_ago) / price_6_months_ago) * 100, 2)
                    else:
                        st.warning(f"No data point found approximately 6 months ago for {ticker_symbol} to calculate change.")
                except IndexError:
                    st.warning(f"Not enough historical data to calculate 6-month change for {ticker_symbol}")
                except Exception as e:
                    st.warning(f"Error calculating 6-month change for {ticker_symbol}: {e}")

            if newsapi and ticker.info.get('shortName'):
                query_term = ticker.info.get('shortName', ticker_symbol).split(' ')[0] 
                news_query_symbol = ticker_symbol.replace(".NS", "")
                news_result = _fetch_news_for_ticker_cached(
                    news_api_key_local, 
                    news_query_symbol,
                    query_term,
                    num_news_items 
                )
                if news_result["status"] == "ok":
                    current_ticker_news = news_result["articles"]
                else:
                    st.warning(f"Could not fetch news for {ticker_symbol} ({query_term}): {news_result['message']}")
                    current_ticker_news = [] # Ensure it's an empty list on error
            
            stock_info_data = {
                "symbol": ticker_symbol,
                "info": ticker.info.get('shortName', ticker_symbol),
                # "historical_data_raw": historical_data, # Commenting out to save memory for now
                "six_month_change": six_month_change_val,
                "technicals": tech_indicators,
                "news": current_ticker_news
            }
            all_stocks_data.append(stock_info_data)
        except Exception as e:
            st.error(f"Error fetching or processing data for {ticker_symbol}: {e}")
        progress_bar.progress((i + 1) / total_tickers, text=f"Processing {ticker_symbol} ({i+1}/{total_tickers})")
    
    progress_bar.empty()
    return all_stocks_data

# --- Candidate Selection Logic (T5) ---
def select_candidate_stocks(user_profile, all_stocks_data_list, num_candidates=3):
    risk = user_profile.get("risk_appetite", "Medium")
    # Initial pool: stocks with 6m change data
    valid_stocks_for_price_change = [s for s in all_stocks_data_list if s.get("six_month_change") is not None]

    if not valid_stocks_for_price_change:
        st.warning("No stocks with valid 6-month performance data found to select candidates.")
        return []

    # TA Filtering
    ta_filtered_stocks = [
        stock for stock in valid_stocks_for_price_change 
        if _passes_ta_filter(stock.get('technicals', {}), risk)
    ]

    candidate_selection_pool = []
    if len(ta_filtered_stocks) >= 1: # If at least one stock passes TA filter
        candidate_selection_pool = ta_filtered_stocks
        if len(ta_filtered_stocks) < num_candidates:
            st.info(f"TA filtering for '{risk}' risk yielded only {len(ta_filtered_stocks)} stock(s) (less than {num_candidates} desired). These will be prioritized for selection based on price change.")
        else:
            st.info(f"Applied TA filtering for '{risk}' risk. {len(ta_filtered_stocks)} stocks met TA criteria and will be used for price-change based selection.")
    elif valid_stocks_for_price_change: # No stocks passed TA filter, fall back to all valid stocks
        candidate_selection_pool = valid_stocks_for_price_change
        st.warning(f"No stocks met TA filtering criteria for '{risk}' risk. Selecting candidates based purely on 6-month price change from all {len(valid_stocks_for_price_change)} stocks with price data.")
    else: # Should not happen if valid_stocks_for_price_change was not empty initially
        return []
    
    if not candidate_selection_pool: # Should ideally not be empty if valid_stocks_for_price_change wasn't
        st.error("Unexpected: Candidate selection pool is empty after TA filtering and fallback.")
        return []

    # Sort the chosen pool based on price change criteria
    final_candidates = []
    pool_size = len(candidate_selection_pool)

    if risk == "High":
        # High risk: Top performers from the pool (highest positive 6m change)
        sorted_pool = sorted(candidate_selection_pool, key=lambda x: x["six_month_change"], reverse=True)
        final_candidates = sorted_pool[:min(num_candidates, pool_size)]
    elif risk == "Low":
        # Low risk: Most stable from the pool (smallest absolute 6m change)
        sorted_pool = sorted(candidate_selection_pool, key=lambda x: abs(x["six_month_change"]))
        final_candidates = sorted_pool[:min(num_candidates, pool_size)]
    elif risk == "Medium":
        # Medium risk: Middle performers from the pool based on actual 6m change
        sorted_pool = sorted(candidate_selection_pool, key=lambda x: x["six_month_change"])
        mid_point = pool_size // 2
        start_index = max(0, mid_point - (num_candidates // 2))
        end_index = min(pool_size, start_index + num_candidates)
        final_candidates = sorted_pool[start_index : end_index]
        # Ensure we get num_candidates if pool is large enough, adjust if slice is too small
        if len(final_candidates) < num_candidates and pool_size >= num_candidates:
            # If centered slice is too small, try to take num_candidates from one end of the middle
            # This part might need refinement if the pool_size is just num_candidates or slightly more.
            # For now, the slice start_index:end_index should be reasonable.
            # If it yielded less than num_candidates (e.g. pool_size=3, num_cand=3, mid=1, start=0, end=3 -> 3 stocks)
            # (e.g. pool_size=4, num_cand=3, mid=2, start=1, end=4 -> 3 stocks)
            # (e.g. pool_size=5, num_cand=3, mid=2, start=1, end=4 -> 3 stocks)
            # The min(num_candidates, pool_size) in High/Low handles small pools.
            # For medium, if pool_size < num_candidates, the slice might be smaller.
            # Let's ensure it takes up to num_candidates if available.
            if pool_size < num_candidates:
                 final_candidates = sorted_pool # take all from the small pool
            # else the slice [start_index:end_index] is probably what we want for "middle"
            # if it's still less than num_candidates, it means the pool itself is small for centering num_candidates.
            # E.g. pool_size=2, num_cand=3. mid=1, start=0, end=2. final_candidates len 2. Good.
            # To ensure exactly num_candidates if pool_size >= num_candidates:
            if pool_size >= num_candidates and len(final_candidates) < num_candidates :
                 # This scenario might indicate an off-by-one or small pool centering issue.
                 # Re-center or broaden the take.
                 # A simple fallback: take top num_candidates after sorting for medium if centering is tricky.
                 # Or take from start_index for num_candidates items.
                 final_candidates = sorted_pool[start_index : min(pool_size, start_index + num_candidates)]


    if not final_candidates and candidate_selection_pool: # Fallback if somehow selection logic failed
        st.warning("Candidate selection based on risk-specific price change logic failed or yielded no results from the pool. Taking top N from pool based on high-risk criteria.")
        final_candidates = sorted(candidate_selection_pool, key=lambda x: x["six_month_change"], reverse=True)[:min(num_candidates, pool_size)]

    return final_candidates

# --- LLM Rationale Generation (T6) ---
def get_llm_rationale_for_stock(user_profile, stock_data, openrouter_key, llm_model_id):
    """Generates investment rationale for a single stock using OpenRouter LLM."""
    if not openrouter_key:
        return "OpenRouter API key not available. Cannot generate rationale."
    if not stock_data:
        return "No stock data provided to LLM."

    # Read the base prompt
    base_prompt = read_prompt_file()
    if not base_prompt:
        return "Could not read prompt file. Cannot generate rationale."

    # Prepare news section
    news_section = ""
    if stock_data.get('news'):
        for news_item in stock_data['news'][:2]:
            news_section += f"  - {news_item.get('title', 'N/A')} ({news_item.get('publishedAt', 'N/A')})\n"
    else:
        news_section = "  - No recent news available.\n"

    # Calculate price vs SMA percentages if available
    price_vs_sma = ""
    if all(k in stock_data.get('technicals', {}) for k in ['SMA50', 'SMA200']):
        sma50 = stock_data['technicals']['SMA50']
        sma200 = stock_data['technicals']['SMA200']
        if sma50 and sma200 and sma200 != 0:
            price_vs_sma = f"Price vs SMA50/200: {((sma50 - sma200) / sma200 * 100):.1f}%"

    # Format the prompt with actual data
    prompt = f"""YOU ARE A PERSONAL-FINANCE ADVISOR AI WITH INSTITUTIONAL-GRADE EQUITY ANALYSIS SKILLS.

TASK: Provide a crisp yet thorough investment rationale and position-size suggestion for a single stock, fully customized to the supplied user profile and investment amount.

User Profile:
- Risk Appetite: {user_profile.get('risk_appetite')}
- Investment Horizon: {user_profile.get('investment_horizon')}
- Finance Experience: {user_profile.get('finance_experience')}
- Investment Amount: ₹{user_profile.get('investment_amount', 0):,}

Stock for Analysis: {stock_data.get('info', stock_data.get('symbol'))} ({stock_data.get('symbol')})

Key Data Points:
- 6-Month Price Change: {stock_data.get('six_month_change', 'N/A')}%
- Technical Indicators: 
    - RSI: {stock_data.get('technicals', {}).get('RSI', 'N/A')}
    - SMA50: {stock_data.get('technicals', {}).get('SMA50', 'N/A')}
    - SMA200: {stock_data.get('technicals', {}).get('SMA200', 'N/A')}
    - MACD Line: {stock_data.get('technicals', {}).get('MACD_line', 'N/A')}
    - MACD Signal: {stock_data.get('technicals', {}).get('MACD_signal', 'N/A')}
- Recent News Snippets (max 2):
{news_section}

{base_prompt}"""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": llm_model_id,
                "messages": [{"role": "user", "content": prompt}]
            }),
            timeout=60
        )
        response.raise_for_status()
        api_response_json = response.json()
        
        if api_response_json.get("choices") and len(api_response_json["choices"]) > 0:
            rationale = api_response_json["choices"][0]["message"]["content"]
            return rationale.strip()
        else:
            st.error(f"LLM API call for {stock_data.get('symbol')} did not return expected choices. Response: {api_response_json}")
            return "Could not generate rationale: Unexpected API response structure."

    except requests.exceptions.RequestException as e:
        st.error(f"LLM API call failed for {stock_data.get('symbol')}: {e}")
        return f"Could not generate rationale: API request error ({e})"
    except Exception as e:
        st.error(f"An unexpected error occurred during LLM rationale generation for {stock_data.get('symbol')}: {e}")
        return "Could not generate rationale due to an unexpected error."

st.markdown("--- ")

# User Input Form
with st.form(key='user_input_form'):
    st.subheader("Your Investment Profile")
    risk_appetite_input = st.selectbox(
        "Risk Appetite",
        ("Low", "Medium", "High"),
        key="risk_appetite_select",
        help="Select your tolerance for investment risk."
    )
    investment_horizon_input = st.selectbox(
        "Investment Horizon",
        ("Short-Term (less than 1 year)", "Medium-Term (1-5 years)", "Long-Term (more than 5 years)"),
        key="investment_horizon_select",
        help="Select your investment time horizon."
    )
    investment_amount_input = st.number_input(
        "Investment Amount (₹)",
        min_value=0,
        step=1000,
        key="investment_amount_input",
        help="Enter the amount you plan to invest."
    )
    finance_experience_input = st.selectbox(
        "Finance Experience",
        ("Beginner", "Intermediate", "Advanced"),
        key="finance_experience_select",
        help="Select your level of financial market knowledge."
    )

    st.subheader("LLM Configuration")
    llm_model_options = {
        "Qwen1.5 32B Chat": "qwen/qwen3-30b-a3b",
        "DeepSeek R1T Chimera": "tngtech/deepseek-r1t-chimera",
        # Adding back some popular free models from previous state if they were removed by user edit
    }
    selected_llm_display_name = st.selectbox(
        "Select LLM Model",
        options=list(llm_model_options.keys()),
        key="llm_model_select",
        help="Choose the LLM for generating insights."
    )

    submit_button = st.form_submit_button(label='Get Investment Idea')

if submit_button:
    st.write("Form Submitted! Processing... (This may take a while for NIFTY 50 stocks)")
    
    user_profile_data = {
        "risk_appetite": risk_appetite_input,
        "investment_horizon": investment_horizon_input,
        "investment_amount": investment_amount_input,
        "finance_experience": finance_experience_input,
        "selected_llm_display_name": selected_llm_display_name,
        "selected_llm_model_id": llm_model_options[selected_llm_display_name]
    }

    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API Key is not configured. LLM rationale generation will be skipped. Please set it in .streamlit/secrets.toml")
        # Allow to proceed without LLM for now, but T6/T7 will be incomplete.
    
    with st.spinner('Gathering stock market data... This might take a few minutes.'):
        detailed_stock_data = gather_stock_data(INDIAN_STOCK_TICKERS, news_api_key_local=NEWSAPI_KEY)
    
    if not detailed_stock_data:
        st.error("Could not gather any stock data. Please check logs or try again later.")
        st.stop()
    
    st.success(f"Data gathering complete for {len(detailed_stock_data)} stocks!")
    
    with st.spinner("Selecting candidate stocks based on your profile..."):
        candidate_stocks = select_candidate_stocks(user_profile_data, detailed_stock_data, num_candidates=5)

    if candidate_stocks:
        st.subheader("Investment Ideas & LLM Rationales:")
        # For MVP, let's process only the first candidate for LLM rationale to save time/API calls
        # In future, could process all or let user choose.
        # num_candidates_for_llm = 1 # Can be increased up to len(candidate_stocks) or num_candidates from select_candidate_stocks
        
        for i, candidate in enumerate(candidate_stocks): # Iterate through all 3 candidates
            st.markdown(f"### Idea {i+1}: {candidate['info']} ({candidate['symbol']})")
            st.write(f"**6-Month Price Change:** {candidate.get('six_month_change', 'N/A')}%   **RSI:** {candidate.get('technicals', {}).get('RSI', 'N/A')}")
            st.write(f"**SMA50:** {candidate.get('technicals', {}).get('SMA50', 'N/A')}   **SMA200:** {candidate.get('technicals', {}).get('SMA200', 'N/A')}")
            
            if candidate.get('news'):
                with st.expander("View Recent News Snippets"):
                    for news_item in candidate['news'][:2]: 
                        st.markdown(f"- [{news_item['title']}]({news_item['url']}) ({news_item['publishedAt']})")
            
            if OPENROUTER_API_KEY: # Only attempt LLM call if key is present
                with st.spinner(f"Generating LLM rationale for {candidate['symbol']}..."):
                    rationale = get_llm_rationale_for_stock(
                        user_profile_data,
                        candidate, 
                        OPENROUTER_API_KEY, 
                        user_profile_data["selected_llm_model_id"]
                    )
                st.markdown("**LLM Rationale:**")
                st.markdown(rationale)
            else:
                st.warning(f"LLM Rationale for {candidate['symbol']} skipped as OpenRouter API key is not configured.")
            st.markdown("--- ")        
    else:
        st.warning("Could not select any candidate stocks based on the current criteria and available data.")

st.markdown("--- ")
st.caption("_Not SEBI-registered; for educational use only._") 