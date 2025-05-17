import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from functools import lru_cache
import plotly.graph_objects as go
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure set_page_config is called only once
if not st.session_state.get("page_config_set", False):
    st.set_page_config(layout="wide", page_title="Stock Dashboard")
    st.session_state.page_config_set = True

# Separate ticker database
TICKER_DB = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Steel": "TATASTEEL.NS",
    "State Bank of India": "SBIN.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
}

# Function to fetch real-time company details
def fetch_company_details(ticker: str, company_name: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        # Basic company info
        details = {
            "industry": info.get("industry", "N/A"),
            "sector": info.get("sector", "N/A"),
            "subsidiaries": [],  # Attempt to scrape
            "products": [],  # Attempt to scrape
            "operations": {
                "employees": info.get("fullTimeEmployees", "N/A"),
                "revenue": f"₹{financials.loc['Total Revenue'].iloc[0]/1e7:.2f} Cr" if 'Total Revenue' in financials.index else "N/A",
                "net_profit": f"₹{financials.loc['Net Income'].iloc[0]/1e7:.2f} Cr" if 'Net Income' in financials.index else "N/A",
                "market_cap": f"₹{info.get('marketCap', 0)/1e7:.2f} Cr",
                "facilities": "N/A",  # To be scraped
                "sustainability": info.get("esgSustainability", {}).get("sustainabilityScore", "N/A")
            }
        }

        # Financial Metrics with robust Altman Z-score calculation
        total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 1
        total_liabilities = balance_sheet.loc['Total Liabilities'].iloc[0] if 'Total Liabilities' in balance_sheet.index else 0
        working_capital = (balance_sheet.loc['Total Current Assets'].iloc[0] - balance_sheet.loc['Total Current Liabilities'].iloc[0]) if ('Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index) else 0
        retained_earnings = balance_sheet.loc['Retained Earnings'].iloc[0] if 'Retained Earnings' in balance_sheet.index else 0
        ebit = financials.loc['Ebit'].iloc[0] if 'Ebit' in financials.index else 0
        revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
        market_cap = info.get('marketCap', 0)

        # Avoid division by zero
        altman_z = 0
        if total_assets != 0 and total_liabilities != 0:
            altman_z = (1.2 * (working_capital / total_assets) +
                        1.4 * (retained_earnings / total_assets) +
                        3.3 * (ebit / total_assets) +
                        0.6 * (market_cap / total_liabilities) +
                        1.0 * (revenue / total_assets))

        details["financial_metrics"] = {
            "altman_z_score": round(altman_z, 2),
            "piotroski_f_score": "N/A",  # Requires detailed checks
            "beneish_m_score": "N/A",  # Requires detailed checks
            "roce": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A",
            "roa": f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else "N/A",
            "ebitda_margin": f"{info.get('ebitdaMargins', 0)*100:.2f}%" if info.get('ebitdaMargins') else "N/A",
            "debt_to_equity": f"{info.get('debtToEquity', 0)/100:.2f}" if info.get('debtToEquity') else "N/A",
            "current_ratio": f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else "N/A",
            "cash_flow_to_debt": "N/A"
        }

        # Valuation Ratios
        details["valuation_ratios"] = {
            "pe_ratio": f"{info.get('trailingPE', 'N/A')}",
            "pb_ratio": f"{info.get('priceToBook', 'N/A')}",
            "peg_ratio": f"{info.get('pegRatio', 'N/A')}",
            "dividend_yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A",
            "ev_ebitda": f"{info.get('enterpriseToEbitda', 'N/A')}",
            "price_to_sales": f"{info.get('priceToSalesTrailing12Months', 'N/A')}"
        }

        # Web scraping for additional details
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = f"https://www.moneycontrol.com/india/stockpricequote/{company_name.lower().replace(' ', '')}/{ticker}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt to scrape subsidiaries and products (simplified)
        subsidiaries = soup.find_all('div', class_='company_info') or []
        details["subsidiaries"] = [s.text.strip() for s in subsidiaries if s.text.strip()] or ["N/A"]
        details["products"] = [{"name": "N/A", "materials": [], "processes": [], "description": "N/A"}]  # Placeholder

        # Competitive Position
        details["competitive_position"] = {
            "market_share": "N/A",
            "moat": info.get("longBusinessSummary", "N/A")[:100] + "...",
            "competitors": ["N/A"],  # Placeholder
            "swot": {
                "strengths": ["N/A"],
                "weaknesses": ["N/A"],
                "opportunities": ["N/A"],
                "threats": ["N/A"]
            }
        }

        # Management Quality
        details["management_quality"] = {
            "ceo": info.get("companyOfficers", [{"name": "N/A"}])[0]["name"],
            "tenure": "N/A",
            "track_record": "N/A",
            "compensation": "N/A",
            "insider_ownership": f"{info.get('heldPercentInsiders', 0)*100:.2f}%",
            "governance": "N/A"
        }

        # Strategic Outlook
        details["strategic_outlook"] = {
            "growth_drivers": ["N/A"],
            "risks": ["N/A"],
            "analyst_consensus": f"{info.get('recommendationMean', 'N/A')} (1=Buy, 5=Sell)",
            "innovation": ["N/A"]
        }

        return details
    except Exception as e:
        logger.error(f"Error fetching details for {ticker}: {str(e)}")
        return {}

# Cache stock data fetching
@lru_cache(maxsize=100)
def fetch_stock_data(ticker: str, period: str, interval: str) -> tuple:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        logger.debug(f"Fetched data for {ticker}: {len(df)} rows")
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame(), {}
        return df, stock.info
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame(), {}

# Function to search tickers
def search_ticker(query: str) -> list:
    query = query.lower().strip()
    return [(company, ticker) for company, ticker in TICKER_DB.items() if query in company.lower()]

# Function to get currency
def get_currency_and_symbol(ticker_info: dict) -> tuple:
    exchange = ticker_info.get('exchange', '')
    country = ticker_info.get('country', '')
    currency_map = {
        'India': ('INR', '₹'), 'United States': ('USD', '$'), 'United Kingdom': ('GBP', '£'),
        'European Union': ('EUR', '€'), 'Japan': ('JPY', '¥'), 'China': ('CNY', '¥'),
        'Australia': ('AUD', 'A$'), 'South Korea': ('KRW', '₩'), 'Hong Kong': ('HKD', 'HK$')
    }
    if country in currency_map:
        return currency_map[country]
    elif 'NSE' in exchange or 'BSE' in exchange:
        return ('INR', '₹')
    return ('USD', '$')

# Function to calculate Fibonacci levels
def calculate_fibonacci_levels(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    return {
        '0.0%': high, '23.6%': high - 0.236 * diff, '38.2%': high - 0.382 * diff,
        '50.0%': high - 0.5 * diff, '61.8%': high - 0.618 * diff, '100.0%': low
    }

# Function to calculate pivot points
def calculate_pivot_points(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 1:
        return 0, 0, 0
    pivot = (df['High'][-1] + df['Low'][-1] + df['Close'][-1]) / 3
    support1 = (2 * pivot) - df['High'][-1]
    resistance1 = (2 * pivot) - df['Low'][-1]
    return pivot, support1, resistance1

# Function to create stock price chart
def create_stock_chart(df: pd.DataFrame, company: str) -> go.Figure:
    if df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')
    ))
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3
    ))
    fig.update_layout(
        title=f"{company} Stock Price and Volume",
        yaxis=dict(title="Price (INR)"),
        yaxis2=dict(title="Volume", overlaying='y', side='right'),
        xaxis=dict(title="Date"),
        template="plotly_white"
    )
    return fig

# Streamlit app
def main():
    st.title("Stock Dashboard")

    # Initialize session state for expanders
    if 'stock_info_expanded' not in st.session_state:
        st.session_state.stock_info_expanded = False
    if 'key_levels_expanded' not in st.session_state:
        st.session_state.key_levels_expanded = False
    if 'company_details_expanded' not in st.session_state:
        st.session_state.company_details_expanded = False

    # Sidebar for inputs
    with st.sidebar:
        st.header("Stock Selection")
        search_query = st.text_input("Search Company Name", "")
        selected_ticker = None
        selected_company = None
        matches = []

        if search_query:
            matches = search_ticker(search_query)
            if matches:
                options = [f"{company} ({ticker})" for company, ticker in matches]
                selected_option = st.selectbox("Select a stock", options)
                if selected_option:
                    selected_company, selected_ticker = matches[options.index(selected_option)]
            else:
                st.warning("No matching stocks found.")

        ticker = st.text_input("Enter Stock Ticker", value=selected_ticker if selected_ticker else "")
        period = st.selectbox("Time Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=3)
        interval = st.selectbox("Data Interval", ['1d', '1wk', '1mo'], index=0)
        analyze = st.button("Analyze")
        find_details = st.button("Find Details")

    if analyze and ticker:
        try:
            with st.spinner("Analyzing..."):
                df, info = fetch_stock_data(ticker, period, interval)
                if df.empty:
                    st.error(f"No data found for {ticker}. Please check the ticker symbol or try again later.")
                    logger.error(f"No data for ticker {ticker}")
                    return

                currency, currency_symbol = get_currency_and_symbol(info)
                fib_levels = calculate_fibonacci_levels(df)
                pivot, support1, resistance1 = calculate_pivot_points(df)

                # Helper function to format values
                def format_value(value, fmt=".2f", default="N/A"):
                    try:
                        return f"{value:{fmt}}" if pd.notnull(value) else default
                    except (ValueError, TypeError) as e:
                        logger.error(f"Formatting error: {str(e)}")
                        return default

                # Stock Information Card
                with st.expander("Stock Information", expanded=st.session_state.stock_info_expanded):
                    st.markdown(f"**Company**: {info.get('longName', 'N/A')}")
                    st.markdown(f"**Current Price**: {currency_symbol}{format_value(df['Close'][-1])}")
                    st.session_state.stock_info_expanded = not st.session_state.stock_info_expanded
                    if st.session_state.stock_info_expanded:
                        st.markdown(f"**Exchange**: {info.get('exchange', 'N/A')}")
                        st.markdown(f"**Currency**: {currency} ({currency_symbol})")
                        st.markdown(f"**52 Week High**: {currency_symbol}{format_value(info.get('fiftyTwoWeekHigh', 'N/A'))}")
                        st.markdown(f"**52 Week Low**: {currency_symbol}{format_value(info.get('fiftyTwoWeekLow', 'N/A'))}")

                # Key Levels Card
                with st.expander("Key Levels", expanded=st.session_state.key_levels_expanded):
                    st.markdown(f"**Pivot Point**: {currency_symbol}{format_value(pivot)}")
                    st.session_state.key_levels_expanded = not st.session_state.key_levels_expanded
                    if st.session_state.key_levels_expanded:
                        st.markdown(f"**Support 1**: {currency_symbol}{format_value(support1)}")
                        st.markdown(f"**Resistance 1**: {currency_symbol}{format_value(resistance1)}")
                        st.markdown("**Fibonacci Levels**:")
                        for level, price in fib_levels.items():
                            st.markdown(f"- {level}: {currency_symbol}{format_value(price)}")

                # Display stock chart
                st.plotly_chart(create_stock_chart(df, info.get('longName', ticker)))

        except Exception as e:
            st.error(f"Error fetching data: {str(e)}. Please try again or check your network connection.")
            logger.error(f"Main loop error: {str(e)}")

    if find_details and selected_company:
        try:
            ticker = TICKER_DB.get(selected_company, "")
            if not ticker:
                st.error(f"No ticker found for {selected_company}.")
                return

            with st.spinner("Fetching company details..."):
                company_details = fetch_company_details(ticker, selected_company)
                if not company_details:
                    st.error(f"No details found for {selected_company}. Please try again later.")
                    return

            with st.expander("Company Details", expanded=st.session_state.company_details_expanded):
                st.markdown(f"**Company**: {selected_company}")
                st.markdown(f"**Industry**: {company_details.get('industry', 'N/A')}")
                st.session_state.company_details_expanded = not st.session_state.company_details_expanded
                if st.session_state.company_details_expanded:
                    # Subsidiaries Table
                    st.subheader("Subsidiaries")
                    subsidiaries_df = pd.DataFrame(company_details.get("subsidiaries", []), columns=["Subsidiary"])
                    st.table(subsidiaries_df)

                    # Operations and Financials
                    st.subheader("Operations and Financials")
                    ops = company_details.get("operations", {})
                    ops_df = pd.DataFrame([
                        {"Metric": "Employees", "Value": ops.get("employees", "N/A")},
                        {"Metric": "Revenue", "Value": ops.get("revenue", "N/A")},
                        {"Metric": "Net Profit", "Value": ops.get("net_profit", "N/A")},
                        {"Metric": "Market Cap", "Value": ops.get("market_cap", "N/A")},
                        {"Metric": "Facilities", "Value": ops.get("facilities", "N/A")},
                        {"Metric": "Sustainability", "Value": ops.get("sustainability", "N/A")}
                    ])
                    st.table(ops_df)

                    # Products Table
                    st.subheader("Products")
                    products_df = pd.DataFrame([
                        {
                            "Name": p.get("name", "N/A"),
                            "Materials": ", ".join(p.get("materials", [])),
                            "Processes": ", ".join(p.get("processes", [])),
                            "Description": p.get("description", "N/A")
                        } for p in company_details.get("products", [])
                    ])
                    st.table(products_df)

                    # Financial Metrics
                    st.subheader("Advanced Financial Metrics")
                    fin_metrics = company_details.get("financial_metrics", {})
                    fin_metrics_df = pd.DataFrame([
                        {"Metric": "Altman Z-Score", "Value": fin_metrics.get("altman_z_score", "N/A"), "Description": "Measures financial distress risk (Safe: >2.99, Grey: 1.81-2.99, Distress: <1.81)"},
                        {"Metric": "Piotroski F-Score", "Value": fin_metrics.get("piotroski_f_score", "N/A"), "Description": "Assesses financial health (0-9, higher is better)"},
                        {"Metric": "Beneish M-Score", "Value": fin_metrics.get("beneish_m_score", "N/A"), "Description": "Detects earnings manipulation (<-2.22 suggests low risk)"},
                        {"Metric": "ROCE", "Value": fin_metrics.get("roce", "N/A"), "Description": "Return on Capital Employed"},
                        {"Metric": "ROA", "Value": fin_metrics.get("roa", "N/A"), "Description": "Return on Assets"},
                        {"Metric": "EBITDA Margin", "Value": fin_metrics.get("ebitda_margin", "N/A"), "Description": "Operational profitability"},
                        {"Metric": "Debt-to-Equity", "Value": fin_metrics.get("debt_to_equity", "N/A"), "Description": "Leverage ratio"},
                        {"Metric": "Current Ratio", "Value": fin_metrics.get("current_ratio", "N/A"), "Description": "Liquidity measure"},
                        {"Metric": "Cash Flow to Debt", "Value": fin_metrics.get("cash_flow_to_debt", "N/A"), "Description": "Ability to service debt"}
                    ])
                    st.table(fin_metrics_df)

                    # Valuation Ratios
                    st.subheader("Valuation Ratios")
                    val_ratios = company_details.get("valuation_ratios", {})
                    val_ratios_df = pd.DataFrame([
                        {"Metric": "P/E Ratio", "Value": val_ratios.get("pe_ratio", "N/A"), "Description": "Price-to-Earnings, market valuation of earnings"},
                        {"Metric": "P/B Ratio", "Value": val_ratios.get("pb_ratio", "N/A"), "Description": "Price-to-Book, market vs. book value"},
                        {"Metric": "PEG Ratio", "Value": val_ratios.get("peg_ratio", "N/A"), "Description": "P/E adjusted for growth"},
                        {"Metric": "Dividend Yield", "Value": val_ratios.get("dividend_yield", "N/A"), "Description": "Dividend return on share price"},
                        {"Metric": "EV/EBITDA", "Value": val_ratios.get("ev_ebitda", "N/A"), "Description": "Enterprise value to EBITDA, overall valuation"},
                        {"Metric": "Price-to-Sales", "Value": val_ratios.get("price_to_sales", "N/A"), "Description": "Market valuation of revenue"}
                    ])
                    st.table(val_ratios_df)

                    # Competitive Position
                    st.subheader("Competitive Position")
                    comp_pos = company_details.get("competitive_position", {})
                    comp_pos_df = pd.DataFrame([
                        {"Metric": "Market Share", "Value": comp_pos.get("market_share", "N/A")},
                        {"Metric": "Economic Moat", "Value": comp_pos.get("moat", "N/A")},
                        {"Metric": "Key Competitors", "Value": ", ".join(comp_pos.get("competitors", []))}
                    ])
                    st.table(comp_pos_df)
                    st.markdown("**SWOT Analysis**:")
                    swot = comp_pos.get("swot", {})
                    for key, value in swot.items():
                        st.markdown(f"- **{key.capitalize()}**: {', '.join(value)}")

                    # Management Quality
                    st.subheader("Management Quality")
                    mgmt = company_details.get("management_quality", {})
                    mgmt_df = pd.DataFrame([
                        {"Metric": "CEO", "Value": mgmt.get("ceo", "N/A")},
                        {"Metric": "Tenure", "Value": mgmt.get("tenure", "N/A")},
                        {"Metric": "Track Record", "Value": mgmt.get("track_record", "N/A")},
                        {"Metric": "Compensation", "Value": mgmt.get("compensation", "N/A")},
                        {"Metric": "Insider Ownership", "Value": mgmt.get("insider_ownership", "N/A")},
                        {"Metric": "Governance", "Value": mgmt.get("governance", "N/A")}
                    ])
                    st.table(mgmt_df)

                    # Strategic Outlook
                    st.subheader("Strategic Outlook")
                    strategy = company_details.get("strategic_outlook", {})
                    strategy_df = pd.DataFrame([
                        {"Metric": "Growth Drivers", "Value": ", ".join(strategy.get("growth_drivers", []))},
                        {"Metric": "Key Risks", "Value": ", ".join(strategy.get("risks", []))},
                        {"Metric": "Analyst Consensus", "Value": strategy.get("analyst_consensus", "N/A")},
                        {"Metric": "Innovation Focus", "Value": ", ".join(strategy.get("innovation", []))}
                    ])
                    st.table(strategy_df)

        except Exception as e:
            st.error(f"Error displaying company details: {str(e)}")
            logger.error(f"Company details error: {str(e)}")

if __name__ == "__main__":
    main()
