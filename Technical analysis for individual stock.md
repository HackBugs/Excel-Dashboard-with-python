## Install These Dependencies

```
pip install pandas
pip install nsepy
pip install yfinance
pip install plotly
pip install dash
pip install dash-bootstrap-components
pip install openpyxl
pip install ta
pip show streamlit
pip install --upgrade streamlit

pip install streamlit yfinance pandas numpy plotly ta
pip install curl_cffi
pip install curl_cffi --only-binary :all:
pip install requests
python --version
```

```
pip install --upgrade streamlit yfinance pandas numpy plotly ta
pip install streamlit==1.38.0
pip cache purge
```

## For running the program

```
python app.py
streamlit run app.py
python -m streamlit run app-5.py
```
## requirements.txt

```
streamlit==1.28.0
yfinance==0.2.31
pandas==2.1.1
numpy==1.26.0
plotly==5.17.0
ta==0.10.2
```

## app.py

```
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Indian Stock Market Dashboard"

# Define sectors and stocks
sectors = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'KOTAKBANK.NS', 'INDUSINDBK.NS', 'BANDHANBNK.NS', 'FEDERALBNK.NS', 'PNB.NS', 'IDFCFIRSTB.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS', 'LTTS.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS'],
    'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'BIOCON.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'ALKEM.NS', 'TORNTPHARM.NS', 'GRANULES.NS'],
    'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BALKRISIND.NS', 'MRF.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'BPCL.NS', 'IOC.NS', 'GAIL.NS', 'ADANIGREEN.NS', 'TATAPOWER.NS', 'HINDPETRO.NS'],
    'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS', 'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS', 'UBL.NS', 'EMAMILTD.NS'],
    'Metal': ['TATASTEEL.NS', 'HINDALCO.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'VEDL.NS', 'JINDALSTEL.NS', 'SAIL.NS', 'NATIONALUM.NS', 'HINDZINC.NS', 'APLAPOLLO.NS'],
    'Realty': ['GODREJPROP.NS', 'DLF.NS', 'PRESTIGE.NS', 'OBEROIRLTY.NS', 'PHOENIXLTD.NS', 'BRIGADE.NS', 'SOBHA.NS', 'SUNTECK.NS', 'MAHLIFE.NS', 'IBREALEST.NS'],
    'Telecom': ['BHARTIARTL.NS', 'IDEA.NS', 'TATACOMM.NS', 'NXTTECHNOLOG.NS', 'ROUTE.NS', 'TANLA.NS', 'LTTS.NS', 'HCLTECH.NS', 'INFY.NS', 'TCS.NS']
}

# Create a flat list of all companies
all_companies = []
for sector_companies in sectors.values():
    all_companies.extend(sector_companies)
all_companies = sorted(list(set(all_companies)))  # Remove duplicates and sort

# Function to fetch stock data
def fetch_stock_data(tickers, start_date, end_date):
    stock_data = {}
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                stock_data[ticker] = data
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return stock_data

# Function to add technical indicators
def add_technical_indicators(df):
    # Add SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Add Bollinger Bands
    df['Bollinger_Mid'] = df['SMA_20']
    df['Bollinger_Std'] = df['Close'].rolling(window=20).std()
    df['Bollinger_High'] = df['Bollinger_Mid'] + (df['Bollinger_Std'] * 2)
    df['Bollinger_Low'] = df['Bollinger_Mid'] - (df['Bollinger_Std'] * 2)
    
    # Add RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Add MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Function to prepare data for dashboard
def prepare_dashboard_data(stock_data):
    dashboard_data = []
    
    for ticker, data in stock_data.items():
        if data.empty:
            continue
        
        # Get latest data
        latest = data.iloc[-1]
        prev_day = data.iloc[-2] if len(data) > 1 else latest
        
        # Calculate change
        change = latest['Close'] - prev_day['Close']
        change_pct = (change / prev_day['Close']) * 100
        
        # Determine sector
        sector = "Index" if ticker in ['^NSEI', '^BSESN'] else next((s for s, companies in sectors.items() if ticker in companies), "Other")
        
        # Get company name
        company_name = ticker.replace('.NS', '')
        if ticker == '^NSEI':
            company_name = 'NIFTY 50'
        elif ticker == '^BSESN':
            company_name = 'SENSEX'
        
        # Add to dashboard data
        dashboard_data.append({
            'Ticker': ticker,
            'Company': company_name,
            'Sector': sector,
            'Date': latest.name.strftime('%Y-%m-%d'),
            'Open': latest['Open'],
            'High': latest['High'],
            'Low': latest['Low'],
            'Close': latest['Close'],
            'Volume': latest['Volume'],
            'Change': change,
            'Change %': change_pct
        })
    
    # Convert to DataFrame
    dashboard_df = pd.DataFrame(dashboard_data)
    
    # Sort by sector and company
    if not dashboard_df.empty:
        dashboard_df = dashboard_df.sort_values(['Sector', 'Company'])
    
    return dashboard_df

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Indian Stock Market Dashboard", style={'textAlign': 'center'}),
        html.P("Real-time analysis of Indian stocks and market indices", style={'textAlign': 'center'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    # Filters and controls
    html.Div([
        dbc.Row([
            # Date range selector
            dbc.Col([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=datetime(2010, 1, 1),
                    max_date_allowed=datetime.now(),
                    start_date=(datetime.now() - timedelta(days=30)).date(),
                    end_date=datetime.now().date()
                )
            ], width=4),
            
            # Sector filter
            dbc.Col([
                html.Label("Sector:"),
                dcc.Dropdown(
                    id='sector-filter',
                    options=[{'label': 'All Sectors', 'value': 'All'}] + 
                            [{'label': sector, 'value': sector} for sector in sectors.keys()],
                    value='All',
                    clearable=False
                )
            ], width=4),
            
            # Company filter
            dbc.Col([
                html.Label("Companies:"),
                dcc.Dropdown(
                    id='company-filter',
                    options=[{'label': company.replace('.NS', ''), 'value': company} for company in all_companies],
                    value=[],
                    multi=True
                )
            ], width=4)
        ], className="mb-4"),
        
        dbc.Row([
            # Fetch data button
            dbc.Col([
                dbc.Button("Fetch Data", id="fetch-data-button", color="primary", className="mr-2"),
                dbc.Button("Export to Excel", id="export-button", color="success", className="ml-2")
            ], width=6),
            
            # Technical indicators selector
            dbc.Col([
                html.Label("Technical Indicators:"),
                dcc.Checklist(
                    id='indicator-selector',
                    options=[
                        {'label': ' Moving Averages', 'value': 'ma'},
                        {'label': ' Bollinger Bands', 'value': 'bollinger'},
                        {'label': ' RSI', 'value': 'rsi'},
                        {'label': ' MACD', 'value': 'macd'}
                    ],
                    value=['ma'],
                    inline=True
                )
            ], width=6)
        ], className="mb-4")
    ], style={'padding': '20px', 'backgroundColor': '#e9ecef'}),
    
    # Market overview cards
    dbc.Row([
        dbc.Col([
            dbc.Card(id='nifty-card', body=True, color="light")
        ], width=6),
        dbc.Col([
            dbc.Card(id='sensex-card', body=True, color="light")
        ], width=6)
    ], className="mb-4", style={'padding': '10px'}),
    
    # Stock data table
    html.Div([
        html.H3("Stock Data"),
        dash_table.DataTable(
            id='stock-table',
            columns=[
                {'name': 'Ticker', 'id': 'Ticker'},
                {'name': 'Company', 'id': 'Company'},
                {'name': 'Sector', 'id': 'Sector'},
                {'name': 'Date', 'id': 'Date'},
                {'name': 'Open (₹)', 'id': 'Open', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                {'name': 'High (₹)', 'id': 'High', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                {'name': 'Low (₹)', 'id': 'Low', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                {'name': 'Close (₹)', 'id': 'Close', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                {'name': 'Volume', 'id': 'Volume', 'type': 'numeric', 'format': {'specifier': ','}},
                {'name': 'Change (₹)', 'id': 'Change', 'type': 'numeric', 'format': {'specifier': '+,.2f'}},
                {'name': 'Change (%)', 'id': 'Change %', 'type': 'numeric', 'format': {'specifier': '+,.2f'}}
            ],
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Change} > 0'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{Change} < 0'
                    },
                    'color': 'red'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'center',
                'font-family': 'Arial',
                'padding': '10px'
            },
            sort_action='native',
            filter_action='native',
            page_size=15
        )
    ], style={'padding': '20px'}),
    
    # Charts
    html.Div([
        dbc.Row([
            # Price chart
            dbc.Col([
                html.H3("Price Comparison"),
                dcc.Graph(id='price-chart')
            ], width=6),
            
            # Sector performance
            dbc.Col([
                html.H3("Sector Performance"),
                dcc.Graph(id='sector-performance')
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            # Technical analysis chart
            dbc.Col([
                html.H3("Technical Analysis"),
                dcc.Graph(id='technical-chart')
            ], width=12)
        ], className="mb-4"),
        
        dbc.Row([
            # Volume analysis
            dbc.Col([
                html.H3("Volume Analysis"),
                dcc.Graph(id='volume-chart')
            ], width=12)
        ])
    ], style={'padding': '20px'}),
    
    # Store component for data
    dcc.Store(id='stored-data'),
    
    # Download component
    dcc.Download(id="download-dataframe-xlsx"),
    
    # Footer
    html.Footer([
        html.P("Indian Stock Market Dashboard © 2023", style={'textAlign': 'center'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'marginTop': '20px'})
])

# Callback to update company dropdown based on sector selection
@app.callback(
    Output('company-filter', 'options'),
    Input('sector-filter', 'value')
)
def update_company_options(selected_sector):
    if selected_sector == 'All':
        companies = all_companies
    else:
        companies = sectors.get(selected_sector, [])
    
    return [{'label': company.replace('.NS', ''), 'value': company} for company in companies]

# Callback to fetch data and update dashboard
# Callback to fetch data and update dashboard
@app.callback(
    [Output('stored-data', 'data'),
     Output('nifty-card', 'children'),
     Output('sensex-card', 'children')],
    Input('fetch-data-button', 'n_clicks'),
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('sector-filter', 'value'),
     State('company-filter', 'value')],
    prevent_initial_call=True
)
def fetch_and_process_data(n_clicks, start_date, end_date, selected_sector, selected_companies):
    if n_clicks is None:
        return None, [], []
    
    # Convert string dates to datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Add indices
    tickers = ['^NSEI', '^BSESN']  # NIFTY 50 and SENSEX
    
    # Add selected companies
    if selected_companies:
        tickers.extend(selected_companies)
    else:
        # If no specific companies selected, add companies from selected sector
        if selected_sector == 'All':
            for sector_companies in sectors.values():
                tickers.extend(sector_companies)
        else:
            tickers.extend(sectors.get(selected_sector, []))
    
    # Remove duplicates
    tickers = list(set(tickers))
    
    # Fetch data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    
    # Prepare dashboard data
    dashboard_df = prepare_dashboard_data(stock_data)
    
    # Create index cards
    nifty_card = []
    sensex_card = []
    
    if '^NSEI' in stock_data and not stock_data['^NSEI'].empty:
        nifty_data = stock_data['^NSEI']
        latest = nifty_data.iloc[-1]
        prev_day = nifty_data.iloc[-2] if len(nifty_data) > 1 else latest
        
        change = latest['Close'] - prev_day['Close']
        change_pct = (change / prev_day['Close']) * 100
        
        nifty_card = [
            html.H4("NIFTY 50", className="card-title"),
            html.H2(f"{latest['Close']:.2f}", className="card-subtitle"),
            html.P([
                f"Change: {change:+.2f} ({change_pct:+.2f}%)",
                html.Span(" ▲" if change >= 0 else " ▼", 
                          style={'color': 'green' if change >= 0 else 'red'})
            ], className="card-text"),
            html.P(f"Range: {latest['Low']:.2f} - {latest['High']:.2f}", className="card-text"),
            html.P(f"Date: {latest.name.strftime('%Y-%m-%d')}", className="text-muted")
        ]
    
    if '^BSESN' in stock_data and not stock_data['^BSESN'].empty:
        sensex_data = stock_data['^BSESN']
        latest = sensex_data.iloc[-1]
        prev_day = sensex_data.iloc[-2] if len(sensex_data) > 1 else latest
        
        change = latest['Close'] - prev_day['Close']
        change_pct = (change / prev_day['Close']) * 100
        
        sensex_card = [
            html.H4("SENSEX", className="card-title"),
            html.H2(f"{latest['Close']:.2f}", className="card-subtitle"),
            html.P([
                f"Change: {change:+.2f} ({change_pct:+.2f}%)",
                html.Span(" ▲" if change >= 0 else " ▼", 
                          style={'color': 'green' if change >= 0 else 'red'})
            ], className="card-text"),
            html.P(f"Range: {latest['Low']:.2f} - {latest['High']:.2f}", className="card-text"),
            html.P(f"Date: {latest.name.strftime('%Y-%m-%d')}", className="text-muted")
        ]
    
    # Convert DataFrame to JSON for storage
    json_data = dashboard_df.to_json(date_format='iso', orient='split') if not dashboard_df.empty else None
    
    return json_data, nifty_card, sensex_card

# Callback to update stock table
@app.callback(
    Output('stock-table', 'data'),
    Input('stored-data', 'data')
)
def update_stock_table(stored_data):
    if stored_data is None:
        return []
    
    df = pd.read_json(stored_data, orient='split')
    return df.to_dict('records')

# Callback to update price comparison chart
@app.callback(
    Output('price-chart', 'figure'),
    [Input('stored-data', 'data'),
     Input('company-filter', 'value')]
)
def update_price_chart(stored_data, selected_companies):
    if stored_data is None:
        return go.Figure()
    
    df = pd.read_json(stored_data, orient='split')
    
    # Get unique tickers
    tickers = df['Ticker'].unique()
    
    # Filter for selected companies if any
    if selected_companies:
        tickers = [t for t in tickers if t in selected_companies]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each ticker
    for ticker in tickers:
        if ticker in ['^NSEI', '^BSESN']:
            continue  # Skip indices
            
        ticker_data = df[df['Ticker'] == ticker]
        
        if not ticker_data.empty:
            company_name = ticker_data['Company'].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=[ticker_data['Date'].iloc[0]],
                y=[ticker_data['Close'].iloc[0]],
                mode='markers+text',
                name=company_name,
                text=[company_name],
                textposition='top center'
            ))
    
    # Update layout
    fig.update_layout(
        title="Stock Price Comparison",
        xaxis_title="Date",
        yaxis_title="Closing Price (₹)",
        height=500,
        showlegend=True,
        hovermode="closest"
    )
    
    return fig

# Callback to update technical analysis chart
@app.callback(
    Output('technical-chart', 'figure'),
    [Input('stored-data', 'data'),
     Input('company-filter', 'value'),
     Input('indicator-selector', 'value')]
)
def update_technical_chart(stored_data, selected_companies, selected_indicators):
    if stored_data is None or not selected_companies:
        return go.Figure()
    
    # Get the first selected company
    ticker = selected_companies[0] if selected_companies else None
    
    if ticker is None:
        return go.Figure()
    
    # Fetch historical data for the selected ticker
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return go.Figure()
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Create figure
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=ticker.replace('.NS', '')
        ))
        
        # Add selected indicators
        if 'ma' in selected_indicators:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='blue', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='orange', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='red', width=1)
            ))
        
        if 'bollinger' in selected_indicators:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Bollinger_High'],
                mode='lines',
                name='Bollinger High',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Bollinger_Low'],
                mode='lines',
                name='Bollinger Low',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
                fill='tonexty'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Technical Analysis - {ticker.replace('.NS', '')}",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating technical chart: {e}")
        return go.Figure()

# Callback to update volume chart
@app.callback(
    Output('volume-chart', 'figure'),
    [Input('stored-data', 'data'),
     Input('company-filter', 'value')]
)
def update_volume_chart(stored_data, selected_companies):
    if stored_data is None or not selected_companies:
        return go.Figure()
    
    # Get the first selected company
    ticker = selected_companies[0] if selected_companies else None
    
    if ticker is None:
        return go.Figure()
    
    # Fetch historical data for the selected ticker
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 months of data
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return go.Figure()
        
        # Create figure
        fig = go.Figure()
        
        # Add volume bars
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                  for i in range(len(data))]
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color=colors
        ))
        
        # Add volume moving average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Volume_MA'],
            mode='lines',
            name='Volume MA (20)',
            line=dict(color='black', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Volume Analysis - {ticker.replace('.NS', '')}",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=400,
            showlegend=True
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating volume chart: {e}")
        return go.Figure()

# Callback to update sector performance chart
@app.callback(
    Output('sector-performance', 'figure'),
    Input('stored-data', 'data')
)
def update_sector_performance(stored_data):
    if stored_data is None:
        return go.Figure()
    
    df = pd.read_json(stored_data, orient='split')
    
    # Calculate average change % by sector
    sector_perf = df.groupby('Sector')['Change %'].mean().reset_index()
    
    # Sort by performance
    sector_perf = sector_perf.sort_values('Change %')
    
    # Create color scale based on performance
    colors = ['red' if x < 0 else 'green' for x in sector_perf['Change %']]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sector_perf['Sector'],
        y=sector_perf['Change %'],
        marker_color=colors,
        text=sector_perf['Change %'].apply(lambda x: f"{x:+.2f}%"),
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title="Sector Performance (Average % Change)",
        xaxis_title="Sector",
        yaxis_title="Average Change (%)",
        height=400,
        hovermode="closest"
    )
    
    return fig

# Callback for Excel export
# Callback for Excel export
@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State('stored-data', 'data'),
    prevent_initial_call=True,
)
def export_to_excel(n_clicks, stored_data):
    if stored_data is None:
        return None
    
    df = pd.read_json(stored_data, orient='split')
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return dcc.send_data_frame(
        df.to_excel, 
        f"indian_stock_market_data_{timestamp}.xlsx",
        sheet_name="Stock Data",
        index=False
    )


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
```
