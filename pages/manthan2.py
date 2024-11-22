import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def analyze_multiple_stocks(tickers, start_date, end_date):
    fig = go.Figure() 
    percent_changes = {}

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        opening_price = data['Open'].iloc[0]
        closing_price = data['Close'].iloc[-1]
        percent_change = (closing_price - opening_price) / opening_price * 100
        percent_changes[ticker] = percent_change
        
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=f'{ticker} Close'))
        
        data['50ma'] = data['Close'].rolling(window=50).mean()
        data['200ma'] = data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['50ma'], mode='lines', name=f'{ticker} 50-Day MA', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=data['200ma'], mode='lines', name=f'{ticker} 200-Day MA', line=dict(dash='dash')))
        
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name=f'{ticker} RSI', line=dict(dash='dot')))
        
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name=f'{ticker} Volume', yaxis='y2'))

    fig.update_layout(
        title="Stock Prices and Technical Indicators",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Stock Tickers",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right"
        )
    )

    return fig, percent_changes, data

st.title("Combined Stock Performance Analyzer")

ticker_input = st.text_input("Enter stock ticker symbols separated by commas (e.g., TCS.NS, INFY.NS):", "AAPL, MSFT")
tickers = [ticker.strip() for ticker in ticker_input.split(",")]

start_date = st.date_input("Select the start date:", value=pd.Timestamp("2020-01-01"))
end_date = st.date_input("Select the end date:", value=pd.Timestamp.today())

if st.button("Analyze Performance"):
    fig, percent_changes, data = analyze_multiple_stocks(tickers, start_date, end_date)

    for ticker, percent_change in percent_changes.items():
        st.write(f"The {ticker} stock price changed by {percent_change:.2f}% between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}.")
    
    st.plotly_chart(fig)

    performance_data = pd.DataFrame({
        "Ticker": tickers,
        "Performance (%)": [percent_changes[ticker] for ticker in tickers]
    })
    st.write("Performance Summary:")
    st.write(performance_data)
    csv_data = data.to_csv()
    st.download_button(
        label="Download Raw Data as CSV",
        data=csv_data,
        file_name=f"stocks_analysis_{','.join(tickers)}.csv",
        mime="text/csv"
    )
