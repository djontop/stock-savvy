import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import base64
import plotly.io as pio

# --- Helper Functions ---

@st.cache_data
def get_stock_data(symbol, start_date, end_date):
    """
    Downloads stock data from Yahoo Finance, handles multi-index columns,
    and validates required columns.
    """
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if stock_data.empty:
            st.warning(f"No data found for {symbol} over the selected date range.")
            return None
        
        # Flatten columns if yfinance returns a multi-index header
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in stock_data.columns for col in required_columns):
            st.error(f"Data from Yahoo Finance is missing required columns for {symbol}")
            return None
            
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data for {symbol}: {e}")
        return None

def plot_candlestick_chart(stock_data, symbol):
    """
    Creates an interactive candlestick chart for a given stock.
    """
    if stock_data is None or len(stock_data) < 2:
        st.warning("Not enough data to plot a candlestick chart.")
        return None
        
    try:
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name=symbol
        )])
        
        fig.update_layout(
            title=f'Candlestick Chart - {symbol}',
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            xaxis_rangeslider_visible=True
        )
        return fig
    except Exception as e:
        st.error(f"Error creating candlestick chart: {e}")
        return None

def get_chart_download_link(figure, filename, linktext="Download HTML"):
    """
    Generates an HTML download link for a Plotly figure.
    """
    if figure is None: return ""
    
    try:
        fig_data = pio.to_html(figure, full_html=False, include_plotlyjs='cdn')
        b64 = base64.b64encode(fig_data.encode()).decode()
        return f'<a href="data:text/html;base64,{b64}" download="{filename}">{linktext}</a>'
    except Exception as e:
        st.error(f"Error generating download link: {e}")
        return ""

# --- Main Streamlit App ---

def main():
    """
    The main function that runs the Streamlit application.
    """
    st.title('📈 Stock Price Analysis 📈')

    # --- User Inputs ---
    symbols_input = st.text_input('Enter Stock Symbols (e.g., TCS.NS, YESBANK.NS):')
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()] if symbols_input else []
    
    start_date = st.date_input('Start Date:')
    end_date = st.date_input('End Date:')

    if start_date >= end_date:
        st.warning("The start date must be before the end date.")
        return

    if st.button('Get Data'):
        if not symbols:
            st.warning("Please enter at least one stock symbol.")
            return
            
        with st.spinner('⏳ Analyzing...'):
            fig_comparison = go.Figure()
            valid_data_found = False
            
            # --- Individual Stock Analysis Loop ---
            for symbol in symbols:
                st.markdown("---")
                st.header(f"Analysis for {symbol}")
                
                stock_data = get_stock_data(symbol, start_date, end_date)
                
                if stock_data is not None and not stock_data.empty:
                    valid_data_found = True
                    
                    st.write(f"### Stock Data for {symbol}")
                    st.dataframe(stock_data)
                    st.write(f"Total records found: {len(stock_data)}")

                    st.subheader("Candlestick Chart")
                    fig_candlestick = plot_candlestick_chart(stock_data, symbol)
                    if fig_candlestick:
                        st.plotly_chart(fig_candlestick, use_container_width=True)
                        st.markdown(get_chart_download_link(fig_candlestick, f"{symbol}_Candlestick_Chart.html"), unsafe_allow_html=True)
                    
                    # Add data to the main comparison chart
                    fig_comparison.add_trace(go.Scatter(
                        x=stock_data.index, 
                        y=stock_data['Close'], 
                        mode='lines', 
                        name=symbol
                    ))
                else:
                    st.error(f"Could not fetch data for {symbol}.")

            # --- Final Comparison Chart ---
            if valid_data_found:
                st.markdown("---")
                st.header("Stock Price Comparison (Log Scale)")
                # The yaxis_type="log" modification solves the scaling problem
                fig_comparison.update_layout(
                    title='Stock Price Comparison',
                    xaxis_title='Date',
                    yaxis_title='Price (INR) - Log Scale',
                    yaxis_type="log",
                    showlegend=True
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                st.markdown(get_chart_download_link(fig_comparison, "Comparison_Log_Scale_Chart.html"), unsafe_allow_html=True)
            elif not valid_data_found:
                st.error("No valid data was found for any of the provided symbols.")

if __name__ == "__main__":
    main()