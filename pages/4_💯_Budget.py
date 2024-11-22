import streamlit as st
import yfinance as yf
import datetime
import os
from langchain.llms import OpenAI
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")

def get_realtime_prices(stocks):
    prices = {}
    for stock_symbol in stocks:
        stock = yf.Ticker(stock_symbol)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        prices[stock_symbol] = current_price
    return prices

def basic_analysis_recommendation(investment_amount, stocks, risk_factor):
    recommendations = {}
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    for stock_symbol in stocks:
        stock_data = yf.download(stock_symbol, start="2020-01-01", end=end_date)
        stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
        avg_daily_return = stock_data['Daily_Return'].mean()
        std_dev_daily_return = stock_data['Daily_Return'].std()
        
        risk_threshold = {"Low": 0.05, "Medium": 0.15, "High": 0.25}[risk_factor]
        
        if avg_daily_return > 0 and std_dev_daily_return < risk_threshold:
            recommendations[stock_symbol] = {"Recommendation": "Buy", "Current_Price": stock_data['Adj Close'].iloc[-1]}
        else:
            recommendations[stock_symbol] = {"Recommendation": "Hold", "Current_Price": stock_data['Adj Close'].iloc[-1]}
    
    return recommendations

def linear_regression_prediction(stock_symbol):
    stock_data = yf.download(stock_symbol, start="2020-01-01", end=datetime.datetime.now().strftime("%Y-%m-%d"))
    
    stock_data.index = pd.to_datetime(stock_data.index)
    
    stock_data['Day'] = (stock_data.index - stock_data.index[0]).days

    X = stock_data['Day'].values.reshape(-1, 1)
    y = stock_data['Close'].values
    model = LinearRegression().fit(X, y)
    
    next_day = np.array([[X[-1][0] + 1]])
    predicted_price = model.predict(next_day)
    
    return predicted_price[0]


def sentiment_analysis(stock_symbol):
    sentiment_score = 0.1
    return "Positive" if sentiment_score > 0 else "Negative"

# Streamlit UI
st.title("Enhanced Stock Recommendation App")

investment_amount = st.number_input("Enter the amount you want to invest:", min_value=1, step=1)
stock_symbols = st.text_input("Enter comma-separated list of stock symbols (e.g., RELIANCE.NS,TCS.NS):")
risk_factor = st.selectbox("Choose the risk factor:", ["Low", "Medium", "High"])
algorithm = st.selectbox("Choose the AI algorithm for analysis:", ["Basic Analysis", "Linear Regression Prediction", "Sentiment Analysis"])

if st.button("Get Recommendations"):
    stocks = [symbol.strip() for symbol in stock_symbols.split(",")]
    recommendations = {}
    
    if algorithm == "Basic Analysis":
        recommendations = basic_analysis_recommendation(investment_amount, stocks, risk_factor)
    
    elif algorithm == "Linear Regression Prediction":
        for stock_symbol in stocks:
            predicted_price = linear_regression_prediction(stock_symbol)
            recommendations[stock_symbol] = {
                "Recommendation": "Predicted Price",
                "Predicted_Price": predicted_price
            }

    st.write("\nREAL TIME DATA\n")
    for stock_symbol, data in recommendations.items():
        st.write(f"{stock_symbol}: {data}")
    
    st.write("\nReal-time Prices:")
    realtime_prices = get_realtime_prices(stocks)
    for stock_symbol, price in realtime_prices.items():
        st.write(f"{stock_symbol}: {price}")

    analysis = llm(
        f"Give a detailed stock analysis using the selected algorithm ({algorithm}). Use available data and provide investment "
        f"recommendations for stocks {recommendations}. The user selected {risk_factor} risk. Write a 5-6 line investment analysis "
        f"that provides actionable insights without specific price mentions."
    )
    
    st.write("\nCONCLUSION\n")
    st.write(analysis)
