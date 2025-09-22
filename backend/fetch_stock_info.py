import json
import time
from bs4 import BeautifulSoup
import re
import requests
from dotenv import load_dotenv
from google import genai
import yfinance as yf
import warnings
import os
warnings.filterwarnings("ignore")

load_dotenv("C:/Users/Jai Jalaram/OneDrive/Desktop/stock-savvy/.env")

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_stock_price(ticker, history=5):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    ticker = ticker + ".NS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    df = df[["Close", "Volume"]]
    df.index = [str(x).split()[0] for x in list(df.index)]
    df.index.rename("Date", inplace=True)
    df = df[-history:]
    return df.to_string()

def google_query(search_term):
    if "news" not in search_term:
        search_term = search_term + " stock news"
    url = f"https://www.google.com/search?q={search_term}&cr=countryIN"
    url = re.sub(r"\s", "+", url)
    return url

def get_recent_stock_news(company_name):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}
    
    g_query = google_query(company_name)
    res = requests.get(g_query, headers=headers).text
    soup = BeautifulSoup(res, "html.parser")
    news = []
    
    for n in soup.find_all("div", "n0jPhd ynAwRc tNxQIb nDgy9d"):
        news.append(n.text)
    for n in soup.find_all("div", "IJl0Z"):
        news.append(n.text)

    if len(news) > 6:
        news = news[:4]
    else:
        news = news
    
    news_string = ""
    for i, n in enumerate(news):
        news_string += f"{i}. {n}\n"
    top5_news = "Recent News:\n\n" + news_string
    
    return top5_news

def get_financial_statements(ticker):
    if "." in ticker:
        ticker = ticker.split(".")[0]
    else:
        ticker = ticker
    ticker = ticker + ".NS"    
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1] >= 3:
        balance_sheet = balance_sheet.iloc[:, :3]    
    balance_sheet = balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

def get_stock_ticker_with_gemini(query):
    """
    Extract company name and ticker using Gemini API
    """
    try:
        prompt = f"""Given the user request: "{query}"
        
        Extract the company name and stock ticker symbol from this request. 
        
        Respond in the following JSON format only:
        {{
            "company_name": "exact company name",
            "ticker_symbol": "stock ticker symbol (without .NS suffix)"
        }}
        
        If you cannot identify a specific company, use your best judgment based on the query context."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        # Try to parse JSON from response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        
        # Parse JSON
        data = json.loads(response_text)
        company_name = data.get("company_name", "Unknown Company")
        ticker_symbol = data.get("ticker_symbol", "UNKNOWN")
        
        return company_name, ticker_symbol
        
    except json.JSONDecodeError:
        # Fallback: try to extract using simple text processing
        response_text = response.text
        lines = response_text.split('\n')
        company_name = "Unknown Company"
        ticker_symbol = "UNKNOWN"
        
        for line in lines:
            if 'company' in line.lower() or 'name' in line.lower():
                if ':' in line:
                    company_name = line.split(':')[1].strip().strip('"')
            elif 'ticker' in line.lower() or 'symbol' in line.lower():
                if ':' in line:
                    ticker_symbol = line.split(':')[1].strip().strip('"')
        
        return company_name, ticker_symbol
    
    except Exception as e:
        print(f"Error in get_stock_ticker_with_gemini: {e}")
        return "Unknown Company", "UNKNOWN"

def analyze_stock_with_gemini(available_information, query, risk, name):
    """
    Analyze stock using Gemini API
    """
    try:
        prompt = f"""Give detailed stock analysis using the available data and provide investment recommendation. 
        At the start, give a clear conclusion to the user about the stock.

        User's Name: {name}
        User's Risk Tolerance: {risk}
        User Question: {query}

        The user is fully aware about investment risks, so don't include generic warnings like 
        'It is recommended to conduct further research and analysis or consult with a financial advisor 
        before making an investment decision' in the answer.

        Write a (5-8) point investment analysis to answer the user query. At the start, give a 
        clear recommendation to the user about the stock.

        Available Information:
        {available_information}

        Please provide a comprehensive analysis with specific insights based on the data provided."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        return response.text
    
    except Exception as e:
        return f"Error generating analysis: {str(e)}. Please try again."

def analyze_stock(query, risk, name):
    try:
        # Get company name and ticker using Gemini
        company_name, ticker = get_stock_ticker_with_gemini(query)
        print({"Query": query, "Company_name": company_name, "Ticker": ticker})
        
        # Get stock data
        stock_data = get_stock_price(ticker, history=10)
        stock_financials = get_financial_statements(ticker)
        stock_news = get_recent_stock_news(company_name)

        available_information = f"Stock Price: {stock_data}\n\nStock Financials: {stock_financials}\n\nStock News: {stock_news}"

        # Generate analysis using Gemini
        analysis = analyze_stock_with_gemini(available_information, query, risk, name)
        
        return analysis
    
    except Exception as e:
        return f"Error in stock analysis: {str(e)}. Please check your input and try again."
