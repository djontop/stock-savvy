import streamlit as st
from backend.fetch_stock_info import analyze_stock
from fpdf import FPDF
import requests
import base64
import os
from dotenv import load_dotenv
from google import genai

# ---------------------- Streamlit Page Config ---------------------- #
st.set_page_config(
    page_title="StockSavvy",
    page_icon="💲",
)

# ---------------------- Title ---------------------- #
st.title("📈 StockSavvy")

# ---------------------- Background Function ---------------------- #
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# ---------------------- Custom CSS ---------------------- #
st.markdown(
    """
    <style>
    body {
        background: #34495e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: white;
    }

    .stButton>button {
        background-color: #3498db;
        color: white;
        border-color: #3498db;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton>button:hover {
        background-color: #D3D3D3;
    }

    .stRadio>div>div>div>label>div:first-child {
        color: white;
        font-weight: bold;
    }
    .stRadio>div>div>div>label>div:last-child {
        color: #bdc3c7;
    }
    .stRadio>div>div>div>label:hover {
        background-color: #2c3e50;
    }
    .stRadio>div>div>div>label:active {
        background-color: #2980b9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Load Environment Variables ---------------------- #
load_dotenv()
stock_keys = os.getenv("stock_keys")

# Initialize Gemini client for news analysis
try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    gemini_client = None

# ---------------------- Helper Functions ---------------------- #
def get_financial_news(query):
    api_key = os.getenv("NEWS_API")
    modified_query = f"{query} AND (finance OR stock OR company)"
    url = f"https://newsapi.org/v2/everything?q={modified_query}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])

    financial_articles = []
    for article in articles:
        if article.get('description'):
            if 'finance' in article['description'].lower() or 'stock' in article['description'].lower():
                financial_articles.append(article)
    return financial_articles

def analyze_news_sentiment_with_gemini(articles, query):
    """
    Analyze news sentiment using Gemini API
    """
    if not gemini_client or not articles:
        return "No news analysis available."
    
    try:
        # Prepare news content for analysis
        news_content = ""
        for i, article in enumerate(articles[:5]):  # Analyze top 5 articles
            news_content += f"{i+1}. {article.get('title', 'No title')}: {article.get('description', 'No description')}\n"
        
        prompt = f"""Analyze the sentiment and key insights from the following financial news related to "{query}":

        News Articles:
        {news_content}

        Please provide:
        1. Overall sentiment (Positive/Negative/Neutral)
        2. Key themes or trends mentioned
        3. Potential impact on stock/investment decisions
        4. Brief summary of important points

        Keep the analysis concise and focused on investment implications."""
        
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        return response.text
    
    except Exception as e:
        return f"Error analyzing news sentiment: {str(e)}"

def get_realtime_stock_data(symbol, stock_keys):
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_keys}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "Global Quote" in data:
            return data["Global Quote"]
        else:
            return None
    else:
        st.error("Failed to fetch real-time data. Please try again later.")
        return None

# ---------------------- Main App ---------------------- #
name = st.text_input("May I know your name ?")

if name:
    st.session_state["user_name"] = name

    st.markdown(f"### Hello, {name}!")
    st.markdown("Please feel free to submit any questions or inquiries related to investments.")

    query = st.text_input('💬 Input the investment query related to a stock:')
    risk_parameter = st.radio("📊 Select Risk Parameter", ["Low", "Medium", "High"])

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col2:
        enter_button = st.button("🚀 Generate")

    with col1:
        news_button = st.button("📰 News")

    with col3:
        clear_button = st.button("🔄 Clear")

    # ---------------------- Generate Analysis ---------------------- #
    if enter_button:
        if query:
            with st.spinner('⏳ Gathering all required information and analyzing. Please wait...'):
                out = analyze_stock(query, risk_parameter, name)

            st.success('✅ Done!')
            st.write(out)

            # PDF Generation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            cleaned_out = out.replace('\u20b9', 'Rs.')
            pdf.multi_cell(0, 10, txt=cleaned_out)
            pdf_output = pdf.output(dest='S').encode('latin-1')

            with col5:
                st.download_button(
                    'Download Report',
                    pdf_output,
                    file_name="Report.pdf",
                    mime="application/octet-stream"
                )

    # ---------------------- Show News with Gemini Analysis ---------------------- #
    elif news_button:
        if query:
            financial_articles = get_financial_news(query)
            if financial_articles:
                st.subheader("📰 Financial News:")
                
                # Display articles
                for article in financial_articles[:5]:  # Show top 5 articles
                    st.write(f"- [{article['title']}]({article['url']})")
                
                # Add Gemini sentiment analysis
                st.subheader("🤖 AI News Analysis:")
                with st.spinner('Analyzing news sentiment...'):
                    sentiment_analysis = analyze_news_sentiment_with_gemini(financial_articles, query)
                
                st.write(sentiment_analysis)
                
            else:
                st.warning("No financial news articles found for the given query.")
        else:
            st.warning('⚠ Please input your query before clicking News.')

    # ---------------------- Clear Input ---------------------- #
    if clear_button:
        st.experimental_rerun()

else:
    st.write("👋 Please enter your name to continue.")
