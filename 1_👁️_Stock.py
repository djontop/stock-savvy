import streamlit as st
import google.generativeai as genai # Import Gemini
from fpdf import FPDF
import requests
import base64
import os
from dotenv import load_dotenv

# Load all environment variables from .env file at the start
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="StockSavvy",
    page_icon="💲",
)

# --- NEW: Gemini-Powered Analysis Function ---
# This function replaces the old backend call
def Analyze_stock(query, risk_parameter, name):
    """
    Analyzes a stock query using the Google Gemini API.
    """
    # Configure the Gemini API with your key
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        return f"Error configuring Gemini API. Make sure your GEMINI_API_KEY is set in the .env file. Details: {e}"

    # Create a detailed prompt for the AI model
    prompt = f"""
    As an expert financial analyst, analyze the investment query: "{query}" for a user named {name} who has a '{risk_parameter}' risk tolerance.

    Your analysis must be comprehensive and structured. Please include the following sections:
    1.  **Executive Summary:** A brief overview of the stock and the investment thesis.
    2.  **Company Overview:** What the company does, its market position, and key business segments.
    3.  **Financial Health:** A summary of its recent financial performance (revenue, profit, debt).
    4.  **Risk Analysis:** Based on the user's '{risk_parameter}' risk parameter, outline the potential risks.
    5.  **Growth Potential:** Discuss potential catalysts for growth and future prospects.
    6.  **Recommendation:** Conclude with a clear recommendation (e.g., Buy, Hold, Sell, or Further Research Needed), justifying your reasoning based on the analysis and the user's risk profile.

    Provide a professional, data-driven, and easy-to-understand report.
    """

    try:
        # Initialize the Gemini model (gemini-1.5-flash is fast and effective)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while analyzing the stock with Gemini: {e}"


# --- Helper Functions (No changes needed here) ---
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

def get_financial_news(query):
    api_key = os.getenv("NEWS_API")
    if not api_key:
        st.error("NEWS_API key not found. Please set it in your .env file.")
        return []
    modified_query = f"{query} AND (finance OR stock OR company)"
    url = f"https://newsapi.org/v2/everything?q={modified_query}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    return [article for article in articles if article.get('description')]

# --- Custom CSS Styling ---
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
        background-color: #2980b9;
        border-color: #2980b9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Application Logic ---
st.title("📈 StockSavvy")

# Initialize session state for the query text input to enable clearing
if 'query' not in st.session_state:
    st.session_state.query = ""

def clear_query():
    st.session_state.query = ""

name = st.text_input("May I know your name?")

if name:
    st.session_state["user_name"] = name
    
    st.markdown(f"### Hello, {name}!")
    st.markdown("Enter a stock symbol or company name to get a detailed analysis.")
    
    # Use the session state key for the text input
    query = st.text_input('💬 Input your investment query:', key="query")
    risk_parameter = st.radio("📊 Select Your Risk Tolerance", ["Low", "Medium", "High"])

    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 5])

    with col1:
        generate_button = st.button("🚀 Generate Analysis")

    with col2:
        news_button = st.button("📰 Fetch News")

    with col3:
        # The 'on_click' callback is the correct way to clear a widget
        st.button("🔄 Clear", on_click=clear_query)

    if generate_button:
        if query:
            with st.spinner('⏳ Analyzing with Gemini... Please wait.'):
                # Call the NEW Gemini-powered function directly
                out = Analyze_stock(query, risk_parameter, name)
            
            st.success('✅ Analysis Complete!')
            st.write(out)
            
            # --- PDF Generation Logic ---
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            # Encode to latin-1 and replace unknown characters to prevent FPDF errors
            cleaned_out = out.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 10, txt=cleaned_out)
            pdf_output = pdf.output(dest='S').encode('latin-1')
            
            with col4:
                 st.download_button(
                    label="📥 Download Report",
                    data=pdf_output,
                    file_name=f"{query.replace(' ', '_')}_Analysis.pdf",
                    mime="application/octet-stream"
                )
        else:
            st.warning('⚠️ Please input a query to generate an analysis.')

    if news_button:
        if query:
            with st.spinner('📰 Fetching the latest news...'):
                financial_articles = get_financial_news(query)
            if financial_articles:
                st.subheader(f"Financial News for: {query}")
                for article in financial_articles[:5]: # Show top 5 articles
                    st.write(f"**[{article['title']}]({article['url']})**")
                    st.caption(f"Source: {article['source']['name']}")
                    st.markdown("---")
            else:
                st.warning("No relevant financial news articles were found.")
        else:
            st.warning('⚠️ Please input a query to fetch news.')

else:
    st.info("👋 Please enter your name to begin.")
