import streamlit as st
import requests
import openai
from crewai import Agent, Crew, Task, Process
__import__('pysqlite3')
import sys,os
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from chromadb import PersistentClient
# OpenAI API Key Setup
openai.api_key = st.secrets["openai"]["api_key"]

# Initialize ChromaDB Client for RAG
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = PersistentClient()

# Alpha Vantage Function to Fetch Daily Stock Data
def fetch_daily_stock_data(ticker):
    """
    Fetch daily stock data for a given ticker symbol using Alpha Vantage API.
    """
    api_key = st.secrets["alpha_vantage"]["api_key"]
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": api_key
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "Time Series (Daily)" in data:
            return data["Time Series (Daily)"]
        elif "Error Message" in data:
            return {"error": data["Error Message"]}
        else:
            return {"error": "Unexpected API response format."}
    except Exception as e:
        return {"error": f"Error fetching stock data for {ticker}: {e}"}

# Alpha Vantage Function to Fetch News
def fetch_market_news():
    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": st.secrets["alpha_vantage"]["api_key"],
            "limit": 50,
            "sort": "RELEVANCE",
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json().get("feed", [])
    except Exception as e:
        st.error(f"Error fetching market news: {e}")
        return []

# Alpha Vantage Function to Fetch Gainers and Losers
def fetch_gainers_losers():
    try:
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": st.secrets["alpha_vantage"]["api_key"],
        }
        response = requests.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching gainers and losers: {e}")
        return {}

# Bespoke Labs Function for Accuracy Scoring
def evaluate_newsletter_accuracy(newsletter_content):
    """
    Evaluate the accuracy of the newsletter using Bespoke Labs API.
    """
    try:
        api_key = st.secrets["bespoke_labs"]["api_key"]
        url = "https://api.bespokelabs.com/v1/evaluate"
        payload = {
            "text": newsletter_content,
            "metrics": ["accuracy_score"],
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data.get("accuracy_score", "N/A")
    except Exception as e:
        st.error(f"Error evaluating newsletter accuracy: {e}")
        return None

# Custom RAG Helper Class
class RAGHelper:
    def __init__(self, client):
        self.client = client

    def add_to_rag(self, collection_name, documents, metadata):
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            collection.add(
                documents=documents,
                metadatas=metadata,
                ids=[str(i) for i in range(len(documents))]
            )
            st.success(f"Data successfully added to the '{collection_name}' collection.")
        except Exception as e:
            st.error(f"Error adding to RAG: {e}")

    def query_from_rag(self, collection_name, query, n_results=5):
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(query_texts=[query], n_results=n_results)
            documents = [doc for sublist in results["documents"] for doc in sublist]  # Flatten nested lists
            return documents
        except Exception as e:
            st.error(f"Error querying RAG: {e}")
            return []

# Market Newsletter Crew
class MarketNewsletterCrew:
    def __init__(self, rag_helper):
        self.rag_helper = rag_helper

    def generate_newsletter(self, company_insights, market_trends):
        """Generate a structured newsletter."""
        try:
            newsletter = f"""
            *Daily Market Newsletter*

            *1. Company Insights*
            {" ".join(company_insights) if company_insights else "No company news available."}

            *2. Market Trends*
            {" ".join(market_trends) if market_trends else "No market trends available."}

            *3. Risk Analysis*
            - Global market conditions remain volatile, with a focus on tech stock trends.
            - Companies like Apple and Tesla are seeing fluctuations based on recent announcements.
            - Inflation rates and monetary policies are critical influencing factors.

            *Highlights:*
            - Top gainers today include technology and finance sectors.
            - Ensure to stay ahead of shifts in macroeconomic factors.

            Stay tuned for tomorrow’s updates!
            """
            return newsletter
        except Exception as e:
            st.error(f"Error generating newsletter: {e}")
            return "Newsletter generation failed due to an error."

# Streamlit Interface
st.title("Market Data Newsletter with Daily Stock Data and Accuracy Scoring")

# Initialize RAG Helper and Crew
rag_helper = RAGHelper(client=st.session_state.chroma_client)
crew_instance = MarketNewsletterCrew(rag_helper)

# Buttons to Fetch News and Trends
if st.button("Fetch and Add News to RAG"):
    news_data = fetch_market_news()
    if news_data:
        documents = [article.get("summary", "No summary available") for article in news_data]
        metadata = [{"title": article.get("title", "")} for article in news_data]
        rag_helper.add_to_rag("news_collection", documents, metadata)

if st.button("Fetch and Add Trends to RAG"):
    gainers_losers = fetch_gainers_losers()
    if gainers_losers:
        documents = [f"{g['ticker']} - ${g['price']} ({g['change_percentage']}%)" for g in gainers_losers.get("top_gainers", [])]
        metadata = [{"ticker": g["ticker"], "price": g["price"], "change": g["change_percentage"]} for g in gainers_losers.get("top_gainers", [])]
        rag_helper.add_to_rag("trends_collection", documents, metadata)

if st.button("Generate Newsletter with Accuracy Score"):
    try:
        company_insights = rag_helper.query_from_rag("news_collection", "latest company news")
        market_trends = rag_helper.query_from_rag("trends_collection", "latest market trends")
        newsletter = crew_instance.generate_newsletter(company_insights, market_trends)
        st.markdown(newsletter)
        
        # Evaluate Accuracy
        accuracy_score = evaluate_newsletter_accuracy(newsletter)
        if accuracy_score is not None:
            st.info(f"Newsletter Accuracy Score: {accuracy_score:.2f}")
            if accuracy_score > 90:
                st.success("The newsletter is highly accurate and reliable!")
            elif accuracy_score > 70:
                st.warning("The newsletter is moderately accurate. Verify critical details.")
            else:
                st.error("The newsletter has a low accuracy score. Consider reviewing the content.")
        else:
            st.warning("Could not retrieve the accuracy score.")
    except Exception as e:
        st.error(f"Error generating newsletter: {e}")
