from src.ml_models.inference import run_predict_model
from core.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from .state import StockAgentState
import yfinance as yf

def fetch_current_price(state: StockAgentState) -> StockAgentState:
    ticker_symbol = state["ticker"]
    if not ticker_symbol.endswith(".NS"):
        ticker_symbol += ".NS"

    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="1d")

    if history.empty:
        raise ValueError(f"No price data found for ticker: {ticker_symbol}")

    state["current_price"] = float(history["Close"].iloc[-1])
    return state

def fetch_prediction(state: StockAgentState) -> StockAgentState:
    pred = run_predict_model(state["ticker"])
    state["predicted_price"] = pred
    return state

def compare_and_decide(state: StockAgentState) -> StockAgentState:
    pred = state.get("predicted_price", 0)
    curr = state.get("current_price", 0)

    if pred > curr:
        state["recommendation"] = "🟢 Consider Buying"
    elif pred < curr:
        state["recommendation"] = "🔴 Consider Selling"
    else:
        state["recommendation"] = "🟡 Consider Holding"

    return state

def generate_final_summary(state: StockAgentState) -> StockAgentState:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)

    prompt = f"""
You are a financial assistant (non-professional or Professional). Based solely on the current price and predicted price below, provide a clear, concise summary and recommendation.

Ticker: {state['ticker']}
Current Price: ₹{state['current_price']}
Predicted Price: ₹{state['predicted_price']}
Recommendation: {state['recommendation']}

Note: This recommendation is based on an ML model trained on 5 years of daily interval data. It is NOT professional financial advice and involves risk. Please consult a financial advisor before making any decisions.
"""

    result = llm.invoke(prompt)
    state["final_summary"] = result.content
    return state
