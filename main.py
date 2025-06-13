import os
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add this line to include the parent directory

""" 
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.ml_models.inference import run_predict_model
from core.logger import logger
from core.settings import settings

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=settings.GOOGLE_API_KEY)

tools = [
    Tool(
        name="StockPricePredictor",
        func=lambda ticker: f"The predicted price for {ticker} is ₹{run_predict_model(ticker):.2f}",
        description="Use to predict stock price of companies like 'RELIANCE', 'TCS'"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
)

if __name__ == "__main__":
    while True:
        query = input("🧑 You: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = agent.run(query)
        print("🤖 Gemini:", response)


 """

from src.agents.langgraph.graph import build_stock_graph
from core.logger import logger
from src.visualizations.plot_history import plot_stock_with_prediction

def run_inference(ticker: str):
    graph = build_stock_graph()
    result = graph.invoke({"ticker": ticker})

    current_price = result.get("current_price", "N/A")
    predicted_price = result.get("predicted_price", "N/A")

    print("🧠 Stock AI - Final Output")
    print("💰 Current Price: ₹", current_price)
    print("🔮 Predicted Price:", predicted_price)
    print("🎯 Recommendation:", result.get("recommendation", "N/A"))
    print("📄 Summary:\n", result.get("final_summary", "N/A"))
    print("────────────────────────────────────────")

    # Plot the stock chart with prediction overlay
    if isinstance(predicted_price, (float, int)):
        plot_stock_with_prediction(ticker, predicted_price)


if __name__ == "__main__":
    while True:
        user_input = input("Enter stock ticker (or 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        run_inference(user_input.upper())