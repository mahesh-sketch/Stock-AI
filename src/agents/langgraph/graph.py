from langgraph.graph import StateGraph, END
from .state import StockAgentState
from .nodes import (
    fetch_current_price,
    fetch_prediction,
    compare_and_decide,
    generate_final_summary,
)

def build_stock_graph():
    workflow = StateGraph(StockAgentState)

    workflow.add_node("FetchCurrentPrice", fetch_current_price)
    workflow.add_node("FetchPrediction", fetch_prediction)
    workflow.add_node("CompareAndDecide", compare_and_decide)
    workflow.add_node("GenerateSummary", generate_final_summary)

    workflow.set_entry_point("FetchCurrentPrice")
    workflow.add_edge("FetchCurrentPrice", "FetchPrediction")
    workflow.add_edge("FetchPrediction", "CompareAndDecide")
    workflow.add_edge("CompareAndDecide", "GenerateSummary")
    workflow.add_edge("GenerateSummary", END)

    return workflow.compile()
