from typing import TypedDict, Optional

class StockAgentState(TypedDict):
    ticker: str
    current_price: Optional[float]
    predicted_price: Optional[float]
    recommendation: Optional[str]
    final_summary: Optional[str]
