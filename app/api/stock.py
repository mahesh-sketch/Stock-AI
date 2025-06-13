from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from src.agents.langgraph.graph import build_stock_graph
from src.visualizations.plot_history import plot_stock_with_prediction

router = APIRouter(prefix="/stock", tags=["Stock Analysis"])

# Main prediction + metadata
@router.get("/{ticker}")
async def analyze_stock(ticker: str):
    try:
        graph = build_stock_graph()
        result = graph.invoke({"ticker": ticker.upper()})

        return {
            "ticker": ticker.upper(),
            "current_price": result.get("current_price", "N/A"),
            "predicted_price": result.get("predicted_price", "N/A"),
            "recommendation": result.get("recommendation", "N/A"),
            "summary": result.get("final_summary", "N/A"),
            "has_chart": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve chart directly from memory
@router.get("/{ticker}/chart", response_class=Response)
async def get_stock_chart(ticker: str):
    try:
        graph = build_stock_graph()
        result = graph.invoke({"ticker": ticker.upper()})
        predicted_price = result.get("predicted_price")

        if not isinstance(predicted_price, (float, int)):
            raise ValueError("Invalid predicted price")

        image_bytes = plot_stock_with_prediction(ticker.upper(), predicted_price)
        return Response(content=image_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
