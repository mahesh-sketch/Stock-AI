from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import stock

app = FastAPI(
    title="🧠 Stock AI GenAI API",
    description="Predicts stock price using ML + LangGraph + GenAI summary",
    version="1.0"
)

# CORS for frontend access (e.g., Streamlit or React later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stock.router)
