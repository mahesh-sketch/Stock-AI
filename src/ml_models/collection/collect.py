import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import yfinance as yf
import pandas as pd
from core.logger import logger

def fetch_fifteenmin_stock_data(ticker,period='5y',interval='1d'):
    full_ticker = f"{ticker}.NS"
    logger.info(f"Fetching intraday data for {full_ticker} with period={period} and interval={interval}...")
    try:
        data = yf.download(full_ticker, period=period, interval=interval)
        if not data.empty:
            return data
        else:
            logger.warning(f"No intraday data found for {full_ticker}.")
            return None
    except Exception as e:
        logger.error(f"Error downloading intraday data for {full_ticker}: {e}")

        return None