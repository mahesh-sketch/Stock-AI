import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pandas as pd
import ta
from core.logger import logger


# Function to flatten multi-index columns in a DataFrame
def fix_multilevel_columns(df):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
    except Exception as e:
        logger.error(f"Error fixing multi-level columns: {e}")
    return df

# Function to clean stock data
def clean_stock_data(df, ticker='UNKNOWN'):
    try:
        logger.info("🧹 Cleaning stock data...")

        # Fix multilevel columns if needed
        df = fix_multilevel_columns(df)

        # Reset index and rename 'index' to 'Date' if necessary
        df.reset_index(inplace=True)
        if 'Date' not in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)

        # Drop NaNs
        df.dropna(inplace=True)

        # Convert 'Date' to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.sort_values(by='Date', inplace=True)
       
        df['Symbol'] = ticker

        # Drop duplicates
        df.drop_duplicates(subset=['Date', 'Symbol'], inplace=True)

        # Remove unwanted columns
        drop_cols = [col for col in df.columns if 'Adj Close' in col]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        # Rename prefixed columns to standard ones
        rename_cols = {
            f'Open_{ticker}.NS': 'Open',
            f'High_{ticker}.NS': 'High',
            f'Low_{ticker}.NS': 'Low',
            f'Close_{ticker}.NS': 'Close',
            f'Price_{ticker}.NS': 'Price',
            f'Volume_{ticker}.NS': 'Volume',
        }
        df.rename(columns=rename_cols, inplace=True)

        # Add feature engineering columns
        df['Hour'] = df['Date'].dt.hour
        df['Minute'] = df['Date'].dt.minute
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['Daily Change'] = df['Close'] - df['Open']
        df['Price Change'] = df['High'] - df['Low']
        df['Volatility'] = (df['High'] - df['Low']) / df['Open']
        df['Price Change %'] = (df['Price Change'] / df['Open']) * 100
        df['Daily Change %'] = (df['Daily Change'] / df['Open']) * 100
        df['Close_lag_1'] = df['Close'].shift(1)
        df['Close_lag_2'] = df['Close'].shift(2)
        df['Close_lag_3'] = df['Close'].shift(3)
        df['MA_3'] = df['Close'].rolling(window=3).mean()
        df['MA_7'] = df['Close'].rolling(window=7).mean()

    except Exception as e:
        logger.error(f"Error cleaning stock data for {ticker}: {e}")
    return df

# Function to add technical indicators to the DataFrame
def add_tech_indicators(df):

    try:
        w_short = 5
        w_mid = 10

        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=w_short).rsi()
        df['MACD'] = ta.trend.MACD(close=df['Close']).macd()
        bb = ta.volatility.BollingerBands(close=df['Close'], window=w_mid, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=w_short).average_true_range()
        df['SMA_5'] = ta.trend.SMAIndicator(close=df['Close'], window=w_short).sma_indicator()
        df['EMA_5'] = ta.trend.EMAIndicator(close=df['Close'], window=w_short).ema_indicator()
        df['Volume_MA_5'] = df['Volume'].rolling(window=w_short).mean()

        df.dropna(inplace=True)
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
    return df

# Function to preprocess stock data
def preprocess_stock_data_rf(df, ticker='UNKNOWN'):
    try:
        logger.info(f"Preprocessing stock data for {ticker}...")

        # Fix multi-level columns if present
        df = fix_multilevel_columns(df)
        
        # Clean the stock data
        df = clean_stock_data(df, ticker)

        # Add technical indicators
        df = add_tech_indicators(df)

        logger.info("Preprocessing complete.")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing stock data for {ticker}: {e}")
        return df


