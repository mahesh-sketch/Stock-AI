from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from core.logger import logger
import joblib


def prepare_data_for_ml(df):
    try:
        df = df.copy()
        df['Target'] = df['Close'].shift(-1)  # next close price

        df.dropna(inplace=True)

        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'BB_High', 'BB_Low', 'BB_Mid',
            'ATR', 'SMA_5', 'EMA_5', 'Volume_MA_5',
            'DayOfWeek', 'Hour', 'Minute', 'Volatility',
            'Price Change %', 'Daily Change %',
            'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'MA_3', 'MA_7'
        ]

        X = df[features]
        y = df['Target']

        return X, y
    except Exception as e:
        logger.error(f"Error in prepare_data_for_ml: {e}")
        return None, None

def split_and_scale(X, y, scaler_type='standard'):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        if scaler_type == 'standard':
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
        else:
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

        X_train_scaled = scaler_x.fit_transform(X_train)
        X_test_scaled = scaler_x.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()

        save_scaler_y(scaler_y, './model/scaler_y.pkl')

        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x, scaler_y, X_train.index, X_test.index
    except Exception as e:
        logger.error(f"Error in split_and_scale: {e}")
        return None

def prepare_data_rf(df, scaler_type='standard'):
    try:
        print("🔍 Preparing data for machine learning...")

        X, y = prepare_data_for_ml(df)
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x, scaler_y, train_index, test_index = split_and_scale(X, y, scaler_type)

        print(f"📊 Training set size: {X_train_scaled.shape[0]}, Test set size: {X_test_scaled.shape[0]}")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_scaled,
            'y_test': y_test_scaled,
            'scaler_x': scaler_x,
            'scaler_y': scaler_y,
            'train_index': train_index,
            'test_index': test_index
        }
    except Exception as e:
        logger.error(f"Error in prepare_ml_data: {e}")
        return None

# Function to save the scaler_y
def save_scaler_y(scaler_y, filename='./model/scaler_y.pkl'):
    try:
        joblib.dump(scaler_y, filename)
        logger.info(f"✅ Scaler saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving scaler_y: {e}")