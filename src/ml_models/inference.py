from src.ml_models.collection.collect import fetch_fifteenmin_stock_data 
from src.ml_models.preprocessing.data_preprocessing import preprocess_stock_data_rf
from src.ml_models.preparation.preparation import prepare_data_rf
from src.ml_models.training.random_forest import train_save_models
from src.ml_models.evaluation.evaluation import evaluate_model

from core.logger import logger
from joblib import load


def run_predict_model(ticker:str) -> float:
    try:
        # Load the trained model
        logger.info(f"🚀 Running prediction for: {ticker}")
        model = load('./model/random_forest_model.pkl')
        # scaler_y = load('./model/scaler_y.pkl')

        df = fetch_fifteenmin_stock_data(ticker, period='5y', interval='1d')
        if df is not None and not df.empty:
            df = preprocess_stock_data_rf(df, ticker=ticker)
            ml_data = prepare_data_rf(df)

            if not ml_data:
                raise ValueError("No data available for prediction after preprocessing and preparation.")
            
            model.fit(ml_data['X_train'], ml_data['y_train'])
            latest_X = ml_data['X_test'][-1].reshape(1, -1)
            prediction_scaled = model.predict(latest_X)

            scaler_y = ml_data['scaler_y']
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

            logger.info(f"📈 Predicted price for {ticker}: ₹{prediction:.2f}")
            return float(prediction)
        else:
            raise ValueError("No data fetched for the specified ticker.")
    except Exception as e:
       raise RuntimeError(f"An error occurred during prediction: {e}")
