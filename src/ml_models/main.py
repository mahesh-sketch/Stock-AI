from collection import collect
from preprocessing import data_preprocessing
from preparation import preparation
from training import random_forest
from evaluation import evaluation
from preprocessing import data_preprocessing
from core.logger import logger
from joblib import load
import pandas as pd


def run_train_model():
  try:
    df = collect.fetch_fifteenmin_stock_data("RELIANCE")
    # Train the Random Forest model and save it
    if df is not None:
      df = data_preprocessing.preprocess_stock_data_rf(df, ticker='RELIANCE')
      ml_data = preparation.prepare_data_rf(df)

      random_forest.train_save_models(
          ml_data['X_train'], ml_data['y_train'], filename='./model/random_forest_model.pkl'
      )

      model = load('./model/random_forest_model.pkl')
      evaluation.evaluate_model(ml_data['y_test'], model.predict(ml_data['X_test']))
      logger.info("✅ Model training and evaluation completed successfully.")
    else:
      logger.warning("No data fetched for the specified ticker.")
  except Exception as e:
    logger.error(f"An error occurred during model training: {e}")
  

def run_predict_model(ticker:str)->float:
    try:
        # Load the trained model
        model = load('./model/random_forest_model.pkl')
        scaler_y = load('./model/scaler_y.pkl')

        df = collect.fetch_fifteenmin_stock_data(ticker, period='5y', interval='1d')
        if df is not None and not df.empty:
            df = data_preprocessing.preprocess_stock_data_rf(df, ticker=ticker)
            ml_data = preparation.prepare_data_rf(df)

            if not ml_data:
                logger.warning("❌ ML data preparation failed.")
                return

            latest_X = ml_data['X_test'][-1].reshape(1, -1) 
            predictions_scaled = model.predict(latest_X)
            real_price_prediction = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()[0]
            logger.info(f"📈 Predicted price for {ticker}: {real_price_prediction}")
        else:
            logger.warning("⚠️ No usable recent data fetched for the specified ticker.")
    except Exception as e:
        logger.error(f"💥 Error during prediction: {e}")


def predict_stock(ticker: str) -> float:
   try:
        model = load('./model/random_forest_model.pkl')
        logger.info(f"🚀 Running prediction for: {ticker}")

        # Fetch and preprocess data
        df = collect.fetch_fifteenmin_stock_data(ticker, period='5y', interval='1d')
        if df is None or df.empty:
            raise ValueError("No data")

        df = data_preprocessing.preprocess_stock_data_rf(df, ticker=ticker)
        ml_data = preparation.prepare_data_rf(df)

        if not ml_data:
            raise ValueError("ML data prep failed")

        # Train model dynamically (not saved)
        model = random_forest.train_random_forest(ml_data['X_train'], ml_data['y_train'])
        if model is None:
            raise ValueError("Model training failed")
        
        model.fit(ml_data['X_train'], ml_data['y_train'])

        latest_X = ml_data['X_test'][-1].reshape(1, -1)
        prediction_scaled = model.predict(latest_X)

        # Inverse transform with ticker-specific y-scaler
        scaler_y = ml_data['scaler_y']
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

        logger.info(f"📈 Predicted price for {ticker}: ₹{prediction:.2f}")
        return float(prediction)
   except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise RuntimeError(f"Prediction failed: {e}")


def main():
  
  """ logger.info("🔄 Starting the machine learning pipeline...")
  run_train_model()
  logger.info("✅ Machine learning pipeline completed successfully.")
 """
  # run_predict_model("RELIANCE")
  predict_stock("TATAMOTORS")

if __name__ == "__main__":
    main()