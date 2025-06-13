from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from core.logger import logger

def evaluate_model(y_true, y_pred):
    try:
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R2:  {r2:.4f}")
        return mse, mae, r2
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None, None, None

def evaluate_model_with_scalers(y_true, y_pred, scaler_y):
    # Inverse transform the predictions
    try:
        y_true_inv = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        logger.info("Evaluating model performance on original scale:")
        mse, mae, r2 = evaluate_model(y_true_inv, y_pred_inv)
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R2:  {r2:.4f}")
        return mse, mae, r2
    except Exception as e:
        logger.error(f"Error evaluating model with scalers: {e}")
        return None, None, None