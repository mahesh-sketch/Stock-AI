from sklearn.ensemble import RandomForestRegressor
from core.logger import logger
# from evaluation.evaluation import evaluate_model
import joblib

# Function to train a Random Forest model
def train_random_forest(X_train, y_train):
    try:
        logger.info("🔍 Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("✅ Random Forest model training complete.")
        return model
    except Exception as e:
        logger.error(f"Error in train_random_forest: {e}")
        return None

def train_save_models(X_train, y_train, filename='random_forest_model.pkl'):
    try:
        model = train_random_forest(X_train, y_train)
        if model:
            joblib.dump(model, filename)
            logger.info(f"✅ Model saved to {filename}")
        else:
            logger.error("Model training failed. Cannot save the model.")
    except Exception as e:
        logger.error(f"Error in train_save_models: {e}")
""" 
# Function to save the trained model
def save_model(model, filename='random_forest_model.pkl'):
    try:
        joblib.dump(model, filename)
        logger.info(f"✅ Model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

# Function to load a trained model
def load_model(filename='random_forest_model.pkl'):
    try:
        model = joblib.load(filename)
        logger.info(f"✅ Model loaded from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
    
# Function to load and evaluate a trained model
def load_and_evaluate_model(filename, X_test, y_test):
    model = load_model(filename)
    if model:
        predictions = model.predict(X_test)
        evaluate_model(y_test, predictions)
    else:
        logger.error("Failed to load the model for evaluation.")
        save_model(model, 'random_forest_model.pkl')

# Function to save and evaluate a trained model
def save_and_evaluate_model(model, X_test, y_test, filename='random_forest_model.pkl'):
    save_model(model, filename)
    predictions = model.predict(X_test)
    evaluate_model(y_test, predictions)

# Function to load, train, save, and evaluate a Random Forest model
def train_save_and_evaluate_random_forest(X_train, y_train, X_test, y_test, filename='random_forest_model.pkl'):
    model = train_random_forest(X_train, y_train)
    if model:
        save_and_evaluate_model(model, X_test, y_test, filename)
    else:
        logger.error("Model training failed. Cannot save or evaluate.")



 """