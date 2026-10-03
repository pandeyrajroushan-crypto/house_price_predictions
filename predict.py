import joblib
import pandas as pd
import os

def load_model():
    model_path = 'models/house_price_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train_model.py first.")
    return joblib.load(model_path)

def predict_price(model, input_data):
    """
    input_data should be a dictionary matching the feature columns.
    """
    df = pd.DataFrame([input_data])
    predicted_price = model.predict(df)[0]
    return predicted_price
