import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv('data/house_prices.csv')
    
    X = df.drop('Price_INR', axis=1)
    y = df['Price_INR']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Preprocessing Pipeline
    numeric_features = ['Area_sqft', 'Bedrooms', 'Bathrooms', 'Floors', 'Age_years']
    categorical_features = ['Location', 'Property_Type', 'Parking']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # 2. Define Models to Compare
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    metrics_report = {}
    
    print("\nTraining and comparing models...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics_report[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"{name} -> R2: {r2:.4f} | RMSE: ₹{rmse:,.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            best_name = name

    print(f"\n🏆 Best Model: {best_name} (R2: {best_r2:.4f})")
    
    # 3. Save Model and Metrics
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/house_price_model.joblib')
    
    final_metrics = {
        "selected_model": best_name,
        "metrics": metrics_report[best_name],
        "training_size": len(X_train),
        "features": list(X.columns)
    }
    
    with open('models/model_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print("✅ Pipeline and metrics saved to 'models/'")

if __name__ == "__main__":
    train_and_evaluate()
