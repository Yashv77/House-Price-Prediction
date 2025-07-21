import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import preprocess_data

# Import models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# --- Model Definitions ---
def get_models():
    """Returns a dictionary of all models to be trained."""
    # Model definitions remain the same
    models = {
        "Linear Regression": LinearRegression(), "Ridge": Ridge(), "Lasso": Lasso(),
        "ElasticNet": ElasticNet(), "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(random_state=42), "LightGBM": lgb.LGBMRegressor(random_state=42),
        "CatBoost": cb.CatBoostRegressor(random_state=42, verbose=0), "SVR": SVR(), "KNN": KNeighborsRegressor()
    }
    return models

# --- Score Conversion ---
def calculate_ratings(scores):
    """Converts RMSE scores to a 1-10 rating."""
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    # Avoid division by zero if all scores are the same
    if max_score == min_score:
        return {name: 10 for name in scores}

    ratings = {}
    for name, score in scores.items():
        # Normalize the score (0-1 range, lower is better)
        normalized_score = (score - min_score) / (max_score - min_score)
        # Invert so higher is better, and scale to 1-10
        rating = 1 + (1 - normalized_score) * 9
        ratings[name] = round(rating, 2)
    return ratings

# --- Main Training Logic ---
def train_and_evaluate():
    """Trains all models, evaluates them, and saves models, scores, and ratings."""
    df = pd.read_csv('data/housing.csv')
    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models()
    model_scores = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        model_scores[name] = rmse
        print(f"  RMSE for {name}: {rmse:.2f}")
        
        joblib.dump(pipeline, f'models/{name.replace(" ", "_")}_model.pkl')

    # Calculate ratings and save both scores and ratings
    model_ratings = calculate_ratings(model_scores)
    
    with open('models/model_scores.json', 'w') as f:
        json.dump({'rmse': model_scores, 'ratings': model_ratings}, f, indent=4)
        
    print("\nAll models trained and files saved successfully!")
    print("Scores and ratings saved to models/model_scores.json")

if __name__ == '__main__':
    train_and_evaluate() 