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

# --- Main Training Logic ---
def train_and_evaluate():
    """Trains all models, evaluates them, and saves models, scores, and ratings."""
    df = pd.read_csv('data/housing.csv')
    X, y, preprocessor = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models()
    model_scores = {}
    model_errors = {}
    model_confidence = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        errors = y_test - y_pred
        model_scores[name] = rmse
        model_errors[name] = errors.tolist()
        model_confidence[name] = {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors))
        }
        print(f"  RMSE for {name}: {rmse:.2f}")
        joblib.dump(pipeline, f'models/{name.replace(" ", "_")}_model.pkl')

    # Calculate ranks (1 = best)
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
    model_ranks = {name: rank+1 for rank, (name, _) in enumerate(sorted_models)}

    with open('models/model_scores.json', 'w') as f:
        json.dump({
            'rmse': model_scores,
            'rank': model_ranks,
            'confidence': model_confidence
        }, f, indent=4)
    print("\nAll models trained and files saved successfully!")
    print("Scores, ranks, and confidence intervals saved to models/model_scores.json")

if __name__ == '__main__':
    train_and_evaluate() 