import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from src.preprocess import preprocess_data
from sklearn.pipeline import Pipeline

# Load the training data
df = pd.read_csv('data/train.csv')

# Preprocess the data
X, y, preprocessor = preprocess_data(df)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(random_state=42))])

# Define hyperparameter grid for tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_features': ['auto', 'sqrt'],
    'regressor__max_depth': [10, 20, None],
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error on Test Set: {rmse}")

# Save the trained model
joblib.dump(best_model, 'house_price_model.pkl')
print("Model saved as house_price_model.pkl") 