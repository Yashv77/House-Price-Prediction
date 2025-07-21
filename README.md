# California House Price Prediction System

This project is an interactive machine learning application that predicts median house values in California. It allows users to select from a variety of regression models, view their performance metrics, and get real-time price predictions based on input features.

## Key Features

-   **Multi-Model Prediction**: Select from a range of ML models (e.g., Linear Regression, Random Forest, XGBoost, CatBoost) to see how different algorithms perform.
-   **Interactive UI**: A user-friendly interface built with Streamlit that allows for easy input of house features.
-   **Model Performance Analysis**: A dedicated tab to compare models based on their RMSE scores and a 1-10 rating system.
-   **Integrated EDA**: An in-app tab showcasing the Exploratory Data Analysis performed on the dataset.

## Project Structure

```
house-price-prediction/
├── app/
│   └── app.py
├── data/
│   └── housing.csv
├── models/
│   ├── CatBoost_model.pkl
│   ├── ... (other trained models)
│   └── model_scores.json
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

-   **app/**: Contains the main Streamlit application code.
-   **data/**: Stores the `housing.csv` dataset.
-   **models/**: Contains all the trained model files (`.pkl`) and a JSON file with their performance scores.
-   **src/**: Holds the Python source code for EDA, preprocessing, and model training.

## How to Run the Project

Follow these steps to set up and run the project locally:

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/house-price-prediction.git
    cd house-price-prediction
    ```

2.  **Set Up a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the Models**
    Run the training script to train all the models and generate their performance scores. This step is crucial as it creates the files the Streamlit app depends on.
    ```bash
    python src/train.py
    ```

5.  **Launch the Streamlit App**
    ```bash
    streamlit run app/app.py
    ```
    This will open the application in your default web browser.

## Using the Application

-   **Prediction Tab**: Choose a model from the dropdown (sorted by performance), input the house features, and click "Predict Price" to get a prediction.
-   **Model Performance Tab**: View a detailed comparison of all trained models, including their RMSE scores and a 1-10 rating.
-   **Project Pipeline & EDA Tab**: Explore the project's workflow and see key visualizations from the exploratory data analysis. 