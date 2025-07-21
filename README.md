# House Price Prediction System

This project is a machine learning system designed to predict house prices based on various features of the property. It includes a data preprocessing pipeline, a model training script with hyperparameter tuning, and a Streamlit web application for interactive predictions.

## Project Structure

The project is organized as follows:

- `app/`: Contains the Streamlit application code.
- `data/`: Stores the dataset files (`train.csv`, `test.csv`).
- `notebooks/`: Holds Jupyter notebooks for exploratory data analysis.
- `src/`: Contains the source code for data preprocessing and model training.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `requirements.txt`: Lists the Python dependencies for the project.
- `README.md`: Provides a detailed description of the project.

## Setup Instructions

To set up and run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/house-price-prediction.git
    cd house-price-prediction
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Exploratory Data Analysis (EDA)

To explore the dataset and understand the relationships between different features, you can use the Jupyter notebook provided:

```bash
jupyter lab notebooks/EDA.ipynb
```

### 2. Model Training

To train the house price prediction model, run the following command in your terminal. This will preprocess the data, train a Random Forest Regressor, and save the trained model as `house_price_model.pkl`.

```bash
python src/train.py
```

### 3. Streamlit Web Application

After training the model, you can launch the interactive web application to get real-time price predictions.

```bash
streamlit run app/app.py
```

This will open a new tab in your web browser with the application interface. You can input the house features in the form and click "Predict Price" to see the estimated price. 