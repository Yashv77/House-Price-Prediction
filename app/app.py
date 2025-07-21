import streamlit as st
import pandas as pd
import joblib
import json
import sys
import os

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import eda

# --- Data Loading Functions ---
def load_model_data():
    """Loads model scores and ratings from the JSON file."""
    with open('models/model_scores.json', 'r') as f:
        return json.load(f)

@st.cache_resource
def load_model(model_name):
    """Loads a specific trained model from disk."""
    model_path = f'models/{model_name.replace(" ", "_")}_model.pkl'
    return joblib.load(model_path)

# --- Page Configuration ---
st.set_page_config(page_title="California House Price Prediction", page_icon="🏠", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.title("About the Project")
    st.info("This app predicts California house prices using various ML models. Choose a model to get started.")
    st.title("Tech Stack")
    st.markdown("- Python\n- Scikit-learn\n- Pandas\n- Streamlit\n- XGBoost\n- LightGBM\n- CatBoost")
    st.markdown("Developed by [Yash Vardhan](https://github.com/Yashv77)")

# --- Main Application ---
st.title('🏠 California House Price Prediction')

tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "Project Pipeline & EDA"])

# --- Prediction Tab ---
with tab1:
    st.header("Make a Prediction")
    
    model_data = load_model_data()
    model_ratings = model_data['ratings']
    model_options = [f"{name} (Rating: {rating}/10)" for name, rating in sorted(model_ratings.items(), key=lambda item: item[1], reverse=True)]
    selected_option = st.selectbox("Choose a Model", model_options)
    selected_model_name = selected_option.split(" (")[0]
    
    with st.form("prediction_form"):
        st.header("Input Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            longitude = st.number_input('Longitude', value=-122.23)
            latitude = st.number_input('Latitude', value=37.88)
            housing_median_age = st.number_input('Housing Median Age', value=41.0)
        with col2:
            total_rooms = st.number_input('Total Rooms', value=880.0)
            total_bedrooms = st.number_input('Total Bedrooms', value=129.0)
            population = st.number_input('Population', value=322.0)
        with col3:
            households = st.number_input('Households', value=126.0)
            median_income = st.number_input('Median Income (x$10,000)', value=8.3252)
            ocean_proximity = st.selectbox('Ocean Proximity', ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
        
        submitted = st.form_submit_button("Predict Price")
        if submitted:
            model = load_model(selected_model_name)
            input_data = pd.DataFrame([{'longitude': longitude, 'latitude': latitude, 'housing_median_age': housing_median_age,
                                        'total_rooms': total_rooms, 'total_bedrooms': total_bedrooms, 'population': population,
                                        'households': households, 'median_income': median_income, 'ocean_proximity': ocean_proximity}])
            prediction = model.predict(input_data)
            st.subheader("Predicted House Price")
            st.metric(label=f"Predicted Value (using {selected_model_name})", value=f"${prediction[0]:,.2f}")

# --- Model Performance Tab ---
with tab2:
    st.header("Model Performance Comparison")
    st.markdown("Here you can see the performance of each model. A lower RMSE (Root Mean Squared Error) is better, and the rating is scaled from 1-10 accordingly.")
    
    model_data = load_model_data()
    performance_df = pd.DataFrame({
        "Model": model_data['ratings'].keys(),
        "Rating (1-10)": model_data['ratings'].values(),
        "RMSE Score": model_data['rmse'].values()
    }).sort_values(by="Rating (1-10)", ascending=False).reset_index(drop=True)
    
    st.dataframe(performance_df)
    
    st.subheader("Model Ratings Visualized")
    st.bar_chart(performance_df.set_index('Model')['Rating (1-10)'])

# --- Project Pipeline & EDA Tab ---
with tab3:
    st.header("Project Pipeline & EDA")
    st.image("https://i.imgur.com/9IZv2zS.png", caption="ML Project Pipeline")
    st.markdown(
        """
        This project follows a standard ML pipeline:
        1.  **Data Loading**: Load the `housing.csv` dataset.
        2.  **EDA**: Perform exploratory data analysis to understand features.
        3.  **Preprocessing**: Clean data and transform features for the models.
        4.  **Model Training**: Train and evaluate multiple regression models.
        5.  **Prediction**: Deploy the models in this Streamlit app for interactive predictions.
        """
    )
    st.header("Exploratory Data Analysis")
    df = eda.load_data()
    st.subheader("1. Dataset Preview")
    st.dataframe(df.head())
    st.subheader("2. Target Variable Distribution")
    st.pyplot(eda.plot_target_distribution(df))
    st.subheader("3. Correlation Heatmap")
    st.pyplot(eda.plot_correlation_heatmap(df))
    st.subheader("4. Missing Values")
    missing_values = eda.get_missing_values(df)
    if not missing_values.empty:
        st.dataframe(missing_values)
    else:
        st.success("No missing values found.") 