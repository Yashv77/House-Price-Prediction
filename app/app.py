import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('house_price_model.pkl')

st.title('House Price Prediction System')
st.write("Enter the details of the house to get a price prediction.")

# Create input fields for house features
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        OverallQual = st.slider('Overall Quality', 1, 10, 5)
        GrLivArea = st.number_input('Above Ground Living Area (sq ft)', min_value=0, value=1500)
        GarageCars = st.slider('Garage Cars', 0, 4, 2)
        TotalBsmtSF = st.number_input('Basement Area (sq ft)', min_value=0, value=850)

    with col2:
        FullBath = st.slider('Full Bathrooms', 0, 4, 2)
        YearBuilt = st.number_input('Year Built', min_value=1800, max_value=2025, value=2000)
        Neighborhood = st.selectbox('Neighborhood', ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'])

    # Create a dictionary from the inputs
    # Add other necessary features with default values for the model to work
    input_data = {
        'OverallQual': OverallQual, 'GrLivArea': GrLivArea, 'GarageCars': GarageCars,
        'TotalBsmtSF': TotalBsmtSF, 'FullBath': FullBath, 'YearBuilt': YearBuilt,
        'Neighborhood': Neighborhood,
        # Default values for other features
        'MSSubClass': 60, 'MSZoning': 'RL', 'LotFrontage': 65.0, 'LotArea': 8450, 'Street': 'Pave', 'Alley': None, 'LotShape': 'Reg', 'LandContour': 'Lvl', 'Utilities': 'AllPub', 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Condition1': 'Norm', 'Condition2': 'Norm', 'BldgType': '1Fam', 'HouseStyle': '2Story', 'OverallCond': 5, 'YearRemodAdd': 2003, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'VinylSd', 'Exterior2nd': 'VinylSd', 'MasVnrType': 'BrkFace', 'MasVnrArea': 196.0, 'ExterQual': 'Gd', 'ExterCond': 'TA', 'Foundation': 'PConc', 'BsmtQual': 'Gd', 'BsmtCond': 'TA', 'BsmtExposure': 'No', 'BsmtFinType1': 'GLQ', 'BsmtFinSF1': 706, 'BsmtFinType2': 'Unf', 'BsmtFinSF2': 0, 'BsmtUnfSF': 150, 'Heating': 'GasA', 'HeatingQC': 'Ex', 'CentralAir': 'Y', 'Electrical': 'SBrkr', '1stFlrSF': 856, '2ndFlrSF': 854, 'LowQualFinSF': 0, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1, 'KitchenQual': 'Gd', 'TotRmsAbvGrd': 8, 'Functional': 'Typ', 'Fireplaces': 0, 'FireplaceQu': None, 'GarageType': 'Attchd', 'GarageYrBlt': 2003.0, 'GarageFinish': 'RFn', 'GarageArea': 548, 'GarageQual': 'TA', 'GarageCond': 'TA', 'PavedDrive': 'Y', 'WoodDeckSF': 0, 'OpenPorchSF': 61, 'EnclosedPorch': 0, '3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': None, 'Fence': None, 'MiscFeature': None, 'MiscVal': 0, 'MoSold': 2, 'YrSold': 2008, 'SaleType': 'WD', 'SaleCondition': 'Normal'
    }
    input_df = pd.DataFrame([input_data])


    submitted = st.form_submit_button("Predict Price")
    if submitted:
        prediction = model.predict(input_df)
        st.success(f"The predicted house price is: ${prediction[0]:,.2f}") 