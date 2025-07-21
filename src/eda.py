import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path='data/housing.csv'):
    """Loads the housing data."""
    return pd.read_csv(path)

def get_data_info(df):
    """Returns basic information and description of the dataframe."""
    return df.info(), df.describe()

def plot_target_distribution(df):
    """Plots the distribution of the target variable."""
    fig, ax = plt.subplots()
    sns.histplot(df['median_house_value'], kde=True, ax=ax)
    ax.set_title('Distribution of Median House Value')
    return fig

def plot_correlation_heatmap(df):
    """Plots the correlation heatmap for numerical features."""
    fig, ax = plt.subplots(figsize=(12, 9))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corrmat = numeric_df.corr()
    sns.heatmap(corrmat, vmax=.8, square=True, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Correlation Heatmap of Numerical Features')
    return fig

def get_missing_values(df):
    """Returns a dataframe with the count and percentage of missing values."""
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data['Total'] > 0] 