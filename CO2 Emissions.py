import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
from streamlit_option_menu import option_menu
# Interactive Visualization
import plotly.graph_objs as go
import plotly.express as px
from scipy import stats
# Define skewness
from scipy.stats import skew
# Box-Cox Transformation for skewness
from scipy.stats import boxcox


st.set_page_config(page_title="Dashboard Prediksi Emisi", page_icon="🌏", layout="wide")

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Fungsi untuk visualisasi data
def visualize_data(data):
    if st.checkbox("Tampilkan Pratinjau Data"):
        st.write("**Pratinjau Data:**")
        st.dataframe(data.head())

    if st.checkbox("Tampilkan Visualisasi Data"):
        st.write("**Visualisasi Data:**")
        columns = data.select_dtypes(include=['float64', 'int64']).columns
        num_columns = 3
        for i in range(0, len(columns), num_columns):
            cols = st.columns(num_columns)
            for col_index in range(num_columns):
                if i + col_index < len(columns):
                    column = columns[i + col_index]
                    fig = px.histogram(data, x=column, nbins=20, title=f'Histogram of {column}', color_discrete_sequence=['darkgreen'])
                    fig.update_layout(height=300, width=300,
                                      title_font_size=12, 
                                      xaxis_title_font_size=10, 
                                      yaxis_title_font_size=10, 
                                      font=dict(size=10)
                                      )
                    cols[col_index].plotly_chart(fig)

# Fungsi untuk membersihkan missing values
def clean_nan(data):
    # Identifikasi kolom dengan missing value > 60%
    threshold = 0.6
    columns_to_drop = []
    for col in data.columns:
        missing_percentage = (data[col].isnull().sum()/data[col].shape[0])
        if missing_percentage > threshold:
            columns_to_drop.append(col)
    # Hapus kolom dengan missing value > threshold
    data = data.drop(columns=columns_to_drop, axis=1)

    # Identifikasi kolom numerik
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    # Mengatasi nilai NaN dengan SimpleImputer untuk kolom numerik
    num_imputer = SimpleImputer(strategy='mean')
    data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

    # Identifikasi kolom kategorikal
    categorical_cols = data.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

    return data

# Fungsi untuk mengidentifikasi dan membersihkan outliers
def clean_outliers(data, threshold=3, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    numeric_data = data.select_dtypes(include=np.number).drop(columns=exclude_columns, errors='ignore')

    for col in numeric_data.columns:
        z_scores = np.abs(stats.zscore(numeric_data[col]))
        outliers = (z_scores > threshold)
        data = data[~outliers]

    return data

# Fungsi untuk mentransformasi data yang skew
def boxcox_skewed(data, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    
    numeric_data = data.select_dtypes(include=np.number).drop(columns=exclude_columns, errors='ignore')

    for col in numeric_data.columns:
        skewness_score = data[col].skew()
        if skewness_score > 0.6:
            # Memastikan semua nilai positive
            shifted_data = data[col] + 1 - data[col].min()
            try:
                # Apply Box-Cox transformation
                transformed_data, _ = stats.boxcox(shifted_data)
                data[col] = transformed_data
            except ValueError as e:
                print(f"Skipping column {col} due to error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while transforming column {col}: {e}")

    return data


# Fungsi untuk membuat model
def build_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return model, predictions, rmse, r2

# Fungsi untuk plot actual vs predicted
def plot_actual_vs_predicted(y_test, predictions):
    fig = go.Figure()

    # Menambahkan sctter plot untuk actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_test,
        y=predictions,
        mode='markers',
        marker=dict(color='forestgreen'),
        name='Predictions'
    ))

    # Menambahkan garis ideal match
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Ideal Match'
    ))

    # Update layout untuk menambahkan judul, label, dan legend
    fig.update_layout(
        title='Actual vs Predicted',
        xaxis_title='Actual',
        yaxis_title='Predicted',
        showlegend=True
    )

    st.plotly_chart(fig)


# Fungsi untuk plot residuals vs predicted
def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    
    # Menambahkan scatter plot untuk residual vs predicted
    fig = px.scatter(
        x=predictions, y=residuals,
        color_discrete_sequence=['green'],
        labels={'x': 'Prediksi', 'y': 'Residual'},
        title='Residual vs Prediksi'
    )

    # Menambahkan garis fungsi horizontal y=0
    fig.add_trace(go.Scatter(
        x=[predictions.min(), predictions.max()],
        y=[0, 0],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Ideal Match'
    ))
    
    # Update layout untuk menambahkan judul, label, dan legend
    fig.update_layout(
        title='Residual vs Prediksi',
        xaxis_title='Prediksi',
        yaxis_title='Residual',
        showlegend=True
    )
    
    st.plotly_chart(fig)


# Halaman Utama
def homepage():
    st.title("🌏Dashboard Prediksi Emisi")
    st.write("Selamat datang di Website Prediksi Emisi CO2!")
    st.write("Silakan menuju ke halaman Prediksi untuk melakukan prediksi.")

# Halaman Prediksi
def prediction_page():
    st.title("📊Prediksi")
    file = st.file_uploader("Unggah file CSV", type=["csv"])
    
    if file:
        data = load_data(file)
        
        st.subheader("Data yang Diunggah")
        visualize_data(data)
    
        # User menentukan kolom target dan kolom unik
        target_column = st.selectbox("Pilih Kolom Target",[""] + data.columns.tolist(), index=0)
        unique_column = st.selectbox("Pilih Kolom ID/Unik", [""] + data.columns.tolist(), index=0)

        if target_column and unique_column:
            with st.spinner("Data Preprocessing sedang berjalan..."):
                clean_nan(data)
            with st.spinner("Data Preprocessing sedang berjalan...."):
                clean_outliers(data, threshold=3, exclude_columns=unique_column)
            with st.spinner("Data Preprocessing sedang berjalan....."):
                boxcox_skewed(data, exclude_columns=[unique_column, target_column])
            
            X = data.drop(columns=[target_column, unique_column])
            y = data[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_name = st.selectbox("Pilih Model", ["Random Forest", "Linear Regression", "Decision Tree Regressor", "Ridge", "Lasso"])
            
            if model_name == "Random Forest":
                model = RandomForestRegressor()
            elif model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            elif model_name == "Ridge":
                model = Ridge()
            elif model_name == "Lasso":
                model = Lasso()
            
            if st.button("Prediksi"):
                model, predictions, rmse, r2 = build_model(model, X_train, y_train, X_test, y_test)
                st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")
                st.write(f"**R^2 Score:** {r2}")

                st.write("**Plot Aktual vs Prediksi:**")
                plot_actual_vs_predicted(y_test, predictions)

                st.write("**Plot Residual vs Prediksi:**")
                plot_residuals(y_test, predictions)

# Navbar Streamlit
with st.sidebar:
    selected = option_menu(
        menu_title="Halaman",
        options=["Dashboard", "Prediksi"],
        icons=["house", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0        
    )

if selected == "Dashboard":
    homepage()
elif selected == "Prediksi":
    prediction_page()
