"""
Streamlit Web Application: Air Quality Index (AQI) Prediction & Analysis
Interactive dashboard for real-time model inference, exploratory data analysis,
and model performance benchmarking for college mini-project demonstration.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AQI Prediction & Analysis System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .category-badge {
        font-size: 1.3rem;
        font-weight: bold;
        padding: 0.4rem 1rem;
        border-radius: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    'Good': '#2ecc71',
    'Satisfactory': '#a8e063',
    'Moderate': '#f1c40f',
    'Poor': '#e67e22',
    'Very Poor': '#e74c3c',
    'Severe': '#8e44ad'
}

HEALTH_ADVISORIES = {
    'Good': {
        'Range': '0 - 50',
        'Impact': 'Minimal health impact. Air quality is considered satisfactory, and pollution poses little or no risk.',
        'Recommendation': 'Ideal for all outdoor activities and exercises.'
    },
    'Satisfactory': {
        'Range': '51 - 100',
        'Impact': 'Minor breathing discomfort to sensitive individuals on prolonged exposure.',
        'Recommendation': 'Safe for general public; sensitive individuals should monitor symptoms.'
    },
    'Moderate': {
        'Range': '101 - 200',
        'Impact': 'Breathing discomfort to individuals with lung diseases (asthma) and heart conditions.',
        'Recommendation': 'Children and elderly should limit prolonged heavy outdoor exertion.'
    },
    'Poor': {
        'Range': '201 - 300',
        'Impact': 'Breathing discomfort to most people on prolonged exposure; respiratory illness in sensitive groups.',
        'Recommendation': 'Avoid prolonged outdoor activities; wear protective masks near roadways.'
    },
    'Very Poor': {
        'Range': '301 - 400',
        'Impact': 'Significant respiratory illness on prolonged exposure; severe distress for cardiac patients.',
        'Recommendation': 'Stay indoors, keep windows closed, and use HEPA air purifiers.'
    },
    'Severe': {
        'Range': '401 - 500',
        'Impact': 'Health emergency. Affects healthy people and seriously impacts those with existing ailments.',
        'Recommendation': 'Avoid all outdoor exposure; run air filtration and consult healthcare professionals if symptomatic.'
    }
}

@st.cache_resource
def load_models():
    scaler = joblib.load('models/aqi_scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    regressor = joblib.load('models/aqi_regressor_rf.pkl')
    classifier = joblib.load('models/aqi_classifier_rf.pkl')
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    with open('models/evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
    return scaler, label_encoder, regressor, classifier, feature_names, metrics

try:
    scaler, label_encoder, regressor, classifier, feature_names, metrics = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure train_models.py has executed.")
    models_loaded = False

# Sidebar: Input Features
st.sidebar.header("🕹️ Sensor & Atmospheric Inputs")

station = st.sidebar.selectbox("Monitoring Station", ['Station_North', 'Station_South', 'Station_East', 'Station_West'])
season = st.sidebar.selectbox("Season", ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'])
date_input = st.sidebar.date_input("Observation Date")

st.sidebar.markdown("---")
st.sidebar.subheader("Criteria Pollutants")
pm25 = st.sidebar.slider("PM2.5 (µg/m³)", min_value=5.0, max_value=450.0, value=65.0, step=0.5)
pm10 = st.sidebar.slider("PM10 (µg/m³)", min_value=10.0, max_value=600.0, value=120.0, step=1.0)
no2 = st.sidebar.slider("NO2 (µg/m³)", min_value=5.0, max_value=250.0, value=35.0, step=0.5)
so2 = st.sidebar.slider("SO2 (µg/m³)", min_value=2.0, max_value=120.0, value=18.0, step=0.5)
co = st.sidebar.slider("CO (mg/m³)", min_value=0.1, max_value=15.0, value=1.0, step=0.1)
o3 = st.sidebar.slider("O3 (µg/m³)", min_value=5.0, max_value=200.0, value=30.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Meteorological Factors")
temp = st.sidebar.slider("Temperature (°C)", min_value=5.0, max_value=48.0, value=26.0, step=0.5)
humidity = st.sidebar.slider("Relative Humidity (%)", min_value=15.0, max_value=98.0, value=60.0, step=1.0)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", min_value=1.5, max_value=32.0, value=12.0, step=0.5)

# Main Dashboard Layout
st.markdown('<div class="main-header">Air Quality Index (AQI) Prediction & Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Mini-Project | Comparative Modeling of Atmospheric Pollutants</div>', unsafe_allow_html=True)

tabs = st.tabs(["🎯 Live Prediction & Health Advisory", "📊 Exploratory Data Analysis", "📈 Model Benchmarks & Results", "🏗️ System Architecture & Data"])

# TAB 1: Live Prediction
with tabs[0]:
    if models_loaded:
        # Preprocess single sample
        input_dict = {
            'Date': str(date_input),
            'Station': station,
            'Season': season,
            'PM2.5': pm25,
            'PM10': pm10,
            'NO2': no2,
            'SO2': so2,
            'CO': co,
            'O3': o3,
            'Temperature': temp,
            'Humidity': humidity,
            'Wind_Speed': wind_speed
        }
        
        df_single = pd.DataFrame([input_dict])
        date_parsed = pd.to_datetime(df_single['Date'].iloc[0])
        df_single['Month'] = date_parsed.month
        df_single['DayOfWeek'] = date_parsed.dayofweek
        df_single['Is_Weekend'] = 1 if date_parsed.dayofweek >= 5 else 0
        
        df_single['PM_Ratio'] = np.clip(pm25 / (pm10 + 1e-5), 0.05, 0.95)
        df_single['Dispersion_Index'] = (wind_speed + 1.0) / (pm25 + 1.0)
        df_single['Heat_Moisture_Index'] = (temp * humidity) / 100.0
        
        for col in feature_names:
            if col not in df_single.columns:
                if col.startswith('Station_'):
                    df_single[col] = 1 if station == col.replace('Station_', '') else 0
                elif col.startswith('Season_'):
                    df_single[col] = 1 if season == col.replace('Season_', '') else 0
                else:
                    df_single[col] = 0.0
                    
        X_in = df_single[feature_names].copy()
        X_scaled = scaler.transform(X_in)
        
        pred_aqi = float(regressor.predict(X_scaled)[0])
        pred_cat_idx = classifier.predict(X_scaled)[0]
        pred_cat = label_encoder.inverse_transform([pred_cat_idx])[0]
        cat_probs = classifier.predict_proba(X_scaled)[0]
        
        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
        
        with col1:
            st.markdown("### 🔢 Predicted AQI Value")
            st.metric(label="Continuous Regression Output", value=f"{pred_aqi:.1f}")
            st.caption(f"Estimated by Random Forest Regressor ($R^2$: {metrics['regression_metrics']['Random Forest']['Test_R2']})")
            
        with col2:
            st.markdown("### 🏷️ AQI Category")
            color = CATEGORY_COLORS.get(pred_cat, '#333333')
            st.markdown(f'<div class="category-badge" style="background-color: {color}; color: white;">{pred_cat}</div>', unsafe_allow_html=True)
            max_p = np.max(cat_probs) * 100
            st.caption(f"Confidence: {max_p:.1f}% (Random Forest Classifier)")
            
        with col3:
            st.markdown("### 🩺 Health Advisory & Impact")
            adv = HEALTH_ADVISORIES.get(pred_cat, {})
            st.info(f"**Impact:** {adv.get('Impact', 'N/A')}\n\n**Recommendation:** {adv.get('Recommendation', 'N/A')}")
            
        st.markdown("---")
        st.subheader("📊 Category Probability Distribution")
        prob_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability (%)': [round(p * 100, 1) for p in cat_probs]
        })
        st.bar_chart(prob_df.set_index('Category'))
        
        st.markdown("---")
        st.subheader("🧬 Computed Domain Features")
        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric("PM2.5 / PM10 Fine Ratio", f"{df_single['PM_Ratio'].iloc[0]:.2f}", help="Indicates the proportion of fine combustion aerosols vs coarse dust.")
        fcol2.metric("Dispersion Ventilation Proxy", f"{df_single['Dispersion_Index'].iloc[0]:.3f}", help="Ratio of wind speed to particulate mass.")
        fcol3.metric("Heat-Moisture Index", f"{df_single['Heat_Moisture_Index'].iloc[0]:.1f}", help="Interaction index between ambient temperature and relative humidity.")

# TAB 2: Exploratory Data Analysis
with tabs[1]:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Exploration of sensor distributions, seasonal cycles, and correlation structures across the monitoring stations.")
    
    eda_plots = [
        ("Pollutant Distribution Histograms & KDE", "plots/eda/pollutant_distributions.png"),
        ("Correlation Matrix Heatmap", "plots/eda/correlation_heatmap.png"),
        ("Seasonal Distribution of AQI", "plots/eda/seasonal_aqi_distribution.png"),
        ("AQI Category Frequency", "plots/eda/aqi_category_distribution.png"),
        ("Atmospheric Driver Regressions", "plots/eda/pollutant_aqi_relationships.png")
    ]
    
    for title, img_path in eda_plots:
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path, use_container_width=True)
            st.markdown("---")

# TAB 3: Model Performance & Comparison
with tabs[2]:
    st.header("📈 Model Evaluation & Comparative Analysis")
    st.write("Benchmarking parametric linear baselines against non-linear decision trees and ensemble random forests.")
    
    if models_loaded:
        st.subheader("1. Regression Task Evaluation (Continuous AQI)")
        reg_df = pd.DataFrame(metrics['regression_metrics']).T
        st.dataframe(reg_df[['Test_R2', 'Test_RMSE', 'Test_MAE', 'CV_R2_Mean', 'CV_R2_Std']].style.highlight_max(subset=['Test_R2'], color='#dcfce7').highlight_min(subset=['Test_RMSE', 'Test_MAE'], color='#dcfce7'))
        
        st.subheader("2. Classification Task Evaluation (AQI Health Category)")
        clf_df = pd.DataFrame(metrics['classification_metrics']).T
        st.dataframe(clf_df[['Test_Accuracy', 'Test_F1_Weighted', 'Test_Precision_Weighted', 'Test_Recall_Weighted', 'CV_Accuracy_Mean']].style.highlight_max(subset=['Test_Accuracy', 'Test_F1_Weighted'], color='#dcfce7'))
        
    st.markdown("---")
    res_plots = [
        ("Actual vs Predicted AQI Regression Scatter", "plots/results/actual_vs_predicted_regression.png"),
        ("Residual Error Distribution & Diagnostics", "plots/results/residual_analysis.png"),
        ("Regression Models Comparison", "plots/results/regression_model_comparison.png"),
        ("Confusion Matrix Heatmap", "plots/results/confusion_matrix_classifier.png"),
        ("Classification Performance Comparison", "plots/results/classification_model_comparison.png"),
        ("Feature Importance Ranking", "plots/results/feature_importance_ranking.png")
    ]
    
    for title, img_path in res_plots:
        if os.path.exists(img_path):
            st.subheader(title)
            st.image(img_path, use_container_width=True)
            st.markdown("---")

# TAB 4: Architecture & Dataset
with tabs[3]:
    st.header("🏗️ System Architecture & Dataset Inspection")
    st.markdown("""
    ### Machine Learning Pipeline Flow
    1. **Data Ingestion**: Multi-station historical sensor monitoring (PM2.5, PM10, NO2, SO2, CO, O3, Temp, Humidity, Wind).
    2. **Data Preprocessing**: Deduplication, station-season stratified median imputation, and temporal parsing.
    3. **Domain Feature Engineering**: Fine-fraction ratio ($PM_{2.5}/PM_{10}$), Ventilation proxy, Heat-moisture index.
    4. **Data Splitting**: 80/20 train-test split strictly executed before scaling to prevent leakage.
    5. **Model Benchmarking**: 
       - Regression: Linear Regression, Decision Tree, Random Forest Regressor.
       - Classification: Logistic Regression, Random Forest Classifier.
    6. **Validation**: 5-Fold Cross Validation.
    7. **Deployment**: Saved artifacts, CLI inference, and interactive web dashboard.
    """)
    
    if os.path.exists('data/air_quality_data_clean.csv'):
        st.subheader("Cleaned Dataset Sample (First 20 records)")
        sample_data = pd.read_csv('data/air_quality_data_clean.csv').head(20)
        st.dataframe(sample_data)
