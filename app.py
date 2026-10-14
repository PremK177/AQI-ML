"""
Streamlit Web Application: Air Quality Index (AQI) Estimation & Analysis
Interactive dashboard for daily estimates from supplied historical-style inputs,
exploratory data analysis, and model performance benchmarking.
"""

import io
import os
import json
import joblib
import numpy as np
import pandas as pd
import sys
import streamlit as st

from src import preprocess as project_preprocess

POLLUTANT_COLUMNS = project_preprocess.POLLUTANT_COLUMNS
HOURLY_SUMMARY_STATS = ('mean', 'median', 'std', 'max', 'min', 'count')
HOURLY_FEATURE_COLUMNS = getattr(
    project_preprocess,
    'HOURLY_FEATURE_COLUMNS',
    [
        f'Hourly_{pollutant}_{stat}'
        for pollutant in POLLUTANT_COLUMNS
        for stat in HOURLY_SUMMARY_STATS
    ],
)


def _aggregate_hourly_features_compat(hourly_df):
    """Support deployments whose preprocess module predates hourly summaries."""
    datetime_column = 'Datetime' if 'Datetime' in hourly_df.columns else 'Date'
    required = {'City', datetime_column}
    missing = required.difference(hourly_df.columns)
    if missing:
        raise ValueError(f'Hourly data is missing required columns: {sorted(missing)}')

    frame = hourly_df[['City', datetime_column] + [
        column for column in POLLUTANT_COLUMNS if column in hourly_df
    ]].copy()
    frame = frame.rename(columns={datetime_column: 'Date'})
    frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce').dt.normalize()
    frame['City'] = frame['City'].astype('string').str.strip()
    frame = frame.dropna(subset=['Date', 'City'])
    for column in POLLUTANT_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors='coerce').replace(
            [np.inf, -np.inf], np.nan
        )

    summaries = frame.groupby(['City', 'Date'], observed=True)[POLLUTANT_COLUMNS].agg(
        list(HOURLY_SUMMARY_STATS)
    )
    summaries.columns = [
        f'Hourly_{pollutant}_{stat}' for pollutant, stat in summaries.columns
    ]
    return summaries.reset_index()


aggregate_hourly_features = getattr(
    project_preprocess,
    'aggregate_hourly_features',
    _aggregate_hourly_features_compat,
)

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

@st.cache_resource
def load_models():
    label_encoder = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))
    regressor = joblib.load(os.path.join(BASE_DIR, 'models', 'aqi_regressor_extra_trees.pkl'))
    classifier = joblib.load(os.path.join(BASE_DIR, 'models', 'aqi_classifier_rf.pkl'))
    hourly_regressor = joblib.load(os.path.join(BASE_DIR, 'models', 'aqi_regressor_extra_trees_hourly.pkl'))
    hourly_classifier = joblib.load(os.path.join(BASE_DIR, 'models', 'aqi_classifier_rf_hourly.pkl'))
    with open(os.path.join(BASE_DIR, 'models', 'evaluation_metrics.json'), 'r') as f:
        metrics = json.load(f)
    return label_encoder, regressor, classifier, hourly_regressor, hourly_classifier, metrics


@st.cache_data(show_spinner=False, ttl=3600, max_entries=3)
def hourly_features_for_selection(csv_bytes, selected_city, selected_date):
    header = pd.read_csv(io.BytesIO(csv_bytes), nrows=0)
    time_column = 'Datetime' if 'Datetime' in header.columns else 'Date'
    if 'City' not in header.columns or time_column not in header.columns:
        raise ValueError('The CSV needs City and Datetime columns (or City and Date).')
    usecols = ['City', time_column] + [c for c in POLLUTANT_COLUMNS if c in header.columns]
    uploaded_frame = pd.read_csv(io.BytesIO(csv_bytes), usecols=usecols, low_memory=False)
    parsed_dates = pd.to_datetime(uploaded_frame[time_column], errors='coerce').dt.normalize()
    city_mask = uploaded_frame['City'].astype('string').str.strip().eq(selected_city)
    date_mask = parsed_dates.eq(pd.Timestamp(selected_date))
    selected_hours = uploaded_frame.loc[city_mask & date_mask].copy()
    if selected_hours.empty:
        return None
    daily_summary = aggregate_hourly_features(selected_hours)
    return {
        'row_count': len(selected_hours),
        'features': daily_summary.iloc[0][HOURLY_FEATURE_COLUMNS].to_dict(),
    }

try:
    label_encoder, regressor, classifier, hourly_regressor, hourly_classifier, metrics = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure train_models.py has executed.")
    models_loaded = False

# Sidebar: Input Features, initialized from the supplied historical dataset.
sidebar_data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'air_quality_data_clean.csv'))
sidebar_data['Date'] = pd.to_datetime(sidebar_data['Date'])
st.sidebar.header("City and Daily Pollutant Inputs")
city = st.sidebar.selectbox("City", sorted(sidebar_data['City'].dropna().unique()))
date_input = st.sidebar.date_input(
    "Observation date",
    value=sidebar_data['Date'].max().date(),
    min_value=sidebar_data['Date'].min().date(),
    max_value=sidebar_data['Date'].max().date()
)
st.sidebar.caption("Enter daily pollutant readings. Optionally upload hourly records for a more detailed daily estimate.")
uploaded_hourly = st.sidebar.file_uploader(
    "Hourly readings CSV (optional)", type=['csv'],
    help="Can be the full city_hour.csv or records for one city and date. Needs City, Datetime, and pollutant columns.",
)
hourly_feature_input = None
hourly_daily_means = {}
if uploaded_hourly is not None:
    try:
        selected_summary = hourly_features_for_selection(uploaded_hourly.getvalue(), city, str(date_input))
        if selected_summary is None:
            st.sidebar.warning("No hourly rows matched the selected city and date. The daily-only model will be used.")
        else:
            hourly_feature_input = selected_summary['features']
            hourly_daily_means = {
                pollutant: hourly_feature_input.get(f'Hourly_{pollutant}_mean')
                for pollutant in POLLUTANT_COLUMNS
            }
            st.sidebar.success(f"Using {selected_summary['row_count']} matched hourly records for the enhanced daily estimate.")
    except Exception as exc:
        st.sidebar.warning(f"Could not use the uploaded hourly file: {exc}. The daily-only model will be used.")

input_dict = {'Date': str(date_input), 'City': city}
for pollutant in POLLUTANT_COLUMNS:
    values = sidebar_data[pollutant].dropna()
    center = float(values.median()) if len(values) else 0.0
    upper = float(values.quantile(0.995)) if len(values) else 100.0
    uploaded_mean = hourly_daily_means.get(pollutant)
    default_value = float(uploaded_mean) if pd.notna(uploaded_mean) else center
    input_max = max(upper * 2, center + 1, default_value if pd.notna(uploaded_mean) else 0.0)
    input_dict[pollutant] = st.sidebar.number_input(
        f"{pollutant} concentration",
        min_value=0.0,
        max_value=input_max,
        value=max(default_value, 0.0),
        step=max(default_value / 20, 0.1),
        format="%.2f",
        disabled=pd.notna(uploaded_mean),
        help="When matching hourly data is uploaded, this value is calculated from that day's hourly readings."
    )

# Main Dashboard Layout
st.markdown('<div class="main-header">Air Quality Index (AQI) Prediction & Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Machine Learning Mini-Project | Comparative Modeling of Atmospheric Pollutants</div>', unsafe_allow_html=True)

tabs = st.tabs(["🎯 AQI Scenario Estimate", "📊 Exploratory Data Analysis", "📈 Model Benchmarks & Results", "🏗️ System Architecture & Data"])

# TAB 1: AQI Scenario Estimate
with tabs[0]:
    if models_loaded:
        enhanced_daily_estimate = hourly_feature_input is not None
        active_regressor = hourly_regressor if enhanced_daily_estimate else regressor
        active_classifier = hourly_classifier if enhanced_daily_estimate else classifier
        active_reg_key = 'Extra Trees + Hourly Summaries' if enhanced_daily_estimate else 'Extra Trees'
        model_inputs = {**input_dict, **(hourly_feature_input or {})}
        X_in = pd.DataFrame([model_inputs])
        pred_aqi = float(active_regressor.predict(X_in)[0])
        pred_cat_idx = active_classifier.predict(X_in)[0]
        pred_cat = label_encoder.inverse_transform([pred_cat_idx])[0]
        cat_probs = active_classifier.predict_proba(X_in)[0]
        
        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
        
        with col1:
            st.markdown("### 🔢 Predicted AQI Value")
            st.metric(label="Continuous Regression Output", value=f"{pred_aqi:.1f}")
            st.caption(f"Estimated by {active_reg_key} ($R^2$: {metrics['regression_metrics'][active_reg_key]['Test_R2']})")
            if enhanced_daily_estimate:
                st.caption("Hourly summaries from the uploaded city-date records were included.")
            else:
                st.caption("Upload hourly records for this city and date to use within-day summaries.")
            
        with col2:
            st.markdown("### 🏷️ AQI Category")
            color = CATEGORY_COLORS.get(pred_cat, '#333333')
            st.markdown(f'<div class="category-badge" style="background-color: {color}; color: white;">{pred_cat}</div>', unsafe_allow_html=True)
            max_p = np.max(cat_probs) * 100
            st.caption(f"Top class probability: {max_p:.1f}% (not calibrated)")
            
        with col3:
            st.markdown("### ℹ️ About this category")
            st.info(
                "This is the model's estimate of the category label supplied in the dataset. "
                "The dataset's AQI conventions and provenance have not been independently verified. "
                "Use an official local source for AQI interpretation or health guidance."
            )
            
        st.markdown("---")
        st.subheader("📊 Category Probability Distribution")
        prob_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability (%)': [round(p * 100, 1) for p in cat_probs]
        })
        st.bar_chart(prob_df.set_index('Category'))
        
        st.markdown("---")
        st.caption("Model estimate trained from historical India city-day observations. It is not a measured AQI, future forecast, or official public-health advisory.")

# TAB 2: Exploratory Data Analysis
with tabs[1]:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("Exploration of pollutant distributions, city observations, calendar seasons, missingness, and AQI relationships.")
    
    eda_plots = [
        ("Pollutant Distribution Histograms & KDE", "plots/eda/pollutant_distributions.png"),
        ("Correlation Matrix Heatmap", "plots/eda/correlation_heatmap.png"),
        ("Seasonal Distribution of AQI", "plots/eda/seasonal_aqi_distribution.png"),
        ("AQI Category Frequency", "plots/eda/aqi_category_distribution.png"),
        ("Atmospheric Driver Regressions", "plots/eda/pollutant_aqi_relationships.png")
    ]
    
    for title, img_path in eda_plots:
        full_img_path = os.path.join(BASE_DIR, img_path)
        if os.path.exists(full_img_path):
            st.subheader(title)
            st.image(full_img_path, width="stretch")
            st.markdown("---")

# TAB 3: Model Performance & Comparison
with tabs[2]:
    st.header("📈 Model Evaluation & Comparative Analysis")
    st.write("Benchmarking linear and tree-based models, including tuned ensemble methods.")
    
    if models_loaded:
        st.subheader("1. Regression Task Evaluation (Continuous AQI)")
        reg_df = pd.DataFrame(metrics['regression_metrics']).T
        st.dataframe(reg_df[['Test_R2', 'Test_RMSE', 'Test_MAE', 'CV_R2_Mean', 'CV_R2_Std']].style.highlight_max(subset=['Test_R2'], color='#dcfce7').highlight_min(subset=['Test_RMSE', 'Test_MAE'], color='#dcfce7'))
        
        st.subheader("2. Classification Task Evaluation (AQI Health Category)")
        clf_df = pd.DataFrame(metrics['classification_metrics']).T
        st.dataframe(clf_df[['Test_Accuracy', 'Test_F1_Weighted', 'Test_F1_Macro', 'Test_Precision_Weighted', 'Test_Recall_Weighted', 'CV_Accuracy_Mean']].style.highlight_max(subset=['Test_Accuracy', 'Test_F1_Weighted', 'Test_F1_Macro'], color='#dcfce7'))
        rf_per_class = metrics['classification_metrics']['Random Forest + Hourly Summaries']['Per_Class']
        st.caption("Hourly-enhanced Random Forest per-class scores")
        st.dataframe(pd.DataFrame(rf_per_class).T.rename_axis('AQI category'))
        
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
        full_img_path = os.path.join(BASE_DIR, img_path)
        if os.path.exists(full_img_path):
            st.subheader(title)
            st.image(full_img_path, width="stretch")
            st.markdown("---")

# TAB 4: Architecture & Dataset
with tabs[3]:
    st.header("🏗️ System Architecture & Dataset Inspection")
    st.markdown("""
    ### Machine Learning Pipeline Flow
    1. **Data Ingestion**: City-day observations from 26 Indian cities (2015–2020), with pollutant measurements and AQI labels.
    2. **Data Preprocessing**: Drop rows without daily AQI targets; retain missing readings; fit city-season median imputation, missingness flags, encoding, and scaling inside each model pipeline.
    3. **Feature Engineering**: Calendar features plus optional city-date hourly pollutant mean, median, spread, extrema, and valid-hour counts.
    4. **Data Splitting**: Chronological 80/20 train-test split; the latest dates form the test period.
    5. **Model Benchmarking**: 
       - Regression: Linear Regression, Decision Tree, Random Forest, and Extra Trees, each with daily-only and hourly-enhanced variants where applicable.
       - Classification: Logistic Regression and a tuned, class-balanced Random Forest with daily-only and hourly-enhanced variants.
    6. **Validation**: Five expanding-window folds grouped by date; preprocessing is refit within every fold. Xylene is excluded due to high missingness.
    7. **Deployment**: Saved daily-only and hourly-enhanced artifacts, CLI inference, and interactive web dashboard.
    """)
    
    st.subheader("Dataset sample")
    st.dataframe(sidebar_data.head(20))
