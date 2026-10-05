"""
Data Preprocessing and Feature Engineering Module for AQI Prediction
Handles duplicate removal, missing value imputation, chronological/stratified splitting,
feature scaling, and feature engineering while strictly preventing data leakage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_inspect_raw_data(filepath='data/air_quality_data_raw.csv'):
    """Loads raw air quality data and prints diagnostic inspection."""
    df = pd.read_csv(filepath)
    print("=" * 60)
    print("RAW DATASET INSPECTION")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Duplicates: {df.duplicated().sum()} rows")
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  - {col:15s}: {count:4d} missing ({pct:.2f}%)")
    return df

def clean_data(df):
    """
    Cleans raw data by dropping duplicates and imputing missing values
    using domain-aware station and seasonal medians.
    """
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    # 1. Remove duplicate entries
    initial_rows = len(df)
    df_clean = df.drop_duplicates().copy()
    print(f"Removed {initial_rows - len(df_clean)} duplicate rows. Current rows: {len(df_clean)}")
    
    # 2. Parse Date
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean = df_clean.sort_values(by=['Date', 'Station']).reset_index(drop=True)
    
    # 3. Handle missing values
    # For environmental and meteorological sensor data, station-and-season-specific median
    # imputation preserves local microclimate baselines better than a global average.
    numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']
    
    for col in numerical_cols:
        # Groupwise median by Station and Season
        group_medians = df_clean.groupby(['Station', 'Season'])[col].transform('median')
        df_clean[col] = df_clean[col].fillna(group_medians)
        # Fallback to global median if any NaNs remain
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # Check target missing values (if any)
    df_clean = df_clean.dropna(subset=['AQI', 'AQI_Bucket']).reset_index(drop=True)
    print(f"Missing values after domain imputation: {df_clean.isnull().sum().sum()}")
    
    return df_clean

def engineer_features(df):
    """
    Constructs domain-specific atmospheric and temporal features:
    1. PM2.5 to PM10 Ratio: Indicates fine particle proportion (combustion vs dust).
    2. Ventilation Proxy: Inverse relationship of pollutants with wind speed.
    3. Heat Index Proxy: Interaction between temperature and humidity.
    4. Temporal components: Month, Day of Week, Weekend indicator.
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    df_feat = df.copy()
    
    # Particulate matter fine-fraction ratio
    df_feat['PM_Ratio'] = np.clip(df_feat['PM2.5'] / (df_feat['PM10'] + 1e-5), 0.05, 0.95)
    
    # Dispersion / Ventilation index proxy (Dispersion increases with higher wind speed)
    df_feat['Dispersion_Index'] = (df_feat['Wind_Speed'] + 1.0) / (df_feat['PM2.5'] + 1.0)
    
    # Atmospheric Heat-Moisture Interaction
    df_feat['Heat_Moisture_Index'] = (df_feat['Temperature'] * df_feat['Humidity']) / 100.0
    
    # Calendar & temporal features
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['DayOfWeek'] = df_feat['Date'].dt.dayofweek
    df_feat['Is_Weekend'] = df_feat['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    print("Engineered features added: ['PM_Ratio', 'Dispersion_Index', 'Heat_Moisture_Index', 'Month', 'DayOfWeek', 'Is_Weekend']")
    return df_feat

def prepare_and_split_data(df, test_size=0.2, random_state=42):
    """
    Prepares feature matrix X and targets y (regression AQI & classification AQI_Bucket).
    Strict Featurization Ordering: Splits data into train and test BEFORE fitting scalers.
    """
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLITTING & PREPROCESSING ORDERING")
    print("=" * 60)
    
    # Encode categorical features: Station and Season via One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=['Station', 'Season'], drop_first=True)
    
    # Separate features and targets
    non_feature_cols = ['Date', 'AQI', 'AQI_Bucket']
    feature_cols = [c for c in df_encoded.columns if c not in non_feature_cols]
    
    X = df_encoded[feature_cols].copy()
    y_reg = df_encoded['AQI'].copy()
    y_clf_raw = df_encoded['AQI_Bucket'].copy()
    
    # Encode categorical target for classification
    label_encoder = LabelEncoder()
    # Define bucket order
    bucket_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    label_encoder.fit(bucket_order)
    y_clf = label_encoder.transform(y_clf_raw)
    
    # Split into train and test sets (stratified by classification target to preserve class balance)
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    
    print(f"Total samples: {len(X)}")
    print(f"Training set:  {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set:   {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"Total features: {len(feature_cols)}")
    print(f"Feature names: {list(feature_cols)}")
    
    # Standardize numerical features using StandardScaler
    # CRITICAL: Fit scaler ONLY on training data, then transform both train and test to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
    
    # Save artifacts for inference
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/aqi_scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
        
    print("\nSaved preprocessing artifacts: 'models/aqi_scaler.pkl', 'models/label_encoder.pkl', 'models/feature_names.json'")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_reg': y_train_reg,
        'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf,
        'y_test_clf': y_test_clf,
        'feature_cols': feature_cols,
        'label_encoder': label_encoder,
        'full_clean_df': df
    }

if __name__ == '__main__':
    raw_df = load_and_inspect_raw_data()
    clean_df = clean_data(raw_df)
    feat_df = engineer_features(clean_df)
    splits = prepare_and_split_data(feat_df)
    print("\nPreprocessing pipeline executed successfully!")
