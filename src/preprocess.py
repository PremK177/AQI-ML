"""
Data Preprocessing and Feature Engineering Module for AQI Prediction
Handles duplicate removal, fold-fitted missing value imputation, chronological splitting,
feature scaling, and feature engineering while strictly preventing data leakage.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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

def clean_data(df, impute=False):
    """
    Removes duplicate rows, parses dates, sorts chronologically, and drops
    rows without targets. Optional imputation is only for descriptive exports;
    model training passes impute=False and learns medians inside each fold.
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
    
    # Keep target-bearing rows only. Predictor imputation is learned later from
    # each training fold by AQIPreprocessor, so validation/test statistics cannot leak.
    df_clean = df_clean.dropna(subset=['AQI', 'AQI_Bucket']).reset_index(drop=True)
    if impute:
        # Optional descriptive-data export only. Model training uses impute=False.
        numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']
        for col in numerical_cols:
            group_medians = df_clean.groupby(['Station', 'Season'])[col].transform('median')
            df_clean[col] = df_clean[col].fillna(group_medians)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        print(f"Missing values after descriptive imputation: {df_clean.isnull().sum().sum()}")
    else:
        print(f"Predictor missing values retained for train-fold imputation: {df_clean.isnull().sum().sum()}")
    
    return df_clean


class AQIPreprocessor(BaseEstimator, TransformerMixin):
    """Fit grouped-median imputation, feature creation, encoding, and scaling per fold."""

    SENSOR_COLUMNS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']

    def fit(self, X, y=None):
        frame = X.copy()
        self.input_columns_ = list(frame.columns)
        self.sensor_columns_ = [c for c in self.SENSOR_COLUMNS if c in frame.columns]
        self.group_medians_ = {}
        self.global_medians_ = {}
        for col in self.sensor_columns_:
            self.global_medians_[col] = float(frame[col].median())
            if not np.isfinite(self.global_medians_[col]):
                raise ValueError(f'No observed training values available for sensor feature {col}')
            self.group_medians_[col] = frame.groupby(['Station', 'Season'])[col].median().to_dict()
        features = self._make_features(frame)
        self.feature_names_out_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(features)
        return self

    def transform(self, X):
        frame = X.copy()
        missing_context = [c for c in ['Date', 'Station', 'Season'] if c not in frame.columns]
        if missing_context:
            raise ValueError(f'Missing required AQI context fields: {missing_context}')
        for col in self.input_columns_:
            if col not in frame:
                frame[col] = np.nan
        frame = frame[self.input_columns_]
        for col in self.sensor_columns_:
            missing = frame[col].isna()
            if missing.any():
                lookup = self.group_medians_[col]
                replacement = [
                    lookup.get((station, season), self.global_medians_[col])
                    for station, season in zip(frame.loc[missing, 'Station'], frame.loc[missing, 'Season'])
                ]
                frame.loc[missing, col] = replacement
            frame[col] = frame[col].fillna(self.global_medians_[col])
        features = self._make_features(frame).reindex(columns=self.feature_names_out_, fill_value=0.0)
        return self.scaler_.transform(features)

    def _make_features(self, frame):
        df = frame.copy()
        dates = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = dates.dt.month
        df['DayOfWeek'] = dates.dt.dayofweek
        df['Is_Weekend'] = dates.dt.dayofweek.ge(5).astype(int)
        df['PM_Ratio'] = np.clip(df['PM2.5'] / (df['PM10'] + 1e-5), 0.05, 0.95)
        df['Dispersion_Index'] = (df['Wind_Speed'] + 1.0) / (df['PM2.5'] + 1.0)
        df['Heat_Moisture_Index'] = (df['Temperature'] * df['Humidity']) / 100.0
        df = df.drop(columns=['Date'])
        return pd.get_dummies(df, columns=['Station', 'Season'], drop_first=True, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

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

def prepare_and_split_data(df, test_size=0.2):
    """
    Prepares feature matrix X and targets y (regression AQI & classification AQI_Bucket).
    Uses a chronological holdout. All learned preprocessing lives in model pipelines.
    """
    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLITTING & PREPROCESSING ORDERING")
    print("=" * 60)
    
    # Preserve raw measurements and Date; the model pipeline fits all
    # preprocessing independently in each chronological CV fold.
    feature_cols = ['Date', 'Station', 'Season'] + AQIPreprocessor.SENSOR_COLUMNS
    X = df[feature_cols].copy()
    y_reg = df['AQI'].copy()
    y_clf_raw = df['AQI_Bucket'].copy()
    
    # Encode categorical target for classification
    label_encoder = LabelEncoder()
    # Define bucket order
    bucket_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    label_encoder.fit(bucket_order)
    y_clf = label_encoder.transform(y_clf_raw)
    
    # Hold out the latest dates so testing measures performance on a later period.
    unique_dates = np.sort(pd.to_datetime(X['Date']).dt.normalize().unique())
    split_at = int(np.floor(len(unique_dates) * (1 - test_size)))
    if split_at <= 0 or split_at >= len(unique_dates):
        raise ValueError('test_size must leave at least one date in both train and test partitions')
    test_start = unique_dates[split_at]
    train_mask = pd.to_datetime(X['Date']).dt.normalize() < test_start
    X_train, X_test = X.loc[train_mask].copy(), X.loc[~train_mask].copy()
    y_train_reg, y_test_reg = y_reg.loc[train_mask], y_reg.loc[~train_mask]
    y_train_clf, y_test_clf = y_clf[train_mask.to_numpy()], y_clf[(~train_mask).to_numpy()]
    
    print(f"Total samples: {len(X)}")
    print(f"Training set:  {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"Testing set:   {len(X_test)} samples ({test_size*100:.0f}%), from {pd.Timestamp(test_start).date()}")
    print(f"Raw input columns: {list(feature_cols)}")

    # The encoder is target metadata; feature preprocessing is bundled in each model pipeline.
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    print("Saved target encoder. Imputation, feature creation, encoding, and scaling are fit inside each model pipeline.")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'test_start': test_start,
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
    clean_df = clean_data(raw_df, impute=False)
    feat_df = engineer_features(clean_df)
    splits = prepare_and_split_data(feat_df)
    print("\nPreprocessing pipeline executed successfully!")
