"""Leakage-safe preprocessing for the India city-day air-quality dataset."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Xylene is excluded from the primary model because 61.3% of its values are
# missing. The other pollutants are retained and imputed within each fit fold.
POLLUTANT_COLUMNS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
    'Benzene', 'Toluene'
]
AQI_CLASSES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']


def load_and_inspect_raw_data(filepath='data/city_day.csv'):
    """Load the supplied city-day CSV and print basic quality checks."""
    df = pd.read_csv(filepath)
    print('=' * 60)
    print('CITY-DAY DATASET INSPECTION')
    print('=' * 60)
    print(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Exact duplicate rows: {df.duplicated().sum()}')
    print(f'Date range: {df["Date"].min()} to {df["Date"].max()}')
    print(f'Cities: {df["City"].nunique()}')
    print('Missing percentage by column:')
    print((100 * df.isna().mean()).round(1).to_string())
    return df


def season_from_month(month):
    """Return broad Indian meteorological seasons from calendar month."""
    if month in (12, 1, 2):
        return 'Winter'
    if month in (3, 4, 5):
        return 'Summer'
    if month in (6, 7, 8, 9):
        return 'Monsoon'
    return 'Post-Monsoon'


def clean_data(df, impute=False):
    """Normalize dates/numerics and retain rows with both modeling targets."""
    del impute  # Kept for compatibility with earlier scripts; no global imputation.
    required = {'City', 'Date', 'AQI', 'AQI_Bucket'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'City-day dataset is missing required columns: {sorted(missing)}')

    out = df.drop_duplicates().copy()
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
    out['City'] = out['City'].astype('string').str.strip()
    out['AQI_Bucket'] = out['AQI_Bucket'].astype('string').str.strip()
    numeric_cols = [c for c in POLLUTANT_COLUMNS + ['Xylene', 'AQI'] if c in out]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce')
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=['Date', 'City', 'AQI', 'AQI_Bucket'])
    out = out[out['AQI_Bucket'].isin(AQI_CLASSES)]
    out['Season'] = out['Date'].dt.month.map(season_from_month)
    out = out.sort_values(['Date', 'City']).reset_index(drop=True)
    return out


class AQIPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-fitted city-season imputation, feature creation, encoding, scaling."""

    def fit(self, X, y=None):
        frame = X.copy()
        self.pollutant_columns_ = [c for c in POLLUTANT_COLUMNS if c in frame.columns]
        missing = set(POLLUTANT_COLUMNS).difference(frame.columns)
        if missing:
            # Old or reduced inputs can omit predictors; keep only available
            # columns while enforcing that the model has at least one pollutant.
            if not self.pollutant_columns_:
                raise ValueError('At least one supported pollutant feature is required')
        self.input_columns_ = ['Date', 'City'] + self.pollutant_columns_
        frame = self._prepare_context(frame)
        self.city_season_medians_ = {}
        self.global_medians_ = {}
        for col in self.pollutant_columns_:
            self.global_medians_[col] = float(frame[col].median())
            if not np.isfinite(self.global_medians_[col]):
                raise ValueError(f'No observed training values for pollutant {col}')
            self.city_season_medians_[col] = frame.groupby(['City', 'Season'], observed=True)[col].median().to_dict()
        frame = self._impute(frame)
        features = self._make_features(frame, fit=True)
        self.feature_names_out_ = list(features.columns)
        self.scaler_ = StandardScaler().fit(features)
        return self

    def transform(self, X):
        frame = X.copy()
        for col in self.input_columns_:
            if col not in frame:
                frame[col] = np.nan
        frame = frame[self.input_columns_]
        frame = self._prepare_context(frame)
        frame = self._impute(frame)
        features = self._make_features(frame, fit=False)
        features = features.reindex(columns=self.feature_names_out_, fill_value=0.0)
        return self.scaler_.transform(features)

    def _impute(self, frame):
        frame = frame.copy()
        for col in self.pollutant_columns_:
            missing = frame[col].isna()
            if missing.any():
                lookup = self.city_season_medians_[col]
                fill_values = [
                    lookup.get((city, season), self.global_medians_[col])
                    for city, season in zip(frame.loc[missing, 'City'], frame.loc[missing, 'Season'])
                ]
                frame.loc[missing, col] = fill_values
            frame[col] = frame[col].fillna(self.global_medians_[col])
        return frame

    def _prepare_context(self, frame):
        frame = frame.copy()
        frame['Date'] = pd.to_datetime(frame['Date'], errors='coerce')
        frame['Season'] = frame['Date'].dt.month.map(season_from_month)
        frame['City'] = frame['City'].astype('string').fillna('Unknown')
        for col in self.pollutant_columns_:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)
            frame[f'{col}_WasMissing'] = frame[col].isna().astype(float)
        return frame

    def _make_features(self, frame, fit):
        df = frame.copy()
        dates = pd.to_datetime(df['Date'], errors='coerce')
        df['Month'] = dates.dt.month.astype(float)
        df['DayOfWeek'] = dates.dt.dayofweek.astype(float)
        df['Is_Weekend'] = dates.dt.dayofweek.ge(5).astype(float)
        df = df.drop(columns=['Date'])
        df = pd.get_dummies(df, columns=['City', 'Season'], drop_first=True, dtype=float)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)


def prepare_and_split_data(df, test_size=0.2):
    """Create a chronological holdout, keeping all same-date city rows together."""
    feature_cols = ['Date', 'City'] + [c for c in POLLUTANT_COLUMNS if c in df.columns]
    if len(feature_cols) <= 2:
        raise ValueError('No supported pollutant columns found in the dataset')
    X = df[feature_cols].copy()
    y_reg = df['AQI'].astype(float).copy()
    label_encoder = LabelEncoder().fit(AQI_CLASSES)
    y_clf = label_encoder.transform(df['AQI_Bucket'].astype(str))

    dates = pd.to_datetime(X['Date']).dt.normalize()
    unique_dates = np.sort(dates.unique())
    split_at = int(np.floor(len(unique_dates) * (1 - test_size)))
    if split_at <= 0 or split_at >= len(unique_dates):
        raise ValueError('test_size must leave dates in both train and test partitions')
    test_start = unique_dates[split_at]
    train_mask = dates < test_start
    X_train, X_test = X.loc[train_mask].copy(), X.loc[~train_mask].copy()
    y_train_reg, y_test_reg = y_reg.loc[train_mask], y_reg.loc[~train_mask]
    y_train_clf = y_clf[train_mask.to_numpy()]
    y_test_clf = y_clf[(~train_mask).to_numpy()]

    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    print(f'Total labeled samples: {len(X)}')
    print(f'Training samples: {len(X_train)}; test samples: {len(X_test)}; test begins {pd.Timestamp(test_start).date()}')
    print(f'Predictor columns ({len(feature_cols) - 2} pollutants): {feature_cols[2:]}')
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train_reg': y_train_reg, 'y_test_reg': y_test_reg,
        'y_train_clf': y_train_clf, 'y_test_clf': y_test_clf,
        'label_encoder': label_encoder, 'test_start': test_start,
        'feature_cols': feature_cols, 'full_clean_df': df
    }
