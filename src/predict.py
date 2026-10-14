"""Command-line AQI estimate from daily readings, optionally enriched by hourly data."""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import HOURLY_FEATURE_COLUMNS, POLLUTANT_COLUMNS, aggregate_hourly_features


class AQIPredictor:
    def __init__(self, models_dir=None):
        models_dir = models_dir or os.path.join(PROJECT_DIR, 'models')
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        self.regressor = joblib.load(os.path.join(models_dir, 'aqi_regressor_extra_trees.pkl'))
        self.classifier = joblib.load(os.path.join(models_dir, 'aqi_classifier_rf.pkl'))
        self.hourly_regressor = joblib.load(os.path.join(models_dir, 'aqi_regressor_extra_trees_hourly.pkl'))
        self.hourly_classifier = joblib.load(os.path.join(models_dir, 'aqi_classifier_rf_hourly.pkl'))

    def predict(self, input_dict, hourly_readings=None):
        enhanced = hourly_readings is not None
        active_regressor = self.hourly_regressor if enhanced else self.regressor
        active_classifier = self.hourly_classifier if enhanced else self.classifier
        model_input = dict(input_dict)
        if enhanced:
            summaries = aggregate_hourly_features(hourly_readings)
            target_date = pd.to_datetime(input_dict['Date']).normalize()
            match = summaries[
                summaries['City'].astype(str).eq(str(input_dict['City']).strip())
                & summaries['Date'].eq(target_date)
            ]
            if match.empty:
                raise ValueError('Hourly readings contain no rows for the requested city and date.')
            summaries = match.iloc[0]
            model_input.update(summaries[HOURLY_FEATURE_COLUMNS].to_dict())
            for pollutant in POLLUTANT_COLUMNS:
                hourly_mean = summaries.get(f'Hourly_{pollutant}_mean')
                if pd.notna(hourly_mean):
                    model_input[pollutant] = float(hourly_mean)
        X = pd.DataFrame([model_input])
        predicted_aqi = float(active_regressor.predict(X)[0])
        predicted_label = int(active_classifier.predict(X)[0])
        category = self.label_encoder.inverse_transform([predicted_label])[0]
        probabilities = active_classifier.predict_proba(X)[0]
        return {
            'predicted_aqi': round(predicted_aqi, 1),
            'predicted_category': category,
            'top_class_probability_not_calibrated': round(float(np.max(probabilities)), 3),
            'model': 'hourly_enhanced' if enhanced else 'daily_only',
            'category_probabilities': {
                label: round(float(prob), 3)
                for label, prob in zip(
                    self.label_encoder.inverse_transform(self.classifier.classes_.astype(int)),
                    probabilities
                )
            }
        }


def run_sample_predictions(hourly_csv=None):
    predictor = AQIPredictor()
    sample = {
        'Date': '2019-11-15', 'City': 'Delhi',
        'PM2.5': 120.0, 'PM10': 210.0, 'NO': 30.0, 'NO2': 45.0,
        'NOx': 70.0, 'NH3': 25.0, 'CO': 1.2, 'SO2': 15.0,
        'O3': 35.0, 'Benzene': 4.0, 'Toluene': 8.0
    }
    hourly_readings = pd.read_csv(hourly_csv, low_memory=False) if hourly_csv else None
    result = predictor.predict(sample, hourly_readings=hourly_readings)
    print('AQI MODEL DEMONSTRATION (historical city-day training data)')
    print(f"Inputs: city={sample['City']}, date={sample['Date']}")
    print(f"Estimated AQI: {result['predicted_aqi']}")
    print(f"Estimated category: {result['predicted_category']}")
    print(f"Model mode: {result['model']}")
    print(f"Top class probability (uncalibrated): {result['top_class_probability_not_calibrated']:.1%}")
    print('This is a model estimate, not a measured value, forecast, or official advisory.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demonstrate a daily AQI estimate.')
    parser.add_argument(
        '--hourly-csv',
        help='Optional hourly readings CSV; use the enhanced daily model when a city/date match is present.',
    )
    run_sample_predictions(parser.parse_args().hourly_csv)
