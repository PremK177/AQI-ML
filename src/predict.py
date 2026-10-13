"""Command-line AQI estimate from city and daily pollutant readings."""

import os
import sys
import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class AQIPredictor:
    def __init__(self, models_dir=None):
        models_dir = models_dir or os.path.join(PROJECT_DIR, 'models')
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
        self.regressor = joblib.load(os.path.join(models_dir, 'aqi_regressor_rf.pkl'))
        self.classifier = joblib.load(os.path.join(models_dir, 'aqi_classifier_rf.pkl'))

    def predict(self, input_dict):
        X = pd.DataFrame([input_dict])
        predicted_aqi = float(self.regressor.predict(X)[0])
        predicted_label = int(self.classifier.predict(X)[0])
        category = self.label_encoder.inverse_transform([predicted_label])[0]
        probabilities = self.classifier.predict_proba(X)[0]
        return {
            'predicted_aqi': round(predicted_aqi, 1),
            'predicted_category': category,
            'top_class_probability_not_calibrated': round(float(np.max(probabilities)), 3),
            'category_probabilities': {
                label: round(float(prob), 3)
                for label, prob in zip(
                    self.label_encoder.inverse_transform(self.classifier.classes_.astype(int)),
                    probabilities
                )
            }
        }


def run_sample_predictions():
    predictor = AQIPredictor()
    sample = {
        'Date': '2019-11-15', 'City': 'Delhi',
        'PM2.5': 120.0, 'PM10': 210.0, 'NO': 30.0, 'NO2': 45.0,
        'NOx': 70.0, 'NH3': 25.0, 'CO': 1.2, 'SO2': 15.0,
        'O3': 35.0, 'Benzene': 4.0, 'Toluene': 8.0
    }
    result = predictor.predict(sample)
    print('AQI MODEL DEMONSTRATION (historical city-day training data)')
    print(f"Inputs: city={sample['City']}, date={sample['Date']}")
    print(f"Estimated AQI: {result['predicted_aqi']}")
    print(f"Estimated category: {result['predicted_category']}")
    print(f"Top class probability (uncalibrated): {result['top_class_probability_not_calibrated']:.1%}")
    print('This is a model estimate, not a measured value, forecast, or official advisory.')


if __name__ == '__main__':
    run_sample_predictions()
