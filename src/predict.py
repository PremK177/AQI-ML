"""
Command-Line Scenario Estimation Module for AQI
Takes pollutant and meteorological readings, performs feature engineering,
and outputs predicted continuous AQI value, AQI Category, and health advisory.
"""

import sys
import os
import joblib
import pandas as pd

# Ensure the custom preprocessing transformer can be imported when saved
# sklearn pipelines are loaded from the project root or from this module.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Health Advisories mapped to AQI categories
HEALTH_ADVISORIES = {
    'Good': {
        'Range': '0 - 50',
        'Impact': 'Minimal impact. Air quality is considered satisfactory, and air pollution poses little or no risk.',
        'Recommendation': 'Ideal air quality for outdoor physical activities and recreation.'
    },
    'Satisfactory': {
        'Range': '51 - 100',
        'Impact': 'Minor breathing discomfort to sensitive people.',
        'Recommendation': 'Safe for normal activities; unusually sensitive individuals should observe mild caution.'
    },
    'Moderate': {
        'Range': '101 - 200',
        'Impact': 'Breathing discomfort to people with lungs, asthma and heart diseases.',
        'Recommendation': 'Children, elderly, and sensitive groups should limit prolonged heavy outdoor exertion.'
    },
    'Poor': {
        'Range': '201 - 300',
        'Impact': 'Breathing discomfort to most people on prolonged exposure.',
        'Recommendation': 'Reduce prolonged or heavy outdoor exertion. Wear N95 masks near traffic.'
    },
    'Very Poor': {
        'Range': '301 - 400',
        'Impact': 'Respiratory illness on prolonged exposure. Significant distress for cardiac patients.',
        'Recommendation': 'Avoid outdoor physical activities. Keep windows closed and use indoor air purifiers.'
    },
    'Severe': {
        'Range': '401 - 500',
        'Impact': 'Affects healthy people and seriously impacts those with existing diseases.',
        'Recommendation': 'Health emergency. Stay indoors, avoid all outdoor exposure, use HEPA purifiers.'
    }
}

class AQIPredictor:
    def __init__(self, models_dir='models'):
        self.label_encoder = joblib.load(f'{models_dir}/label_encoder.pkl')
        self.regressor = joblib.load(f'{models_dir}/aqi_regressor_rf.pkl')
        self.classifier = joblib.load(f'{models_dir}/aqi_classifier_rf.pkl')

    def prepare_input(self, input_dict):
        """Return raw inputs; the saved model pipeline handles transformations."""
        return pd.DataFrame([input_dict])

    def predict(self, input_dict):
        X = self.prepare_input(input_dict)
        pred_aqi = float(self.regressor.predict(X)[0])
        pred_cat_idx = self.classifier.predict(X)[0]
        pred_cat = self.label_encoder.inverse_transform([pred_cat_idx])[0]
        
        # Predict class probabilities
        probs = self.classifier.predict_proba(X)[0]
        prob_dict = {cls: round(float(prob), 3) for cls, prob in zip(self.label_encoder.classes_, probs)}
        
        advisory = HEALTH_ADVISORIES.get(pred_cat, {})
        
        return {
            'predicted_aqi': round(pred_aqi, 1),
            'predicted_category': pred_cat,
            'confidence_score': round(float(np.max(probs)) * 100, 1),
            'category_probabilities': prob_dict,
            'advisory': advisory
        }

def run_sample_predictions():
    predictor = AQIPredictor()
    
    samples = [
        {
            'name': 'Sample 1: Clean Day (Monsoon / Low Traffic)',
            'data': {
                'Date': '2024-07-15',
                'Station': 'Station_South',
                'Season': 'Monsoon',
                'PM2.5': 18.5,
                'PM10': 35.0,
                'NO2': 16.0,
                'SO2': 8.5,
                'CO': 0.45,
                'O3': 22.0,
                'Temperature': 28.5,
                'Humidity': 85.0,
                'Wind_Speed': 16.5
            }
        },
        {
            'name': 'Sample 2: Moderate Urban Day (Summer)',
            'data': {
                'Date': '2024-04-20',
                'Station': 'Station_North',
                'Season': 'Summer',
                'PM2.5': 65.0,
                'PM10': 140.0,
                'NO2': 45.0,
                'SO2': 22.0,
                'CO': 1.10,
                'O3': 58.0,
                'Temperature': 38.0,
                'Humidity': 38.0,
                'Wind_Speed': 11.0
            }
        },
        {
            'name': 'Sample 3: Winter Smog Episode (Severe Pollution)',
            'data': {
                'Date': '2024-12-10',
                'Station': 'Station_North',
                'Season': 'Winter',
                'PM2.5': 285.0,
                'PM10': 460.0,
                'NO2': 110.0,
                'SO2': 48.0,
                'CO': 4.20,
                'O3': 18.0,
                'Temperature': 11.5,
                'Humidity': 78.0,
                'Wind_Speed': 3.2
            }
        }
    ]
    
    print("=" * 70)
    print("AIR QUALITY INDEX PREDICTION INFERENCE SAMPLES")
    print("=" * 70)
    print("Note: models use synthetic data; advisory text is illustrative, not official.")
    
    for item in samples:
        print(f"\n>>> {item['name']}")
        print(f"Inputs: PM2.5={item['data']['PM2.5']}, PM10={item['data']['PM10']}, Wind={item['data']['Wind_Speed']} km/h, Temp={item['data']['Temperature']} C")
        res = predictor.predict(item['data'])
        print(f"-> Predicted AQI Value:    {res['predicted_aqi']} (Continuous Regression)")
        print(f"-> Predicted Category:     {res['predicted_category']} (Top class probability, not calibrated: {res['confidence_score']}%)")
        print(f"-> Category Health Impact: {res['advisory'].get('Impact')}")
        print(f"-> Recommended Action:     {res['advisory'].get('Recommendation')}")

if __name__ == '__main__':
    run_sample_predictions()
