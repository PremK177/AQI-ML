"""
Command-Line Inference Module for AQI Prediction
Takes pollutant and meteorological readings, performs feature engineering,
and outputs predicted continuous AQI value, AQI Category, and health advisory.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import datetime

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
        self.scaler = joblib.load(f'{models_dir}/aqi_scaler.pkl')
        self.label_encoder = joblib.load(f'{models_dir}/label_encoder.pkl')
        self.regressor = joblib.load(f'{models_dir}/aqi_regressor_rf.pkl')
        self.classifier = joblib.load(f'{models_dir}/aqi_classifier_rf.pkl')
        with open(f'{models_dir}/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)

    def prepare_input(self, input_dict):
        """Converts raw input dictionary into transformed, scaled feature vector."""
        df = pd.DataFrame([input_dict])
        
        # Parse date components
        date = pd.to_datetime(df['Date'].iloc[0])
        df['Month'] = date.month
        df['DayOfWeek'] = date.dayofweek
        df['Is_Weekend'] = 1 if date.dayofweek >= 5 else 0
        
        # Domain feature engineering
        df['PM_Ratio'] = np.clip(df['PM2.5'] / (df['PM10'] + 1e-5), 0.05, 0.95)
        df['Dispersion_Index'] = (df['Wind_Speed'] + 1.0) / (df['PM2.5'] + 1.0)
        df['Heat_Moisture_Index'] = (df['Temperature'] * df['Humidity']) / 100.0
        
        # One-Hot Encoding alignment for Station and Season
        for col in self.feature_names:
            if col not in df.columns:
                if col.startswith('Station_'):
                    station_val = col.replace('Station_', '')
                    df[col] = 1 if df['Station'].iloc[0] == station_val else 0
                elif col.startswith('Season_'):
                    season_val = col.replace('Season_', '')
                    df[col] = 1 if df['Season'].iloc[0] == season_val else 0
                else:
                    df[col] = 0.0

        # Filter and order exactly as during training
        X = df[self.feature_names].copy()
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_names)
        return X_scaled

    def predict(self, input_dict):
        X_scaled = self.prepare_input(input_dict)
        pred_aqi = float(self.regressor.predict(X_scaled)[0])
        pred_cat_idx = self.classifier.predict(X_scaled)[0]
        pred_cat = self.label_encoder.inverse_transform([pred_cat_idx])[0]
        
        # Predict class probabilities
        probs = self.classifier.predict_proba(X_scaled)[0]
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
    
    for item in samples:
        print(f"\n>>> {item['name']}")
        print(f"Inputs: PM2.5={item['data']['PM2.5']}, PM10={item['data']['PM10']}, Wind={item['data']['Wind_Speed']} km/h, Temp={item['data']['Temperature']} C")
        res = predictor.predict(item['data'])
        print(f"-> Predicted AQI Value:    {res['predicted_aqi']} (Continuous Regression)")
        print(f"-> Predicted Category:     {res['predicted_category']} (Confidence: {res['confidence_score']}%)")
        print(f"-> Category Health Impact: {res['advisory'].get('Impact')}")
        print(f"-> Recommended Action:     {res['advisory'].get('Recommendation')}")

if __name__ == '__main__':
    run_sample_predictions()
