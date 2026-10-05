"""
Air Quality Dataset Generator
Simulates realistic daily air quality monitoring data across multiple stations.
Incorporates seasonal trends, meteorological effects, realistic correlations,
and Indian CPCB / US EPA standard AQI sub-index calculations.
"""

import numpy as np
import pandas as pd
import datetime

np.random.seed(42)

def calculate_sub_index(val, breakpoints):
    """Calculates sub-index using standard piecewise linear interpolation."""
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= val <= c_high:
            return i_low + (i_high - i_low) / (c_high - c_low) * (val - c_low)
    if val > breakpoints[-1][1]:
        return 500.0
    return 0.0

# Standard CPCB Breakpoints: (C_low, C_high, I_low, I_high)
BP_PM25 = [
    (0, 30, 0, 50),
    (30.1, 60, 51, 100),
    (60.1, 90, 101, 200),
    (90.1, 120, 201, 300),
    (120.1, 250, 301, 400),
    (250.1, 500, 401, 500)
]

BP_PM10 = [
    (0, 50, 0, 50),
    (50.1, 100, 51, 100),
    (100.1, 250, 101, 200),
    (250.1, 350, 201, 300),
    (350.1, 430, 301, 400),
    (430.1, 600, 401, 500)
]

BP_NO2 = [
    (0, 40, 0, 50),
    (40.1, 80, 51, 100),
    (80.1, 180, 101, 200),
    (180.1, 280, 201, 300),
    (280.1, 400, 301, 400),
    (400.1, 500, 401, 500)
]

BP_SO2 = [
    (0, 40, 0, 50),
    (40.1, 80, 51, 100),
    (80.1, 380, 101, 200),
    (380.1, 800, 201, 300),
    (800.1, 1600, 301, 400),
    (1600.1, 2000, 401, 500)
]

BP_CO = [
    (0, 1.0, 0, 50),
    (1.01, 2.0, 51, 100),
    (2.01, 10.0, 101, 200),
    (10.01, 17.0, 201, 300),
    (17.01, 34.0, 301, 400),
    (34.01, 50.0, 401, 500)
]

BP_O3 = [
    (0, 50, 0, 50),
    (50.1, 100, 51, 100),
    (100.1, 168, 101, 200),
    (168.1, 208, 201, 300),
    (208.1, 748, 301, 400),
    (748.1, 1000, 401, 500)
]

def get_aqi_bucket(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

def generate_data(num_days=730, stations=['Station_North', 'Station_South', 'Station_East', 'Station_West']):
    start_date = datetime.date(2023, 1, 1)
    records = []

    for station_idx, station in enumerate(stations):
        # Station baseline emission modifiers
        station_bias = (station_idx + 1) * 0.15

        for day in range(num_days):
            current_date = start_date + datetime.timedelta(days=day)
            day_of_year = current_date.timetuple().tm_yday
            month = current_date.month

            # Seasonal factors (in northern hemisphere/subtropical context)
            # Winter: Nov, Dec, Jan, Feb -> High PM, low temp, low wind speed, stagnant air
            # Summer: Mar, Apr, May -> High temp, moderate ozone, moderate dust
            # Monsoon: Jun, Jul, Aug, Sep -> High humidity, rain washout (low PM), moderate temp
            # Post-Monsoon: Oct -> Transition, crop residue, rising PM
            if month in [12, 1, 2]:
                season = 'Winter'
                temp_base = 14.0
                hum_base = 65.0
                wind_base = 7.0
                pm_mult = 2.2
            elif month in [3, 4, 5]:
                season = 'Summer'
                temp_base = 36.0
                hum_base = 40.0
                wind_base = 14.0
                pm_mult = 1.1
            elif month in [6, 7, 8, 9]:
                season = 'Monsoon'
                temp_base = 28.0
                hum_base = 82.0
                wind_base = 16.0
                pm_mult = 0.55
            else:
                season = 'Post-Monsoon'
                temp_base = 24.0
                hum_base = 55.0
                wind_base = 8.5
                pm_mult = 1.8

            # Add stochastic weather variations
            temperature = np.clip(np.random.normal(temp_base, 3.5), 5.0, 48.0)
            humidity = np.clip(np.random.normal(hum_base, 8.0), 15.0, 98.0)
            wind_speed = np.clip(np.random.normal(wind_base, 3.0), 1.5, 32.0)

            # Atmospheric dispersion factor (inverse relation with wind speed)
            dispersion = 12.0 / (wind_speed + 2.0)

            # Generate pollutants
            # PM2.5 and PM10 are heavily affected by season and dispersion
            pm25 = np.clip(np.random.gamma(shape=3.5, scale=18.0) * pm_mult * dispersion * (1 + station_bias) * 0.45, 5.0, 450.0)
            pm10 = np.clip(pm25 * np.random.uniform(1.4, 2.2) + np.random.normal(15, 5), pm25 + 5.0, 600.0)

            # NO2 and CO: vehicle & combustion emissions
            no2 = np.clip(np.random.gamma(shape=3.0, scale=12.0) * dispersion * (1 + station_bias * 0.7) * 0.7, 5.0, 220.0)
            co = np.clip(np.random.gamma(shape=2.2, scale=0.6) * dispersion * (1 + station_bias * 0.5) * 0.6, 0.1, 15.0)

            # SO2: industrial emissions
            so2 = np.clip(np.random.gamma(shape=2.5, scale=5.0) * (1 + station_bias * 1.2), 2.0, 110.0)

            # O3 (Ozone): Photochemical production requires sunlight/heat
            o3_solar = max(0.1, (temperature - 15.0) / 25.0)
            o3 = np.clip(np.random.normal(45.0, 15.0) * o3_solar * (1.0 + no2 / 150.0), 5.0, 240.0)

            # Calculate individual sub-indices
            si_pm25 = calculate_sub_index(pm25, BP_PM25)
            si_pm10 = calculate_sub_index(pm10, BP_PM10)
            si_no2 = calculate_sub_index(no2, BP_NO2)
            si_so2 = calculate_sub_index(so2, BP_SO2)
            si_co = calculate_sub_index(co, BP_CO)
            si_o3 = calculate_sub_index(o3, BP_O3)

            # Standard environmental AQI is determined by the maximum dominant sub-index
            # with slight realistic measurement noise
            raw_aqi = max(si_pm25, si_pm10, si_no2, si_so2, si_co, si_o3)
            aqi = float(np.clip(raw_aqi + np.random.normal(0, 1.5), 10.0, 500.0))
            aqi_bucket = get_aqi_bucket(aqi)

            records.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Station': station,
                'Season': season,
                'PM2.5': round(pm25, 2),
                'PM10': round(pm10, 2),
                'NO2': round(no2, 2),
                'SO2': round(so2, 2),
                'CO': round(co, 2),
                'O3': round(o3, 2),
                'Temperature': round(temperature, 1),
                'Humidity': round(humidity, 1),
                'Wind_Speed': round(wind_speed, 1),
                'AQI': round(aqi, 1),
                'AQI_Bucket': aqi_bucket
            })

    df = pd.DataFrame(records)

    # Save complete clean ground truth for verification
    df.to_csv('data/air_quality_data_clean.csv', index=False)
    print(f"Generated clean dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Inject realistic missing values and slight sensor noise into raw data (~2.5% missing in some columns)
    raw_df = df.copy()
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']
    for col in pollutant_cols:
        mask = np.random.rand(len(raw_df)) < 0.025  # 2.5% missing
        raw_df.loc[mask, col] = np.nan

    # Add a few duplicate rows to demonstrate cleaning
    duplicates = raw_df.sample(n=15, random_state=42)
    raw_df = pd.concat([raw_df, duplicates], ignore_index=True)

    raw_df.to_csv('data/air_quality_data_raw.csv', index=False)
    print(f"Generated raw dataset with missing values and duplicates: {raw_df.shape[0]} rows.")

if __name__ == '__main__':
    generate_data()
