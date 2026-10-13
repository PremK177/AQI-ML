"""Create the walkthrough notebook for the city-day AQI project."""

import json
from pathlib import Path


def markdown(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': source.splitlines(keepends=True)}


cells = [
    markdown("""# AQI Estimation with India City-Day Data

This walkthrough uses the supplied daily air-quality observations for 26 Indian cities (2015–2020). It compares same-day AQI regression and AQI category classification.

> The dataset contains observed AQI labels and substantial pollutant missingness. Results describe agreement with the dataset labels in a later date period; they do not validate a live service, give an official AQI reading, or forecast future air quality.
"""),
    code("""from pathlib import Path
import os
import sys
import json
import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from preprocess import load_and_inspect_raw_data, clean_data, prepare_and_split_data
from train_models import train_and_evaluate_all
from predict import AQIPredictor
"""),
    markdown("""## 1. Inspect the dataset

The CSV has 29,531 rows, 26 cities, and dates from 2015-01-01 to 2020-07-01. The target is present on 24,850 rows. Xylene is omitted from the primary predictors because more than 60% of its values are missing. Other pollutant readings are retained as missing until each model fold fits its own imputer.
"""),
    code("""raw_df = load_and_inspect_raw_data('data/city_day.csv')
raw_df.head()
"""),
    code("""missing_pct = (100 * raw_df.isna().mean()).sort_values(ascending=False).round(1)
missing_pct.rename('Missing (%)').to_frame()
"""),
    markdown("""## 2. Prepare a chronological holdout

Rows without either AQI target are removed. The latest 20% of distinct dates form the test period; all cities recorded on a date stay together. The number of rows is uneven across dates, so the resulting row counts are not 80/20.
"""),
    code("""clean_df = clean_data(raw_df)
split = prepare_and_split_data(clean_df, test_size=0.2)
print('Train rows:', len(split['X_train']))
print('Test rows:', len(split['X_test']))
print('Test begins:', pd.Timestamp(split['test_start']).date())
print('Train cities:', split['X_train']['City'].nunique())
print('Test cities:', split['X_test']['City'].nunique())
"""),
    markdown("""## 3. Train and compare models

Each pipeline learns City-by-Season pollutant medians and global fallbacks from the current training fold, adds missingness and calendar features, encodes categories, and scales. Five expanding date folds validate model choices before the final later-date holdout is scored. The Random Forest classifier uses balanced class weights, selected by training-period macro-F1 validation.
"""),
    code("""results = train_and_evaluate_all()
"""),
    code("""regression = pd.DataFrame(results['reg_results']).T
regression[['CV_R2_Mean', 'CV_R2_Std', 'Test_R2', 'Test_RMSE', 'Test_MAE']].sort_values('Test_R2', ascending=False)
"""),
    code("""classification = pd.DataFrame(results['clf_results']).T
classification[['CV_Accuracy_Mean', 'CV_F1_Macro_Mean', 'Test_Accuracy', 'Test_F1_Weighted', 'Test_F1_Macro']]
"""),
    markdown("""## 4. Check performance by AQI category

Macro F1 gives each category equal weight. Weighted F1 gives more influence to common categories. Per-class recall and support show where the model still misses labels.
"""),
    code("""rf_per_class = results['clf_results']['Random Forest Classifier']['Per_Class']
pd.DataFrame(rf_per_class).T.rename_axis('AQI category')
"""),
    code("""display(Image(filename='plots/results/confusion_matrix_classifier.png'))
display(Image(filename='plots/results/actual_vs_predicted_regression.png'))
"""),
    markdown("""## 5. Try a same-day estimate

Provide a city, date, and pollutant readings. The saved pipeline applies fold-fitted transformations learned from the training period. Class probabilities are not calibrated.
"""),
    code("""predictor = AQIPredictor()
observation = {
    'Date': '2019-11-15', 'City': 'Delhi',
    'PM2.5': 120.0, 'PM10': 210.0, 'NO': 30.0, 'NO2': 45.0,
    'NOx': 70.0, 'NH3': 25.0, 'CO': 1.2, 'SO2': 15.0,
    'O3': 35.0, 'Benzene': 4.0, 'Toluene': 8.0
}
predictor.predict(observation)
"""),
    markdown("""## Conclusion

The chronological Random Forest results are strongest in this benchmark, but the available data has substantial missingness, uneven city coverage, and extreme AQI values. A stronger next evaluation would validate the data's provenance and units, test against independent monitoring records, and hold out entire cities. This pipeline estimates AQI for supplied same-day readings; future forecasting would require a separate future-horizon target.
""")
]

notebook = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.x'}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

output = Path(__file__).with_name('aqi_prediction_walkthrough.ipynb')
output.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
print(f'Wrote {output}')
