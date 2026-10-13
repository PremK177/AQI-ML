# AQI Estimation and Category Classification

This project compares regression and classification models using the supplied India city-day air-quality dataset. It estimates the AQI and its supplied category from same-day pollutant readings.

> **Scope:** These are estimates from historical observations, not future forecasts, official AQI calculations, or public-health advice. AQI and category labels are present in the dataset. Scores measure agreement with those labels on a later time period; they do not establish performance for a live monitoring service.

## Dataset

The active input file is [`data/city_day.csv`](data/city_day.csv). It contains **29,531** city-day rows from **2015-01-01 to 2020-07-01**, for **26 Indian cities**. There are no exact duplicate rows. AQI and category are both present for **24,850** rows; the other rows are excluded from the two supervised tasks because a target is missing.

The pollutant fields are PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, and Xylene. Missingness is uneven: PM10 is missing in 37.7% of raw rows, NH3 in 35.0%, and Xylene in 61.3%. Xylene is excluded from the primary models due to its high missingness. Other selected pollutant readings remain missing until each training fold fits its imputer. The target AQI contains values up to 2049, so the dataset's labels and outliers are retained as supplied rather than silently clipped.

The CSV does not include temperature, humidity, or wind speed; those fields from the earlier synthetic version are no longer used. A broad season is derived from each date for city-season imputation and analysis. The original synthetic CSVs and generator remain in `data/` for reference, but they are not used by the current training pipeline.

## Modeling tasks

- **Regression:** estimate continuous `AQI`; compare Linear Regression, Decision Tree Regressor, and Random Forest Regressor.
- **Classification:** estimate `AQI_Bucket`; compare Logistic Regression and Random Forest Classifier.
- **Inputs:** City, Date, and 11 pollutant measurements (excluding Xylene).
- **Evaluation:** chronological holdout plus five expanding-window validation folds grouped by date. All city rows for a date stay in the same partition.

The latest 20% of distinct dates are held out. In this dataset the test period begins **2019-05-27** and contains 8,832 rows; training contains 16,018. The row counts are not an 80/20 split because the number of cities with observations varies substantially by date. Splitting by distinct dates preserves the intended time boundary.

## Preprocessing

1. Parse dates, normalize City text, convert pollutant and target columns to numeric, and remove exact duplicates.
2. Derive Winter (Dec–Feb), Summer (Mar–May), Monsoon (Jun–Sep), and Post-Monsoon (Oct–Nov) from the date.
3. Drop rows without AQI or AQI_Bucket for the joint model comparison.
4. Inside each fold's model pipeline, impute each pollutant using the training-fold City × Season median, then the training-fold global median as a fallback. Validation/test rows never contribute to these medians.
5. Add missing-reading indicators and calendar features (month, weekday, weekend), one-hot encode City and Season, and scale features. Each pipeline refits these steps within each validation fold and on the full training period before holdout evaluation.

## Latest results

Results below are from the chronological holdout and are reproduced in `models/evaluation_metrics.json`.

### Regression

| Model | 5-fold CV R² (mean ± SD) | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Linear Regression | 0.7657 ± 0.0501 | 32.55 | 45.61 | 0.8075 |
| Decision Tree | 0.7446 ± 0.0790 | 22.34 | 38.08 | 0.8658 |
| Random Forest | **0.8196 ± 0.0394** | **17.68** | **32.03** | **0.9051** |

### Classification

| Model | 5-fold CV Accuracy | 5-fold CV Macro F1 | Test Accuracy | Test Weighted F1 | Test Macro F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6474 ± 0.0476 | 0.5454 ± 0.0291 | 0.7285 | 0.7132 | 0.6601 |
| Random Forest (balanced class weights) | 0.6743 ± 0.0803 | **0.6094 ± 0.0674** | **0.7711** | **0.7713** | **0.7534** |

Class weighting was selected using training-period macro-F1 folds to improve attention to less frequent categories. Holdout Random Forest per-class results:

| AQI category | Precision | Recall | F1 | Test rows |
|---|---:|---:|---:|---:|
| Good | 0.6847 | 0.7411 | 0.7118 | 923 |
| Moderate | 0.8217 | 0.6890 | 0.7495 | 2,775 |
| Poor | 0.5877 | 0.7702 | 0.6667 | 644 |
| Satisfactory | 0.8000 | 0.8370 | 0.8181 | 3,884 |
| Severe | 0.8089 | 0.8767 | 0.8414 | 227 |
| Very Poor | 0.7614 | 0.7071 | 0.7332 | 379 |

Macro F1 gives every class equal weight; weighted F1 is influenced more by common classes. The per-class table should be read alongside the overall scores.

## Project structure

```text
AQIPredict/
├── app.py                              # Streamlit interface for city/day estimates
├── data/
│   ├── city_day.csv                    # Supplied active dataset
│   ├── air_quality_data_clean.csv      # Labeled rows and derived season, regenerated by training
│   ├── test_predictions.csv            # Chronological holdout predictions
│   └── generate_dataset.py             # Legacy synthetic-data generator, not active input
├── models/
│   ├── aqi_regressor_lr.pkl             # Fitted preprocessing + Linear Regression
│   ├── aqi_regressor_dt.pkl             # Fitted preprocessing + Decision Tree
│   ├── aqi_regressor_rf.pkl             # Fitted preprocessing + Random Forest
│   ├── aqi_classifier_rf.pkl            # Fitted preprocessing + balanced Random Forest classifier
│   ├── label_encoder.pkl                # AQI category mapping
│   ├── evaluation_metrics.json          # CV and holdout results, including class-level metrics
│   └── feature_names.json               # Feature names emitted by the fitted pipeline
├── notebooks/
│   ├── aqi_prediction_walkthrough.ipynb # Guided project walkthrough
│   └── generate_notebook.py             # Rebuild the notebook
├── plots/
│   ├── eda/                             # Distributions, missingness, correlations, city-day patterns
│   └── results/                         # Regression/classification diagnostics and importance plots
└── src/
    ├── preprocess.py                    # Cleaning, chronological split, fold-local transforms
    ├── train_models.py                  # Training, date CV, metrics, artifact export
    ├── eda.py                           # Dataset summary and EDA plots
    ├── visualize_results.py              # Holdout plots
    └── predict.py                       # CLI estimate using saved pipelines
```

## Run the project

Install dependencies from the project folder:

```bash
pip install -r requirements.txt
```

Train models and recreate cleaned data, metrics, and holdout predictions:

```bash
python src/train_models.py
```

Generate exploratory and model-result plots, then try the CLI:

```bash
python src/eda.py
python src/visualize_results.py
python src/predict.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

The dashboard accepts a city, date, and pollutant readings. Missing pollutant values are estimated by the saved preprocessing pipeline. Its probability scores are not calibrated, and the displayed category/advisory is not an official determination.

## Limitations and next improvements

- The dataset has large and uneven gaps in pollutant coverage and city/date coverage. Fold-local imputation prevents leakage but cannot recover information that was never measured.
- Random Forest is strongest overall on the holdout, but no model is uniformly best for every category. Continue to inspect class-level recall and validate class weighting on new time periods.
- The target AQI contains extreme values and may reflect source-specific conventions. Verify provenance, AQI calculation method, units, and outlier records before operational use.
- A chronological split tests later dates across the same collection of cities. A stronger external check would hold out one or more entire cities and compare against an independent monitoring source.
- This is same-day estimation. A forecasting project would need a stated prediction horizon and future AQI targets constructed without future input leakage.
