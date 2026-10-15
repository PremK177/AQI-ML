# Air Quality Index (AQI) Prediction and Analysis
### College Mini-Project | Machine Learning Minor

---

> **Scope:** This project estimates daily AQI and its category from city-day pollutant readings, with an optional enhanced mode that derives daily summaries from hourly pollutant readings. The supplied files do not include a data dictionary or source provenance. Results measure agreement with their AQI labels in a later date period; they do not validate an official AQI calculation, a live monitoring service, a future forecast, or health advice.

## 1. The Objectives of the Project

Air pollution is an important environmental and public health concern. The goal of this college mini-project is to build an end-to-end machine-learning pipeline that studies pollutant observations and estimates the **Air Quality Index (AQI)** and the category label supplied with each record.

The project also documents how the data is prepared, how models are compared, and where the current results have limits.

### Core Project Objectives

1. **Dual prediction tasks**
   - **Regression:** estimate the numeric `AQI` value from the supplied city, date, and pollutant readings.
   - **Classification:** estimate the supplied `AQI_Bucket` category: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.
2. **Leakage-aware preprocessing:** retain missing pollutant readings until each training fold fits its own imputation, encoding, and scaling steps.
3. **Exploratory analysis:** summarize distributions, missingness, seasonal patterns, and pollutant-to-AQI correlations.
4. **Algorithm comparison:** compare linear baselines, decision trees, Random Forest, and Extra Trees using time-ordered validation and a later-date holdout, with and without hourly-derived daily summaries.
5. **Usable outputs:** provide saved model pipelines, analysis plots, a command-line example, a notebook, and a Streamlit dashboard.

---

## 2. Proposed System and Block Diagram

The project uses a modular six-stage workflow. Every validation fold fits its transformations on earlier dates only, and all city records for one date remain together in the same split.

```mermaid
flowchart TD
    A[Supplied city_day.csv<br/>29,531 records, 26 cities] --> B[Inspect columns, dates, missingness]
    U[Supplied city_hour.csv<br/>707,875 hourly records] --> V[Aggregate pollutant statistics by city and date]
    B --> C[Remove duplicates and rows without both targets]
    C --> D[Join hourly pollutant summaries by city and date]
    V --> D
    D --> E[Chronological holdout and fold-fitted preprocessing<br/>City × Season imputation, missing flags, calendar features, encoding, scaling]
    E --> F[Regression models<br/>Linear, Decision Tree, Random Forest, Extra Trees]
    E --> G[Classification models<br/>Logistic Regression, tuned Random Forest]
    F --> H[MAE, RMSE, R² and regression plots]
    G --> I[Accuracy, weighted/macro F1,<br/>per-class metrics and confusion matrix]
    H --> J[Saved metrics, models and predictions]
    I --> J
    J --> K[CLI and Streamlit dashboard]
```

---

## 3. The Dataset Description

The daily labels come from [`data/city_day.csv`](data/city_day.csv), with **29,531 city-day records** across **26 cities** from **2015-01-01 to 2020-07-01**. There are no exact duplicate rows. Both `AQI` and `AQI_Bucket` are present for **24,850 records**; records missing either target are excluded from the joint supervised comparison.

The companion [`data/city_hour.csv`](data/city_hour.csv) contains **707,875 hourly records** covering the same cities and date range, with **578,795 hourly AQI labels**. Hourly pollutant missingness is somewhat higher than in the daily file; Xylene is 64.4% missing and remains excluded. For the enhanced daily models, the code aggregates pollutant-only hourly statistics by city and date: mean, median, standard deviation, maximum, minimum, and valid-hour count for each of the 11 modeled pollutants. This creates 66 daily summary features. Hourly `AQI` and `AQI_Bucket` are never included among the predictors. Each city-date still contributes only one labeled daily training row.

The file contains 16 columns. It does not specify pollutant units, AQI calculation conventions, or source provenance. The maximum supplied AQI is **2,049**. Values are retained as provided; they are not clipped to a standard range.

| Field | Role | Notes |
| :--- | :--- | :--- |
| `City` | Input metadata | 26 city names; normalized as text before modeling |
| `Date` | Input time field | 2015-01-01 to 2020-07-01; also used to form chronological partitions |
| `PM2.5`, `PM10` | Pollutant inputs | Numeric readings; units are not specified in the supplied file |
| `NO`, `NO2`, `NOx`, `NH3` | Pollutant inputs | Numeric readings; units are not specified in the supplied file |
| `CO`, `SO2`, `O3` | Pollutant inputs | Numeric readings; units are not specified in the supplied file |
| `Benzene`, `Toluene`, `Xylene` | Pollutant inputs | Numeric readings; Xylene is excluded from the primary models due to missingness |
| `AQI` | Target for regression | Continuous label supplied in the file; observed range 13 to 2,049 among labeled rows |
| `AQI_Bucket` | Target for classification | Supplied category: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe |
| `Season` | Derived field | Created from `Date`; not present as a source column |

### Missingness in the Supplied File

| Column | Missing (%) | Column | Missing (%) |
| :--- | ---: | :--- | ---: |
| PM2.5 | 15.6 | PM10 | 37.7 |
| NO | 12.1 | NO2 | 12.1 |
| NOx | 14.2 | NH3 | 35.0 |
| CO | 7.0 | SO2 | 13.1 |
| O3 | 13.6 | Benzene | 19.0 |
| Toluene | 27.2 | Xylene | 61.3 |
| AQI | 15.9 | AQI_Bucket | 15.9 |

The original synthetic-data files and generator remain in `data/` for reference. The current training pipeline uses `city_day.csv` for daily labels and `city_hour.csv` for optional daily pollutant summaries.

---

## 4. Info on Data Preprocessing

Preprocessing converts the supplied file into model-ready records while keeping each fold's validation period separate from the statistics learned on its training period.

### 1. Basic Validation and Cleaning
- Check for required fields (`City`, `Date`, `AQI`, and `AQI_Bucket`) and remove exact duplicate rows.
- Parse dates, trim city and category text, convert pollutant and AQI fields to numeric, and treat non-finite numbers as missing.
- Remove rows with invalid dates, missing cities, or a missing AQI target or category. Keep only the six category names present in the project label mapping.

### 2. Missing Pollutant Values
- Xylene is omitted from the primary model inputs because **61.3%** of its values are missing.
- The other 11 pollutant columns are kept with their missing values. For each pollutant, `AQIPreprocessor` learns a **City × Season median** from the current training fold, then uses that fold's overall pollutant median when no group value is available.
- A missing-value indicator is created for every modeled pollutant before imputation, so the model can distinguish an imputed reading from an observed one.
- Validation and test observations never contribute to the learned imputation values.

```text
Fit on an earlier training fold:
  learn city-season and global medians
  learn feature scaling values

Transform the next validation fold:
  apply the saved medians and scaling values
```

### 3. Calendar Features and Encoding
- Derive Winter (December–February), Summer (March–May), Monsoon (June–September), or Post-Monsoon (October–November) from the date.
- Add month, day of week, and weekend indicators.
- One-hot encode City and Season, then standardize the resulting features.
- Keep imputation, feature creation, encoding, and scaling together inside each model pipeline, so cross-validation refits them within every fold.

### 4. Chronological Holdout
The latest **20% of distinct labeled dates** are held out. The test period begins on **2019-05-27** and contains **8,832 rows**; the earlier training period contains **16,018 rows**. These row counts are not an 80/20 ratio because the number of cities represented varies by date. All rows with the same date stay in one partition.

---

## 5. Exploratory Data Analysis (EDA)

EDA summarizes the labeled records before modeling. The scripts save plots and supporting tables under `plots/eda/`.

### Selected Summary Statistics

| Variable | Mean | Median | Maximum | Skewness |
| :--- | ---: | ---: | ---: | ---: |
| PM2.5 | 67.48 | 48.79 | 914.94 | 2.96 |
| PM10 | 118.45 | 96.18 | 917.08 | 1.93 |
| NO2 | 28.98 | 22.10 | 362.21 | 2.44 |
| CO | 2.35 | 0.93 | 175.81 | 8.63 |
| AQI | 166.46 | 118.00 | 2,049.00 | 3.40 |

### Key EDA Findings
1. **Uneven pollutant coverage:** PM10 and NH3 are missing in over one-third of the raw rows; Xylene is missing in 61.3%.
2. **Pollutant associations:** among labeled records, AQI has the strongest Pearson correlation with PM10 (`r = 0.803`), followed by CO (`r = 0.683`) and PM2.5 (`r = 0.659`). Correlation describes association, not causation.
3. **Seasonal pattern:** the observed median AQI is 171 in calendar-derived Winter and 89 in Monsoon. City and date coverage differ across the dataset and can contribute to this pattern.
4. **Skew and extremes:** pollutant distributions are right-skewed, and AQI includes values above 500. Summary metrics and plots should be interpreted with those extremes in mind.
5. **Label frequency:** Moderate and Satisfactory are the most common supplied categories. Category-level scores are included with overall classification metrics.

---

## 6. Feature Selection and Engineering

This version does not perform a separate statistical feature-selection procedure. It chooses inputs based on the available columns and missingness, then records the features generated by the fitted preprocessing pipeline.

### 1. Pollutant Inputs
Eleven measured fields are used: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, and Toluene. Xylene is excluded from the primary models due to 61.3% missingness.

### 2. Missingness Indicators
For each modeled pollutant, a binary `*_WasMissing` field records whether the original input was absent before fold-fitted imputation.

### 3. Calendar and Location Features
- `Month`, `DayOfWeek`, and `Is_Weekend` represent calendar timing.
- `City_*` and `Season_*` one-hot fields represent the city and date-derived season.

**Final feature space:** the daily-only pipeline emits **46 model features**; the hourly-enhanced pipeline emits **178**, including hourly summary values and their missingness indicators. Feature importance plots are model diagnostics; they are not a causal analysis or an independent feature-selection stage.

---

## 7. ML Methodology Used

### Problem Formulation
1. **Regression:** predict continuous `AQI` from City, Date, and the 11 selected daily pollutants; enhanced models also use the 66 hourly-summary features.
2. **Classification:** predict the supplied `AQI_Bucket` category from the same daily and optional hourly-summary inputs.

The two targets are modeled separately. AQI category is not recalculated from the regression output.

### Validation Strategy
- **Later-date holdout:** latest 20% of distinct labeled dates, beginning 2019-05-27.
- **Five-fold expanding-window cross-validation:** each validation block contains later dates than its training data; all records on a date remain in one block.
- Every preprocessing step, including imputation and scaling for the hourly summaries, is fit inside the model pipeline on each training fold. Candidate settings were compared on the same five chronological folds. Extra Trees was selected for regression by mean CV R²; the balanced Random Forest classifier was tuned for macro F1. The hourly-enhanced variants were evaluated against their daily-only counterparts on the same folds and later-date holdout.

### Evaluation Metrics
- **Regression:** MAE measures average absolute error in AQI points; RMSE penalizes larger errors more strongly; R² compares explained variance with a mean-only baseline.
- **Classification:** accuracy measures the fraction of correct labels; weighted F1 reflects class frequency; macro F1 gives each category equal weight. Per-class precision, recall, F1, support, and a confusion matrix show where errors occur.

---

## 8. Algorithms Used

The project compares linear baselines, a single decision tree, Random Forest, and Extra Trees. The equations below summarize the fitted model families.

### Regression Algorithms

#### 1. Linear Regression

```text
y_hat = beta_0 + sum(j = 1..p, beta_j * x_j)
```

The model fits coefficients by minimizing squared residuals, `Σ(yᵢ − ŷᵢ)²`. It provides a simple baseline and is easy to inspect, but a single linear equation cannot represent every nonlinear interaction in the data.

#### 2. Decision Tree Regressor

At each split, the tree chooses a partition that reduces within-leaf squared error. For leaf region `Rₘ`:

```text
MSE(R_m) = (1 / N_m) * sum(i in R_m, (y_i - y_bar_m)^2)
```

The tree captures threshold rules and nonlinear patterns without requiring feature scaling for its splits. The project limits it to `max_depth=6` with `min_samples_split=10` to constrain complexity. A single tree can still vary substantially with the training data.

#### 3. Random Forest Regressor (comparison baseline)

The forest fits 100 trees to bootstrap samples and averages their predictions:

```text
y_hat_RF(x) = (1 / B) * sum(b = 1..B, T_b(x)), where B = 100
```

Random feature subsets decorrelate trees, while averaging reduces the variance of an individual tree. The comparison estimator uses `max_depth=12` and `min_samples_split=5`.

#### 4. Extra Trees Regressor (daily-only model)

Extra Trees fits 160 randomized trees. It introduces additional randomness when choosing split thresholds and averages the tree outputs:

```text
y_hat_ET(x) = (1 / B) * sum(b = 1..B, T_b(x)), where B = 160
```

The fitted model uses `min_samples_leaf=2`. Its additional split randomness can reduce variance. An enhanced version of this model also uses the hourly summaries described below.

#### Hourly-summary features

For each modeled pollutant, the daily pipeline can also use six statistics from that city's hourly records: mean, median, standard deviation, maximum, minimum, and number of valid hourly readings. These summarize the within-day pattern that one daily value cannot show. The daily-only model remains available when hourly readings are not supplied.

### Classification Algorithms

#### 1. Logistic Regression

For `K` classes, the multinomial model applies softmax to class scores:

```text
P(Y = k | x) = exp(beta_k^T * x) / sum(j = 1..K, exp(beta_j^T * x))
```

It is a linear classification baseline and produces class probabilities. The project uses `max_iter=1000`; the output probabilities are not calibrated.

#### 2. Random Forest Classifier

The selected classifier combines 160 decision trees and predicts the class with the most votes. Its trees use `max_features=0.8`, `min_samples_leaf=2`, and balanced class weights. The enhanced classifier also receives the hourly-summary features. Tree splits use Gini impurity:

```text
G = 1 - sum(k = 1..K, p_k^2)
```

The classifier's final prediction is the class receiving the most tree votes:

```text
y_hat_RF(x) = mode(T_1(x), T_2(x), ..., T_B(x))
```

Class weights give less frequent training categories more influence. The selected tree settings were chosen using training-period macro-F1 validation.

---

## 9. Model Training, Validating, and Testing

1. Read `data/city_day.csv`, inspect data quality, and retain labeled rows for the two supervised tasks.
2. Reserve the later-date holdout before fitting model pipelines.
3. Compare daily-only and hourly-enhanced regression and classification pipelines using the same fold-local preprocessing.
4. Use five expanding date folds for model validation. Classification cross-validation reports both accuracy and macro F1.
5. Fit each final estimator on the complete training period and evaluate it once on the held-out later dates.
6. Save model pipelines, feature names, metrics, and test predictions under `models/` and `data/`.

The test rows come later in time but from the same 26-city collection. This tests temporal transfer within the supplied file; it does not replace validation on unseen cities or independent monitoring data.

---

## 10. Project Structure and How to Run

```text
AQIPredict/
├── app.py                                  # Streamlit interface for daily estimates and analysis
├── data/
│   ├── city_day.csv                        # Daily AQI labels and daily pollutant readings
│   ├── city_hour.csv                       # Hourly readings used for optional daily summaries
│   ├── air_quality_data_clean.csv          # Labeled rows and derived season, regenerated by training
│   ├── test_predictions.csv                # Later-date holdout targets and predictions
│   └── generate_dataset.py                 # Legacy synthetic-data generator; not used by training
├── models/
│   ├── aqi_regressor_lr.pkl                 # Preprocessing pipeline + Linear Regression
│   ├── aqi_regressor_dt.pkl                 # Preprocessing pipeline + Decision Tree Regressor
│   ├── aqi_regressor_rf.pkl                 # Preprocessing pipeline + Random Forest Regressor
│   ├── aqi_regressor_extra_trees.pkl         # Daily-only Extra Trees pipeline
│   ├── aqi_regressor_extra_trees_hourly.pkl  # Enhanced Extra Trees pipeline using hourly summaries
│   ├── aqi_regressor_rf_hourly.pkl           # Enhanced Random Forest regression comparison
│   ├── aqi_classifier_rf.pkl                # Daily-only tuned Random Forest classifier
│   ├── aqi_classifier_rf_hourly.pkl         # Enhanced classifier using hourly summaries
│   ├── label_encoder.pkl                    # Mapping between category names and model labels
│   ├── evaluation_metrics.json              # Cross-validation, holdout, and per-class metrics
│   ├── feature_names.json                   # Daily-only model features
│   └── feature_names_hourly.json             # Features including hourly summaries
├── notebooks/
│   ├── aqi_prediction_walkthrough.ipynb     # Guided walkthrough
│   └── generate_notebook.py                 # Rebuild the walkthrough notebook
├── plots/
│   ├── eda/                                 # Distributions, missingness, correlations, and seasonality
│   └── results/                             # Regression/classification diagnostics and importance
└── src/
    ├── preprocess.py                        # Cleaning, chronological split, fold-fitted transformer
    ├── train_models.py                      # Model fitting, date CV, metrics, and artifact export
    ├── eda.py                               # Dataset summary and EDA figures/tables
    ├── visualize_results.py                 # Holdout comparisons and diagnostics
    └── predict.py                           # CLI estimate using saved model pipelines
```

### Run the Project

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the models and recreate cleaned data, metrics, and holdout predictions**
   ```bash
   python src/train_models.py
   ```
3. **Create EDA and model-result plots**
   ```bash
   python src/eda.py
   python src/visualize_results.py
   ```
4. **Try daily-only or hourly-enhanced command-line estimates**
   ```bash
   python src/predict.py
   python src/predict.py --hourly-csv data/city_hour.csv
   ```
5. **Launch the interactive dashboard**
   ```bash
   streamlit run app.py
   ```

The dashboard and CLI return daily estimates from City, Date, and pollutant readings. Upload hourly records matching the selected city and date to use the enhanced model; otherwise the daily-only model runs. These tools do not generate future forecasts or official AQI or health determinations. The displayed classifier probabilities are not calibrated.

---

## 11. Model Evaluation

The following results come from the held-out dates beginning 2019-05-27. Cross-validation scores were calculated from training-period dates only. Values are reproduced in `models/evaluation_metrics.json`.

### 1. Regression Models (`AQI`)

| Model | CV R² (mean ± SD) | Train R² | Test R² | Test RMSE | Test MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Regression | 0.7657 ± 0.0501 | 0.8456 | 0.8075 | 45.61 | 32.55 |
| Decision Tree Regressor | 0.7446 ± 0.0790 | 0.8934 | 0.8658 | 38.08 | 22.34 |
| Random Forest Regressor (daily-only) | 0.8196 ± 0.0394 | 0.9596 | 0.9051 | 32.03 | 17.68 |
| Extra Trees Regressor (daily-only) | 0.8380 ± 0.0252 | 0.9887 | 0.9193 | 29.53 | 16.92 |
| Random Forest + Hourly Summaries | 0.8471 ± 0.0280 | 0.9664 | 0.9237 | 28.72 | 15.05 |
| **Extra Trees + Hourly Summaries** | **0.8602 ± 0.0250** | **0.9950** | **0.9251** | **28.45** | **14.98** |

The hourly-enhanced Extra Trees model has the strongest regression scores. Compared with the daily-only Extra Trees model, its test MAE fell by 1.94 AQI points (11.5%), RMSE by 1.08 points (3.7%), and R² rose by 0.0058. Its mean chronological CV R² also rose from 0.8380 to 0.8602. These values describe agreement with the supplied targets, including the large values in that file.

### 2. Classification Models (`AQI_Bucket`)

| Model | CV Accuracy (mean ± SD) | CV Macro F1 (mean ± SD) | Test Accuracy | Test Weighted F1 | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6474 ± 0.0476 | 0.5454 ± 0.0291 | 0.7285 | 0.7132 | 0.6601 |
| Tuned Random Forest (daily-only) | 0.7009 ± 0.0550 | 0.6420 ± 0.0580 | 0.7868 | 0.7866 | 0.7634 |
| **Tuned Random Forest + Hourly Summaries** | **0.7161 ± 0.0439** | **0.6674 ± 0.0392** | **0.8036** | **0.8036** | **0.7832** |

The hourly summaries improved holdout accuracy by 1.7 percentage points and macro F1 by 2.0 points over the daily-only classifier. Cross-validation macro F1 rose by 2.5 points.

#### Random Forest Per-Category Results

| Supplied category | Precision | Recall | F1 | Test rows |
| :--- | ---: | ---: | ---: | ---: |
| Good | 0.7044 | 0.7616 | 0.7319 | 923 |
| Moderate | 0.8672 | 0.7366 | 0.7966 | 2,775 |
| Poor | 0.6612 | 0.7547 | 0.7049 | 644 |
| Satisfactory | 0.8160 | 0.8702 | 0.8423 | 3,884 |
| Severe | 0.8319 | 0.8722 | 0.8516 | 227 |
| Very Poor | 0.7901 | 0.7546 | 0.7719 | 379 |

Macro F1 gives each category equal weight; weighted F1 gives common classes more influence. Poor has the lowest per-class F1 in this holdout, so the overall accuracy alone does not tell the full story.

---

## 12. Graphical Representation of Results

The generated figures are saved under `plots/`. The regression scatter plot is explicitly zoomed to 0–520 for readability; the metrics above use all holdout records, including AQI outliers above that range.

1. **Actual vs. predicted AQI:** `plots/results/actual_vs_predicted_regression.png` compares six daily-only and hourly-enhanced regression models with the ideal prediction line.
2. **Residual analysis:** `plots/results/residual_analysis.png` shows residuals from the hourly-enhanced Extra Trees model.
3. **Regression comparison:** `plots/results/regression_model_comparison.png` compares holdout R² and RMSE.
4. **Classification confusion matrix:** `plots/results/confusion_matrix_classifier.png` shows category assignments from the hourly-enhanced Random Forest.
5. **Classification comparison:** `plots/results/classification_model_comparison.png` compares the daily-only and hourly-enhanced classifiers with Logistic Regression.
6. **Feature importance:** `plots/results/feature_importance_ranking.png` shows impurity-based importance from the hourly-enhanced Extra Trees and Random Forest models. These scores are model-specific associations and should not be interpreted as causal effects.
7. **EDA figures and tables:** `plots/eda/` contains pollutant distributions, category counts, seasonal AQI, the correlation heatmap and matrix, pollutant-AQI relationships, summary statistics, and missingness percentages.

---

## 13. Conclusion

This project provides an end-to-end workflow for estimating numeric AQI and the category label supplied with historical city-day observations. The time-ordered split, fold-fitted preprocessing, and class-level metrics make the benchmark more informative than a random split or accuracy-only report.

### Key Conclusions and Next Steps
1. **Model comparison:** the hourly-enhanced Extra Trees model performs best for regression, while the hourly-enhanced Random Forest performs best for classification in this benchmark.
2. **Data limits:** pollutant readings and city/date coverage are uneven; imputation cannot recreate measurements that were not collected.
3. **Verify the source:** confirm dataset provenance, units, AQI calculation method, and extreme AQI records before using results beyond this project.
4. **Test city transfer:** hold out one or more entire cities and evaluate against an independent monitoring source.
5. **Separate forecasting from estimation:** a future forecast requires a stated prediction horizon and targets built without using future information.
6. **Keep category errors visible:** continue reviewing per-class recall and F1 as the class distribution or model changes.

---
*Developed for College Mini-Project (Machine Learning Minor).*
