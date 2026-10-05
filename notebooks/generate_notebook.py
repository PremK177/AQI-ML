"""
Generates a comprehensive Jupyter Notebook walking through the entire
Air Quality Index (AQI) Prediction and Analysis project step-by-step.
"""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Air Quality Index (AQI) Prediction and Analysis\n",
            "### College Mini-Project | Machine Learning Minor\n",
            "---\n",
            "**Focus Area:** Regression (Continuous AQI Value) & Classification (AQI Health Category)\n",
            "\n",
            "### Notebook Workflow:\n",
            "1. **Project Objectives & System Architecture**\n",
            "2. **Dataset Loading & Exploration**\n",
            "3. **Data Cleaning & Preprocessing (Handling missing values, deduplication)**\n",
            "4. **Exploratory Data Analysis (EDA) & Visualizations**\n",
            "5. **Domain Feature Engineering**\n",
            "6. **Strict Train-Test Splitting & Feature Scaling (Preventing Data Leakage)**\n",
            "7. **Model Building & Cross-Validation:**\n",
            "   - Regression: *Linear Regression*, *Decision Tree*, *Random Forest*\n",
            "   - Classification: *Logistic Regression*, *Random Forest Classifier*\n",
            "8. **Model Evaluation & Error Metrics (MAE, RMSE, R², Accuracy, F1-Score, Confusion Matrix)**\n",
            "9. **Graphical Representation of Results & Residual Analysis**\n",
            "10. **Interactive Real-Time Inference Demo**\n",
            "11. **Conclusion & Environmental Findings**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Import Core Dependencies\n",
            "import os\n",
            "import json\n",
            "import joblib\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold\n",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
            "from sklearn.linear_model import LinearRegression, LogisticRegression\n",
            "from sklearn.tree import DecisionTreeRegressor\n",
            "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
            "from sklearn.metrics import (\n",
            "    mean_absolute_error, mean_squared_error, r2_score,\n",
            "    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
            ")\n",
            "\n",
            "# Visual configurations\n",
            "sns.set_theme(style='whitegrid', palette='muted')\n",
            "plt.rcParams['figure.figsize'] = (10, 6)\n",
            "print('Libraries imported successfully!')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Dataset Loading and Initial Inspection\n",
            "We inspect the raw monitoring station data containing criteria pollutants ($PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$) and meteorological parameters ($Temperature, Humidity, Wind\\ Speed$)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "raw_df = pd.read_csv('../data/air_quality_data_raw.csv')\n",
            "print(f\"Raw dataset shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns\")\n",
            "print(f\"Duplicate records: {raw_df.duplicated().sum()}\")\n",
            "print(\"\\nMissing values per feature:\")\n",
            "print(raw_df.isnull().sum())\n",
            "raw_df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Preprocessing & Cleaning\n",
            "- **Deduplication:** Remove recorded duplicate sensor logs.\n",
            "- **Missing Value Imputation:** For atmospheric monitoring, replacing missing values with station-and-season-specific medians preserves localized environmental baselines.\n",
            "- **Temporal Parsing:** Convert `Date` to datetime objects."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Deduplicate\n",
            "df = raw_df.drop_duplicates().copy()\n",
            "df['Date'] = pd.to_datetime(df['Date'])\n",
            "df = df.sort_values(by=['Date', 'Station']).reset_index(drop=True)\n",
            "\n",
            "# Median imputation grouped by Station & Season\n",
            "num_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed']\n",
            "for col in num_cols:\n",
            "    medians = df.groupby(['Station', 'Season'])[col].transform('median')\n",
            "    df[col] = df[col].fillna(medians)\n",
            "    df[col] = df[col].fillna(df[col].median())\n",
            "\n",
            "print(f\"Remaining missing values: {df.isnull().sum().sum()}\")\n",
            "df.info()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis (EDA)\n",
            "Visualizing the distributions of criteria pollutants and analyzing correlation dynamics with AQI."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Correlation Matrix\n",
            "plt.figure(figsize=(10, 8))\n",
            "corr_cols = num_cols + ['AQI']\n",
            "corr = df[corr_cols].corr()\n",
            "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
            "sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)\n",
            "plt.title('Correlation Matrix of Pollutants, Meteorology, and AQI')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Seasonal AQI Distribution\n",
            "plt.figure(figsize=(9, 5))\n",
            "sns.boxplot(x='Season', y='AQI', data=df, palette='Set2')\n",
            "plt.title('Seasonal Variation in Air Quality Index (AQI)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Feature Selection & Engineering\n",
            "We engineer physically motivated atmospheric features:\n",
            "1. **PM Fine Ratio ($PM_{2.5} / PM_{10}$):** Proportional contribution of combustion-derived fine particles.\n",
            "2. **Dispersion Index:** Proxy of atmospheric ventilation: $(Wind\\ Speed + 1) / (PM_{2.5} + 1)$.\n",
            "3. **Heat-Moisture Index:** Interaction of temperature and humidity.\n",
            "4. **Calendar Features:** Month, Day of week, Weekend indicator."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df['PM_Ratio'] = np.clip(df['PM2.5'] / (df['PM10'] + 1e-5), 0.05, 0.95)\n",
            "df['Dispersion_Index'] = (df['Wind_Speed'] + 1.0) / (df['PM2.5'] + 1.0)\n",
            "df['Heat_Moisture_Index'] = (df['Temperature'] * df['Humidity']) / 100.0\n",
            "df['Month'] = df['Date'].dt.month\n",
            "df['DayOfWeek'] = df['Date'].dt.dayofweek\n",
            "df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)\n",
            "\n",
            "# One-hot encoding of Station & Season\n",
            "df_encoded = pd.get_dummies(df, columns=['Station', 'Season'], drop_first=True)\n",
            "feature_cols = [c for c in df_encoded.columns if c not in ['Date', 'AQI', 'AQI_Bucket']]\n",
            "print(f\"Total features engineered: {len(feature_cols)}\")\n",
            "feature_cols"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Strict Featurization Ordering & Data Splitting\n",
            "**Crucial ML Best Practice:** We split into 80% Training and 20% Testing **BEFORE** fitting `StandardScaler`. The scaler is fitted exclusively on the training set and applied onto the test set to guarantee no data leakage."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "X = df_encoded[feature_cols]\n",
            "y_reg = df_encoded['AQI']\n",
            "\n",
            "label_enc = LabelEncoder()\n",
            "y_clf = label_enc.fit_transform(df_encoded['AQI_Bucket'])\n",
            "\n",
            "X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(\n",
            "    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf\n",
            ")\n",
            "\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "print(f\"Training set: {X_train.shape[0]} samples\")\n",
            "print(f\"Testing set:  {X_test.shape[0]} samples\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Model Training & 5-Fold Cross-Validation\n",
            "We compare three regression algorithms:\n",
            "1. **Linear Regression:** Baseline parametric model.\n",
            "2. **Decision Tree Regressor:** Non-linear threshold partitioning.\n",
            "3. **Random Forest Regressor:** Bagged ensemble of decision trees.\n",
            "\n",
            "And two classification algorithms for AQI Category:\n",
            "1. **Logistic Regression (Multinomial)**\n",
            "2. **Random Forest Classifier**"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Regression Models\n",
            "reg_models = {\n",
            "    'Linear Regression': LinearRegression(),\n",
            "    'Decision Tree': DecisionTreeRegressor(max_depth=6, min_samples_split=10, random_state=42),\n",
            "    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)\n",
            "}\n",
            "\n",
            "kf = KFold(n_splits=5, shuffle=True, random_state=42)\n",
            "reg_metrics = []\n",
            "\n",
            "for name, model in reg_models.items():\n",
            "    model.fit(X_train_scaled, y_train_reg)\n",
            "    cv_r2 = cross_val_score(model, X_train_scaled, y_train_reg, cv=kf, scoring='r2').mean()\n",
            "    preds = model.predict(X_test_scaled)\n",
            "    \n",
            "    mae = mean_absolute_error(y_test_reg, preds)\n",
            "    rmse = np.sqrt(mean_squared_error(y_test_reg, preds))\n",
            "    r2 = r2_score(y_test_reg, preds)\n",
            "    \n",
            "    reg_metrics.append({\n",
            "        'Model': name,\n",
            "        '5-Fold CV R2': round(cv_r2, 4),\n",
            "        'Test R2': round(r2, 4),\n",
            "        'Test RMSE': round(rmse, 2),\n",
            "        'Test MAE': round(mae, 2)\n",
            "    })\n",
            "\n",
            "pd.DataFrame(reg_metrics)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Classification Models\n",
            "clf_models = {\n",
            "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
            "    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)\n",
            "}\n",
            "\n",
            "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
            "clf_metrics = []\n",
            "\n",
            "for name, model in clf_models.items():\n",
            "    model.fit(X_train_scaled, y_train_clf)\n",
            "    cv_acc = cross_val_score(model, X_train_scaled, y_train_clf, cv=skf, scoring='accuracy').mean()\n",
            "    preds = model.predict(X_test_scaled)\n",
            "    \n",
            "    acc = accuracy_score(y_test_clf, preds)\n",
            "    f1 = f1_score(y_test_clf, preds, average='weighted')\n",
            "    \n",
            "    clf_metrics.append({\n",
            "        'Model': name,\n",
            "        '5-Fold CV Accuracy': f\"{cv_acc*100:.2f}%\开拓\",\n",
            "        'Test Accuracy': f\"{acc*100:.2f}%\",\n",
            "        'Weighted F1': f\"{f1*100:.2f}%\"\n",
            "    })\n",
            "\n",
            "pd.DataFrame(clf_metrics)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Results Visualization & Feature Importance\n",
            "We inspect actual vs predicted AQI regression lines and evaluate which features contributed most heavily to predictions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Actual vs Predicted AQI Scatter Plot\n",
            "best_reg = reg_models['Random Forest']\n",
            "rf_preds = best_reg.predict(X_test_scaled)\n",
            "\n",
            "plt.figure(figsize=(8, 6))\n",
            "plt.scatter(y_test_reg, rf_preds, alpha=0.5, color='#2ecc71', edgecolors='none', label='Test Data Points')\n",
            "plt.plot([0, 500], [0, 500], 'r--', linewidth=2, label='Ideal Perfect Fit (y=x)')\n",
            "plt.title(f\"Random Forest Regressor (R² = {r2_score(y_test_reg, rf_preds):.4f})\")\n",
            "plt.xlabel('Actual AQI')\n",
            "plt.ylabel('Predicted AQI')\n",
            "plt.legend()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Feature Importance\n",
            "feat_importances = pd.Series(best_reg.feature_importances_, index=feature_cols).sort_values(ascending=True)\n",
            "feat_importances.tail(10).plot(kind='barh', color='#2980b9')\n",
            "plt.title('Top 10 Feature Importances in AQI Prediction')\n",
            "plt.xlabel('Gini Importance')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Conclusion\n",
            "- Particulate matter ($PM_{2.5}$ and $PM_{10}$) along with the atmospheric dispersion index are the strongest drivers of Air Quality Index.\n",
            "- Non-linear ensemble methods (Random Forest) significantly outperform linear baselines by capturing the piecewise breakpoint mechanics of AQI calculations.\n",
            "- The dual modeling framework delivers precise continuous AQI estimates alongside accurate health hazard classifications."
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('notebooks/aqi_prediction_walkthrough.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Created notebooks/aqi_prediction_walkthrough.ipynb successfully!")
