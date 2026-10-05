"""
Model Training and Cross-Validation Module
Trains and compares baseline and ensemble models for both Regression and Classification:
- Regression: Linear Regression, Decision Tree Regressor, Random Forest Regressor
- Classification: Logistic Regression, Random Forest Classifier
Includes 5-Fold Cross Validation and model persistence.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

from preprocess import load_and_inspect_raw_data, clean_data, engineer_features, prepare_and_split_data

def train_and_evaluate_all():
    print("=" * 60)
    print("STEP 1: PREPROCESSING & DATA SPLITTING")
    print("=" * 60)
    raw_df = load_and_inspect_raw_data('data/air_quality_data_raw.csv')
    clean_df = clean_data(raw_df)
    feat_df = engineer_features(clean_df)
    data = prepare_and_split_data(feat_df, test_size=0.2, random_state=42)
    
    X_train_s = data['X_train_scaled']
    X_test_s = data['X_test_scaled']
    X_train_raw = data['X_train']
    X_test_raw = data['X_test']
    
    y_train_reg = data['y_train_reg']
    y_test_reg = data['y_test_reg']
    y_train_clf = data['y_train_clf']
    y_test_clf = data['y_test_clf']
    label_encoder = data['label_encoder']
    classes = list(label_encoder.classes_)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots/results', exist_ok=True)
    
    # -------------------------------------------------------------
    # REGRESSION MODELS (Predicting AQI Continuous Value)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING REGRESSION MODELS")
    print("=" * 60)
    
    reg_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=6, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
    }
    
    reg_results = {}
    reg_predictions = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in reg_models.items():
        print(f"\n--- Training {name} Regressor ---")
        # For tree models, scaled or unscaled yields identical trees, but scaled features keep inputs uniform
        model.fit(X_train_s, y_train_reg)
        
        # 5-fold cross-validation on training data to assess generalization
        cv_scores = cross_val_score(model, X_train_s, y_train_reg, cv=kf, scoring='r2')
        print(f"5-Fold CV R2 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_train_pred = model.predict(X_train_s)
        y_test_pred = model.predict(X_test_s)
        reg_predictions[name] = y_test_pred
        
        # Metrics
        train_mae = mean_absolute_error(y_train_reg, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred))
        train_r2 = r2_score(y_train_reg, y_train_pred)
        
        test_mae = mean_absolute_error(y_test_reg, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred))
        test_r2 = r2_score(y_test_reg, y_test_pred)
        
        reg_results[name] = {
            'CV_R2_Mean': float(round(cv_scores.mean(), 4)),
            'CV_R2_Std': float(round(cv_scores.std(), 4)),
            'Train_MAE': float(round(train_mae, 4)),
            'Train_RMSE': float(round(train_rmse, 4)),
            'Train_R2': float(round(train_r2, 4)),
            'Test_MAE': float(round(test_mae, 4)),
            'Test_RMSE': float(round(test_rmse, 4)),
            'Test_R2': float(round(test_r2, 4))
        }
        
        print(f"Train - MAE: {train_mae:.2f} | RMSE: {train_rmse:.2f} | R2: {train_r2:.4f}")
        print(f"Test  - MAE: {test_mae:.2f} | RMSE: {test_rmse:.2f} | R2: {test_r2:.4f}")
        
    # Save best regression model
    joblib.dump(reg_models['Linear Regression'], 'models/aqi_regressor_lr.pkl')
    joblib.dump(reg_models['Decision Tree'], 'models/aqi_regressor_dt.pkl')
    joblib.dump(reg_models['Random Forest'], 'models/aqi_regressor_rf.pkl')
    
    # -------------------------------------------------------------
    # CLASSIFICATION MODELS (Predicting AQI Category)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING CLASSIFICATION MODELS")
    print("=" * 60)
    
    clf_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1)
    }
    
    clf_results = {}
    clf_predictions = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in clf_models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_s, y_train_clf)
        
        # 5-fold cross-validation
        cv_scores = cross_val_score(model, X_train_s, y_train_clf, cv=skf, scoring='accuracy')
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        y_train_pred = model.predict(X_train_s)
        y_test_pred = model.predict(X_test_s)
        clf_predictions[name] = y_test_pred
        
        test_acc = accuracy_score(y_test_clf, y_test_pred)
        test_precision = precision_score(y_test_clf, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test_clf, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test_clf, y_test_pred, average='weighted', zero_division=0)
        
        clf_results[name] = {
            'CV_Accuracy_Mean': float(round(cv_scores.mean(), 4)),
            'CV_Accuracy_Std': float(round(cv_scores.std(), 4)),
            'Test_Accuracy': float(round(test_acc, 4)),
            'Test_Precision_Weighted': float(round(test_precision, 4)),
            'Test_Recall_Weighted': float(round(test_recall, 4)),
            'Test_F1_Weighted': float(round(test_f1, 4))
        }
        
        print(f"Test Accuracy: {test_acc:.4f} | Weighted Precision: {test_precision:.4f} | Weighted Recall: {test_recall:.4f} | Weighted F1: {test_f1:.4f}")
        
    joblib.dump(clf_models['Random Forest Classifier'], 'models/aqi_classifier_rf.pkl')
    
    # Save evaluation summary to JSON
    summary = {
        'regression_metrics': reg_results,
        'classification_metrics': clf_results
    }
    with open('models/evaluation_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    # Save predictions and test data for plotting
    test_data_export = X_test_raw.copy()
    test_data_export['Actual_AQI'] = y_test_reg.values
    test_data_export['Actual_Category'] = label_encoder.inverse_transform(y_test_clf)
    for model_name, preds in reg_predictions.items():
        test_data_export[f'Pred_AQI_{model_name}'] = preds
    for model_name, preds in clf_predictions.items():
        test_data_export[f'Pred_Category_{model_name}'] = label_encoder.inverse_transform(preds)
        
    test_data_export.to_csv('data/test_predictions.csv', index=False)
    print("\nSaved evaluation artifacts to 'models/evaluation_metrics.json' and 'data/test_predictions.csv'")
    
    return {
        'reg_results': reg_results,
        'clf_results': clf_results,
        'reg_models': reg_models,
        'clf_models': clf_models,
        'data': data,
        'reg_predictions': reg_predictions,
        'clf_predictions': clf_predictions
    }

if __name__ == '__main__':
    train_and_evaluate_all()
