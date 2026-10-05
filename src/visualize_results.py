"""
Visualization of Results Module
Generates high-resolution graphical representations of model performance:
1. Actual vs Predicted AQI Scatter Plot (Comparison of Linear Reg, Decision Tree, Random Forest)
2. Residual Error Analysis (Distribution & Residuals vs Predicted)
3. Regression Performance Comparison Bar Chart (R2 & RMSE)
4. Multi-class Confusion Matrix for AQI Classification
5. Classification Metrics Comparison Bar Chart (Accuracy & F1-score)
6. Feature Importance Analysis (Atmospheric driver rankings)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

def generate_result_visualizations(output_dir='plots/results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test predictions and metrics
    test_df = pd.read_csv('data/test_predictions.csv')
    with open('models/evaluation_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
        
    rf_reg = joblib.load('models/aqi_regressor_rf.pkl')
    rf_clf = joblib.load('models/aqi_classifier_rf.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    print("=" * 60)
    print("GENERATING GRAPHICAL REPRESENTATIONS OF RESULTS")
    print("=" * 60)
    
    # -----------------------------------------------------------------
    # 1. ACTUAL VS PREDICTED SCATTER PLOTS (Regression Models)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    reg_models = [
        ('Linear Regression', 'Pred_AQI_Linear Regression', '#3498db'),
        ('Decision Tree', 'Pred_AQI_Decision Tree', '#e67e22'),
        ('Random Forest', 'Pred_AQI_Random Forest', '#2ecc71')
    ]
    
    actual = test_df['Actual_AQI']
    min_val, max_val = 0, 520
    
    for ax, (name, col, color) in zip(axes, reg_models):
        pred = test_df[col]
        r2 = metrics['regression_metrics'][name]['Test_R2']
        rmse = metrics['regression_metrics'][name]['Test_RMSE']
        mae = metrics['regression_metrics'][name]['Test_MAE']
        
        ax.scatter(actual, pred, alpha=0.45, color=color, edgecolors='none', s=25, label='Observations')
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.8, label='Ideal Fit (y = x)')
        
        ax.set_title(f'{name}\n$R^2$: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}', fontweight='bold')
        ax.set_xlabel('Actual AQI', fontweight='bold')
        ax.set_ylabel('Predicted AQI', fontweight='bold')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend(loc='upper left')
        
    plt.suptitle('Actual vs Predicted AQI across Regression Models on Test Set', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_regression.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'actual_vs_predicted_regression.png')}")
    
    # -----------------------------------------------------------------
    # 2. RESIDUAL ANALYSIS (Random Forest Regressor)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    rf_preds = test_df['Pred_AQI_Random Forest']
    residuals = actual - rf_preds
    
    # Residual distribution
    sns.histplot(residuals, kde=True, ax=axes[0], color='#27ae60', bins=35)
    axes[0].set_title('Residual Error Distribution (Random Forest)', fontweight='bold')
    axes[0].set_xlabel('Residual (Actual - Predicted AQI)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1.5)
    axes[0].axvline(residuals.mean(), color='black', linestyle=':', label=f'Mean Error: {residuals.mean():.2f}')
    axes[0].legend()
    
    # Residuals vs Predicted
    axes[1].scatter(rf_preds, residuals, alpha=0.5, color='#2980b9', edgecolors='none', s=25)
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title('Residuals vs Predicted Values (Homoscedasticity Check)', fontweight='bold')
    axes[1].set_xlabel('Predicted AQI')
    axes[1].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'residual_analysis.png')}")
    
    # -----------------------------------------------------------------
    # 3. REGRESSION METRICS COMPARISON (Bar Chart)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models_list = list(metrics['regression_metrics'].keys())
    r2_scores = [metrics['regression_metrics'][m]['Test_R2'] for m in models_list]
    rmse_scores = [metrics['regression_metrics'][m]['Test_RMSE'] for m in models_list]
    
    colors = ['#3498db', '#e67e22', '#2ecc71']
    
    # R2 Comparison
    b1 = axes[0].bar(models_list, r2_scores, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title('Test $R^2$ Score (Higher is Better)', fontweight='bold')
    axes[0].set_ylabel('$R^2$ Score')
    axes[0].set_ylim(0.85, 1.02)
    for bar in b1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
    # RMSE Comparison
    b2 = axes[1].bar(models_list, rmse_scores, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title('Test RMSE Error (Lower is Better)', fontweight='bold')
    axes[1].set_ylabel('RMSE (AQI Units)')
    axes[1].set_ylim(0, max(rmse_scores) * 1.2)
    for bar in b2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
    plt.suptitle('Regression Model Performance Benchmark', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_model_comparison.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'regression_model_comparison.png')}")
    
    # -----------------------------------------------------------------
    # 4. CONFUSION MATRIX (Classification Task)
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6.5))
    classes = list(label_encoder.classes_)
    y_true_clf = test_df['Actual_Category']
    y_pred_clf = test_df['Pred_Category_Random Forest Classifier']
    
    cm = confusion_matrix(y_true_clf, y_pred_clf, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title('Confusion Matrix - Random Forest Classifier (AQI Category)', fontsize=14, pad=15)
    plt.xlabel('Predicted Category', fontweight='bold')
    plt.ylabel('Actual Category', fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_classifier.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'confusion_matrix_classifier.png')}")
    
    # -----------------------------------------------------------------
    # 5. CLASSIFICATION METRICS COMPARISON (Bar Chart)
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    clf_names = list(metrics['classification_metrics'].keys())
    accuracies = [metrics['classification_metrics'][m]['Test_Accuracy'] * 100 for m in clf_names]
    f1_scores = [metrics['classification_metrics'][m]['Test_F1_Weighted'] * 100 for m in clf_names]
    
    x = np.arange(len(clf_names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#3498db', edgecolor='black', alpha=0.85)
    plt.bar(x + width/2, f1_scores, width, label='Weighted F1 (%)', color='#2ecc71', edgecolor='black', alpha=0.85)
    
    plt.ylabel('Score (%)', fontweight='bold')
    plt.title('Classification Performance: Logistic Regression vs Random Forest', fontsize=14, pad=15)
    plt.xticks(x, clf_names, fontweight='bold')
    plt.ylim(65, 105)
    
    for i in range(len(clf_names)):
        plt.text(x[i] - width/2, accuracies[i] + 1, f'{accuracies[i]:.1f}%', ha='center', fontweight='bold')
        plt.text(x[i] + width/2, f1_scores[i] + 1, f'{f1_scores[i]:.1f}%', ha='center', fontweight='bold')
        
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_model_comparison.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'classification_model_comparison.png')}")
    
    # -----------------------------------------------------------------
    # 6. FEATURE IMPORTANCE RANKING (Random Forest Models)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Regressor feature importance
    reg_importances = pd.Series(rf_reg.feature_importances_, index=feature_names).sort_values(ascending=True)
    # Top 10 features
    top_reg = reg_importances.tail(10)
    top_reg.plot(kind='barh', ax=axes[0], color='#2980b9', edgecolor='black', alpha=0.85)
    axes[0].set_title('Top 10 Feature Importances (Random Forest Regressor)', fontweight='bold')
    axes[0].set_xlabel('Relative Gini Importance')
    for i, v in enumerate(top_reg):
        axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
        
    # Classifier feature importance
    clf_importances = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(ascending=True)
    top_clf = clf_importances.tail(10)
    top_clf.plot(kind='barh', ax=axes[1], color='#27ae60', edgecolor='black', alpha=0.85)
    axes[1].set_title('Top 10 Feature Importances (Random Forest Classifier)', fontweight='bold')
    axes[1].set_xlabel('Relative Gini Importance')
    for i, v in enumerate(top_clf):
        axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
        
    plt.suptitle('Atmospheric Predictor Importance Ranking', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_ranking.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'feature_importance_ranking.png')}")
    
    print("\nAll result visualizations successfully created!")

if __name__ == '__main__':
    generate_result_visualizations()
