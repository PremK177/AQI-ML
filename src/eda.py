"""
Exploratory Data Analysis (EDA) Module
Computes summary statistics and generates publication-grade visualizations
for air quality metrics, meteorological factors, seasonal variations, and AQI relationships.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic styling
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

def perform_eda(clean_data_path='data/air_quality_data_clean.csv', output_dir='plots/eda'):
    """Performs statistical EDA and produces all visual plots."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(clean_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 1. Summary Statistics
    numerical_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'Wind_Speed', 'AQI']
    stats = df[numerical_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats['skewness'] = df[numerical_cols].skew()
    print("\nDescriptive Statistics & Skewness:")
    print(stats.to_string())
    stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    # 2. Pollutant Distributions Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#2980b9', '#8e44ad']
    
    for ax, col, color in zip(axes.flatten(), pollutants, colors):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=30, alpha=0.6)
        ax.set_title(f'Distribution of {col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        median_val = df[col].median()
        ax.axvline(median_val, color='black', linestyle='--', linewidth=1.2, label=f'Median: {median_val:.1f}')
        ax.legend()
        
    plt.suptitle('Distribution of Criteria Air Pollutants with KDE', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pollutant_distributions.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'pollutant_distributions.png')}")
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[numerical_cols].corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}
    )
    plt.title('Correlation Matrix of Atmospheric Variables and AQI', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'correlation_heatmap.png')}")
    
    # 4. Seasonal Variation of AQI (Boxplot + Strip plot)
    plt.figure(figsize=(10, 6))
    season_order = ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon']
    palette = {'Winter': '#34495e', 'Summer': '#e67e22', 'Monsoon': '#3498db', 'Post-Monsoon': '#9b59b6'}
    sns.boxplot(x='Season', y='AQI', data=df, order=season_order, palette=palette, boxprops=dict(alpha=0.8), showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"8"})
    plt.title('Seasonal Distribution of Air Quality Index (AQI)', fontsize=14, pad=15)
    plt.xlabel('Season', fontweight='bold')
    plt.ylabel('AQI Value', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_aqi_distribution.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'seasonal_aqi_distribution.png')}")
    
    # 5. AQI Category Class Distribution
    plt.figure(figsize=(9, 5))
    bucket_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    bucket_colors = ['#2ecc71', '#a8e063', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    counts = df['AQI_Bucket'].value_counts().reindex(bucket_order).fillna(0)
    
    bars = plt.bar(counts.index, counts.values, color=bucket_colors, edgecolor='black', alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + 15, f'{int(height)}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.title('Frequency Distribution of AQI Health Categories', fontsize=14, pad=15)
    plt.xlabel('AQI Category', fontweight='bold')
    plt.ylabel('Observation Count', fontweight='bold')
    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aqi_category_distribution.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'aqi_category_distribution.png')}")
    
    # 6. Scatter plots: Pollutant vs AQI relationships
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # PM2.5 vs AQI
    sns.regplot(data=df, x='PM2.5', y='AQI', ax=axes[0, 0], scatter_kws={'alpha': 0.4, 'color': '#e74c3c', 's': 15}, line_kws={'color': 'darkred', 'linewidth': 2})
    axes[0, 0].set_title('PM2.5 vs AQI (Primary Driver)', fontweight='bold')
    
    # PM10 vs AQI
    sns.regplot(data=df, x='PM10', y='AQI', ax=axes[0, 1], scatter_kws={'alpha': 0.4, 'color': '#e67e22', 's': 15}, line_kws={'color': 'darkorange', 'linewidth': 2})
    axes[0, 1].set_title('PM10 vs AQI', fontweight='bold')
    
    # Wind Speed vs AQI (Atmospheric Dispersion Effect)
    sns.regplot(data=df, x='Wind_Speed', y='AQI', ax=axes[1, 0], scatter_kws={'alpha': 0.4, 'color': '#2980b9', 's': 15}, line_kws={'color': 'darkblue', 'linewidth': 2})
    axes[1, 0].set_title('Wind Speed vs AQI (Dispersion Ventilation)', fontweight='bold')
    
    # Temperature vs Ozone (Photochemical Production)
    sns.regplot(data=df, x='Temperature', y='O3', ax=axes[1, 1], scatter_kws={'alpha': 0.4, 'color': '#8e44ad', 's': 15}, line_kws={'color': 'purple', 'linewidth': 2})
    axes[1, 1].set_title('Temperature vs Ozone (Photochemical Reaction)', fontweight='bold')
    
    plt.suptitle('Key Atmospheric Driver Relationships', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pollutant_aqi_relationships.png'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'pollutant_aqi_relationships.png')}")
    
    print("\nEDA completed successfully! All plots and statistics generated.")

if __name__ == '__main__':
    perform_eda()
