"""EDA for the supplied India city-day air quality observations."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from preprocess import POLLUTANT_COLUMNS

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 180})


def perform_eda(clean_data_path='data/air_quality_data_clean.csv', output_dir='plots/eda'):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(clean_data_path, parse_dates=['Date'])
    pollutants = [c for c in POLLUTANT_COLUMNS if c in df]

    print('=' * 60)
    print('EXPLORATORY DATA ANALYSIS — CITY-DAY DATA')
    print('=' * 60)
    print(f'Labeled observations: {len(df):,}; cities: {df.City.nunique()}; dates: {df.Date.min().date()} to {df.Date.max().date()}')
    missing = (100 * df[pollutants].isna().mean()).round(2).rename('Missing_Percent')
    missing.to_csv(os.path.join(output_dir, 'pollutant_missingness.csv'))

    numerical_cols = pollutants + ['AQI']
    stats = df[numerical_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats['skewness'] = df[numerical_cols].skew()
    stats['missing_percent'] = (100 * df[numerical_cols].isna().mean())
    stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    print('\nSummary statistics:')
    print(stats.round(2).to_string())

    hist_cols = [c for c in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'] if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, hist_cols):
        sns.histplot(df[col].dropna(), ax=ax, bins=35, color='#3688b7')
        ax.set_title(f'{col} distribution')
        ax.set_xlabel(f'{col} concentration (dataset units)')
        ax.set_ylabel('Daily observations')
    fig.suptitle('Pollutant distributions (missing values omitted)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pollutant_distributions.png'), bbox_inches='tight')
    plt.close(fig)

    corr_cols = [c for c in pollutants + ['AQI'] if c in df]
    corr = df[corr_cols].corr(method='pearson')
    corr.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)), annot=True,
                fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title('Pollutant and AQI correlation matrix (pairwise complete rows)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x='Season', y='AQI', order=['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'],
                showfliers=False, ax=ax, color='#83b8d1')
    ax.set_title('Observed AQI by calendar-derived season')
    ax.set_ylabel('AQI')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'seasonal_aqi_distribution.png'), bbox_inches='tight')
    plt.close(fig)

    counts = df['AQI_Bucket'].value_counts().reindex(
        ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe'], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=['#43a66b', '#91bd55', '#e3c642', '#e28e35', '#cf584d', '#78508d'])
    ax.bar_label(bars, fmt='%d', padding=3)
    ax.set_title('AQI category counts among labeled observations')
    ax.set_ylabel('Daily observations')
    ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'aqi_category_distribution.png'), bbox_inches='tight')
    plt.close(fig)

    scatter_cols = [c for c in ['PM2.5', 'PM10', 'NO2', 'O3'] if c in df]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, col in zip(axes.flat, scatter_cols):
        sample = df[[col, 'AQI']].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        sns.regplot(data=sample, x=col, y='AQI', lowess=False,
                    scatter_kws={'alpha': 0.18, 's': 10}, line_kws={'color': '#bb4430'}, ax=ax)
        ax.set_title(f'{col} versus observed AQI')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'pollutant_aqi_relationships.png'), bbox_inches='tight')
    plt.close(fig)

    print(f'EDA outputs saved under {output_dir}')


if __name__ == '__main__':
    perform_eda()
