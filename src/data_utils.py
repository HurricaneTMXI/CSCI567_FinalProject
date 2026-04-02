"""
Data utilities for gasoline price forecasting project.
Handles data loading, feature engineering, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
import os

# Column name mapping for the EIA gasoline/diesel dataset
COLUMN_NAMES = {
    'A1': 'AllGrades_AllFormulations',
    'A2': 'AllGrades_Conventional',
    'A3': 'AllGrades_Reformulated',
    'R1': 'Regular_AllFormulations',
    'R2': 'Regular_Conventional',
    'R3': 'Regular_Reformulated',
    'M1': 'Midgrade_AllFormulations',
    'M2': 'Midgrade_Conventional',
    'M3': 'Midgrade_Reformulated',
    'P1': 'Premium_AllFormulations',
    'P2': 'Premium_Conventional',
    'P3': 'Premium_Reformulated',
    'D1': 'Diesel',
}

TARGET_COL = 'A1'  # All Grades, All Formulations


def load_raw_data(data_path=None):
    """Load the raw CSV dataset."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'raw',
            'PET_PRI_GND_DCUS_NUS_W.csv'
        )
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def clean_data(df):
    """Clean the dataset: handle missing values, ensure numeric types."""
    # Convert all price columns to numeric (coerce errors to NaN)
    price_cols = [c for c in df.columns if c != 'Date']
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Forward-fill then back-fill any missing values (rare in this dataset)
    df[price_cols] = df[price_cols].ffill().bfill()

    return df


def add_features(df, target_col=TARGET_COL, lags=None, rolling_windows=None):
    """
    Add time series features for modeling.

    Parameters
    ----------
    df : DataFrame with Date and price columns
    target_col : target price column
    lags : list of lag periods (default: [1, 2, 3, 4, 8, 12])
    rolling_windows : list of rolling window sizes (default: [4, 8, 12])

    Returns
    -------
    DataFrame with added features
    """
    if lags is None:
        lags = [1, 2, 3, 4, 8, 12]
    if rolling_windows is None:
        rolling_windows = [4, 8, 12]

    df = df.copy()

    # Lag features of target
    for lag in lags:
        df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)

    # Rolling statistics
    for w in rolling_windows:
        df[f'{target_col}_rolling_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        df[f'{target_col}_rolling_std_{w}'] = df[target_col].shift(1).rolling(w).std()

    # Price change features
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_4'] = df[target_col].diff(4)
    df[f'{target_col}_pct_change_1'] = df[target_col].pct_change(1)
    df[f'{target_col}_pct_change_4'] = df[target_col].pct_change(4)

    # Date features
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['year'] = df['Date'].dt.year
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    # Month sin/cos encoding for cyclical nature
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Other grade prices as features (lagged by 1 to avoid leakage)
    other_price_cols = [c for c in ['R1', 'M1', 'P1', 'D1'] if c != target_col]
    for col in other_price_cols:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)

    return df


def create_target(df, target_col=TARGET_COL, horizon=1):
    """
    Create the prediction target: price at t+horizon.

    Returns df with a 'target' column. Rows where target is NaN (end of series)
    should be dropped before modeling.
    """
    df = df.copy()
    df['target'] = df[target_col].shift(-horizon)
    return df


def get_feature_columns(df, mode='full'):
    """
    Get feature column names based on mode.

    Parameters
    ----------
    mode : 'basic' (only lag features) or 'full' (all engineered features)

    Returns
    -------
    list of column names
    """
    exclude = {'Date', 'target', TARGET_COL,
                'A1', 'A2', 'A3', 'R1', 'R2', 'R3',
                'M1', 'M2', 'M3', 'P1', 'P2', 'P3', 'D1'}

    if mode == 'basic':
        # Only lag features of the target
        return [c for c in df.columns
                if c.startswith(f'{TARGET_COL}_lag') and c in df.columns]
    else:
        # All engineered features
        return [c for c in df.columns if c not in exclude]


def train_val_test_split(df, train_ratio=0.70, val_ratio=0.15):
    """
    Chronological train/val/test split.

    Parameters
    ----------
    df : DataFrame (must be sorted by Date)
    train_ratio : fraction for training
    val_ratio : fraction for validation
    test_ratio = 1 - train_ratio - val_ratio

    Returns
    -------
    train_df, val_df, test_df
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def prepare_data(horizon=1, feature_mode='full'):
    """
    End-to-end data preparation pipeline.

    Returns
    -------
    dict with keys: train, val, test (DataFrames), feature_cols, dates
    """
    df = load_raw_data()
    df = clean_data(df)
    df = add_features(df)
    df = create_target(df, horizon=horizon)

    # Drop rows with NaN (from lag/rolling features and target shift)
    df = df.dropna().reset_index(drop=True)

    feature_cols = get_feature_columns(df, mode=feature_mode)
    train_df, val_df, test_df = train_val_test_split(df)

    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'feature_cols': feature_cols,
        'full_df': df,
    }
