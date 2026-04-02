"""
Evaluation utilities for gasoline price forecasting project.
Provides metrics, result saving/loading, and comparison visualizations.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred):
    """Compute all metrics and return as dict."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        'RMSE': round(rmse(y_true, y_pred), 6),
        'MAE': round(mae(y_true, y_pred), 6),
        'MAPE': round(mape(y_true, y_pred), 4),
    }


def save_results(model_name, horizon, metrics, predictions=None,
                 dates=None, feature_mode='full'):
    """
    Save model results to JSON file.

    Parameters
    ----------
    model_name : str, e.g. 'Linear Regression'
    horizon : int, prediction horizon (1 or 4)
    metrics : dict with RMSE, MAE, MAPE (should have 'val' and 'test' keys)
    predictions : dict with 'val' and 'test' arrays (optional)
    dates : dict with 'val' and 'test' date arrays (optional)
    feature_mode : 'basic' or 'full'
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    safe_name = model_name.replace(' ', '_').replace('/', '_').lower()
    filename = f'{safe_name}_h{horizon}_{feature_mode}.json'

    result = {
        'model_name': model_name,
        'horizon': horizon,
        'feature_mode': feature_mode,
        'metrics': metrics,
    }

    if predictions is not None:
        result['predictions'] = {
            k: v.tolist() if hasattr(v, 'tolist') else list(v)
            for k, v in predictions.items()
        }

    if dates is not None:
        result['dates'] = {
            k: [str(d) for d in v] for k, v in dates.items()
        }

    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {filepath}")


def load_all_results():
    """Load all saved results from the results directory."""
    results = []
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                results.append(json.load(f))
    return results


def build_comparison_table(results=None):
    """
    Build a comparison DataFrame from all saved results.

    Returns DataFrame with columns: Model, Horizon, Features, RMSE, MAE, MAPE
    """
    if results is None:
        results = load_all_results()

    rows = []
    for r in results:
        # Use test metrics if available, else val
        metrics = r['metrics'].get('test', r['metrics'].get('val', r['metrics']))
        rows.append({
            'Model': r['model_name'],
            'Horizon': r['horizon'],
            'Features': r['feature_mode'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE (%)': metrics['MAPE'],
        })

    df = pd.DataFrame(rows)
    return df.sort_values(['Horizon', 'RMSE']).reset_index(drop=True)


def plot_predictions_vs_actual(dates, y_true, y_pred, model_name, title=None):
    """Plot predicted vs actual prices."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, label='Actual', color='black', linewidth=1.5)
    ax.plot(dates, y_pred, label=f'{model_name}', color='tab:blue',
            linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($/gallon)')
    ax.set_title(title or f'{model_name}: Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparison_bar(comparison_df, metric='RMSE', horizon=1):
    """Bar chart comparing models on a given metric and horizon."""
    subset = comparison_df[comparison_df['Horizon'] == horizon].copy()
    subset = subset.sort_values(metric)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(subset)))
    bars = ax.barh(
        subset['Model'] + ' (' + subset['Features'] + ')',
        subset[metric],
        color=colors
    )
    ax.set_xlabel(metric)
    ax.set_title(f'Model Comparison - {metric} (Horizon={horizon} week)')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f' {width:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig
