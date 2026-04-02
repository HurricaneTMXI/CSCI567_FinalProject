# CSCI 567 Final Project Plan
## Weekly Gasoline Price Forecasting - Comparative ML Study

Team: Simou Chen, Khalid Ali, Mario Prado, Jasper Fan-Chiang

---

## Overview

**Research Question:** How do different ML models compare when forecasting weekly U.S. gasoline prices? Do complex models (neural networks, ensembles) meaningfully outperform simpler regression approaches? How does performance vary across stable vs volatile market conditions?

**Dataset:** U.S. Gasoline and Diesel Retail Prices (1995-2021), ~1361 weekly observations, from Kaggle/EIA.

**Primary Target:** `A1` (All Grades, All Formulations retail price). Other grade columns used as additional features.

---

## Notebook Structure

### `01_data_cleaning_and_eda.ipynb`
- Load and inspect raw CSV
- Parse dates, handle missing values
- Visualize price trends, distributions, volatility over time
- Identify stable vs volatile periods (for later analysis)
- Feature engineering: lag features, rolling stats, date features, price momentum
- Chronological train/val/test split (70/15/15)
- Save cleaned & feature-engineered data to `data/cleaned/`

### `02_baseline_and_linear_models.ipynb`
- Naive baseline (predict last known price)
- Linear Regression
- Ridge Regression (with alpha tuning)
- Lasso Regression (with alpha tuning)
- Evaluate all on 1-week and 4-week horizons
- Compare basic vs engineered features
- Save results to `results/`

### `03_tree_based_models.ipynb`
- Random Forest (with hyperparameter tuning)
- XGBoost (with hyperparameter tuning)
- Feature importance analysis
- Evaluate on 1-week and 4-week horizons
- Save results to `results/`

### `04_neural_network_models.ipynb`
- MLP (varying depth, width, activation)
- LSTM (sequence-based, varying architecture)
- Training curves and convergence analysis
- Evaluate on 1-week and 4-week horizons
- Save results to `results/`

### `05_model_comparison_and_analysis.ipynb`
- Load all saved results
- **Comparison 1:** Full results table (RMSE, MAE, MAPE for all models x both horizons)
- **Comparison 2:** Simple vs complex -- is extra complexity worth it?
- **Comparison 3:** Stable vs volatile periods -- which models handle price spikes?
- **Comparison 4:** 1-week vs 4-week horizon -- how does accuracy degrade?
- **Comparison 5:** Feature engineering impact across models
- Prediction vs actual plots for all models
- Error distribution analysis
- Key findings and takeaways for the final report

---

## Shared Utilities

- `src/data_utils.py` -- data loading, feature engineering, train/val/test split
- `src/evaluate.py` -- RMSE, MAE, MAPE, result saving/loading, comparison tables & plots

---

## Project Structure
```
CSCI567_FinalProject/
├── PLAN.md
├── data/
│   ├── raw/              # Original downloaded data
│   └── cleaned/          # Processed data
├── notebooks/
│   ├── 01_data_cleaning_and_eda.ipynb
│   ├── 02_baseline_and_linear_models.ipynb
│   ├── 03_tree_based_models.ipynb
│   ├── 04_neural_network_models.ipynb
│   └── 05_model_comparison_and_analysis.ipynb
├── src/
│   ├── data_utils.py
│   └── evaluate.py
├── results/              # Saved model results (JSON)
└── report/
    └── final_report.pdf
```

## Key Deadlines
| Milestone | Date |
|-----------|------|
| Pre-final check-in meeting | April 20-24, 2026 |
| Final report | Finals week (TBD) |

## Task Division (Suggested)
Split so each person owns 1-2 model families + shared work:
- **Person A:** Data cleaning, EDA, Linear/Ridge/Lasso
- **Person B:** Random Forest / XGBoost, feature importance
- **Person C:** MLP, hyperparameter tuning
- **Person D:** LSTM, results aggregation & analysis
- **Everyone:** Final report writing, review
