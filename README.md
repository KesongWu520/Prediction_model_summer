# Prediction_model_summer
### Global Probabilistic Forecasting and Evaluation Pipeline
### Recent Update
Code/0216_model_optuna/Optuna_Find_Optimal_Parameter_V1.py - Using optuna package to find the optimal hyperparameter of model

Code/0220_model_optuna_enhancer/optuna_enhancer_tune_v2.py - Using optuna package to find the optimal hyperparameter of model and enhancer data with following step:
[1] Log transform on target (log1p / expm1)

[2] PRED_NUM_SAMPLES = 1000

[3] LR scheduler: ReduceLROnPlateau

Code/Evaluation_R_code - Evaluation R code using slingshot package.



---

## 📖 Overview
This repository implements a **Global Probabilistic Time Series Forecasting Pipeline** designed for weekly data across multiple countries. By leveraging global models—where a single model is trained on data from all regions—this framework aims to capture cross-country patterns and improve predictive accuracy.

The project focuses on:
* **Global Training:** Training unified models across diverse geographical datasets.
* **Probabilistic Output:** Generating quantile forecasts to quantify uncertainty.
* **Rigorous Evaluation:** Utilizing rolling backtesting and the **Weighted Interval Score (WIS)** to assess performance.

---

## 🚀 Key Features
* **Data Alignment:** Synchronizes multi-country time series to a regular weekly grid and handles missing values.
* **Global Modeling:** Trains a single, shared forecasting model shared by all countries.
* **Rolling Backtests:** Performs evaluation across multiple forecast origins to ensure robustness.
* **Standardized Output:** Generates forecasts in a consistent quantile-based format.
* **Advanced Metrics:** Evaluates forecast quality using calibration and accuracy metrics in **R**.

---

## 🧠 Supported Models
The pipeline supports state-of-the-art global architectures:

* **Transformer:** Leverages self-attention mechanisms to capture complex temporal dependencies.

> **Note:** All models produce probabilistic forecasts via **Quantile Regression**, providing a full distribution of potential outcomes rather than a single point estimate.

---

## 📂 Outputs
The pipeline generates the following data and artifacts:
* **Quantile Forecasts:** CSV files organized by country and forecast date.
* **Backtest Summaries:** Detailed logs containing WIS scores for each rolling window.
* **Evaluation Plots:** Summary tables and visualizations covering:
    * **WIS (Weighted Interval Score)**
    * **Coverage Analysis**
    * **Forecast Horizon Analysis**

---

## 📊 Evaluation Metrics
To ensure consistent and reproducible results, we focus on:
1. **Weighted Interval Score (WIS):** The primary proper scoring rule used to evaluate quantile forecasts.
2. **Calibration:** Assessing the reliability of predicted intervals.
3. **Sharpness:** Measuring the concentration of the predictive distributions.

---

## 🎯 Purpose
This project is intended for **research and methodological evaluation** of global probabilistic forecasting models, particularly for multi-country or multi-region time series data. It provides a standardized framework to compare different neural network architectures in a unified environment.
