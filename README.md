# Prediction_model_summer
Global Probabilistic Forecasting and Evaluation Pipeline

Overview

This repository implements a global probabilistic time series forecasting pipeline for weekly data across multiple countries.

The project focuses on:
	•	Training single global models across all countries
	•	Generating probabilistic (quantile) forecasts using rolling backtesting
	•	Evaluating forecast quality using proper scoring rules, mainly the Weighted Interval Score (WIS)

The goal of this project is to compare forecasting models and assess predictive uncertainty in a consistent and reproducible way.

What This Project Does
	•	Aligns multi-country time series to a regular weekly grid and handles missing values
	•	Trains a global forecasting model shared by all countries
	•	Performs rolling backtests with multiple forecast origins
	•	Outputs forecasts in a standard quantile-based format
	•	Evaluates forecasts using calibration and accuracy metrics in R

Models

The pipeline supports the following global models:
	•	Transformer

All models produce probabilistic forecasts via quantile regression.

Outputs
	•	Quantile forecast CSV files for each country and forecast date
	•	Rolling backtest summary with WIS scores
	•	Evaluation plots and summary tables (WIS, coverage, horizon analysis)

Purpose

This project is intended for research and methodological evaluation of global probabilistic forecasting models, particularly for multi-country or multi-region time series data.
