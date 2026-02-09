# Benchmark_Darts040_AutoARIMA.py
# Darts==0.40 ARIMA-family benchmark using AutoARIMA (statsforecast backend).
# Produces probabilistic forecasts via sampling and writes outputs in EXACT same schema
# as your deep-learning generate_output_format().
# 2026-02-04

import os
import re
import warnings
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from darts import TimeSeries
from darts.models.forecasting.sf_auto_arima import AutoARIMA  # Darts 0.40: AutoARIMA

# ============================================================
# Configuration (match your DL script)
# ============================================================

DATA_PATH = "data/training_data.csv"
GOOGLE_TREND_PATH = "data/Google_Trend.csv"  # <-- NEW: Google Trend data path
OUTPUT_DIR = "./output_darts_arima_benchmark/"

TEST_WEEKS = 4
MAX_BACKTEST_WEEKS = 52
STEP_WEEKS = 1
MIN_TRAIN_WEEKS = 52

FREQ = "7D"

# Filling strategy for missing values after reindexing to weekly grid:
#   "zero"               -> fill missing with 0
#   "ffill_bfill_zero"   -> forward fill, then backward fill, then 0
#   "linear"             -> linear interpolation, then 0
FILL_STRATEGY = "ffill_bfill_zero"

# WIS/quantile configuration (same as DL)
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES = sorted(list(set([a / 2 for a in ALPHAS] + [1 - a / 2 for a in ALPHAS] + [0.5])))

# Sampling size for probabilistic forecast
PRED_NUM_SAMPLES = 500

# AutoARIMA kwargs (weekly: 52 often works; adjust if needed)
AUTO_ARIMA_KWARGS = dict(
    season_length=52,
)

# Mode: keep as "samples" to guarantee quantiles differ
MODE = "samples"  # "samples" (recommended). "direct_quantiles" optional.


# ============================================================
# Utilities (aligned with your DL script)
# ============================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Expected columns: date, country_name, target, value
    (Your file may contain extra columns; we only use these.)
    """
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_name", "target", "date"]).reset_index(drop=True)

    print("\n[Data Overview]")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Countries: {df['country_name'].unique().tolist()}")
    print(f"Targets: {df['target'].unique().tolist()}")

    return df


# ============================================================
# NEW: Google Trend loading and panel building
# ============================================================

def load_google_trend(filepath: str) -> pd.DataFrame:
    """
    Load Google Trend data.
    Expected columns: h (date), flu (trend value), Location (country name)
    Returns DataFrame with columns: date, country_name, flu
    """
    gt = pd.read_csv(filepath)
    gt = gt.rename(columns={"h": "date", "Location": "country_name"})
    gt["date"] = pd.to_datetime(gt["date"])
    gt["flu"] = pd.to_numeric(gt["flu"], errors="coerce")
    gt = gt.sort_values(["country_name", "date"]).reset_index(drop=True)

    print("\n[Google Trend Data Overview]")
    print(f"Shape: {gt.shape}")
    print(f"Date range: {gt['date'].min().date()} to {gt['date'].max().date()}")
    print(f"Countries: {gt['country_name'].unique().tolist()}")

    return gt


def build_google_trend_panel(gt_df: pd.DataFrame, full_idx: pd.DatetimeIndex, countries: list) -> pd.DataFrame:
    """
    Build a weekly panel for Google Trend 'flu' values aligned to the same
    full_idx used by the target panel.

    For each country:
      - Reindex to full_idx
      - Fill missing values using FILL_STRATEGY

    Returns DataFrame with index=full_idx, columns=countries, values=float32.
    Countries not present in Google Trend data will have all-zero columns.
    """
    panel = pd.DataFrame(index=full_idx)

    gt_countries = gt_df["country_name"].unique().tolist()

    for c in countries:
        if c in gt_countries:
            s = gt_df[gt_df["country_name"] == c].set_index("date")["flu"].sort_index()
            s = s.groupby(level=0).mean()
            s = s.reindex(full_idx)
            s = fill_series_values(s, FILL_STRATEGY)
        else:
            # Country not in Google Trend data -> fill with zeros
            s = pd.Series(0.0, index=full_idx)

        panel[c] = s.astype(np.float32)

    panel = panel.sort_index()
    return panel


def build_country_covariate_from_panel(panel: pd.DataFrame, country: str) -> TimeSeries:
    """
    Build a Darts TimeSeries for a covariate from a filled weekly panel column.
    """
    dfc = panel[[country]].copy()
    dfc = dfc.reset_index().rename(columns={"index": "date", country: "value"})

    return TimeSeries.from_dataframe(
        dfc,
        time_col="date",
        value_cols="value",
        fill_missing_dates=False,
        freq=FREQ,
    )


# ============================================================
# (Original utilities unchanged)
# ============================================================

def country_to_location_code(country: str) -> str:
    mapping = {
        "Belgium": "BE", "Czech Republic": "CZ", "France": "FR",
        "Poland": "PL", "Romania": "RO", "Austria": "AT",
        "Germany": "DE", "Italy": "IT", "Spain": "ES",
        "Netherlands": "NL", "Portugal": "PT", "Sweden": "SE",
        "Norway": "NO", "Denmark": "DK", "Finland": "FI",
        "Ireland": "IE", "United Kingdom": "GB", "Switzerland": "CH",
        "Hungary": "HU", "Slovakia": "SK", "Slovenia": "SI",
        "Croatia": "HR", "Bulgaria": "BG", "Greece": "GR",
        "Estonia": "EE", "Latvia": "LV", "Lithuania": "LT",
        "Luxembourg": "LU", "Malta": "MT", "Cyprus": "CY",
    }
    return mapping.get(country, country[:2].upper())


def build_full_weekly_index(dates: pd.Series) -> pd.DatetimeIndex:
    start = pd.to_datetime(dates.min())
    end = pd.to_datetime(dates.max())
    return pd.date_range(start=start, end=end, freq=FREQ)


def fill_series_values(s: pd.Series, strategy: str) -> pd.Series:
    if strategy == "zero":
        return s.fillna(0.0)
    if strategy == "ffill_bfill_zero":
        return s.ffill().bfill().fillna(0.0)
    if strategy == "linear":
        s2 = s.astype(float).interpolate(method="time")
        return s2.fillna(0.0)
    raise ValueError(f"Unknown FILL_STRATEGY='{strategy}'")


def build_filled_panel_for_target(df: pd.DataFrame, target: str, countries: list) -> pd.DataFrame:
    """
    Build a weekly wide panel for one target, with filled values.
    index: weekly dates
    columns: countries
    """
    sub = df[df["target"] == target].copy()
    sub = sub[sub["country_name"].isin(countries)]
    sub["date"] = pd.to_datetime(sub["date"])
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")

    if sub.empty:
        return pd.DataFrame()

    full_idx = build_full_weekly_index(sub["date"])
    panel = pd.DataFrame(index=full_idx)

    for c in countries:
        s = sub[sub["country_name"] == c].set_index("date")["value"].sort_index()
        s = s.groupby(level=0).mean()   # handle duplicates
        s = s.reindex(full_idx)         # weekly grid
        s = fill_series_values(s, FILL_STRATEGY)
        panel[c] = s.astype(np.float32)

    return panel.sort_index()


def build_darts_series(panel: pd.DataFrame, country: str) -> TimeSeries:
    dfc = panel[[country]].copy().reset_index().rename(columns={"index": "date", country: "value"})
    return TimeSeries.from_dataframe(
        dfc,
        time_col="date",
        value_cols="value",
        fill_missing_dates=False,
        freq=FREQ,
    )


def split_train_test_by_offset(ts: TimeSeries, test_weeks: int, offset: int, min_train_weeks: int):
    """
    Rolling split: same logic as your DL code.
    """
    n = len(ts)
    test_end = n - offset
    test_start = test_end - test_weeks
    train_end = test_start

    if train_end < min_train_weeks:
        return None, None, None

    train_ts = ts[:train_end]
    test_ts = ts[test_start:test_end]
    forecast_date = test_ts.start_time() - timedelta(days=7)
    return train_ts, test_ts, forecast_date


# ============================================================
# NEW: Split future covariate aligned with target split
# ============================================================

def split_future_covariate_for_arima(cov_ts: TimeSeries, train_end: int, test_weeks: int):
    """
    Split future covariate for AutoARIMA.
    future_covariates must cover the training period AND extend into the prediction horizon.
    So we need covariates from start up to train_end + test_weeks.
    """
    cov_end = train_end + test_weeks
    cov_end = min(cov_end, len(cov_ts))
    return cov_ts[:cov_end]


def compute_max_rounds(ts_len: int, test_weeks: int, min_train_weeks: int, step_weeks: int, user_max_backtest_weeks: int):
    max_offset = ts_len - (min_train_weeks + test_weeks)
    if max_offset < 0:
        return 0
    max_rounds_len = max_offset // step_weeks + 1
    max_rounds_user = user_max_backtest_weeks // step_weeks
    return int(min(max_rounds_len, max_rounds_user))


# ============================================================
# Forecast conversion: TimeSeries -> q_dict
# ============================================================

def forecast_to_qdict(pred_ts: TimeSeries) -> Dict[float, np.ndarray]:
    """
    Convert forecast TimeSeries to q_dict required by WIS/output.

    Primary path (recommended): stochastic samples -> compute quantiles across samples.
    Fallback: try parse quantile columns from to_dataframe().
    Last fallback: replicate point forecast across quantiles.
    """
    q_dict: Dict[float, np.ndarray] = {}
    vals = pred_ts.all_values(copy=False)

    # Case 1: stochastic forecast with sample dimension: (time, component, sample)
    if vals.ndim == 3 and vals.shape[2] > 1:
        sample_matrix = vals[:, 0, :].T  # (n_samples, horizon)
        for q in QUANTILES:
            q_dict[q] = np.percentile(sample_matrix, q * 100.0, axis=0)
        return q_dict

    # Case 2: deterministic multi-component - parse quantiles from column names
    dfq = pred_ts.to_dataframe()  # Darts 0.40 compatible

    col_to_q = {}
    for col in dfq.columns:
        s = str(col)
        m = re.search(r"([01](?:\.\d+)?)", s)
        if m:
            q_val = float(m.group(1))
            if 0.0 <= q_val <= 1.0:
                col_to_q[q_val] = col

    if col_to_q:
        for q in QUANTILES:
            if q in col_to_q:
                q_dict[q] = dfq[col_to_q[q]].to_numpy().flatten()
            else:
                nearest = min(col_to_q.keys(), key=lambda x: abs(x - q))
                q_dict[q] = dfq[col_to_q[nearest]].to_numpy().flatten()
        return q_dict

    # Case 3: last fallback - replicate point forecast
    point = pred_ts.values().flatten()
    for q in QUANTILES:
        q_dict[q] = point
    return q_dict


def generate_output_format_from_qdict(forecast_date, target, country, time_index, q_dict) -> pd.DataFrame:
    """
    EXACT same schema as your DL generate_output_format().
    """
    location = country_to_location_code(country)
    horizon = len(time_index)

    rows = []
    for h in range(1, horizon + 1):
        target_end_date = pd.Timestamp(time_index[h - 1])

        # Median row
        rows.append({
            "origin_date": forecast_date.strftime("%Y-%m-%d"),
            "target": target,
            "target_end_date": target_end_date.strftime("%Y-%m-%d"),
            "horizon": h,
            "location": location,
            "output_type": "median",
            "output_type_id": "",
            "value": float(q_dict[0.5][h - 1]),
        })

        # Quantile rows
        for q in QUANTILES:
            rows.append({
                "origin_date": forecast_date.strftime("%Y-%m-%d"),
                "target": target,
                "target_end_date": target_end_date.strftime("%Y-%m-%d"),
                "horizon": h,
                "location": location,
                "output_type": "quantile",
                "output_type_id": q,
                "value": float(q_dict[q][h - 1]),
            })

    return pd.DataFrame(rows)


# ============================================================
# Rolling Backtest - main
# ============================================================

def rolling_backtest_darts040_autoarima(
    data_path: str,
    output_dir: str,
    test_weeks: int,
    max_backtest_weeks: int,
    step_weeks: int,
    google_trend_path: str = GOOGLE_TREND_PATH,  # <-- NEW parameter
):
    print("=" * 70)
    print("Rolling Backtest - Darts 0.40 AutoARIMA Benchmark (probabilistic)")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    df = load_and_preprocess_data(data_path)

    # ---- NEW: Load Google Trend data ----
    gt_df = None
    if google_trend_path and os.path.exists(google_trend_path):
        gt_df = load_google_trend(google_trend_path)
    else:
        print(f"\n[WARNING] Google Trend file not found at '{google_trend_path}'. Running without covariates.")

    countries = df["country_name"].unique().tolist()
    targets = df["target"].unique().tolist()

    for target in targets:
        print(f"\n{'='*70}")
        print(f"Processing target: {target}")
        print("=" * 70)

        panel = build_filled_panel_for_target(df, target, countries)
        if panel.empty:
            print("  Panel is empty. Skipping.")
            continue

        series_map = {c: build_darts_series(panel, c) for c in countries}
        ts_len = len(next(iter(series_map.values())))

        # ---- NEW: Build Google Trend covariate panel aligned to target panel ----
        gt_panel = None
        full_cov_series = {}
        if gt_df is not None:
            full_idx = panel.index
            gt_panel = build_google_trend_panel(gt_df, full_idx, countries)
            full_cov_series = {c: build_country_covariate_from_panel(gt_panel, c) for c in countries}
            gt_countries_available = [c for c in countries if c in gt_df["country_name"].unique()]
            print(f"  Google Trend covariate built for {len(gt_countries_available)} countries: {gt_countries_available}")

        n_rounds = compute_max_rounds(ts_len, test_weeks, MIN_TRAIN_WEEKS, step_weeks, max_backtest_weeks)
        if n_rounds <= 0:
            print(f"  Not enough data for min_train={MIN_TRAIN_WEEKS} + test={test_weeks}. Skipping.")
            continue

        print(f"  Weekly panel length: {len(panel)} weeks | Fill strategy: {FILL_STRATEGY}")
        print(f"  Total backtest rounds: {n_rounds}")
        print(f"  MODE: {MODE}")
        print(f"  AUTO_ARIMA_KWARGS: {AUTO_ARIMA_KWARGS}")
        print(f"  PRED_NUM_SAMPLES: {PRED_NUM_SAMPLES}")

        for round_idx in range(n_rounds):
            offset = round_idx * step_weeks
            print(f"\n  Round {round_idx + 1}/{n_rounds} | offset={offset}")

            for country in countries:
                train_ts, test_ts, forecast_date = split_train_test_by_offset(
                    series_map[country], test_weeks, offset, MIN_TRAIN_WEEKS
                )
                if train_ts is None:
                    continue

                try:
                    # ---- NEW: Prepare future covariate for this country ----
                    fut_cov = None
                    if full_cov_series and country in full_cov_series:
                        cov_ts = full_cov_series[country]
                        train_end = len(train_ts)
                        fut_cov = split_future_covariate_for_arima(cov_ts, train_end, test_weeks)

                    model = AutoARIMA(**AUTO_ARIMA_KWARGS, quantiles=QUANTILES)

                    # ---- MODIFIED: fit with future_covariates ----
                    model.fit(train_ts, future_covariates=fut_cov)

                    # ---- MODIFIED: predict with future_covariates ----
                    if MODE == "direct_quantiles":
                        pred_ts = model.predict(
                            n=test_weeks,
                            future_covariates=fut_cov,
                            predict_likelihood_parameters=True,
                        )
                    else:
                        pred_ts = model.predict(
                            n=test_weeks,
                            future_covariates=fut_cov,
                            num_samples=PRED_NUM_SAMPLES,
                        )

                    q_dict = forecast_to_qdict(pred_ts)
                    out_df = generate_output_format_from_qdict(
                        forecast_date=forecast_date,
                        target=target,
                        country=country,
                        time_index=pred_ts.time_index,
                        q_dict=q_dict,
                    )

                    safe_country = country.replace(" ", "_")
                    safe_target = target.replace(" ", "_")
                    fname = f"forecast_output_{safe_country}_{safe_target}_{forecast_date.strftime('%Y-%m-%d')}.csv"
                    out_df.to_csv(os.path.join(output_dir, fname), index=False)

                    print(f"    {country}: wrote {fname}")

                except Exception as e:
                    print(f"    {country}: AutoARIMA failed: {e}")

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)
    print(
        f"""
Output Files:
  - {OUTPUT_DIR}forecast_output_[country]_[target]_[date].csv

Format:
  - Same columns/rows as your DL generate_output_format()

Covariates:
  - Google Trend 'flu' search index used as future_covariates for AutoARIMA (when available).
  - Countries without Google Trend data run without covariates.

Notes:
  - MODE='samples' ensures quantiles are computed from sampled forecasts, so they differ.
  - If AutoARIMA fails due to missing dependency, install:
      pip install statsforecast
"""
    )


if __name__ == "__main__":
    rolling_backtest_darts040_autoarima(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        test_weeks=TEST_WEEKS,
        max_backtest_weeks=MAX_BACKTEST_WEEKS,
        step_weeks=STEP_WEEKS,
        google_trend_path=GOOGLE_TREND_PATH,  # <-- NEW
    )