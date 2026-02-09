# 2026-02-06 (v4) FULL SCRIPT - WEATHER ONLY (NO HOLIDAY)
# Global Darts Transformer backtest with:
# - ERA5 weather covariates from .nc files (2021-2025): 2m_temperature, 2m_dewpoint_temperature, total_precipitation
# - Weekly alignment to Sundays (W-SUN), matching truth_date (Sunday)
# - total_precipitation aggregated as weekly SUM and converted from m -> mm
# - Derived absolute_humidity (g/m^3) from temperature + dewpoint
# - Weather lags (1-4 weeks)
# - Avoid covariates "all() veto": always provide covariates, fill missing with zeros
# - Avoid bfill leakage: fill targets with ffill + 0 only
# - Force float32 for ALL covariates to fix dtype error: double != float
# - Longer input chunk + seasonal encoders for Transformer
#
# Expected files:
#   data/new_train_data.csv
#   ./era5_2m_temperature_2021.nc ... _2025.nc
#   ./era5_2m_dewpoint_temperature_2021.nc ... _2025.nc
#   ./era5_total_precipitation_2021.nc ... _2025.nc
#
# Output:
#   ./output_darts_transformer_weather_only_v4/wis_summary.csv
#   ./output_darts_transformer_weather_only_v4/forecast_output_*.csv

import os
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scoring import weighted_interval_score_fast

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, RNNModel, TCNModel
from darts.utils.likelihood_models import QuantileRegression

# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/new_train_data.csv"
OUTPUT_DIR = "./output_darts_transformer_weather_only_v4/"

WEATHER_DATA_DIR = "."  # where the .nc files are
WEATHER_VARIABLES_RAW = ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"]
WEATHER_YEARS = list(range(2021, 2026))

# Derived + lags
WEATHER_BASE_COLS = ["absolute_humidity", "2m_temperature", "total_precipitation"]
WEATHER_LAGS = [1, 2, 3, 4]

TEST_WEEKS = 4
MAX_BACKTEST_WEEKS = 52
STEP_WEEKS = 1
MIN_TRAIN_WEEKS = 52

FREQ = "7D"              # target weekly grid used throughout
WEEK_RESAMPLE = "W-SUN"  # aligns to truth_date (Sunday)

FILL_STRATEGY = "ffill_zero"  # avoid bfill leakage

# WIS configuration
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES = sorted(list(set([a / 2 for a in ALPHAS] + [1 - a / 2 for a in ALPHAS] + [0.5])))

PRED_NUM_SAMPLES = 300
MODEL_TYPE = "transformer"

# Country approximate bounding boxes [lat_min, lat_max, lon_min, lon_max]
COUNTRY_BBOX = {
    "Belgium":        [49.5, 51.5, 2.5, 6.4],
    "Czech Republic": [48.5, 51.1, 12.1, 18.9],
    "France":         [41.3, 51.1, -5.1, 9.6],
    "Italy":          [36.6, 47.1, 6.6, 18.5],
    "Poland":         [49.0, 54.8, 14.1, 24.1],
    "Spain":          [36.0, 43.8, -9.3, 3.3],
}

MODEL_KWARGS_MAP = {
    "transformer": dict(
        input_chunk_length=52,
        output_chunk_length=TEST_WEEKS,
        batch_size=32,
        n_epochs=100,
        d_model=32,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        random_state=42,
        save_checkpoints=False,
        force_reset=True,
        add_encoders={
            "cyclic": {"future": ["weekofyear"]},
            "datetime_attribute": {"future": ["month"]},
        },
    ),
    "rnn": dict(
        input_chunk_length=26,
        output_chunk_length=TEST_WEEKS,
        batch_size=32,
        n_epochs=100,
        model="LSTM",
        hidden_dim=32,
        n_rnn_layers=2,
        dropout=0.1,
        random_state=42,
        force_reset=True,
    ),
    "tcn": dict(
        input_chunk_length=52,
        output_chunk_length=TEST_WEEKS,
        batch_size=32,
        n_epochs=100,
        num_filters=16,
        kernel_size=3,
        dilation_base=2,
        dropout=0.1,
        random_state=42,
        force_reset=True,
    ),
}

# ============================================================
# WIS helpers
# ============================================================

def evaluate_model_wis(y_true: np.ndarray, q_dict: dict):
    y_true = np.asarray(y_true).flatten()

    if len(y_true) == 0 or np.isnan(y_true).any():
        return "N/A"

    for q, arr in q_dict.items():
        arr = np.asarray(arr).flatten()
        if len(arr) != len(y_true) or np.isnan(arr).any():
            return "N/A"

    try:
        wis_total, _, _ = weighted_interval_score_fast(
            observations=y_true,
            alphas=ALPHAS,
            q_dict=q_dict,
            weights=None,
            percent=False,
            check_consistency=True,
        )
        mean_wis = float(np.nanmean(wis_total))
        if np.isnan(mean_wis):
            return "N/A"
        return round(mean_wis, 4)
    except Exception:
        return "N/A"


def forecast_to_qdict(pred_ts: TimeSeries) -> dict:
    q_dict = {}
    vals = pred_ts.all_values(copy=False)

    # sample-based forecasts
    if vals.ndim == 3 and vals.shape[2] > 1:
        sample_matrix = vals[:, 0, :].T
        for q in QUANTILES:
            q_dict[q] = np.percentile(sample_matrix, q * 100.0, axis=0)
        return q_dict

    dfq = pred_ts.pd_dataframe()
    col_to_q = {}
    for col in dfq.columns:
        s = str(col)
        m = re.search(r"([01](?:\.\d+)?)", s)
        if m:
            q_val = float(m.group(1))
            if 0.0 <= q_val <= 1.0:
                col_to_q[q_val] = col

    for q in QUANTILES:
        if q not in col_to_q:
            raise ValueError(f"Quantile {q} not found in forecast columns: {list(dfq.columns)}")
        q_dict[q] = dfq[col_to_q[q]].to_numpy().flatten()

    return q_dict


# ============================================================
# Output formatting
# ============================================================

def country_to_location_code(country: str) -> str:
    mapping = {
        "Belgium": "BE", "Czech Republic": "CZ", "France": "FR",
        "Italy": "IT", "Poland": "PL", "Spain": "ES",
    }
    return mapping.get(country, country[:2].upper())


def generate_output_format(forecast_date, target, country, pred_ts) -> pd.DataFrame:
    location = country_to_location_code(country)
    times = pred_ts.time_index
    horizon = len(pred_ts)
    q_dict = forecast_to_qdict(pred_ts)

    rows = []
    for h in range(1, horizon + 1):
        target_end_date = pd.Timestamp(times[h - 1])

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
# Data loading / panel building
# ============================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Robust date parsing (your file uses 2021/6/27)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["date"])

    df["country_name"] = df["country_name"].astype(str)
    df = df.sort_values(["country_name", "target", "date"]).reset_index(drop=True)

    print("\n[Data Overview]")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Countries: {df['country_name'].unique().tolist()}")
    print(f"Targets: {df['target'].unique().tolist()}")

    return df


def build_full_weekly_index(dates: pd.Series) -> pd.DatetimeIndex:
    start = pd.to_datetime(dates.min())
    end = pd.to_datetime(dates.max())
    return pd.date_range(start=start, end=end, freq=FREQ)


def fill_series_values(s: pd.Series, strategy: str) -> pd.Series:
    if strategy == "zero":
        return s.fillna(0.0)
    if strategy == "ffill_zero":
        return s.ffill().fillna(0.0)
    if strategy == "ffill_bfill_zero":
        return s.ffill().bfill().fillna(0.0)
    if strategy == "linear":
        s2 = s.astype(float).interpolate(method="time")
        return s2.fillna(0.0)
    raise ValueError(f"Unknown FILL_STRATEGY='{strategy}'")


def build_filled_panel_for_target(df: pd.DataFrame, target: str, countries: list) -> pd.DataFrame:
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
        s = s.groupby(level=0).mean()
        s = s.reindex(full_idx)
        s = fill_series_values(s, FILL_STRATEGY)
        panel[c] = s.astype(np.float32)

    return panel.sort_index()


def build_country_series_from_panel(panel: pd.DataFrame, country: str) -> TimeSeries:
    dfc = panel[[country]].copy()
    dfc = dfc.reset_index().rename(columns={"index": "date", country: "value"})
    dfc["value"] = dfc["value"].astype(np.float32)
    return TimeSeries.from_dataframe(dfc, time_col="date", value_cols="value", fill_missing_dates=False, freq=FREQ)


# ============================================================
# ERA5 loading (W-SUN aligned) + AH + lags
# ============================================================

def load_era5_weekly_country_means(weather_dir: str, variables: list, years: list, countries: list) -> dict:
    import xarray as xr

    print("\n[Loading ERA5 Weather Data - aligned W-SUN]")

    var_datasets = {}
    for var in variables:
        files = []
        for y in years:
            f = os.path.join(weather_dir, f"era5_{var}_{y}.nc")
            if os.path.exists(f):
                files.append(f)
        if not files:
            print(f"  WARNING: no files for {var}")
            continue
        var_datasets[var] = xr.open_mfdataset(files, combine="by_coords")
        print(f"  Loaded {var}: {len(files)} files")

    if not var_datasets:
        return {}

    country_weather = {}

    for country in countries:
        if country not in COUNTRY_BBOX:
            continue

        lat_min, lat_max, lon_min, lon_max = COUNTRY_BBOX[country]
        out_hourly = {}

        for var, ds in var_datasets.items():
            lat_dim = "latitude" if "latitude" in ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds.dims else "lon"

            lat_vals = ds[lat_dim].values
            lat_slice = slice(lat_max, lat_min) if lat_vals[0] > lat_vals[-1] else slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)

            sub = ds.sel(**{lat_dim: lat_slice, lon_dim: lon_slice})
            data_vars = list(sub.data_vars)
            if not data_vars:
                continue
            da = sub[data_vars[0]]

            time_dim = "valid_time" if "valid_time" in da.dims else (
                "time" if "time" in da.dims else [d for d in da.dims if "time" in d.lower()][0]
            )

            spatial_dims = [d for d in da.dims if d != time_dim]
            s = da.mean(dim=spatial_dims).to_series()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index().astype(np.float64)

            # unit fix: tp m -> mm
            if var == "total_precipitation":
                s = s * 1000.0

            out_hourly[var] = s

        if not out_hourly:
            continue

        df_hourly = pd.concat(out_hourly, axis=1).sort_index()

        weekly_index = df_hourly.resample(WEEK_RESAMPLE).mean().index
        weekly = pd.DataFrame(index=weekly_index)

        # weekly mean for temperature / dewpoint
        if "2m_temperature" in df_hourly.columns:
            weekly["2m_temperature"] = df_hourly["2m_temperature"].resample(WEEK_RESAMPLE).mean()
        if "2m_dewpoint_temperature" in df_hourly.columns:
            weekly["2m_dewpoint_temperature"] = df_hourly["2m_dewpoint_temperature"].resample(WEEK_RESAMPLE).mean()

        # weekly sum for precipitation
        if "total_precipitation" in df_hourly.columns:
            weekly["total_precipitation"] = df_hourly["total_precipitation"].resample(WEEK_RESAMPLE).sum()

        # derive absolute humidity (g/m^3)
        if "2m_temperature" in weekly.columns and "2m_dewpoint_temperature" in weekly.columns:
            T_k = weekly["2m_temperature"].astype(np.float64)
            Td_c = weekly["2m_dewpoint_temperature"].astype(np.float64) - 273.15
            e = 6.112 * np.exp((17.67 * Td_c) / (Td_c + 243.5))  # hPa
            weekly["absolute_humidity"] = 216.7 * e / T_k

        weekly = weekly.ffill().fillna(0.0)
        country_weather[country] = weekly

    print(f"  Total countries with weather: {len(country_weather)}")
    return country_weather


def align_weather_to_panel(
    country_weather: dict,
    panel_index: pd.DatetimeIndex,
    country: str,
) -> pd.DataFrame:
    # weather aligned to panel
    if country in country_weather and not country_weather[country].empty:
        wdf = country_weather[country].reindex(panel_index).ffill().fillna(0.0)
    else:
        wdf = pd.DataFrame(index=panel_index, columns=WEATHER_BASE_COLS, data=0.0)

    base_cols = [c for c in WEATHER_BASE_COLS if c in wdf.columns]
    if not base_cols:
        wdf = pd.DataFrame(index=panel_index, columns=WEATHER_BASE_COLS, data=0.0)
        base_cols = WEATHER_BASE_COLS

    wdf = wdf[base_cols].copy()

    # lags
    for lag in WEATHER_LAGS:
        for col in base_cols:
            wdf[f"{col}_lag{lag}"] = wdf[col].shift(lag)

    # fill and enforce float32 (fix dtype mismatch)
    wdf = wdf.ffill().fillna(0.0)
    wdf = wdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    wdf = wdf.astype(np.float32)

    return wdf


def build_weather_covariate_series(
    country_weather: dict,
    panel_index: pd.DatetimeIndex,
    country: str,
) -> TimeSeries:
    cdf = align_weather_to_panel(country_weather, panel_index, country)
    cdf = cdf.astype(np.float32)
    cdf = cdf.reset_index().rename(columns={"index": "date"})
    return TimeSeries.from_dataframe(
        cdf,
        time_col="date",
        value_cols=[c for c in cdf.columns if c != "date"],
        fill_missing_dates=False,
        freq=FREQ,
    )


# ============================================================
# Model Factory
# ============================================================

def build_model(model_type: str, target: str, round_idx: int):
    model_type = (model_type or "transformer").lower().strip()
    if model_type not in MODEL_KWARGS_MAP:
        raise ValueError(f"Unknown MODEL_TYPE='{model_type}'. Choose from {list(MODEL_KWARGS_MAP.keys())}.")

    common = dict(
        likelihood=QuantileRegression(quantiles=QUANTILES),
        model_name=f"{model_type}_global_{target.replace(' ', '_')}_r{round_idx+1}",
    )

    if model_type == "transformer":
        return TransformerModel(**common, **MODEL_KWARGS_MAP[model_type])
    if model_type == "rnn":
        return RNNModel(**common, **MODEL_KWARGS_MAP[model_type])
    if model_type == "tcn":
        return TCNModel(**common, **MODEL_KWARGS_MAP[model_type])

    raise ValueError(f"Unhandled MODEL_TYPE='{model_type}'")


# ============================================================
# Rolling backtest
# ============================================================

def split_train_test_by_offset(ts: TimeSeries, test_weeks: int, offset: int, min_train_weeks: int):
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


def compute_max_rounds(ts_len: int, test_weeks: int, min_train_weeks: int, step_weeks: int, user_max_backtest_weeks: int):
    max_offset = ts_len - (min_train_weeks + test_weeks)
    if max_offset < 0:
        return 0
    max_rounds_len = max_offset // step_weeks + 1
    max_rounds_user = user_max_backtest_weeks // step_weeks
    return int(min(max_rounds_len, max_rounds_user))


def rolling_backtest_global(
    data_path: str,
    output_dir: str,
    test_weeks: int,
    max_backtest_weeks: int,
    step_weeks: int,
    model_type: str = MODEL_TYPE,
) -> pd.DataFrame:
    print("=" * 70)
    print("Rolling Backtest - Global Model (ERA5 Weather ONLY)")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    df = load_and_preprocess_data(data_path)

    countries = df["country_name"].unique().tolist()
    targets = df["target"].unique().tolist()

    # Weather series
    country_weather = load_era5_weekly_country_means(
        weather_dir=WEATHER_DATA_DIR,
        variables=WEATHER_VARIABLES_RAW,
        years=WEATHER_YEARS,
        countries=countries,
    )
    use_weather = len(country_weather) > 0
    print(f"[Weather] enabled={use_weather} (countries with weather={len(country_weather)})")

    wis_records = []

    for target in targets:
        print(f"\n{'='*70}")
        print(f"Processing target: {target}")
        print("=" * 70)

        panel = build_filled_panel_for_target(df, target, countries)
        if panel.empty:
            print("  Panel is empty. Skipping.")
            continue

        print(f"  Weekly panel length: {len(panel)} weeks | Fill strategy: {FILL_STRATEGY}")

        # Target series per country
        full_series = {c: build_country_series_from_panel(panel, c) for c in countries}

        # Weather covariates per country (always built, filled with 0 if missing)
        full_cov = {c: build_weather_covariate_series(country_weather, panel.index, c) for c in countries}
        any_country = countries[0]
        try:
            print(f"  Weather covariates width (features): {full_cov[any_country].width}")
        except Exception:
            pass

        ts_len = len(next(iter(full_series.values())))
        n_rounds = compute_max_rounds(ts_len, test_weeks, MIN_TRAIN_WEEKS, step_weeks, max_backtest_weeks)
        if n_rounds <= 0:
            print(f"  Not enough data for min_train={MIN_TRAIN_WEEKS} + test={test_weeks}. Skipping.")
            continue

        print(f"  Total backtest rounds: {n_rounds}")

        for round_idx in range(n_rounds):
            offset = round_idx * step_weeks
            print(f"\n  Round {round_idx + 1}/{n_rounds} | offset={offset}")

            train_list, meta = [], []
            cov_train_list, cov_full_list = [], []

            for c in countries:
                train_ts, test_ts, forecast_date = split_train_test_by_offset(
                    full_series[c], test_weeks, offset, MIN_TRAIN_WEEKS
                )
                if train_ts is None:
                    continue

                train_list.append(train_ts)
                meta.append((c, forecast_date, test_ts))

                n = len(full_series[c])
                test_end = n - offset
                train_end = test_end - test_weeks

                cov_full = full_cov[c]
                cov_train = cov_full[:train_end]
                cov_pred = cov_full[:test_end]

                cov_train_list.append(cov_train)
                cov_full_list.append(cov_pred)

            if not train_list:
                print("    No valid series for this round. Stopping.")
                break

            # Scale targets
            scaler_y = Scaler()
            train_scaled = scaler_y.fit_transform(train_list)

            # Scale covariates
            scaler_x = Scaler()
            cov_train_scaled = scaler_x.fit_transform(cov_train_list)
            cov_full_scaled = scaler_x.transform(cov_full_list)

            # Debug dtype (optional)
            # print("dtype y:", train_scaled[0].values().dtype, "dtype x:", cov_train_scaled[0].values().dtype)

            model = build_model(model_type=model_type, target=target, round_idx=round_idx)

            model.fit(
                series=train_scaled,
                past_covariates=cov_train_scaled,
                verbose=True,
            )

            preds_scaled = []
            for i in range(len(meta)):
                preds_scaled.append(
                    model.predict(
                        n=test_weeks,
                        series=train_scaled[i],
                        past_covariates=cov_full_scaled[i],
                        num_samples=PRED_NUM_SAMPLES,
                    )
                )

            preds = scaler_y.inverse_transform(preds_scaled)

            for i, (country, forecast_date, test_ts) in enumerate(meta):
                pred = preds[i]
                y_true = test_ts.values().flatten()

                try:
                    q_dict = forecast_to_qdict(pred)
                    wis = evaluate_model_wis(y_true, q_dict)
                except Exception as e:
                    print(f"    {country}: failed to compute WIS/quantiles: {e}")
                    wis = "N/A"

                wis_records.append({
                    "Country": country,
                    "Target": target,
                    "Model": f"Darts-{model_type.upper()}(QuantileRegression)+ERA5",
                    "Forecast_Date": forecast_date.strftime("%Y-%m-%d"),
                    "Backtest_Round": round_idx + 1,
                    "Offset_Weeks": offset,
                    "Train_Weeks": len(train_list[i]),
                    "WIS": wis,
                })

                try:
                    out_df = generate_output_format(forecast_date, target, country, pred)
                    safe_country = country.replace(" ", "_")
                    safe_target = str(target).replace(" ", "_")
                    fname = f"forecast_output_{safe_country}_{safe_target}_{forecast_date.strftime('%Y-%m-%d')}.csv"
                    out_df.to_csv(os.path.join(output_dir, fname), index=False)
                except Exception as e:
                    print(f"    {country}: failed to write output CSV: {e}")

                print(f"    {country}: Train={len(train_list[i])}w Test={test_weeks}w -> WIS={wis}")

    print("\n" + "=" * 70)
    print("Saving WIS Summary")
    print("=" * 70)

    if not wis_records:
        print("  No records saved.")
        return pd.DataFrame()

    wis_summary = pd.DataFrame(wis_records).sort_values(["Target", "Country", "Backtest_Round"])
    wis_path = os.path.join(output_dir, "wis_summary.csv")
    wis_summary.to_csv(wis_path, index=False)
    print(f"  Saved: {wis_path} ({len(wis_summary)} rows)")

    return wis_summary


if __name__ == "__main__":
    rolling_backtest_global(
        DATA_PATH,
        OUTPUT_DIR,
        TEST_WEEKS,
        MAX_BACKTEST_WEEKS,
        STEP_WEEKS,
        model_type=MODEL_TYPE,
    )

    print("\n" + "=" * 70)
    print("Rolling Backtest Complete!")
    print("=" * 70)
    print(
        f"""
Output Files:
  - {OUTPUT_DIR}wis_summary.csv
  - {OUTPUT_DIR}forecast_output_[country]_[target]_[date].csv

Target filling:
  - Weekly grid: FREQ='{FREQ}', aligned to Sundays
  - Missing values filled: FILL_STRATEGY='{FILL_STRATEGY}'

Covariates:
  - ERA5 raw: {WEATHER_VARIABLES_RAW} (tp weekly SUM, tp m->mm)
  - Derived: absolute_humidity + lags {WEATHER_LAGS}
"""
    )