# 2026-02-06
# Modified: integrated ERA5 weather covariates (2m_temperature, 2m_dewpoint_temperature, total_precipitation)

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

DATA_PATH = "data/training_data.csv"
OUTPUT_DIR = "./output_darts_transformer_filled_rnn/"

# --- ERA5 weather data configuration ---
WEATHER_DATA_DIR = "."  # directory containing era5_*.nc files
WEATHER_VARIABLES = ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"]
WEATHER_YEARS = list(range(2021, 2026))

# Country approximate bounding boxes [lat_min, lat_max, lon_min, lon_max]
# Used to compute country-level weekly mean from ERA5 gridded data
COUNTRY_BBOX = {
    "Belgium":        [49.5, 51.5, 2.5, 6.4],
    "Czech Republic": [48.5, 51.1, 12.1, 18.9],
    "France":         [41.3, 51.1, -5.1, 9.6],
    "Italy":          [36.6, 47.1, 6.6, 18.5],
    "Poland":         [49.0, 54.8, 14.1, 24.1],
    "Spain":          [36.0, 43.8, -9.3, 3.3],
    "Austria":        [46.4, 49.0, 9.5, 17.2],
    "Germany":        [47.3, 55.1, 5.9, 15.0],
    "Netherlands":    [50.8, 53.5, 3.4, 7.2],
    "Portugal":       [36.9, 42.2, -9.5, -6.2],
    "Sweden":         [55.3, 69.1, 11.1, 24.2],
    "Norway":         [58.0, 71.2, 4.6, 31.1],
    "Denmark":        [54.6, 57.8, 8.1, 15.2],
    "Finland":        [59.8, 70.1, 20.6, 31.6],
    "Ireland":        [51.4, 55.4, -10.5, -5.9],
    "United Kingdom": [49.9, 60.9, -8.2, 1.8],
    "Switzerland":    [45.8, 47.8, 5.9, 10.5],
    "Hungary":        [45.7, 48.6, 16.1, 22.9],
    "Slovakia":       [47.7, 49.6, 16.8, 22.6],
    "Slovenia":       [45.4, 46.9, 13.4, 16.6],
    "Croatia":        [42.4, 46.6, 13.5, 19.4],
    "Bulgaria":       [41.2, 44.2, 22.4, 28.6],
    "Greece":         [34.8, 41.7, 19.4, 29.6],
    "Estonia":        [57.5, 59.7, 21.8, 28.2],
    "Latvia":         [55.7, 58.1, 20.9, 28.2],
    "Lithuania":      [53.9, 56.5, 20.9, 26.8],
    "Luxembourg":     [49.4, 50.2, 5.7, 6.5],
    "Romania":        [43.6, 48.3, 20.3, 29.7],
    "Malta":          [35.8, 36.1, 14.3, 14.6],
    "Cyprus":         [34.6, 35.7, 32.3, 34.6],
}

TEST_WEEKS = 4
MAX_BACKTEST_WEEKS = 52
STEP_WEEKS = 1
MIN_TRAIN_WEEKS = 52

# Weekly frequency
FREQ = "7D"

# Filling strategy for missing values after reindexing to weekly grid:
FILL_STRATEGY = "ffill_bfill_zero"

# WIS configuration
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES = sorted(list(set([a / 2 for a in ALPHAS] + [1 - a / 2 for a in ALPHAS] + [0.5])))

# ============================================================
# Model selection
# ============================================================
MODEL_TYPE = "transformer"

MODEL_KWARGS_MAP = {
    "transformer": dict(
        input_chunk_length=26,
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
    ),
    "rnn": dict(
        input_chunk_length=16,
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
        input_chunk_length=26,
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

PRED_NUM_SAMPLES = 500

# ============================================================
# ERA5 Weather Data Loading
# ============================================================

def load_era5_weekly_country_means(
    weather_dir: str,
    variables: list,
    years: list,
    countries: list,
    freq: str = "7D",
) -> dict:
    """
    Load ERA5 .nc files and compute weekly country-level means for each variable.

    Returns:
        dict[country] -> pd.DataFrame with DatetimeIndex (weekly) and columns = variables
    """
    import xarray as xr

    print("\n[Loading ERA5 Weather Data]")

    # Load and concatenate all years for each variable
    var_datasets = {}
    for var in variables:
        ds_list = []
        for year in years:
            nc_file = os.path.join(weather_dir, f"era5_{var}_{year}.nc")
            if not os.path.exists(nc_file):
                print(f"  WARNING: {nc_file} not found, skipping.")
                continue
            ds_list.append(xr.open_dataset(nc_file))
        if ds_list:
            var_datasets[var] = xr.concat(ds_list, dim="valid_time")
            print(f"  Loaded {var}: {len(ds_list)} year files")
        else:
            print(f"  WARNING: No files found for {var}")

    if not var_datasets:
        print("  No weather data loaded. Returning empty.")
        return {}

    # For each country, extract spatial mean over bbox, then resample to weekly
    country_weather = {}

    for country in countries:
        if country not in COUNTRY_BBOX:
            print(f"  WARNING: No bbox defined for {country}, skipping weather.")
            continue

        lat_min, lat_max, lon_min, lon_max = COUNTRY_BBOX[country]

        dfs = []
        for var, ds in var_datasets.items():
            # Determine latitude/longitude dimension names
            if "latitude" in ds.dims:
                lat_dim, lon_dim = "latitude", "longitude"
            elif "lat" in ds.dims:
                lat_dim, lon_dim = "lat", "lon"
            else:
                # Fallback: try to find coordinate names
                lat_dim = [d for d in ds.dims if "lat" in d.lower()][0]
                lon_dim = [d for d in ds.dims if "lon" in d.lower()][0]

            # ERA5 latitude may be descending (90 to -90)
            lat_vals = ds[lat_dim].values
            if lat_vals[0] > lat_vals[-1]:
                # Descending latitude
                lat_slice = slice(lat_max, lat_min)
            else:
                lat_slice = slice(lat_min, lat_max)
            lon_slice = slice(lon_min, lon_max)

            sub = ds.sel(**{lat_dim: lat_slice, lon_dim: lon_slice})

            # Get the data variable name (first non-coordinate variable)
            data_vars = list(sub.data_vars)
            if len(data_vars) == 0:
                continue
            da = sub[data_vars[0]]

            # Determine time dimension name
            if "valid_time" in da.dims:
                time_dim = "valid_time"
            elif "time" in da.dims:
                time_dim = "time"
            else:
                time_dim = [d for d in da.dims if "time" in d.lower()][0]

            # Spatial mean
            spatial_dims = [d for d in da.dims if d != time_dim]
            daily_mean = da.mean(dim=spatial_dims).to_series()
            daily_mean.index = pd.to_datetime(daily_mean.index)
            daily_mean = daily_mean.sort_index()

            # Resample to weekly (matching the 7D grid of the target data)
            weekly_mean = daily_mean.resample("7D").mean()
            weekly_mean.name = var
            dfs.append(weekly_mean)

        if dfs:
            weather_df = pd.concat(dfs, axis=1)
            weather_df = weather_df.sort_index()
            # Fill any NaN from partial weeks at edges
            weather_df = weather_df.ffill().bfill().fillna(0.0)
            country_weather[country] = weather_df
            print(f"  {country}: weather shape {weather_df.shape}, "
                  f"range {weather_df.index.min().date()} to {weather_df.index.max().date()}")

    print(f"  Total countries with weather: {len(country_weather)}")
    return country_weather


def align_weather_to_panel(
    country_weather: dict,
    panel_index: pd.DatetimeIndex,
    country: str,
) -> pd.DataFrame:
    """
    Align weather data for a country to the target panel's weekly DatetimeIndex.
    Returns a DataFrame with the same index as panel_index, columns = weather variables.
    Missing values are forward/backward filled then zeroed.
    """
    if country not in country_weather or country_weather[country].empty:
        return pd.DataFrame(index=panel_index)

    wdf = country_weather[country].copy()
    # Reindex to panel's weekly dates
    wdf = wdf.reindex(panel_index, method="nearest", tolerance=pd.Timedelta("4D"))
    wdf = wdf.ffill().bfill().fillna(0.0)
    return wdf


def build_weather_covariate_series(
    country_weather: dict,
    panel_index: pd.DatetimeIndex,
    country: str,
    freq: str = "7D",
) -> TimeSeries:
    """
    Build a Darts TimeSeries of weather covariates for one country,
    aligned to the target panel's weekly index.
    Returns None if no weather data is available.
    """
    wdf = align_weather_to_panel(country_weather, panel_index, country)
    if wdf.empty or wdf.shape[1] == 0:
        return None

    wdf = wdf.reset_index().rename(columns={"index": "date"})
    return TimeSeries.from_dataframe(
        wdf,
        time_col="date",
        value_cols=[c for c in wdf.columns if c != "date"],
        fill_missing_dates=False,
        freq=freq,
    )


# ============================================================
# Utilities (unchanged from original)
# ============================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_name", "target", "date"]).reset_index(drop=True)

    print("\n[Data Overview]")
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Countries: {df['country_name'].unique().tolist()}")
    print(f"Targets: {df['target'].unique().tolist()}")

    return df


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

    panel = panel.sort_index()
    return panel


def build_country_series_from_panel(panel: pd.DataFrame, country: str) -> TimeSeries:
    dfc = panel[[country]].copy()
    dfc = dfc.reset_index().rename(columns={"index": "date", country: "value"})

    return TimeSeries.from_dataframe(
        dfc,
        time_col="date",
        value_cols="value",
        fill_missing_dates=False,
        freq=FREQ,
    )


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


def forecast_to_qdict(pred_ts: TimeSeries) -> dict:
    q_dict = {}
    vals = pred_ts.all_values(copy=False)

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
# Rolling Backtest (with weather covariates)
# ============================================================

def rolling_backtest_global_transformer(
    data_path,
    output_dir,
    test_weeks,
    max_backtest_weeks,
    step_weeks,
    model_type: str = MODEL_TYPE,
):
    print("=" * 70)
    print("Rolling Backtest - Global Transformer (with ERA5 Weather Covariates)")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    df = load_and_preprocess_data(data_path)

    countries = df["country_name"].unique().tolist()
    targets = df["target"].unique().tolist()

    # --- Load ERA5 weather data once ---
    country_weather = load_era5_weekly_country_means(
        weather_dir=WEATHER_DATA_DIR,
        variables=WEATHER_VARIABLES,
        years=WEATHER_YEARS,
        countries=countries,
    )
    use_weather = len(country_weather) > 0
    if use_weather:
        print(f"\n  Weather covariates enabled for {len(country_weather)} countries.")
    else:
        print("\n  WARNING: No weather data loaded. Running without covariates.")

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

        # Build one TimeSeries per country (target)
        full_series = {c: build_country_series_from_panel(panel, c) for c in countries}

        # Build weather covariate TimeSeries per country (aligned to panel index)
        full_weather_cov = {}
        if use_weather:
            for c in countries:
                cov_ts = build_weather_covariate_series(
                    country_weather, panel.index, c, freq=FREQ
                )
                if cov_ts is not None:
                    full_weather_cov[c] = cov_ts

            print(f"  Weather covariates built for {len(full_weather_cov)}/{len(countries)} countries.")

        ts_len = len(next(iter(full_series.values())))

        n_rounds = compute_max_rounds(ts_len, test_weeks, MIN_TRAIN_WEEKS, step_weeks, max_backtest_weeks)
        if n_rounds <= 0:
            print(f"  Not enough data for min_train={MIN_TRAIN_WEEKS} + test={test_weeks}. Skipping.")
            continue

        print(f"  Total backtest rounds: {n_rounds}")

        for round_idx in range(n_rounds):
            offset = round_idx * step_weeks
            print(f"\n  Round {round_idx + 1}/{n_rounds} | offset={offset}")

            train_list, test_list, meta = [], [], []
            cov_train_list = []  # past_covariates for training
            cov_full_list = []   # past_covariates full (for predict)

            for c in countries:
                train_ts, test_ts, forecast_date = split_train_test_by_offset(
                    full_series[c], test_weeks, offset, MIN_TRAIN_WEEKS
                )
                if train_ts is None:
                    continue
                train_list.append(train_ts)
                test_list.append(test_ts)
                meta.append((c, forecast_date, test_ts))

                # Split weather covariates in the same way
                if c in full_weather_cov:
                    n = len(full_series[c])
                    test_end = n - offset
                    train_end = test_end - test_weeks

                    cov_full = full_weather_cov[c]
                    # past_covariates for fit: same length as train series
                    cov_train = cov_full[:train_end]
                    cov_train_list.append(cov_train)
                    # past_covariates for predict: need up to train_end + test_weeks
                    # (past_covariates must cover the prediction horizon)
                    cov_pred = cov_full[:test_end]
                    cov_full_list.append(cov_pred)
                else:
                    cov_train_list.append(None)
                    cov_full_list.append(None)

            if not train_list:
                print("    No valid series for this round. Stopping.")
                break

            # Fit scaler on ALL training series
            scaler = Scaler()
            train_scaled = scaler.fit_transform(train_list)

            # Fit weather covariate scaler (separate from target scaler)
            has_covariates = all(c is not None for c in cov_train_list) and len(cov_train_list) > 0
            cov_train_scaled = None
            cov_full_scaled = None
            cov_scaler = None

            if has_covariates:
                cov_scaler = Scaler()
                cov_train_scaled = cov_scaler.fit_transform(cov_train_list)
                cov_full_scaled = cov_scaler.transform(cov_full_list)

            model = build_model(model_type=model_type, target=target, round_idx=round_idx)

            if has_covariates:
                model.fit(
                    series=train_scaled,
                    past_covariates=cov_train_scaled,
                    verbose=True,
                )
            else:
                model.fit(series=train_scaled, verbose=True)

            # Batch predict
            preds_scaled = []
            for i in range(len(meta)):
                predict_kwargs = dict(
                    n=test_weeks,
                    series=train_scaled[i],
                    num_samples=PRED_NUM_SAMPLES,
                )
                if has_covariates:
                    predict_kwargs["past_covariates"] = cov_full_scaled[i]

                preds_scaled.append(model.predict(**predict_kwargs))

            preds = scaler.inverse_transform(preds_scaled)

            # Evaluate and write outputs
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
                    "Model": f"Darts-{model_type.upper()}(QuantileRegression)+Weather",
                    "Forecast_Date": forecast_date.strftime("%Y-%m-%d"),
                    "Backtest_Round": round_idx + 1,
                    "Offset_Weeks": offset,
                    "Train_Weeks": len(train_list[i]),
                    "WIS": wis,
                })

                try:
                    out_df = generate_output_format(forecast_date, target, country, pred)
                    safe_country = country.replace(" ", "_")
                    safe_target = target.replace(" ", "_")
                    fname = f"forecast_output_{safe_country}_{safe_target}_{forecast_date.strftime('%Y-%m-%d')}.csv"
                    out_df.to_csv(os.path.join(output_dir, fname), index=False)
                except Exception as e:
                    print(f"    {country}: failed to write output CSV: {e}")

                print(f"    {country}: Train={len(train_list[i])}w Test={test_weeks}w -> WIS={wis}")

    # Save summary
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
    rolling_backtest_global_transformer(
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

Filling:
  - Missing weekly timestamps are inserted on a 7D grid.
  - Missing values are filled using FILL_STRATEGY='{FILL_STRATEGY}'.
  
Weather Covariates:
  - ERA5 variables: {WEATHER_VARIABLES}
  - Years: {WEATHER_YEARS}
  - Used as past_covariates in Darts model.
"""
    )