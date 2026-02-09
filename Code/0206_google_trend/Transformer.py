# 2026-02-06

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
GOOGLE_TREND_PATH = "data/Google_Trend.csv"  # <-- NEW: Google Trend data path
OUTPUT_DIR = "./output_darts_transformer_filled_v2/"

TEST_WEEKS = 4
MAX_BACKTEST_WEEKS = 52
STEP_WEEKS = 1
MIN_TRAIN_WEEKS = 52

# Weekly frequency
FREQ = "7D"

# Filling strategy for missing values after reindexing to weekly grid:
#   "zero"               -> fill missing with 0
#   "ffill_bfill_zero"   -> forward fill, then backward fill, then 0 (recommended general-purpose)
#   "linear"             -> linear interpolation, then 0
FILL_STRATEGY = "ffill_bfill_zero"

# WIS configuration
ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES = sorted(list(set([a / 2 for a in ALPHAS] + [1 - a / 2 for a in ALPHAS] + [0.5])))

# ============================================================
# Model selection
# ============================================================
# Choose model type: "transformer", "rnn", "tcn"
MODEL_TYPE = "transformer"

# Model hyperparameters by type (tune as needed)
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
        input_chunk_length=26,
        output_chunk_length=TEST_WEEKS,
        batch_size=32,
        n_epochs=100,
        model="LSTM",            # "RNN", "LSTM", or "GRU"
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
# Utilities
# ============================================================

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load and basic preprocess.
    Expected columns: date, country_name, target, value
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

def evaluate_model_wis(y_true: np.ndarray, q_dict: dict):
    """
    Compute mean WIS. Returns float or 'N/A'.
    """
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
    """
    Convert country name to an ISO-like code.
    """
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
    """
    Build a full weekly DatetimeIndex (7D) spanning [min_date, max_date].
    """
    start = pd.to_datetime(dates.min())
    end = pd.to_datetime(dates.max())
    # Ensure the index is strictly weekly at 7-day step.
    return pd.date_range(start=start, end=end, freq=FREQ)


def fill_series_values(s: pd.Series, strategy: str) -> pd.Series:
    """
    Fill missing values for a reindexed weekly series.
    """
    if strategy == "zero":
        return s.fillna(0.0)

    if strategy == "ffill_bfill_zero":
        return s.ffill().bfill().fillna(0.0)

    if strategy == "linear":
        # Linear interpolation along time, then zeros for any remaining
        # (interpolate requires numeric dtype)
        s2 = s.astype(float).interpolate(method="time")
        return s2.fillna(0.0)

    raise ValueError(f"Unknown FILL_STRATEGY='{strategy}'")


def build_filled_panel_for_target(df: pd.DataFrame, target: str, countries: list) -> pd.DataFrame:
    """
    Build a weekly panel (wide format) for one target:
      - Create a full weekly index covering the target's date span
      - For each country: reindex to this full index and fill missing values
      - Return panel with columns in the same order as `countries`

    Output:
      index: weekly dates (regular)
      columns: countries
      values: float32 (filled)
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

        # If there are duplicate dates, aggregate (mean)
        s = s.groupby(level=0).mean()

        # Reindex to full weekly grid
        s = s.reindex(full_idx)

        # Fill missing values
        s = fill_series_values(s, FILL_STRATEGY)

        panel[c] = s.astype(np.float32)

    panel = panel.sort_index()
    return panel


def build_country_series_from_panel(panel: pd.DataFrame, country: str) -> TimeSeries:
    """
    Build a Darts TimeSeries from a filled weekly panel column.
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


def split_train_test_by_offset(ts: TimeSeries, test_weeks: int, offset: int, min_train_weeks: int):
    """
    Rolling split for one TimeSeries.
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
# NEW: Split covariate series aligned with target split
# ============================================================

def split_covariate_by_offset(cov_ts: TimeSeries, target_ts: TimeSeries, train_end: int, test_weeks: int):
    """
    Split covariate TimeSeries to align with the target's train/test split.
    For past_covariates: we need covariates covering up to train_end + test_weeks
    (so the model can use them during prediction).
    The covariate is sliced to cover the same time range as train + test of target.
    """
    # past_covariates should cover the full range: from start up to end of test period
    cov_end = train_end + test_weeks
    cov_end = min(cov_end, len(cov_ts))
    return cov_ts[:cov_end]


def compute_max_rounds(ts_len: int, test_weeks: int, min_train_weeks: int, step_weeks: int, user_max_backtest_weeks: int):
    """
    Max number of rolling rounds allowed by series length and user cap.
    """
    max_offset = ts_len - (min_train_weeks + test_weeks)
    if max_offset < 0:
        return 0
    max_rounds_len = max_offset // step_weeks + 1
    max_rounds_user = user_max_backtest_weeks // step_weeks
    return int(min(max_rounds_len, max_rounds_user))


def forecast_to_qdict(pred_ts: TimeSeries) -> dict:
    """
    Convert forecast TimeSeries to q_dict required by WIS.

    Handles:
      - Stochastic forecasts (time, component, sample) -> percentiles across samples
      - Deterministic multi-component forecasts -> parse quantiles from component names
    """
    q_dict = {}
    vals = pred_ts.all_values(copy=False)

    # Case 1: stochastic forecast with sample dimension
    if vals.ndim == 3 and vals.shape[2] > 1:
        sample_matrix = vals[:, 0, :].T  # (n_samples, horizon)
        for q in QUANTILES:
            q_dict[q] = np.percentile(sample_matrix, q * 100.0, axis=0)
        return q_dict

    # Case 2: deterministic multi-component
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
    """
    Standard output schema (same as your original).
    """
    location = country_to_location_code(country)
    times = pred_ts.time_index
    horizon = len(pred_ts)

    q_dict = forecast_to_qdict(pred_ts)

    rows = []
    for h in range(1, horizon + 1):
        target_end_date = pd.Timestamp(times[h - 1])

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
# Model Factory
# ============================================================

def build_model(model_type: str, target: str, round_idx: int):
    """Factory to build a Darts model based on MODEL_TYPE."""
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

    # Defensive fallback
    raise ValueError(f"Unhandled MODEL_TYPE='{model_type}'")


# ============================================================
# Rolling Backtest
# ============================================================

def rolling_backtest_global_transformer(
    data_path,
    output_dir,
    test_weeks,
    max_backtest_weeks,
    step_weeks,
    model_type: str = MODEL_TYPE,
    google_trend_path: str = GOOGLE_TREND_PATH,  # <-- NEW parameter
):
    print("=" * 70)
    print("Rolling Backtest - Global Transformer (Filling Missing Weekly Values)")
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

    wis_records = []

    for target in targets:
        print(f"\n{'='*70}")
        print(f"Processing target: {target}")
        print("=" * 70)

        # Build a filled, regular weekly panel for this target
        panel = build_filled_panel_for_target(df, target, countries)
        if panel.empty:
            print("  Panel is empty. Skipping.")
            continue

        print(f"  Weekly panel length: {len(panel)} weeks | Fill strategy: {FILL_STRATEGY}")

        # ---- NEW: Build Google Trend covariate panel aligned to target panel ----
        gt_panel = None
        full_cov_series = {}
        if gt_df is not None:
            full_idx = panel.index
            gt_panel = build_google_trend_panel(gt_df, full_idx, countries)
            full_cov_series = {c: build_country_covariate_from_panel(gt_panel, c) for c in countries}
            gt_countries_available = [c for c in countries if c in gt_df["country_name"].unique()]
            print(f"  Google Trend covariate built for {len(gt_countries_available)} countries: {gt_countries_available}")

        # Build one TimeSeries per country
        full_series = {c: build_country_series_from_panel(panel, c) for c in countries}
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
            past_cov_train_list = []  # <-- NEW: covariates for training
            past_cov_pred_list = []   # <-- NEW: covariates for prediction
            for c in countries:
                train_ts, test_ts, forecast_date = split_train_test_by_offset(
                    full_series[c], test_weeks, offset, MIN_TRAIN_WEEKS
                )
                if train_ts is None:
                    continue
                train_list.append(train_ts)
                test_list.append(test_ts)
                meta.append((c, forecast_date, test_ts))

                # ---- NEW: Split covariate aligned with target ----
                if full_cov_series:
                    cov_ts = full_cov_series[c]
                    train_end = len(train_ts)
                    # For training: covariate covering the training period
                    past_cov_train_list.append(cov_ts[:train_end])
                    # For prediction: covariate covering train + test period
                    # (past_covariates need to extend up to prediction horizon)
                    cov_for_pred = split_covariate_by_offset(cov_ts, train_ts, train_end, test_weeks)
                    past_cov_pred_list.append(cov_for_pred)

            if not train_list:
                print("    No valid series for this round. Stopping.")
                break

            # Fit scaler on ALL training series
            scaler = Scaler()
            train_scaled = scaler.fit_transform(train_list)

            # ---- NEW: Scale covariates ----
            cov_scaler = None
            past_cov_train_scaled = None
            past_cov_pred_scaled = None
            if past_cov_train_list:
                cov_scaler = Scaler()
                past_cov_train_scaled = cov_scaler.fit_transform(past_cov_train_list)
                past_cov_pred_scaled = cov_scaler.transform(past_cov_pred_list)

            model = build_model(model_type=model_type, target=target, round_idx=round_idx)

            # ---- MODIFIED: fit with past_covariates ----
            model.fit(
                series=train_scaled,
                past_covariates=past_cov_train_scaled,  # <-- NEW
                verbose=True,
            )

            # Batch predict and batch inverse_transform to avoid scaler mismatch
            preds_scaled = []
            for i in range(len(meta)):
                # ---- MODIFIED: predict with past_covariates ----
                pred_kwargs = dict(
                    n=test_weeks,
                    series=train_scaled[i],
                    num_samples=PRED_NUM_SAMPLES,
                )
                if past_cov_pred_scaled is not None:
                    pred_kwargs["past_covariates"] = past_cov_pred_scaled[i]

                preds_scaled.append(model.predict(**pred_kwargs))

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
                    "Model": f"Darts-{model_type.upper()}(QuantileRegression)",
                    "Forecast_Date": forecast_date.strftime("%Y-%m-%d"),
                    "Backtest_Round": round_idx + 1,
                    "Offset_Weeks": offset,
                    "Train_Weeks": len(train_list[i]),
                    "WIS": wis,
                })

                # Write forecast output file
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
        google_trend_path=GOOGLE_TREND_PATH,  # <-- NEW
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

Covariates:
  - Google Trend 'flu' search index used as past_covariates (when available).
  - Countries without Google Trend data use zero-filled covariates.
"""
    )