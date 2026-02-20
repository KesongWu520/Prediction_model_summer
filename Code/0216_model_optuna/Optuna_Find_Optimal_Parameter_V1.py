# ============================================================
# Conservative Optuna Tuning + Full Backtest Output
#
# Two phases:
#   Phase 1: Optuna tunes 3 params using full 52-round WIS evaluation
#   Phase 2: Re-run full backtest with best params, outputting
#            forecast files compatible with R evaluation script
#
# Output format (same as retrain_optimized.py):
#   - forecast_{Country}_{Target}_{date}.csv  (per-round per-country)
#   - all_forecasts.csv                       (consolidated)
#   Columns: origin_date, target, target_end_date, horizon, location,
#            output_type, output_type_id, value, backtest_round, known_weeks
#
# Install: pip install optuna
# ============================================================

import os
import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import optuna
from optuna.samplers import TPESampler

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import Callback

from scoring import weighted_interval_score_fast

# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/new_train_data.csv"
TUNING_DIR = "./output_optuna_conservative/"    # Optuna logs
FORECAST_DIR = "./output_optuna_best_forecast/" # Final forecast output for R

HOLIDAY_COL = "is_holiday_week"

WEATHER_DATA_DIR = "."
WEATHER_VARIABLES = ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"]
WEATHER_YEARS = list(range(2021, 2026))

COUNTRY_BBOX = {
    "Belgium":        [49.5, 51.5, 2.5, 6.4],
    "Czech Republic": [48.5, 51.1, 12.1, 18.9],
    "France":         [41.3, 51.1, -5.1, 9.6],
    "Italy":          [36.6, 47.1, 6.6, 18.5],
    "Poland":         [49.0, 54.8, 14.1, 24.1],
    "Spain":          [36.0, 43.8, -9.3, 3.3],
}

TEST_WEEKS = 4
MAX_BACKTEST_WEEKS = 52
STEP_WEEKS = 1
MIN_TRAIN_WEEKS = 52
FREQ = "7D"
FILL_STRATEGY = "ffill_bfill_zero"

ALPHAS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANTILES = sorted(list(set(
    [a / 2 for a in ALPHAS] + [1 - a / 2 for a in ALPHAS] + [0.5]
)))

PRED_NUM_SAMPLES = 500

# ============================================================
# Optuna Configuration
# ============================================================

N_TRIALS = 15
STUDY_NAME = "transformer_conservative_hpo"

# ============================================================
# FIXED architecture (proven best, NOT tuned)
# ============================================================
FIXED_PARAMS = dict(
    d_model=32,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    batch_size=32,
    activation="relu",
    random_state=42,
    save_checkpoints=False,
    force_reset=True,
    log_tensorboard=False,
)


# ============================================================
# Loss Logger
# ============================================================

class LossLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_epoch_losses.append(loss.item())


def plot_loss_curve(losses, target, round_idx, output_dir):
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = list(range(1, len(losses) + 1))
    ax.plot(epochs, losses, "b-", linewidth=1.5, label="Train Loss")
    min_idx = int(np.argmin(losses))
    ax.axvline(x=min_idx + 1, color="r", linestyle="--", alpha=0.6)
    ax.scatter([min_idx + 1], [losses[min_idx]], color="r", s=60, zorder=5,
              label=f"Min = {losses[min_idx]:.4f} @ epoch {min_idx+1}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss - {target} (Round {round_idx+1})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    safe = target.replace(" ", "_")
    fig.savefig(os.path.join(output_dir, f"loss_curve_{safe}_round{round_idx+1}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_summary(all_round_losses, target, output_dir):
    if not all_round_losses:
        return
    final_losses = [l[-1] for l in all_round_losses if l]
    best_losses = [min(l) for l in all_round_losses if l]
    best_epochs = [int(np.argmin(l)) + 1 for l in all_round_losses if l]
    rounds = list(range(1, len(final_losses) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(rounds, final_losses, "bo-", markersize=3, linewidth=0.8)
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Final Loss"); axes[0].set_title("Final Loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rounds, best_losses, "ro-", markersize=3, linewidth=0.8)
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Best Loss"); axes[1].set_title("Best Loss")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(rounds, best_epochs, "go-", markersize=3, linewidth=0.8)
    axes[2].axhline(y=np.median(best_epochs), color="gray", linestyle="--", alpha=0.6,
                    label=f"Median={np.median(best_epochs):.0f}")
    axes[2].set_xlabel("Round"); axes[2].set_ylabel("Best Epoch"); axes[2].set_title("Convergence")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Training Summary - {target}", fontsize=14, y=1.02)
    fig.tight_layout()
    safe = target.replace(" ", "_")
    fig.savefig(os.path.join(output_dir, f"loss_summary_{safe}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss summary saved: loss_summary_{safe}.png")


# ============================================================
# Data Loading (same as original)
# ============================================================

def load_era5_weekly_country_means(weather_dir, variables, years, countries):
    import xarray as xr
    var_datasets = {}
    for var in variables:
        ds_list = []
        for year in years:
            nc_file = os.path.join(weather_dir, f"era5_{var}_{year}.nc")
            if os.path.exists(nc_file):
                ds_list.append(xr.open_dataset(nc_file))
        if ds_list:
            var_datasets[var] = xr.concat(ds_list, dim="valid_time")
    if not var_datasets:
        return {}

    country_weather = {}
    for country in countries:
        if country not in COUNTRY_BBOX:
            continue
        lat_min, lat_max, lon_min, lon_max = COUNTRY_BBOX[country]
        dfs = []
        for var, ds in var_datasets.items():
            if "latitude" in ds.dims:
                lat_dim, lon_dim = "latitude", "longitude"
            elif "lat" in ds.dims:
                lat_dim, lon_dim = "lat", "lon"
            else:
                lat_dim = [d for d in ds.dims if "lat" in d.lower()][0]
                lon_dim = [d for d in ds.dims if "lon" in d.lower()][0]
            lat_vals = ds[lat_dim].values
            lat_slice = (slice(lat_max, lat_min) if lat_vals[0] > lat_vals[-1]
                         else slice(lat_min, lat_max))
            sub = ds.sel(**{lat_dim: lat_slice, lon_dim: slice(lon_min, lon_max)})
            data_vars = list(sub.data_vars)
            if not data_vars:
                continue
            da = sub[data_vars[0]]
            time_dim = ("valid_time" if "valid_time" in da.dims
                        else "time" if "time" in da.dims
                        else [d for d in da.dims if "time" in d.lower()][0])
            spatial_dims = [d for d in da.dims if d != time_dim]
            daily_mean = da.mean(dim=spatial_dims).to_series()
            daily_mean.index = pd.to_datetime(daily_mean.index)
            weekly_mean = daily_mean.sort_index().resample("7D").mean()
            weekly_mean.name = var
            dfs.append(weekly_mean)
        if dfs:
            wdf = pd.concat(dfs, axis=1).sort_index().ffill().bfill().fillna(0.0)
            country_weather[country] = wdf
    return country_weather


def align_weather_to_panel(country_weather, panel_index, country):
    if country not in country_weather or country_weather[country].empty:
        return pd.DataFrame(index=panel_index, columns=WEATHER_VARIABLES, data=0.0)
    wdf = country_weather[country].copy()
    wdf = wdf.reindex(panel_index, method="nearest", tolerance=pd.Timedelta("4D"))
    return wdf.ffill().bfill().fillna(0.0)


def build_holiday_series(df_all, panel_index, country):
    if HOLIDAY_COL not in df_all.columns:
        return pd.Series(index=panel_index, data=0.0, dtype=np.float32)
    sub = df_all[df_all["country_name"] == country][["date", HOLIDAY_COL]].copy()
    if sub.empty:
        return pd.Series(index=panel_index, data=0.0, dtype=np.float32)
    sub[HOLIDAY_COL] = pd.to_numeric(sub[HOLIDAY_COL], errors="coerce").fillna(0).astype(int)
    s = sub.groupby("date")[HOLIDAY_COL].max().sort_index()
    return s.reindex(panel_index).fillna(0.0).astype(np.float32)


def build_covariate_series(country_weather, df_all, panel_index, country):
    wdf = align_weather_to_panel(country_weather, panel_index, country)
    wdf = wdf.copy()
    wdf[HOLIDAY_COL] = build_holiday_series(df_all, panel_index, country)
    wdf = wdf.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    wdf = wdf.reset_index().rename(columns={"index": "date"})
    return TimeSeries.from_dataframe(
        wdf, time_col="date",
        value_cols=[c for c in wdf.columns if c != "date"],
        fill_missing_dates=False, freq=FREQ)


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["date"])
    return df.sort_values(["country_name", "target", "date"]).reset_index(drop=True)


def country_code(country):
    mapping = {
        "Belgium": "BE", "Czech Republic": "CZ", "France": "FR",
        "Poland": "PL", "Italy": "IT", "Spain": "ES",
    }
    return mapping.get(country, country[:2].upper())


def build_weekly_index(dates):
    return pd.date_range(start=dates.min(), end=dates.max(), freq=FREQ)


def fill_series_values(s):
    return s.ffill().bfill().fillna(0.0)


def build_panel(df, target, countries):
    sub = df[(df["target"] == target) & (df["country_name"].isin(countries))].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    if sub.empty:
        return pd.DataFrame()
    full_idx = build_weekly_index(sub["date"])
    panel = pd.DataFrame(index=full_idx)
    for c in countries:
        s = (sub[sub["country_name"] == c]
             .set_index("date")["value"].sort_index().groupby(level=0).mean())
        panel[c] = fill_series_values(s.reindex(full_idx)).astype(np.float32)
    return panel.sort_index()


def panel_to_ts(panel, country):
    dfc = panel[[country]].reset_index().rename(
        columns={"index": "date", country: "value"})
    dfc["value"] = dfc["value"].astype(np.float32)
    return TimeSeries.from_dataframe(
        dfc, time_col="date", value_cols="value",
        fill_missing_dates=False, freq=FREQ)


def split_train_test_by_offset(ts, test_weeks, offset, min_train_weeks):
    n = len(ts)
    test_end = n - offset
    test_start = test_end - test_weeks
    train_end = test_start
    if train_end < min_train_weeks:
        return None, None, None
    return (ts[:train_end], ts[test_start:test_end],
            ts[test_start:test_end].start_time() - timedelta(days=7))


def compute_max_rounds(ts_len, test_weeks, min_train, step, user_max):
    max_offset = ts_len - (min_train + test_weeks)
    if max_offset < 0:
        return 0
    return int(min(max_offset // step + 1, user_max // step))


def forecast_to_qdict(pred_ts):
    vals = pred_ts.all_values(copy=False)
    if vals.ndim == 3 and vals.shape[2] > 1:
        sample_matrix = vals[:, 0, :].T
        return {q: np.percentile(sample_matrix, q * 100.0, axis=0) for q in QUANTILES}
    raise ValueError("Expected sampled forecast")


def evaluate_wis(y_true, q_dict):
    y_true = np.asarray(y_true).flatten()
    if len(y_true) == 0 or np.isnan(y_true).any():
        return np.nan
    try:
        wis_total, _, _ = weighted_interval_score_fast(
            observations=y_true, alphas=ALPHAS, q_dict=q_dict,
            weights=None, percent=False, check_consistency=True)
        return float(np.nanmean(wis_total))
    except Exception:
        return np.nan


def generate_output(forecast_date, target, country, pred_ts):
    """Generate R-compatible output DataFrame for one forecast."""
    location = country_code(country)
    times = pred_ts.time_index
    q_dict = forecast_to_qdict(pred_ts)
    rows = []
    for h in range(1, len(pred_ts) + 1):
        ted = times[h - 1].strftime("%Y-%m-%d")
        od = forecast_date.strftime("%Y-%m-%d")
        rows.append({
            "origin_date": od, "target": target, "target_end_date": ted,
            "horizon": h, "location": location,
            "output_type": "median", "output_type_id": "",
            "value": float(q_dict[0.5][h - 1]),
        })
        for q in QUANTILES:
            rows.append({
                "origin_date": od, "target": target, "target_end_date": ted,
                "horizon": h, "location": location,
                "output_type": "quantile", "output_type_id": q,
                "value": float(q_dict[q][h - 1]),
            })
    return pd.DataFrame(rows)


# ============================================================
# Preload data (shared across Phase 1 and Phase 2)
# ============================================================

print("=" * 70)
print("Preloading data...")
print("=" * 70)

DF = load_data(DATA_PATH)
COUNTRIES = DF["country_name"].unique().tolist()
TARGETS = DF["target"].unique().tolist()

COUNTRY_WEATHER = load_era5_weekly_country_means(
    WEATHER_DATA_DIR, WEATHER_VARIABLES, WEATHER_YEARS, COUNTRIES)

PRECOMPUTED = {}
for target in TARGETS:
    panel = build_panel(DF, target, COUNTRIES)
    if panel.empty:
        continue
    full_series = {c: panel_to_ts(panel, c) for c in COUNTRIES}
    full_cov = {c: build_covariate_series(COUNTRY_WEATHER, DF, panel.index, c)
                for c in COUNTRIES}
    ts_len = len(next(iter(full_series.values())))
    n_rounds = compute_max_rounds(
        ts_len, TEST_WEEKS, MIN_TRAIN_WEEKS, STEP_WEEKS, MAX_BACKTEST_WEEKS)
    PRECOMPUTED[target] = dict(
        panel=panel, full_series=full_series, full_cov=full_cov,
        ts_len=ts_len, n_rounds=n_rounds)
    print(f"  {target}: {ts_len} weeks, {n_rounds} rounds")


# ============================================================
# Core backtest function (used by both phases)
# ============================================================

def run_one_backtest(input_chunk_length, dropout, n_epochs,
                     verbose=False, save_forecasts=False, forecast_dir=None):
    """Run a complete backtest and return mean WIS.

    Args:
        verbose: print progress
        save_forecasts: save individual + consolidated forecast CSV files
        forecast_dir: directory for forecast files (required if save_forecasts=True)

    Returns:
        mean_wis (float)
    """
    all_wis = []
    all_outputs = []
    all_round_losses = {}

    for target in PRECOMPUTED:
        data = PRECOMPUTED[target]
        full_series = data["full_series"]
        full_cov = data["full_cov"]
        n_rounds = data["n_rounds"]
        safe_target = target.replace(" ", "_")

        target_wis = []
        target_losses = []

        for round_idx in range(n_rounds):
            offset = round_idx * STEP_WEEKS

            train_list, meta = [], []
            cov_train_list, cov_full_list = [], []

            for c in COUNTRIES:
                train_ts, test_ts, forecast_date = split_train_test_by_offset(
                    full_series[c], TEST_WEEKS, offset, MIN_TRAIN_WEEKS)
                if train_ts is None:
                    continue
                if len(train_ts) < input_chunk_length + TEST_WEEKS:
                    continue

                train_list.append(train_ts)
                meta.append((c, forecast_date, test_ts))

                n = len(full_series[c])
                test_end = n - offset
                train_end = test_end - TEST_WEEKS
                cov_train_list.append(full_cov[c][:train_end])
                cov_full_list.append(full_cov[c][:test_end])

            if not train_list:
                continue

            y_scaler = Scaler()
            train_scaled = y_scaler.fit_transform(train_list)
            x_scaler = Scaler()
            cov_train_scaled = x_scaler.fit_transform(cov_train_list)
            cov_full_scaled = x_scaler.transform(cov_full_list)

            try:
                loss_logger = LossLogger() if verbose else None
                pl_kwargs = {"callbacks": [loss_logger]} if verbose else {}

                model = TransformerModel(
                    input_chunk_length=input_chunk_length,
                    output_chunk_length=TEST_WEEKS,
                    n_epochs=n_epochs,
                    dropout=dropout,
                    **FIXED_PARAMS,
                    likelihood=QuantileRegression(quantiles=QUANTILES),
                    model_name=f"bt_r{round_idx}_{safe_target}",
                    pl_trainer_kwargs=pl_kwargs,
                )

                model.fit(series=train_scaled,
                          past_covariates=cov_train_scaled, verbose=False)

                if verbose and loss_logger and loss_logger.train_epoch_losses:
                    target_losses.append(loss_logger.train_epoch_losses)

                preds_scaled = []
                for i in range(len(meta)):
                    preds_scaled.append(
                        model.predict(
                            n=TEST_WEEKS, series=train_scaled[i],
                            past_covariates=cov_full_scaled[i],
                            num_samples=PRED_NUM_SAMPLES))
                preds = y_scaler.inverse_transform(preds_scaled)

                for i, (country, forecast_date, test_ts) in enumerate(meta):
                    y_true = test_ts.values().flatten()
                    try:
                        q_dict = forecast_to_qdict(preds[i])
                        wis = evaluate_wis(y_true, q_dict)
                        if not np.isnan(wis):
                            target_wis.append(wis)
                    except Exception:
                        pass

                    # Save forecast output
                    if save_forecasts:
                        try:
                            out_df = generate_output(
                                forecast_date, target, country, preds[i])
                            out_df["backtest_round"] = round_idx + 1
                            out_df["known_weeks"] = len(train_list[i])
                            all_outputs.append(out_df)

                            safe_country = country.replace(" ", "_")
                            fname = (f"forecast_{safe_country}_{safe_target}_"
                                     f"{forecast_date.strftime('%Y-%m-%d')}.csv")
                            out_df.to_csv(
                                os.path.join(forecast_dir, fname), index=False)
                        except Exception as e:
                            if verbose:
                                print(f"      {country}: output error - {e}")

            except Exception as e:
                if verbose:
                    print(f"    Round {round_idx}: error - {e}")
                continue

            if verbose and (round_idx == 0 or round_idx == n_rounds - 1
                            or (round_idx + 1) % 10 == 0):
                mean_so_far = np.mean(target_wis) if target_wis else float('nan')
                print(f"    Round {round_idx+1}/{n_rounds}: "
                      f"train={len(train_list[0])}w, "
                      f"running WIS={mean_so_far:.4f}")

        if target_wis:
            mean_t = np.mean(target_wis)
            all_wis.extend(target_wis)
            if verbose:
                print(f"  {target}: mean WIS = {mean_t:.4f} ({len(target_wis)} evals)")

        if verbose and target_losses:
            all_round_losses[target] = target_losses

    mean_wis = np.mean(all_wis) if all_wis else float("inf")

    # Save consolidated forecasts
    if save_forecasts and all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)
        combined_path = os.path.join(forecast_dir, "all_forecasts.csv")
        combined.to_csv(combined_path, index=False)
        if verbose:
            print(f"\n  Saved: all_forecasts.csv ({len(combined)} rows)")
            print(f"  Saved: {len(all_outputs)} individual forecast CSVs")

    # Save loss curves
    if verbose and all_round_losses:
        for target, losses_list in all_round_losses.items():
            for r_idx, losses in enumerate(losses_list):
                if (r_idx == 0 or r_idx == len(losses_list) - 1
                        or (r_idx + 1) % 10 == 0):
                    plot_loss_curve(losses, target, r_idx, forecast_dir)
            plot_loss_summary(losses_list, target, forecast_dir)

    return mean_wis


# ============================================================
# Phase 1: Optuna Objective
# ============================================================

def objective(trial):
    input_chunk_length = trial.suggest_categorical(
        "input_chunk_length", [13, 20, 26, 33, 39, 52])
    dropout = trial.suggest_float(
        "dropout", 0.05, 0.25, step=0.05)
    n_epochs = trial.suggest_categorical(
        "n_epochs", [50, 75, 100, 125, 150])

    print(f"\n  Trial {trial.number}: "
          f"input_chunk={input_chunk_length}, "
          f"dropout={dropout}, n_epochs={n_epochs}")

    mean_wis = run_one_backtest(
        input_chunk_length, dropout, n_epochs,
        verbose=False, save_forecasts=False)

    print(f"  Trial {trial.number} RESULT: mean WIS = {mean_wis:.4f}")
    return mean_wis


# ============================================================
# Main: Phase 1 (tune) + Phase 2 (output for R)
# ============================================================

def main():
    os.makedirs(TUNING_DIR, exist_ok=True)
    os.makedirs(FORECAST_DIR, exist_ok=True)

    # ========================================
    # Phase 1: Optuna Tuning
    # ========================================
    print(f"\n{'='*70}")
    print("Phase 1: Optuna Hyperparameter Tuning")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Full 52-round evaluation per trial")
    print(f"  Tuning: input_chunk_length, dropout, n_epochs")
    print(f"  Fixed:  d_model=32, 2+2 layers, relu, batch=32")
    print(f"{'='*70}\n")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        sampler=TPESampler(seed=42),
    )

    # Enqueue baseline as Trial 0
    study.enqueue_trial({
        "input_chunk_length": 26,
        "dropout": 0.1,
        "n_epochs": 100,
    })

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ---- Print Phase 1 results ----
    best = study.best_trial
    baseline_wis = study.trials[0].value

    print(f"\n{'='*70}")
    print("Phase 1 Complete!")
    print(f"{'='*70}")
    print(f"\n  Baseline (Trial 0): WIS = {baseline_wis:.4f}")
    print(f"  Best (Trial {best.number}):    WIS = {best.value:.4f}")
    if baseline_wis and best.value < baseline_wis:
        print(f"  Improvement: {(baseline_wis - best.value) / baseline_wis * 100:.2f}%")
    elif best.number == 0:
        print("  Baseline was already optimal!")

    print(f"\n  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save Optuna logs
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(TUNING_DIR, "optuna_trials.csv"), index=False)
    pd.DataFrame([{**best.params, "wis": best.value}]).to_csv(
        os.path.join(TUNING_DIR, "best_params.csv"), index=False)

    # Print all trials sorted
    print(f"\n  All trials (sorted by WIS):")
    print(f"  {'Trial':>6} {'WIS':>10} {'input_chunk':>12} {'dropout':>8} {'n_epochs':>9}")
    print(f"  {'-'*50}")
    for t in sorted(study.trials, key=lambda t: t.value if t.value else float('inf')):
        if t.value is not None:
            p = t.params
            marker = " <-- baseline" if t.number == 0 else ""
            if t.number == best.number:
                marker = " <-- BEST"
            print(f"  {t.number:>6} {t.value:>10.4f} "
                  f"{p['input_chunk_length']:>12} "
                  f"{p['dropout']:>8.2f} "
                  f"{p['n_epochs']:>9}{marker}")

    # ========================================
    # Phase 2: Full Backtest + Forecast Output
    # ========================================
    best_params = best.params

    print(f"\n{'='*70}")
    print("Phase 2: Full backtest with best params -> forecast files for R")
    print(f"  input_chunk_length = {best_params['input_chunk_length']}")
    print(f"  dropout            = {best_params['dropout']}")
    print(f"  n_epochs           = {best_params['n_epochs']}")
    print(f"  Output dir:          {FORECAST_DIR}")
    print(f"{'='*70}\n")

    final_wis = run_one_backtest(
        input_chunk_length=best_params["input_chunk_length"],
        dropout=best_params["dropout"],
        n_epochs=best_params["n_epochs"],
        verbose=True,
        save_forecasts=True,
        forecast_dir=FORECAST_DIR,
    )

    # ---- Final summary ----
    p = best_params
    print(f"""
{'='*70}
ALL DONE
{'='*70}

Phase 1 result: Best WIS = {best.value:.4f} (Trial {best.number})
Phase 2 result: Final WIS = {final_wis:.4f} (full backtest with output)

Best hyperparameters:

  TRANSFORMER_KWARGS = dict(
      input_chunk_length={p['input_chunk_length']},
      output_chunk_length=TEST_WEEKS,
      batch_size=32,
      n_epochs={p['n_epochs']},
      d_model=32,
      nhead=4,
      num_encoder_layers=2,
      num_decoder_layers=2,
      dim_feedforward=128,
      dropout={p['dropout']},
      activation="relu",
      random_state=42,
      save_checkpoints=False,
      force_reset=True,
      log_tensorboard=False,
  )

Forecast files for R evaluation:
  {FORECAST_DIR}all_forecasts.csv           <- consolidated (R reads this)
  {FORECAST_DIR}forecast_*.csv              <- individual files

To evaluate in R, set in evaluation_analysis.R:
  OUTPUT_DIR <- "{FORECAST_DIR}"
""")


if __name__ == "__main__":
    main()