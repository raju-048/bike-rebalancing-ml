"""
Feature Engineering for Citi Bike Demand Prediction (2-year, memory-safe)
==========================================================================

Input : data/processed/<YYYY-MM>_station_time.parquet  (10-min, UTC, complete grid)
Output: data/features/features_<YYYY-MM>.parquet        (one file per month)
        outputs/metrics/feature_report.json

Modelling resolution
    The 10-min counts are summed to `model_interval_minutes` (default 60).
    Hourly is the natural resolution here: weather is hourly, rebalancing
    trucks act over hours, and 10-min station counts are ~75% zeros.

One row = (station, interval t). Targets are the demand IN interval t:
    target_pickups, target_dropoffs
Every feature uses only information available BEFORE t, except weather at t
(at prediction time this would come from a short-range weather forecast).

Features
    Recent demand  : lags 1, 2, 3 intervals, same time yesterday, last week
    Smoothed demand: mean of last 3 h, last 24 h, last 28 days
    Seasonal       : mean of the same hour-of-week over the last 4 weeks
    Calendar       : hour, weekday, month, weekend, US federal holiday, cyclic encodings
    Station        : lat / lng
    Weather        : temperature, humidity, precipitation, wind speed,
                     weather code, is_rainy, precipitation over the previous 3 h

Memory
    Each month is processed together with the previous month (history for the
    lag / 28-day features) using dense (station × time) numpy arrays, so peak
    memory is a few hundred MB regardless of the total dataset size.
"""

import os
import json
import glob
import logging

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

logger = logging.getLogger(__name__)

LOCAL_TZ = "America/New_York"

WEATHER_COLS = [
    "temperature", "humidity", "precipitation", "wind_speed",
    "weather_code", "is_rainy", "precip_prev_3h",
]

NON_FEATURE_COLS = {"station_id", "interval_start", "target_pickups", "target_dropoffs"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def processed_months(processed_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(processed_dir, "*_station_time.parquet")))
    return [os.path.basename(f)[:7] for f in files]


def load_month_counts(processed_dir: str, month: str, model_minutes: int):
    """
    Load one processed month and aggregate to `model_minutes`.

    Returns (station_ids: list[str], times: DatetimeIndex (UTC naive),
             P: int32 [stations × T], D: int32 [stations × T])
    """
    path = os.path.join(processed_dir, f"{month}_station_time.parquet")
    df = pd.read_parquet(path)
    sid = df["station_id"]
    if not isinstance(sid.dtype, pd.CategoricalDtype):
        sid = sid.astype(str).astype("category")
    codes = sid.cat.codes.to_numpy()
    if np.any(np.diff(codes) < 0):
        # Not in (station, time) order: sort once.
        order = np.lexsort((df["interval_start"].to_numpy(), codes))
        df = df.iloc[order].reset_index(drop=True)
        sid = sid.iloc[order].reset_index(drop=True)
    ids = [str(c) for c in sid.cat.categories]
    k = len(ids)
    n = len(df) // k
    ts = df["interval_start"].to_numpy()[:n]
    step_min = int((ts[1] - ts[0]) / np.timedelta64(1, "m")) if n > 1 else model_minutes
    if model_minutes % step_min:
        raise ValueError(f"model_interval_minutes={model_minutes} is not a multiple of "
                         f"the processed step {step_min} min")
    f = model_minutes // step_min
    if n % f:
        raise ValueError(f"{month}: {n} intervals not divisible by {f}")
    P = df["pickups"].to_numpy(np.int32).reshape(k, n // f, f).sum(axis=2)
    D = df["dropoffs"].to_numpy(np.int32).reshape(k, n // f, f).sum(axis=2)
    times = pd.DatetimeIndex(ts[::f])
    return ids, times, P, D


# ---------------------------------------------------------------------------
# Array helpers (all operate along axis 1 = time)
# ---------------------------------------------------------------------------

def _lag(X: np.ndarray, L: int) -> np.ndarray:
    out = np.full_like(X, np.nan)
    if L < X.shape[1]:
        out[:, L:] = X[:, :-L]
    return out


def _past_mean(X: np.ndarray, w: int) -> np.ndarray:
    """Mean of X[:, t-w : t] (excludes t), NaN-aware; NaN if no valid value."""
    valid = ~np.isnan(X)
    Xz = np.where(valid, X, 0.0)
    k, T = X.shape
    C = np.zeros((k, T + 1), dtype=np.float64)
    N = np.zeros((k, T + 1), dtype=np.float64)
    np.cumsum(Xz, axis=1, out=C[:, 1:])
    np.cumsum(valid, axis=1, out=N[:, 1:])
    t = np.arange(T)
    lo = np.maximum(t - w, 0)
    s = C[:, t] - C[:, lo]
    c = N[:, t] - N[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    m[c == 0] = np.nan
    return m.astype(np.float32)


def _nanmean_stack(arrs: list) -> np.ndarray:
    st = np.stack(arrs, axis=0)
    valid = ~np.isnan(st)
    cnt = valid.sum(axis=0)
    tot = np.where(valid, st, 0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = tot / cnt
    m[cnt == 0] = np.nan
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Weather / calendar (per time step, later tiled across stations)
# ---------------------------------------------------------------------------

def load_weather_utc(weather_path: str, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Hourly weather indexed by UTC-naive hour."""
    wx = pd.read_parquet(weather_path)
    wx["timestamp"] = pd.to_datetime(wx["timestamp"])
    loc = wx["timestamp"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    wx = wx[loc.notna()].copy()
    wx["timestamp"] = loc[loc.notna()].dt.tz_convert("UTC").dt.tz_localize(None)
    wx = wx.set_index("timestamp").sort_index()
    wx = wx[~wx.index.duplicated(keep="first")]
    full = pd.date_range(wx.index.min(), wx.index.max(), freq="1h")
    wx = wx.reindex(full)
    num = ["temperature", "humidity", "wind_speed", "precipitation", "weather_code"]
    wx[num] = wx[num].astype("float64").interpolate(limit=3).ffill().bfill()
    wx["is_rainy"] = (wx["precipitation"] > 0).astype(np.float32)
    wx["precip_prev_3h"] = wx["precipitation"].shift(1).rolling(3, min_periods=1).sum()
    return wx


def weather_for_times(wx: pd.DataFrame, times: pd.DatetimeIndex) -> dict:
    hours = times.floor("1h")
    sub = wx.reindex(hours)
    return {c: sub[c].to_numpy(np.float32) for c in WEATHER_COLS}


def calendar_for_times(times: pd.DatetimeIndex, tz: str = LOCAL_TZ) -> dict:
    loc = times.tz_localize("UTC").tz_convert(tz)
    hour = loc.hour.to_numpy()
    minute = loc.minute.to_numpy()
    dow = loc.dayofweek.to_numpy()
    month = loc.month.to_numpy()
    doy = loc.dayofyear.to_numpy()
    days = loc.normalize().tz_localize(None)
    hol = USFederalHolidayCalendar().holidays(start=days.min(), end=days.max())
    is_hol = days.isin(hol)
    frac_hour = hour + minute / 60.0
    return {
        "hour": hour.astype(np.int8),
        "day_of_week": dow.astype(np.int8),
        "month": month.astype(np.int8),
        "is_weekend": (dow >= 5).astype(np.int8),
        "is_holiday": is_hol.astype(np.int8),
        "is_workday": ((dow < 5) & ~is_hol).astype(np.int8),
        "sin_hour": np.sin(2 * np.pi * frac_hour / 24).astype(np.float32),
        "cos_hour": np.cos(2 * np.pi * frac_hour / 24).astype(np.float32),
        "sin_dow": np.sin(2 * np.pi * dow / 7).astype(np.float32),
        "cos_dow": np.cos(2 * np.pi * dow / 7).astype(np.float32),
        "sin_doy": np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# One month
# ---------------------------------------------------------------------------

def build_month_features(processed_dir: str, month: str, prev_month, model_minutes: int,
                         wx: pd.DataFrame, station_coords: pd.DataFrame) -> pd.DataFrame:
    ids, times, P, D = load_month_counts(processed_dir, month, model_minutes)
    k, Tm = P.shape
    steps_h = 60 // model_minutes if model_minutes <= 60 else None
    if steps_h is None:
        raise ValueError("model_interval_minutes must be <= 60")
    day, week = 24 * steps_h, 168 * steps_h
    history = 4 * week  # 28 days

    # --- History from the previous month (aligned to this month's stations) ---
    Pp = Dp = None
    if prev_month is not None:
        pids, ptimes, Pprev, Dprev = load_month_counts(processed_dir, prev_month, model_minutes)
        if len(ptimes) and ptimes[-1] + pd.Timedelta(minutes=model_minutes) == times[0]:
            take = min(history, len(ptimes))
            pos = pd.Index(pids).get_indexer(ids)
            Pp = np.zeros((k, take), np.float32)
            Dp = np.zeros((k, take), np.float32)
            have = pos >= 0
            Pp[have] = Pprev[pos[have], -take:]
            Dp[have] = Dprev[pos[have], -take:]
            # Stations not active last month had (almost) no trips: 0 is the right history.
        else:
            logger.warning("  %s is not contiguous with %s — no history used", prev_month, month)
    if Pp is None:
        # No history (first month): NaN so lag features are missing, not fake zeros.
        Pp = np.full((k, history), np.nan, np.float32)
        Dp = np.full((k, history), np.nan, np.float32)
    H = Pp.shape[1]
    Pw = np.concatenate([Pp, P.astype(np.float32)], axis=1)
    Dw = np.concatenate([Dp, D.astype(np.float32)], axis=1)

    feats = {}
    for name, X in (("pickup", Pw), ("dropoff", Dw)):
        lags = {f"{name}_lag_1": _lag(X, 1), f"{name}_lag_2": _lag(X, 2),
                f"{name}_lag_3": _lag(X, 3), f"{name}_lag_1d": _lag(X, day),
                f"{name}_lag_1w": _lag(X, week)}
        feats.update(lags)
        feats[f"{name}_mean_3h"] = _past_mean(X, 3 * steps_h)
        feats[f"{name}_mean_24h"] = _past_mean(X, day)
        feats[f"{name}_mean_28d"] = _past_mean(X, history)
        feats[f"{name}_how_mean_4w"] = _nanmean_stack(
            [lags[f"{name}_lag_1w"], _lag(X, 2 * week), _lag(X, 3 * week), _lag(X, 4 * week)])
    feats["net_lag_1"] = feats["dropoff_lag_1"] - feats["pickup_lag_1"]
    feats["net_mean_24h"] = feats["dropoff_mean_24h"] - feats["pickup_mean_24h"]

    # --- Assemble rows for THIS month only ---
    n_rows = k * Tm
    out = {
        "station_id": pd.Categorical.from_codes(
            np.repeat(np.arange(k, dtype=np.int32), Tm), categories=ids),
        "interval_start": np.tile(times.values, k),
        "target_pickups": P.ravel().astype(np.float32),
        "target_dropoffs": D.ravel().astype(np.float32),
    }
    for c, arr in feats.items():
        out[c] = arr[:, H:].ravel()
    for c, arr in calendar_for_times(times).items():
        out[c] = np.tile(arr, k)
    if wx is not None:
        for c, arr in weather_for_times(wx, times).items():
            out[c] = np.tile(arr, k)
    if station_coords is not None:
        sc = station_coords.reindex(ids)
        out["lat"] = np.repeat(sc["lat"].to_numpy(np.float32), Tm)
        out["lng"] = np.repeat(sc["lng"].to_numpy(np.float32), Tm)
    df = pd.DataFrame(out)
    assert len(df) == n_rows
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def feature_files(features_dir: str, months_filter=None) -> list:
    files = sorted(glob.glob(os.path.join(features_dir, "features_????-??.parquet")))
    if months_filter:
        keep = set(months_filter)
        files = [f for f in files if os.path.basename(f)[9:16] in keep]
    return files


def get_feature_cols(columns) -> list:
    return sorted(c for c in columns if c not in NON_FEATURE_COLS)


def build_features(config: dict):
    processed_dir = config["processed_data_path"]
    features_dir = config.get("features_data_path", "data/features")
    outputs_path = config.get("outputs_path", "outputs")
    model_minutes = int(config.get("model_interval_minutes", 60))
    use_weather = config.get("use_weather", True)
    weather_path = config.get("weather_data_path", "data/processed/weather_hourly.parquet")
    months_filter = config.get("months_filter") or None

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_path, "metrics"), exist_ok=True)

    months = processed_months(processed_dir)
    if months_filter:
        months_run = [m for m in months if m in set(months_filter)]
    else:
        months_run = months
    if not months_run:
        raise FileNotFoundError(f"No processed months found in {processed_dir}")

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING START  (%d-min resolution, %d months)",
                model_minutes, len(months_run))
    logger.info("=" * 60)

    wx = None
    if use_weather:
        if os.path.exists(weather_path):
            wx = load_weather_utc(weather_path)
            logger.info("Weather loaded: %s → %s (UTC)", wx.index.min(), wx.index.max())
        else:
            logger.warning("Weather file %s not found — run --stage weather. "
                           "Continuing WITHOUT weather features.", weather_path)

    sm_path = os.path.join(processed_dir, "station_master.parquet")
    coords = None
    if os.path.exists(sm_path):
        sm = pd.read_parquet(sm_path)
        sm["station_id"] = sm["station_id"].astype(str)
        coords = sm.drop_duplicates("station_id").set_index("station_id")[["lat", "lng"]]

    month_set = set(months)
    per_month = []
    for m in months_run:
        prev = (pd.Period(m, "M") - 1).strftime("%Y-%m")
        prev = prev if prev in month_set else None
        df = build_month_features(processed_dir, m, prev, model_minutes, wx, coords)
        path = os.path.join(features_dir, f"features_{m}.parquet")
        df.to_parquet(path, engine="pyarrow", index=False)
        feat_cols = get_feature_cols(df.columns)
        nan_rows = int(df[feat_cols].isna().any(axis=1).sum())
        info = {
            "month": m, "rows": len(df), "stations": int(df["station_id"].nunique()),
            "rows_with_nan_features": nan_rows,
            "pickups": float(df["target_pickups"].sum()),
            "dropoffs": float(df["target_dropoffs"].sum()),
            "mb": round(os.path.getsize(path) / 1e6, 1),
        }
        per_month.append(info)
        logger.info("  %s: %d rows, %d stations, %d rows with NaN features, %.0f MB",
                    m, info["rows"], info["stations"], nan_rows, info["mb"])
        del df

    report = {
        "model_interval_minutes": model_minutes,
        "weather_used": wx is not None,
        "feature_columns": feat_cols,
        "n_features": len(feat_cols),
        "months": per_month,
        "total_rows": int(sum(p["rows"] for p in per_month)),
    }
    with open(os.path.join(outputs_path, "metrics", "feature_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING COMPLETE: %d rows, %d features",
                report["total_rows"], len(feat_cols))
    logger.info("=" * 60)
    return report
