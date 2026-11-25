"""
Model Training (2-year, memory-safe)
====================================

Predicts next-interval (default: next hour) pickups and dropoffs per station.

Data
    Reads the monthly feature files. The TRAIN set is a uniform random sample
    of rows from every training month (capped by `max_train_rows`), so the
    model sees all seasons and weather conditions without loading 2 years
    into memory. Validation is sampled the same way (`max_val_rows`).
    The test set is never touched here (see evaluate.py).

Models (separately for pickups and dropoffs)
    1. Baseline      — seasonal average: same hour-of-week over the last 4 weeks
                        (falls back to the 28-day mean). No fitting.
    2. RandomForest  — small random search, trained on `rf_max_rows` rows.
    3. XGBoost       — random search incl. count objectives (Poisson / Tweedie)
                        with early stopping on validation MAE.
    4. XGBoost (no weather) — best XGBoost params retrained WITHOUT weather
                        features, to measure how much weather improves accuracy.

Best model per target = lowest validation MAE (among 1-3).

Saved
    models/baseline.json, rf_<target>.joblib, xgb_<target>.joblib,
    xgb_noweather_<target>.joblib, feature_names.json
    outputs/metrics/training_report.json, split_dates.json
"""

import os
import json
import time
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler

from src.feature_engineering.features import (
    feature_files, get_feature_cols, WEATHER_COLS,
)
from src.models.split import compute_split, split_bounds

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

TARGETS = [("target_pickups", "pickups"), ("target_dropoffs", "dropoffs")]
BASELINE_COL = {"pickups": ("pickup_how_mean_4w", "pickup_mean_28d"),
                "dropoffs": ("dropoff_how_mean_4w", "dropoff_mean_28d")}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, counts: bool = True) -> dict:
    """counts=True: non-negative targets (clip preds at 0, add RMSLE).
    counts=False: signed targets such as net flow."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if counts:
        y_pred = np.clip(y_pred, 0, None)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else 0.0
    out = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}
    if counts:
        out["rmsle"] = round(float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))), 4)
    return out


def direction_metrics(y_true, y_pred, thr: float = 2.0) -> dict:
    """For signed net flow: how often surplus/deficit direction is right when it matters."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mat = np.abs(y_true) >= thr
    return {
        "direction_accuracy": round(float((np.sign(y_pred[mat]) == np.sign(y_true[mat])).mean()), 4)
        if mat.any() else None,
        "deficit_recall": round(float((y_pred[y_true <= -thr] <= -thr / 2).mean()), 4)
        if (y_true <= -thr).any() else None,
        "surplus_recall": round(float((y_pred[y_true >= thr] >= thr / 2).mean()), 4)
        if (y_true >= thr).any() else None,
    }


def baseline_predict(df: pd.DataFrame, label: str) -> np.ndarray:
    main, fallback = BASELINE_COL[label]
    p = df[main].to_numpy(np.float32)
    fb = df[fallback].to_numpy(np.float32)
    p = np.where(np.isnan(p), fb, p)
    return np.nan_to_num(p, nan=0.0)


# ---------------------------------------------------------------------------
# Sampling rows from monthly files
# ---------------------------------------------------------------------------

def _mask(ts: pd.Series, lo, hi) -> np.ndarray:
    m = np.ones(len(ts), dtype=bool)
    if lo is not None:
        m &= (ts >= lo).to_numpy()
    if hi is not None:
        m &= (ts < hi).to_numpy()
    return m


def sample_rows(files: list, columns: list, lo, hi, max_rows, seed: int,
                dropna_cols: list) -> pd.DataFrame:
    """Uniform random sample of complete rows with lo <= interval_start < hi."""
    counts = []
    for f in files:
        ts = pd.read_parquet(f, columns=["interval_start"])["interval_start"]
        counts.append(int(_mask(ts, lo, hi).sum()))
    total = sum(counts)
    if total == 0:
        return pd.DataFrame(columns=columns)
    frac = 1.0 if not max_rows else min(1.0, max_rows / total)
    rng = np.random.default_rng(seed)
    parts = []
    for f, c in zip(files, counts):
        if c == 0:
            continue
        df = pd.read_parquet(f, columns=columns)
        df = df[_mask(df["interval_start"], lo, hi)]
        df = df.dropna(subset=dropna_cols)
        if frac < 1.0:
            df = df[rng.random(len(df)) < frac]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    logger.info("  Sampled %d of %d rows (%.1f%%) from %d month file(s)",
                len(out), total, 100 * len(out) / total, sum(c > 0 for c in counts))
    return out


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

RF_PARAM_SPACE = {
    "n_estimators": [100, 150],
    "max_depth": [16, 24, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features": [0.33, 0.5],
}


def train_random_forest(X, y, Xv, yv, n_iter, seed):
    best = (np.inf, None, None)
    results = []
    for i, params in enumerate(ParameterSampler(RF_PARAM_SPACE, n_iter=n_iter, random_state=seed)):
        t0 = time.time()
        model = RandomForestRegressor(**params, n_jobs=-1, random_state=seed)
        model.fit(X, y)
        m = compute_metrics(yv, model.predict(Xv))
        dt = round(time.time() - t0, 1)
        results.append({"params": params, "val_metrics": m, "train_time_s": dt})
        logger.info("  RF  [%d/%d] val MAE=%.4f RMSE=%.4f  %s  (%.0fs)",
                    i + 1, n_iter, m["mae"], m["rmse"], params, dt)
        if m["mae"] < best[0]:
            best = (m["mae"], model, params)
        else:
            del model
    return best[1], best[2], results


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

XGB_PARAM_SPACE = {
    "objective": ["count:poisson", "reg:tweedie", "reg:squarederror"],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1.0, 5.0],
}


def _xgb_model(params: dict, seed: int):
    return xgb.XGBRegressor(
        **params, n_estimators=3000, early_stopping_rounds=50,
        eval_metric="mae", tree_method="hist", max_bin=256,
        random_state=seed, n_jobs=-1,
    )


# Net flow (dropoffs − pickups) is signed, so only symmetric objectives apply.
XGB_NET_PARAM_SPACE = {
    "objective": ["reg:squarederror", "reg:pseudohubererror"],
    "max_depth": [6, 8, 10],
    "learning_rate": [0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1.0, 5.0],
}


def train_xgboost(X, y, Xv, yv, n_iter, seed, space=None, counts=True):
    best = (np.inf, None, None)
    results = []
    space = space or XGB_PARAM_SPACE
    for i, params in enumerate(ParameterSampler(space, n_iter=n_iter, random_state=seed)):
        t0 = time.time()
        model = _xgb_model(params, seed)
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        m = compute_metrics(yv, model.predict(Xv), counts=counts)
        dt = round(time.time() - t0, 1)
        results.append({"params": params, "val_metrics": m, "train_time_s": dt,
                        "best_iteration": int(model.best_iteration)})
        logger.info("  XGB [%d/%d] val MAE=%.4f RMSE=%.4f  trees=%d  %s  (%.0fs)",
                    i + 1, n_iter, m["mae"], m["rmse"], model.best_iteration + 1, params, dt)
        if m["mae"] < best[0]:
            best = (m["mae"], model, params)
    return best[1], best[2], results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(config: dict):
    features_dir = config.get("features_data_path", "data/features")
    models_dir = config.get("models_path", "models")
    outputs_path = config.get("outputs_path", "outputs")
    seed = int(config.get("random_seed", 42))
    n_iter = int(config.get("n_iter_search", 6))
    n_iter_rf = int(config.get("n_iter_search_rf", min(n_iter, 3)))
    max_train = config.get("max_train_rows", 2_000_000)
    max_val = config.get("max_val_rows", 1_000_000)
    rf_max = config.get("rf_max_rows", 500_000)
    train_rf = config.get("train_random_forest", True)
    lrs = config.get("xgb_learning_rates")  # optional override, e.g. [0.1, 0.15] for speed
    space_counts = dict(XGB_PARAM_SPACE, **({"learning_rate": lrs} if lrs else {}))
    space_net = dict(XGB_NET_PARAM_SPACE, **({"learning_rate": lrs} if lrs else {}))

    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is required: pip install xgboost")

    files = feature_files(features_dir, config.get("months_filter") or None)
    if not files:
        raise FileNotFoundError(f"No features_YYYY-MM.parquet in {features_dir}. "
                                "Run --stage features first.")
    os.makedirs(models_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING START  (%d monthly feature files)", len(files))
    logger.info("=" * 60)
    split = compute_split(files, config.get("train_ratio", 0.70),
                          config.get("validation_ratio", 0.15), outputs_path)

    all_cols = pd.read_parquet(files[0]).columns
    feature_cols = get_feature_cols(all_cols)
    weather_cols = [c for c in WEATHER_COLS if c in feature_cols]
    noweather_cols = [c for c in feature_cols if c not in weather_cols]
    with open(os.path.join(models_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump({"all": feature_cols, "no_weather": noweather_cols}, f, indent=2)
    logger.info("Features (%d): %s", len(feature_cols), feature_cols)
    logger.info("Weather features (%d): %s", len(weather_cols), weather_cols)

    cols = ["interval_start"] + feature_cols + [t for t, _ in TARGETS]
    need = feature_cols + [t for t, _ in TARGETS]
    logger.info("Loading TRAIN sample ...")
    tr = sample_rows(files, cols, *split_bounds(split, "train"), max_train, seed, need)
    logger.info("Loading VALIDATION sample ...")
    va = sample_rows(files, cols, *split_bounds(split, "val"), max_val, seed + 1, need)
    if tr.empty or va.empty:
        raise ValueError("Empty train or validation set — need more months of features.")
    logger.info("Train rows: %d (%s → %s) | Val rows: %d",
                len(tr), tr.interval_start.min(), tr.interval_start.max(), len(va))

    X = tr[feature_cols].to_numpy(np.float32)
    Xv = va[feature_cols].to_numpy(np.float32)
    rng = np.random.default_rng(seed)
    rf_idx = rng.choice(len(tr), size=min(len(tr), rf_max), replace=False) if rf_max else None

    report = {
        "split_dates": split,
        "train_rows": len(tr), "val_rows": len(va),
        "feature_cols": feature_cols, "weather_cols": weather_cols,
        "models": {}, "best_model": {}, "weather_impact": {}, "feature_importance": {},
    }

    for target, label in TARGETS:
        logger.info("-" * 60)
        logger.info("TARGET: %s  (train mean=%.3f, val mean=%.3f)",
                    label, tr[target].mean(), va[target].mean())
        y, yv = tr[target].to_numpy(np.float32), va[target].to_numpy(np.float32)
        res = {}

        # 1. Baseline
        m_bl = compute_metrics(yv, baseline_predict(va, label))
        res["Baseline"] = {"val_metrics": m_bl, "key": "baseline"}
        logger.info("  Baseline (4-week same-hour average) val MAE=%.4f RMSE=%.4f",
                    m_bl["mae"], m_bl["rmse"])

        # 2. Random Forest
        if train_rf:
            t0 = time.time()
            Xr, yr = (X[rf_idx], y[rf_idx]) if rf_idx is not None else (X, y)
            rf, rf_params, rf_res = train_random_forest(Xr, yr, Xv, yv, n_iter_rf, seed)
            path = os.path.join(models_dir, f"rf_{label}.joblib")
            joblib.dump(rf, path, compress=3)
            res["RandomForest"] = {"val_metrics": compute_metrics(yv, rf.predict(Xv)),
                                   "best_params": rf_params, "search": rf_res,
                                   "train_rows": len(yr), "model_path": path, "key": "rf",
                                   "train_time_s": round(time.time() - t0, 1)}
            del rf

        # 3. XGBoost (with weather)
        t0 = time.time()
        xg, xg_params, xg_res = train_xgboost(X, y, Xv, yv, n_iter, seed, space=space_counts)
        path = os.path.join(models_dir, f"xgb_{label}.joblib")
        joblib.dump(xg, path)
        m_xg = compute_metrics(yv, xg.predict(Xv))
        res["XGBoost"] = {"val_metrics": m_xg, "best_params": xg_params, "search": xg_res,
                          "best_iteration": int(xg.best_iteration), "model_path": path,
                          "key": "xgb", "train_time_s": round(time.time() - t0, 1)}
        gain = xg.get_booster().get_score(importance_type="gain")
        imp = {feature_cols[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v
               for k, v in gain.items()}
        tot = sum(imp.values()) or 1.0
        report["feature_importance"][label] = dict(sorted(
            ((k, round(v / tot, 5)) for k, v in imp.items()), key=lambda kv: -kv[1]))

        # 4. XGBoost without weather (same params) → weather impact
        if weather_cols:
            iw = [feature_cols.index(c) for c in noweather_cols]
            xn = _xgb_model(xg_params, seed)
            xn.fit(X[:, iw], y, eval_set=[(Xv[:, iw], yv)], verbose=False)
            path = os.path.join(models_dir, f"xgb_noweather_{label}.joblib")
            joblib.dump(xn, path)
            m_xn = compute_metrics(yv, xn.predict(Xv[:, iw]))
            res["XGBoost (no weather)"] = {"val_metrics": m_xn, "best_params": xg_params,
                                           "model_path": path, "key": "xgb_noweather",
                                           "selectable": False}
            rain = va["is_rainy"].to_numpy() > 0
            p_w, p_n = xg.predict(Xv), xn.predict(Xv[:, iw])
            report["weather_impact"][label] = {
                "val_mae_with_weather": m_xg["mae"],
                "val_mae_without_weather": m_xn["mae"],
                "mae_reduction_pct": round(100 * (m_xn["mae"] - m_xg["mae"]) / m_xn["mae"], 2),
                "rainy_hours_mae_with": compute_metrics(yv[rain], p_w[rain])["mae"] if rain.any() else None,
                "rainy_hours_mae_without": compute_metrics(yv[rain], p_n[rain])["mae"] if rain.any() else None,
            }
            logger.info("  Weather impact: val MAE %.4f (with) vs %.4f (without) → %.1f%% better",
                        m_xg["mae"], m_xn["mae"],
                        report["weather_impact"][label]["mae_reduction_pct"])
            del xn

        cands = [(n, r["val_metrics"]["mae"]) for n, r in res.items() if r.get("selectable", True)]
        best_name = min(cands, key=lambda c: c[1])[0]
        report["models"][label] = res
        report["best_model"][label] = {"name": best_name, "key": res[best_name]["key"],
                                       "val_mae": res[best_name]["val_metrics"]["mae"],
                                       "model_path": res[best_name].get("model_path")}
        logger.info("  BEST for %s: %s (val MAE=%.4f)", label, best_name,
                    res[best_name]["val_metrics"]["mae"])
        del xg

    # ---- Net flow model: predicts surplus (+) / deficit (−) directly ----
    # Subtracting two separately-tuned count models shrinks the imbalance
    # towards 0; a model trained on the difference keeps the signal.
    logger.info("-" * 60)
    logger.info("TARGET: net flow (dropoffs − pickups)")
    y_n = (tr["target_dropoffs"] - tr["target_pickups"]).to_numpy(np.float32)
    yv_n = (va["target_dropoffs"] - va["target_pickups"]).to_numpy(np.float32)
    p_bl = baseline_predict(va, "dropoffs") - baseline_predict(va, "pickups")
    sep = {}
    for _, label in TARGETS:
        best_key = report["best_model"][label]["key"]
        if best_key == "baseline":
            sep[label] = baseline_predict(va, label)
        else:
            sep[label] = np.clip(joblib.load(report["best_model"][label]["model_path"]).predict(Xv), 0, None)
    p_sep = sep["dropoffs"] - sep["pickups"]
    t0 = time.time()
    xn, xn_params, xn_res = train_xgboost(X, y_n, Xv, yv_n, n_iter, seed,
                                          space=space_net, counts=False)
    path = os.path.join(models_dir, "xgb_net.joblib")
    joblib.dump(xn, path)
    p_net = xn.predict(Xv)
    net_res = {
        "Baseline": {"val_metrics": {**compute_metrics(yv_n, p_bl, counts=False),
                                     **direction_metrics(yv_n, p_bl)}},
        "Pickup/dropoff models (difference)": {
            "val_metrics": {**compute_metrics(yv_n, p_sep, counts=False),
                            **direction_metrics(yv_n, p_sep)}},
        "XGBoost net-flow model": {
            "val_metrics": {**compute_metrics(yv_n, p_net, counts=False),
                            **direction_metrics(yv_n, p_net)},
            "best_params": xn_params, "search": xn_res, "model_path": path,
            "best_iteration": int(xn.best_iteration),
            "train_time_s": round(time.time() - t0, 1)},
    }
    report["models"]["net_flow"] = net_res
    report["best_model"]["net_flow"] = {"name": "XGBoost net-flow model", "key": "xgb_net",
                                        "model_path": path,
                                        "val_mae": net_res["XGBoost net-flow model"]["val_metrics"]["mae"]}
    for name, r in net_res.items():
        m = r["val_metrics"]
        logger.info("  %-36s MAE=%.4f dir_acc=%s deficit_recall=%s surplus_recall=%s",
                    name, m["mae"], m["direction_accuracy"], m["deficit_recall"], m["surplus_recall"])
    del xn

    with open(os.path.join(models_dir, "baseline.json"), "w", encoding="utf-8") as f:
        json.dump({"description": "same hour-of-week mean over last 4 weeks, "
                                  "fallback 28-day mean", "columns": BASELINE_COL}, f, indent=2)
    with open(os.path.join(outputs_path, "metrics", "training_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    for label in ("pickups", "dropoffs"):
        b = report["best_model"][label]
        logger.info("  %-8s best=%s val MAE=%.4f", label, b["name"], b["val_mae"])
    logger.info("=" * 60)
    return report
