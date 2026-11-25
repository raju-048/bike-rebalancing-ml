"""
Evaluation on the held-out TEST period (full, not sampled)
==========================================================

For every trained model (Baseline, RandomForest, XGBoost, XGBoost-no-weather)
and both targets, computes test MAE / RMSE / RMSLE / R² by streaming the test
months one file at a time. Breakdowns for the best model:
    by local hour, weekday vs weekend, rain vs dry, temperature band.

Surplus / deficit evaluation
    Net flow = dropoffs − pickups. Positive = surplus (bikes pile up),
    negative = deficit (station empties). Aggregated per station over
    3-hour local blocks, we compare predicted vs actual net flow and check
    how often the model gets the direction right when the imbalance is
    material (|actual net| >= `imbalance_threshold` bikes).

Outputs
    outputs/predictions/test_predictions.parquet
    outputs/metrics/evaluation_report.json
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import joblib

from src.feature_engineering.features import feature_files
from src.models.split import load_split, split_bounds
from src.models.train import compute_metrics, baseline_predict, TARGETS

logger = logging.getLogger(__name__)

LOCAL_TZ = "America/New_York"


class _Acc:
    """Streaming accumulator for MAE / RMSE / RMSLE / R² (optionally grouped)."""

    def __init__(self, n_groups: int = 1, counts: bool = True):
        z = lambda: np.zeros(n_groups)
        self.n, self.ae, self.se, self.sle, self.sy, self.syy = z(), z(), z(), z(), z(), z()
        self.sp = z()
        self.counts = counts

    def add(self, y, p, g=None):
        y = y.astype(np.float64)
        p = p.astype(np.float64)
        if self.counts:
            p = np.clip(p, 0, None)
        g = np.zeros(len(y), dtype=np.int64) if g is None else g
        L = len(self.n)
        bc = lambda w: np.bincount(g, weights=w, minlength=L)
        self.n += np.bincount(g, minlength=L)
        self.ae += bc(np.abs(y - p))
        self.se += bc((y - p) ** 2)
        if self.counts:
            self.sle += bc((np.log1p(p) - np.log1p(y)) ** 2)
        self.sy += bc(y)
        self.syy += bc(y * y)
        self.sp += bc(p)

    def result(self, i: int = 0) -> dict:
        n = self.n[i]
        if n == 0:
            return {"n": 0}
        var = self.syy[i] - self.sy[i] ** 2 / n
        out = {"n": int(n),
               "mae": round(self.ae[i] / n, 4),
               "rmse": round(float(np.sqrt(self.se[i] / n)), 4),
               "r2": round(1 - self.se[i] / var, 4) if var > 0 else None,
               "mean_actual": round(self.sy[i] / n, 4),
               "mean_pred": round(self.sp[i] / n, 4)}
        if self.counts:
            out["rmsle"] = round(float(np.sqrt(self.sle[i] / n)), 4)
        return out


TEMP_BINS = [-50, 0, 10, 20, 27, 60]
TEMP_LABELS = ["< 0°C", "0–10°C", "10–20°C", "20–27°C", "> 27°C"]


def evaluate(config: dict):
    features_dir = config.get("features_data_path", "data/features")
    models_dir = config.get("models_path", "models")
    outputs_path = config.get("outputs_path", "outputs")
    thr = float(config.get("imbalance_threshold", 3))
    pred_dir = os.path.join(outputs_path, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    with open(os.path.join(outputs_path, "metrics", "training_report.json"), encoding="utf-8") as f:
        tr = json.load(f)
    split = load_split(outputs_path)
    lo, _ = split_bounds(split, "test")
    with open(os.path.join(models_dir, "feature_names.json"), encoding="utf-8") as f:
        names = json.load(f)
    feat_all, feat_nw = names["all"], names["no_weather"]

    # Load models
    models = {}
    for _, label in TARGETS:
        models[label] = {}
        for name, info in tr["models"][label].items():
            if info["key"] == "baseline":
                models[label][name] = ("baseline", None)
            else:
                models[label][name] = (info["key"], joblib.load(info["model_path"]))
    best = {label: tr["best_model"][label]["name"] for _, label in TARGETS}
    net_info = tr.get("best_model", {}).get("net_flow")
    net_model = joblib.load(net_info["model_path"]) if net_info else None
    acc_net = _Acc(counts=False)

    files = feature_files(features_dir, config.get("months_filter") or None)
    acc = {label: {name: _Acc() for name in models[label]} for _, label in TARGETS}
    grp = {label: {"hour": _Acc(24), "daytype": _Acc(2), "rain": _Acc(2),
                   "temp": _Acc(len(TEMP_LABELS))} for _, label in TARGETS}
    pred_parts = []

    logger.info("=" * 60)
    logger.info("EVALUATION on test period from %s", lo)
    logger.info("=" * 60)
    for f in files:
        ts = pd.read_parquet(f, columns=["interval_start"])["interval_start"]
        if not (ts >= lo).any():
            continue
        df = pd.read_parquet(f)
        df = df[df["interval_start"] >= lo]
        df = df.dropna(subset=feat_all + [t for t, _ in TARGETS]).reset_index(drop=True)
        logger.info("  %s: %d test rows", os.path.basename(f), len(df))
        X = df[feat_all].to_numpy(np.float32)
        Xn = df[feat_nw].to_numpy(np.float32)
        g_hour = df["hour"].to_numpy(np.int64)
        g_day = df["is_weekend"].to_numpy(np.int64)
        g_rain = (df["is_rainy"].to_numpy() > 0).astype(np.int64) if "is_rainy" in df else None
        g_temp = (pd.cut(df["temperature"], TEMP_BINS, labels=False).fillna(0).astype(np.int64).to_numpy()
                  if "temperature" in df else None)

        out = {"station_id": df["station_id"].astype(str), "interval_start": df["interval_start"]}
        for c in ("hour", "is_weekend", "temperature", "precipitation", "is_rainy"):
            if c in df:
                out[c] = df[c].to_numpy()
        for target, label in TARGETS:
            y = df[target].to_numpy()
            out[f"actual_{label}"] = y.astype(np.float32)
            for name, (key, model) in models[label].items():
                if key == "baseline":
                    p = baseline_predict(df, label)
                elif key == "xgb_noweather":
                    p = model.predict(Xn)
                else:
                    p = model.predict(X)
                p = np.clip(p, 0, None).astype(np.float32)
                acc[label][name].add(y, p)
                if name == best[label]:
                    out[f"pred_{label}"] = p
                    G = grp[label]
                    G["hour"].add(y, p, g_hour)
                    G["daytype"].add(y, p, g_day)
                    if g_rain is not None:
                        G["rain"].add(y, p, g_rain)
                    if g_temp is not None:
                        G["temp"].add(y, p, g_temp)
                if key == "baseline":
                    out[f"base_{label}"] = p
        if net_model is not None:
            pn = net_model.predict(X).astype(np.float32)
            out["pred_net"] = pn
            acc_net.add(out["actual_dropoffs"] - out["actual_pickups"], pn)
        pred_parts.append(pd.DataFrame(out))
        del df, X, Xn

    if not pred_parts:
        raise ValueError("No test rows found — check split / months_filter.")
    pred = pd.concat(pred_parts, ignore_index=True)
    pred["station_id"] = pred["station_id"].astype("category")
    pred_path = os.path.join(pred_dir, "test_predictions.parquet")
    pred.to_parquet(pred_path, index=False)
    logger.info("Predictions saved: %s (%d rows)", pred_path, len(pred))

    # ---- Surplus / deficit (net flow) per station × 3-hour local block ----
    loc = pred["interval_start"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ)
    block = loc.dt.tz_localize(None).dt.floor("3h")
    cols = {
        "station_id": pred["station_id"], "block": block,
        "actual": pred["actual_dropoffs"] - pred["actual_pickups"],
        "pickup_dropoff_models": pred["pred_dropoffs"] - pred["pred_pickups"],
        "baseline": pred["base_dropoffs"] - pred["base_pickups"],
    }
    if "pred_net" in pred:
        cols["net_flow_model"] = pred["pred_net"]
    nf = pd.DataFrame(cols).groupby(["station_id", "block"], observed=True).sum().reset_index()
    material = nf["actual"].abs() >= thr
    net_eval = {"blocks": int(len(nf)), "material_blocks": int(material.sum()),
                "threshold_bikes": thr,
                "hourly_net_flow_model": acc_net.result() if net_model is not None else None}
    whos = [c for c in ("net_flow_model", "pickup_dropoff_models", "baseline") if c in nf]
    for who in whos:
        e = nf[who] - nf["actual"]
        sign_ok = np.sign(nf.loc[material, who]) == np.sign(nf.loc[material, "actual"])
        net_eval[who] = {
            "net_flow_mae": round(float(e.abs().mean()), 4),
            "net_flow_mae_material": round(float(e[material].abs().mean()), 4),
            "direction_accuracy_material": round(float(sign_ok.mean()), 4),
            "deficit_recall": round(float(((nf[who] <= -thr / 2) & (nf["actual"] <= -thr)).sum()
                                          / max(1, (nf["actual"] <= -thr).sum())), 4),
            "surplus_recall": round(float(((nf[who] >= thr / 2) & (nf["actual"] >= thr)).sum()
                                          / max(1, (nf["actual"] >= thr).sum())), 4),
            # Of the blocks flagged as deficit/surplus, how many really were one
            "deficit_precision": round(float(((nf[who] <= -thr) & (nf["actual"] <= -thr / 2)).sum()
                                             / max(1, (nf[who] <= -thr).sum())), 4),
            "surplus_precision": round(float(((nf[who] >= thr) & (nf["actual"] >= thr / 2)).sum()
                                             / max(1, (nf[who] >= thr).sum())), 4),
        }
    for who in whos:
        r = net_eval[who]
        logger.info("Surplus/deficit 3-h blocks [%s]: net MAE=%.3f direction acc=%.1f%% "
                    "deficit recall=%.1f%% surplus recall=%.1f%%", who, r["net_flow_mae"],
                    100 * r["direction_accuracy_material"], 100 * r["deficit_recall"],
                    100 * r["surplus_recall"])

    results = {}
    for _, label in TARGETS:
        results[label] = {name: a.result() for name, a in acc[label].items()}
        G = grp[label]
        results[label]["_best_breakdown"] = {
            "model": best[label],
            "by_hour": {h: G["hour"].result(h) for h in range(24)},
            "by_daytype": {"weekday": G["daytype"].result(0), "weekend": G["daytype"].result(1)},
            "by_rain": {"dry": G["rain"].result(0), "rainy": G["rain"].result(1)},
            "by_temperature": {lab: G["temp"].result(i) for i, lab in enumerate(TEMP_LABELS)},
        }

    report = {"split_dates": split, "best_model": best, "test_metrics": results,
              "net_flow_evaluation": net_eval, "predictions_path": pred_path}
    with open(os.path.join(outputs_path, "metrics", "evaluation_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    for _, label in TARGETS:
        for name, r in results[label].items():
            if name.startswith("_"):
                continue
            logger.info("  %-8s %-22s MAE=%.4f RMSE=%.4f R²=%s%s", label, name, r["mae"],
                        r["rmse"], r["r2"], "  <= best" if name == best[label] else "")
    logger.info("=" * 60)
    return report
