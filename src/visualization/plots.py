"""
Visualization Module
====================

Figures saved to outputs/figures/:

    01  Monthly trip volume (2 years)
    02  Hourly demand — weekday vs weekend
    03  Weather effect on demand — temperature and rain
    04  Top 20 stations by demand
    05  Model comparison on the test period (MAE / RMSE)
    06  Weather impact — model error with vs without weather features
    07  Feature importance (XGBoost)
    08  Actual vs predicted — busiest station, one test week
    09  Test error by hour of day and by weather condition
    10  Surplus / deficit — predicted vs actual net flow (3-h blocks)
    11  Rebalancing strategies — remaining shortage
    12  Station map for a sample peak window (HTML, folium)
"""

import os
import json
import glob
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQ = (8, 6)
DPI = 150
LOCAL_TZ = "America/New_York"


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


def _json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Data-exploration plots
# ---------------------------------------------------------------------------

def plot_monthly_volume(processed_dir, fig_dir):
    rec = []
    for f in sorted(glob.glob(os.path.join(processed_dir, "*_station_time.parquet"))):
        s = pd.read_parquet(f, columns=["pickups"])["pickups"].sum()
        rec.append((os.path.basename(f)[:7], s / 1e6))
    d = pd.DataFrame(rec, columns=["month", "trips_m"])
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.bar(d["month"], d["trips_m"])
    ax.set_ylabel("Trips (millions)")
    ax.set_title("Monthly Citi Bike trips, 2022–2023")
    plt.xticks(rotation=60, ha="right")
    _save(fig, os.path.join(fig_dir, "01_monthly_volume.png"))


def load_feature_sample(features_dir, n_per_month=150_000, seed=42):
    cols = ["station_id", "interval_start", "hour", "is_weekend", "is_workday",
            "temperature", "precipitation", "is_rainy", "target_pickups", "target_dropoffs"]
    parts = []
    for f in sorted(glob.glob(os.path.join(features_dir, "features_????-??.parquet"))):
        d = pd.read_parquet(f, columns=[c for c in cols if c != "station_id"] + ["station_id"])
        if len(d) > n_per_month:
            d = d.sample(n_per_month, random_state=seed)
        d["station_id"] = d["station_id"].astype(str)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def plot_hourly_weekday_weekend(fs, fig_dir):
    a = fs.groupby(["hour", "is_weekend"])["target_pickups"].mean().unstack()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(a.index, a[0], marker="o", label="Weekday")
    ax.plot(a.index, a[1], marker="o", label="Weekend")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day (NYC local)")
    ax.set_ylabel("Mean pickups per station-hour")
    ax.set_title("Demand by hour — weekday vs weekend")
    ax.legend()
    _save(fig, os.path.join(fig_dir, "02_hourly_weekday_weekend.png"))


def plot_weather_effect(fs, fig_dir):
    d = fs.dropna(subset=["temperature"]).copy()
    d["temp_bin"] = (d["temperature"] // 3 * 3).clip(-12, 36)
    day = d[(d["hour"] >= 7) & (d["hour"] <= 21)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    t = day.groupby("temp_bin")[["target_pickups", "target_dropoffs"]].mean()
    axes[0].plot(t.index, t["target_pickups"], marker="o", label="pickups")
    axes[0].plot(t.index, t["target_dropoffs"], marker="s", label="dropoffs")
    axes[0].set_xlabel("Temperature (°C, 3° bins)")
    axes[0].set_ylabel("Mean per station-hour (07–21 h)")
    axes[0].set_title("Demand vs temperature")
    axes[0].legend()
    day = day.assign(rain_level=pd.cut(day["precipitation"], [-0.01, 0, 0.5, 2, 100],
                                       labels=["dry", "light (<0.5mm)", "moderate", "heavy (>2mm)"]))
    r = (day.groupby(["rain_level", "is_workday"], observed=True)["target_pickups"].mean()
         .unstack().rename(columns={0: "Weekend/holiday", 1: "Workday"}))
    r.plot(kind="bar", ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean pickups per station-hour (07–21 h)")
    axes[1].set_title("Demand vs rain")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "03_weather_effect_on_demand.png"))


def plot_top_stations(processed_dir, sm, fig_dir):
    tot = None
    for f in sorted(glob.glob(os.path.join(processed_dir, "*_station_time.parquet"))):
        d = pd.read_parquet(f, columns=["station_id", "pickups"])
        s = d.groupby(d["station_id"].astype(str))["pickups"].sum()
        tot = s if tot is None else tot.add(s, fill_value=0)
    top = tot.nlargest(20)
    names = sm.set_index("station_id")["station_name"].reindex(top.index).fillna(pd.Series(top.index, index=top.index))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names[::-1], top[::-1] / 1e3)
    ax.set_xlabel("Total pickups 2022–2023 (thousands)")
    ax.set_title("Top 20 stations by pickups")
    _save(fig, os.path.join(fig_dir, "04_top_stations.png"))


# ---------------------------------------------------------------------------
# Model plots
# ---------------------------------------------------------------------------

def plot_model_comparison(ev, fig_dir):
    rec = []
    for label, models in ev.get("test_metrics", {}).items():
        for name, m in models.items():
            if not name.startswith("_") and m.get("n"):
                rec.append({"model": name, "target": label, "MAE": m["mae"], "RMSE": m["rmse"]})
    d = pd.DataFrame(rec)
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        d.pivot(index="model", columns="target", values=metric).plot(kind="bar", ax=ax)
        ax.set_title(f"Test {metric} (lower is better)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "05_model_comparison.png"))


def plot_weather_impact(ev, fig_dir):
    rec = []
    for label, models in ev.get("test_metrics", {}).items():
        if "XGBoost" in models and "XGBoost (no weather)" in models:
            rec.append({"target": label, "with weather": models["XGBoost"]["mae"],
                        "without weather": models["XGBoost (no weather)"]["mae"]})
    if not rec:
        return
    d = pd.DataFrame(rec).set_index("target")
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    d.plot(kind="bar", ax=ax)
    for i, (_, r) in enumerate(d.iterrows()):
        pct = 100 * (r["without weather"] - r["with weather"]) / r["without weather"]
        ax.text(i, max(r) * 1.01, f"{pct:+.1f}% better" if pct > 0 else f"{pct:.1f}%", ha="center")
    ax.set_ylabel("Test MAE (bikes per station-hour)")
    ax.set_title("Does weather help? XGBoost with vs without weather features")
    ax.tick_params(axis="x", rotation=0)
    _save(fig, os.path.join(fig_dir, "06_weather_impact.png"))


def plot_feature_importance(tr, fig_dir):
    fi = tr.get("feature_importance", {})
    if not fi:
        return
    weather = set(tr.get("weather_cols", []))
    fig, axes = plt.subplots(1, len(fi), figsize=(7 * len(fi), 7))
    axes = np.atleast_1d(axes)
    for ax, (label, imp) in zip(axes, fi.items()):
        s = pd.Series(imp).nlargest(20)[::-1]
        ax.barh(s.index, s.values, color=["tab:orange" if c in weather else "tab:blue" for c in s.index])
        ax.set_title(f"XGBoost importance (gain) — {label}\norange = weather")
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "07_feature_importance.png"))


def plot_actual_vs_predicted(pred, fig_dir):
    busiest = pred.groupby("station_id", observed=True)["actual_pickups"].sum().idxmax()
    s = pred[pred["station_id"] == busiest].sort_values("interval_start")
    t0 = s["interval_start"].min() + pd.Timedelta(days=7)
    s = s[(s["interval_start"] >= t0) & (s["interval_start"] < t0 + pd.Timedelta(days=7))]
    x = s["interval_start"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, lab in zip(axes, ["pickups", "dropoffs"]):
        ax.plot(x, s[f"actual_{lab}"], label="actual", lw=1.5)
        ax.plot(x, s[f"pred_{lab}"], label="model", lw=1.5, ls="--")
        ax.plot(x, s[f"base_{lab}"], label="4-week average", lw=1, alpha=0.6)
        ax.set_ylabel(f"{lab} / hour")
        ax.legend(loc="upper right")
    axes[0].set_title(f"Actual vs predicted — station {busiest}, one test week")
    _save(fig, os.path.join(fig_dir, "08_actual_vs_predicted_week.png"))


def plot_error_breakdown(ev, fig_dir):
    tm = ev.get("test_metrics", {})
    if not tm:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label in tm:
        b = tm[label]["_best_breakdown"]
        h = pd.DataFrame(b["by_hour"]).T
        axes[0].plot(h.index.astype(int), h["mae"], marker="o", label=label)
        w = pd.DataFrame(b["by_rain"]).T
        axes[1].bar([f"{label}\n{k}" for k in w.index], w["mae"])
        t = pd.DataFrame(b["by_temperature"]).T
        if "mae" in t:
            t = t.dropna(subset=["mae"])
            axes[2].plot(t.index, t["mae"], marker="o", label=label)
    axes[0].set_title("Test MAE by hour (local)")
    axes[0].set_xlabel("hour")
    axes[0].legend()
    axes[1].set_title("Test MAE: dry vs rainy hours")
    axes[2].set_title("Test MAE by temperature")
    axes[2].legend()
    for ax in axes:
        ax.set_ylabel("MAE")
    fig.tight_layout()
    _save(fig, os.path.join(fig_dir, "09_error_by_hour_and_weather.png"))


def plot_net_flow(pred, fig_dir):
    loc = pred["interval_start"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    nf = pd.DataFrame({
        "station_id": pred["station_id"], "block": loc.dt.floor("3h"),
        "actual": pred["actual_dropoffs"] - pred["actual_pickups"],
        "pred": pred["pred_net"] if "pred_net" in pred else pred["pred_dropoffs"] - pred["pred_pickups"],
    }).groupby(["station_id", "block"], observed=True).sum()
    smp = nf.sample(min(len(nf), 60_000), random_state=42)
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    ax.scatter(smp["actual"], smp["pred"], s=4, alpha=0.2)
    lim = np.percentile(np.abs(smp["actual"]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual net flow (dropoffs − pickups)")
    ax.set_ylabel("Predicted net flow")
    ax.set_title("Surplus (+) / deficit (−) per station, 3-hour blocks, test period")
    _save(fig, os.path.join(fig_dir, "10_net_flow_pred_vs_actual.png"))


def plot_rebalancing(rb, fig_dir):
    s = rb.get("summary")
    if not s:
        return
    names = {"none": "No rebalancing", "historical": "Historical avg",
             "ml": "ML forecast", "oracle": "Perfect forecast\n(upper bound)"}
    d = pd.DataFrame(s).T
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    bars = ax.bar([names.get(i, i) for i in d.index], d["total_shortage_bikes"] / 1e3,
                  color=["grey", "tab:blue", "tab:green", "tab:purple"][: len(d)])
    for b, pct in zip(bars, d["shortage_reduction_pct"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"-{pct:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Remaining bike shortage, test peaks (thousands)")
    ax.set_title("Rebalancing driven by each forecast")
    _save(fig, os.path.join(fig_dir, "11_rebalancing_strategies.png"))


def plot_station_map(rb, fig_dir):
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping map")
        return
    smp = rb.get("sample_window_ml_plan")
    if not smp:
        return
    st = pd.DataFrame(smp["stations"])
    fmap = folium.Map(location=[st["lat"].median(), st["lng"].median()], zoom_start=13)
    for r in st.itertuples():
        color = "red" if r.pred_net < 0 else "green"
        folium.CircleMarker([r.lat, r.lng], radius=min(12, max(2, abs(r.pred_net) / 2)),
                            color=color, fill=True, fill_opacity=0.6,
                            popup=f"<b>{r.station_name}</b><br>predicted net {r.pred_net:+.1f}"
                                  f"<br>actual net {r.actual_net:+.1f}").add_to(fmap)
    pos = st.set_index("station_id")[["lat", "lng"]]
    for m in smp["moves"]:
        if m["from_station_id"] in pos.index and m["to_station_id"] in pos.index:
            a, b = pos.loc[m["from_station_id"]], pos.loc[m["to_station_id"]]
            folium.PolyLine([[a.lat, a.lng], [b.lat, b.lng]], color="blue", weight=3, opacity=0.7,
                            tooltip=f"{m['bikes']} bikes: {m['from_station_name']} → "
                                    f"{m['to_station_name']}").add_to(fmap)
    title = (f"<div style='position:fixed;top:10px;left:50px;z-index:1000;background:white;"
             f"padding:8px;border:1px solid #999'><b>{smp['date']} {smp['window']}</b> — "
             f"red = predicted deficit, green = surplus, blue = truck moves</div>")
    fmap.get_root().html.add_child(folium.Element(title))
    path = os.path.join(fig_dir, "12_station_map.html")
    fmap.save(path)
    logger.info("  Saved: %s", path)


def visualize(config: dict):
    processed_dir = config.get("processed_data_path", "data/processed")
    features_dir = config.get("features_data_path", "data/features")
    outputs_path = config.get("outputs_path", "outputs")
    fig_dir = os.path.join(outputs_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    met = os.path.join(outputs_path, "metrics")
    tr = _json(os.path.join(met, "training_report.json"))
    ev = _json(os.path.join(met, "evaluation_report.json"))
    rb = _json(os.path.join(met, "rebalancing_report.json"))
    sm = pd.read_parquet(os.path.join(processed_dir, "station_master.parquet"))
    sm["station_id"] = sm["station_id"].astype(str)
    pred_path = os.path.join(outputs_path, "predictions", "test_predictions.parquet")
    pred = pd.read_parquet(pred_path) if os.path.exists(pred_path) else None

    logger.info("=" * 60)
    logger.info("VISUALIZATION START")
    fs = None
    try:
        fs = load_feature_sample(features_dir)
    except Exception as e:
        logger.warning("Feature sample failed: %s", e)

    jobs = [
        ("01", lambda: plot_monthly_volume(processed_dir, fig_dir)),
        ("02", lambda: fs is not None and plot_hourly_weekday_weekend(fs, fig_dir)),
        ("03", lambda: fs is not None and plot_weather_effect(fs, fig_dir)),
        ("04", lambda: plot_top_stations(processed_dir, sm, fig_dir)),
        ("05", lambda: plot_model_comparison(ev, fig_dir)),
        ("06", lambda: plot_weather_impact(ev, fig_dir)),
        ("07", lambda: plot_feature_importance(tr, fig_dir)),
        ("08", lambda: pred is not None and plot_actual_vs_predicted(pred, fig_dir)),
        ("09", lambda: plot_error_breakdown(ev, fig_dir)),
        ("10", lambda: pred is not None and plot_net_flow(pred, fig_dir)),
        ("11", lambda: plot_rebalancing(rb, fig_dir)),
        ("12", lambda: plot_station_map(rb, fig_dir)),
    ]
    for num, fn in jobs:
        try:
            fn()
        except Exception as e:
            logger.warning("Plot %s failed: %s", num, e, exc_info=True)
    logger.info("VISUALIZATION COMPLETE — %s", fig_dir)
    logger.info("=" * 60)
