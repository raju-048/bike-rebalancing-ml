"""
Rebalancing from predicted surplus / deficit
============================================

For every planning window in the test period (by default the morning
07–10 and evening 16–19 local peaks of each day):

  1. Station net flow over the window = dropoffs − pickups
        > 0  surplus  (bikes accumulate)       < 0  deficit (station empties)
  2. A greedy planner moves bikes from predicted-surplus stations to the
     nearest predicted-deficit stations (truck capacity, max distance,
     limited number of trucks per window, minimum move size).
  3. The plan is scored against what ACTUALLY happened:
        shortage(station) = max(0, −(actual_net + delivered − removed))
     i.e. bikes still missing at deficit stations after the moves. Taking
     bikes from a station that actually runs short makes things worse.

Strategies compared on the same windows:
    No rebalancing | Historical average (baseline forecast) | ML forecast
    | Oracle (plans with actual net flow — upper bound)

Limitation: trip data has no dock inventory, so this measures demand
pressure (flow imbalance), not exact empty/full dock events.

Outputs: outputs/metrics/rebalancing_report.json
"""

import os
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOCAL_TZ = "America/New_York"
EARTH_R_KM = 6371.0


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def distance_matrix(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    return haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :]).astype(np.float32)


def plan_moves(net: np.ndarray, dist: np.ndarray, n_vehicles: int, capacity: int,
               max_km: float, min_move: int) -> list:
    """
    Greedy plan. `net` = forecast net flow per station for the window.
    Each truck serves one deficit station (worst first) and collects up to
    `capacity` bikes from the nearest surplus stations within `max_km`.
    Returns list of (from_idx, to_idx, bikes, km).
    """
    need = np.floor(np.maximum(0.0, -net))
    avail = np.floor(np.maximum(0.0, net))
    moves = []
    trucks = 0
    for d in np.argsort(-need):
        if trucks >= n_vehicles or need[d] < min_move:
            break
        cand = np.flatnonzero((avail >= 1) & (dist[d] <= max_km))
        if cand.size == 0:
            continue
        cand = cand[np.argsort(dist[d, cand])]
        cap = min(capacity, need[d])
        load = []
        for s in cand:
            if cap <= 0:
                break
            q = min(avail[s], cap)
            load.append((s, q))
            cap -= q
        total = sum(q for _, q in load)
        if total < min_move:
            continue
        for s, q in load:
            avail[s] -= q
            moves.append((int(s), int(d), int(q), float(dist[d, s])))
        need[d] -= total
        trucks += 1
    return moves


def score(actual_net: np.ndarray, moves: list) -> dict:
    delta = np.zeros_like(actual_net, dtype=np.float64)
    for s, d, q, _ in moves:
        delta[s] -= q
        delta[d] += q
    shortage = np.maximum(0.0, -(actual_net + delta)).sum()
    return {"shortage": float(shortage),
            "bikes_moved": int(sum(m[2] for m in moves)),
            "km": float(sum(m[3] for m in moves)),
            "trucks": len({m[1] for m in moves})}


def rebalance(config: dict):
    processed_dir = config.get("processed_data_path", "data/processed")
    outputs_path = config.get("outputs_path", "outputs")
    capacity = int(config.get("vehicle_capacity", 20))
    n_vehicles = int(config.get("num_vehicles", 15))
    max_km = float(config.get("max_route_distance_km", 3.0))
    min_move = int(config.get("min_move_bikes", 3))
    windows = config.get("rebalance_windows", [["07:00", "10:00"], ["16:00", "19:00"]])

    pred_path = os.path.join(outputs_path, "predictions", "test_predictions.parquet")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"{pred_path} not found — run --stage evaluate first.")
    logger.info("=" * 60)
    logger.info("REBALANCING START  (trucks/window=%d, capacity=%d, max %.1f km)",
                n_vehicles, capacity, max_km)
    logger.info("=" * 60)

    import pyarrow.parquet as pq
    have = set(pq.read_schema(pred_path).names)
    cols = ["station_id", "interval_start", "actual_pickups", "actual_dropoffs",
            "pred_pickups", "pred_dropoffs", "base_pickups", "base_dropoffs"]
    pred = pd.read_parquet(pred_path, columns=cols + (["pred_net"] if "pred_net" in have else []))
    ml_net = pred["pred_net"] if "pred_net" in pred else pred["pred_dropoffs"] - pred["pred_pickups"]
    logger.info("ML strategy uses: %s", "net-flow model" if "pred_net" in pred
                else "pickup/dropoff model difference")
    sm = pd.read_parquet(os.path.join(processed_dir, "station_master.parquet"))
    sm["station_id"] = sm["station_id"].astype(str)
    sm = sm.drop_duplicates("station_id").set_index("station_id")

    loc = pred["interval_start"].dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    minutes = loc.dt.hour * 60 + loc.dt.minute
    win_id = pd.Series(-1, index=pred.index)
    win_names = []
    for i, (a, b) in enumerate(windows):
        ha, ma = map(int, a.split(":"))
        hb, mb = map(int, b.split(":"))
        m = (minutes >= ha * 60 + ma) & (minutes < hb * 60 + mb)
        win_id[m] = i
        win_names.append(f"{a}-{b}")
    keep = win_id >= 0
    df = pd.DataFrame({
        "station_id": pred["station_id"].astype(str)[keep],
        "date": loc.dt.date[keep],
        "win": win_id[keep],
        "actual": (pred["actual_dropoffs"] - pred["actual_pickups"])[keep],
        "ml": ml_net[keep],
        "hist": (pred["base_dropoffs"] - pred["base_pickups"])[keep],
    })
    del pred
    agg = df.groupby(["date", "win", "station_id"]).sum().reset_index()
    stations = sorted(agg["station_id"].unique())
    sidx = {s: i for i, s in enumerate(stations)}
    coords = sm.reindex(stations)
    has_xy = coords["lat"].notna().to_numpy()
    lat = coords["lat"].fillna(0).to_numpy()
    lng = coords["lng"].fillna(0).to_numpy()
    dist = distance_matrix(lat, lng)
    dist[~has_xy, :] = np.inf
    dist[:, ~has_xy] = np.inf
    logger.info("Windows: %d days × %d windows, %d stations",
                agg["date"].nunique(), len(windows), len(stations))

    strategies = ["none", "historical", "ml", "oracle"]
    totals = {s: {"shortage": 0.0, "bikes_moved": 0, "km": 0.0, "trucks": 0} for s in strategies}
    per_window = []
    sample = None
    for (date, w), g in agg.groupby(["date", "win"]):
        idx = g["station_id"].map(sidx).to_numpy()
        vec = lambda col: np.bincount(idx, weights=g[col].to_numpy(), minlength=len(stations))
        actual, ml, hist = vec("actual"), vec("ml"), vec("hist")
        plans = {"none": [], "historical": plan_moves(hist, dist, n_vehicles, capacity, max_km, min_move),
                 "ml": plan_moves(ml, dist, n_vehicles, capacity, max_km, min_move),
                 "oracle": plan_moves(actual, dist, n_vehicles, capacity, max_km, min_move)}
        row = {"date": str(date), "window": win_names[w]}
        for s in strategies:
            r = score(actual, plans[s])
            for k in totals[s]:
                totals[s][k] += r[k]
            row[s] = round(r["shortage"], 1)
        per_window.append(row)
        if pd.Timestamp(date).dayofweek < 5 and w == 0:
            sample = (str(date), win_names[w], actual, ml, plans["ml"])

    base = totals["none"]["shortage"] or 1.0
    oracle_gain = (base - totals["oracle"]["shortage"]) or 1.0
    summary = {}
    for s in strategies:
        t = totals[s]
        summary[s] = {
            "total_shortage_bikes": round(t["shortage"], 1),
            "shortage_reduction_pct": round(100 * (base - t["shortage"]) / base, 2),
            "pct_of_perfect_forecast_benefit": round(100 * (base - t["shortage"]) / oracle_gain, 1),
            "bikes_moved": int(t["bikes_moved"]), "truck_km": round(t["km"], 1),
            "truck_trips": int(t["trucks"]),
            "shortage_avoided_per_bike_moved": round((base - t["shortage"]) / t["bikes_moved"], 3)
            if t["bikes_moved"] else None,
        }

    sample_out = None
    if sample is not None:
        date, wname, actual, ml, moves = sample
        name = lambda i: str(sm["station_name"].get(stations[i], stations[i]))
        sample_out = {
            "date": date, "window": wname,
            "moves": [{"from_station_id": stations[s], "from_station_name": name(s),
                       "to_station_id": stations[d], "to_station_name": name(d),
                       "bikes": q, "km": round(km, 3)} for s, d, q, km in moves],
            "stations": [{"station_id": stations[i], "station_name": name(i),
                          "lat": float(lat[i]), "lng": float(lng[i]),
                          "pred_net": round(float(ml[i]), 2), "actual_net": round(float(actual[i]), 2)}
                         for i in range(len(stations)) if has_xy[i] and (abs(ml[i]) >= 0.5 or abs(actual[i]) >= 1)],
        }

    report = {
        "settings": {"vehicle_capacity": capacity, "num_vehicles_per_window": n_vehicles,
                     "max_route_distance_km": max_km, "min_move_bikes": min_move,
                     "windows_local_time": win_names},
        "n_windows": len(per_window), "summary": summary,
        "per_window": per_window, "sample_window_ml_plan": sample_out,
        "note": ("Shortage = bikes still missing at stations whose actual net flow is negative "
                 "after the planned moves. Trip data has no dock inventory, so this measures "
                 "flow imbalance (demand pressure), not exact empty-dock events."),
    }
    with open(os.path.join(outputs_path, "metrics", "rebalancing_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("%-12s %12s %9s %9s %9s %9s", "strategy", "shortage", "reduced%",
                "of_ideal%", "moved", "truck_km")
    for s in strategies:
        r = summary[s]
        logger.info("%-12s %12.0f %8.1f%% %8.1f%% %9d %9.0f", s, r["total_shortage_bikes"],
                    r["shortage_reduction_pct"], r["pct_of_perfect_forecast_benefit"],
                    r["bikes_moved"], r["truck_km"])
    logger.info("=" * 60)
    logger.info("REBALANCING COMPLETE")
    logger.info("=" * 60)
    return report
