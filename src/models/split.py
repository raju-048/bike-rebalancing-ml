"""
Chronological Train / Validation / Test Split
==============================================

The split is defined by two timestamps computed from the full time range of
the feature files and the configured ratios, snapped to NYC local midnight:

    train :            t <  train_end
    val   : train_end <= t <  val_end
    test  : val_end   <= t

It is never random, so the model is always evaluated on the future.
The boundaries are saved to outputs/metrics/split_dates.json and reused by
the train / evaluate / rebalance stages.
"""

import os
import json
import logging

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

LOCAL_TZ = "America/New_York"


def _snap_local_midnight(ts_utc: pd.Timestamp) -> pd.Timestamp:
    loc = ts_utc.tz_localize("UTC").tz_convert(LOCAL_TZ).normalize()
    return loc.tz_convert("UTC").tz_localize(None)


def _time_range(files: list) -> tuple:
    first = pq.read_table(files[0], columns=["interval_start"]).column(0).to_pandas()
    last = pq.read_table(files[-1], columns=["interval_start"]).column(0).to_pandas()
    return first.min(), last.max()


def compute_split(files: list, train_ratio: float = 0.70, val_ratio: float = 0.15,
                  outputs_path: str = "outputs") -> dict:
    t_min, t_max = _time_range(files)
    span = t_max - t_min
    train_end = _snap_local_midnight(t_min + span * train_ratio)
    val_end = _snap_local_midnight(t_min + span * (train_ratio + val_ratio))
    split = {
        "train_start": str(t_min),
        "train_end": str(train_end),
        "val_end": str(val_end),
        "test_end": str(t_max),
        "note": "UTC-naive; train < train_end <= val < val_end <= test",
    }
    logger.info("Chronological split (UTC):")
    logger.info("  Train: %s → %s", t_min, train_end)
    logger.info("  Val:   %s → %s", train_end, val_end)
    logger.info("  Test:  %s → %s", val_end, t_max)
    os.makedirs(os.path.join(outputs_path, "metrics"), exist_ok=True)
    with open(os.path.join(outputs_path, "metrics", "split_dates.json"), "w",
              encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def split_bounds(split: dict, part: str) -> tuple:
    """(lo, hi) timestamps for 'train' | 'val' | 'test' (hi exclusive, None = open)."""
    tr, va = pd.Timestamp(split["train_end"]), pd.Timestamp(split["val_end"])
    return {"train": (None, tr), "val": (tr, va), "test": (va, None)}[part]


def in_part(ts: pd.Series, split: dict, part: str) -> pd.Series:
    lo, hi = split_bounds(split, part)
    m = pd.Series(True, index=ts.index)
    if lo is not None:
        m &= ts >= lo
    if hi is not None:
        m &= ts < hi
    return m


def load_split(outputs_path: str) -> dict:
    path = os.path.join(outputs_path, "metrics", "split_dates.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — run --stage train first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
