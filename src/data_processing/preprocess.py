"""
Data Preprocessing Pipeline for Citi Bike Trip Data
=====================================================

Processes raw Citi Bike trip data from nested ZIP archives into
station-level fixed-interval aggregation Parquet files.

Data structure handled:
    Year ZIP (e.g. 2022-citibike-tripdata.zip)
      └── Monthly ZIP (e.g. 202201-citibike-tripdata.zip)
            └── CSV files (e.g. 202201-citibike-tripdata_1.csv)

Output per month (<YYYY-MM>_station_time.parquet):
    station_id      category   Citi Bike station id (string values)
    interval_start  datetime   tz-naive UTC, floored to `time_interval_minutes`
    pickups         int32      trips that STARTED at this station in the interval
    dropoffs        int32      trips that ENDED at this station in the interval

    The grid is complete: every active station × every interval of the
    calendar month (NYC local month, expressed in UTC). Missing combinations
    are real zeros (no trips), not missing data.

Key cleaning decisions:
    * Raw Citi Bike timestamps are NYC local time. They are converted to UTC
      (downstream features/weather join expect UTC-naive timestamps). The
      ambiguous DST fall-back hour is resolved as the first (DST) occurrence;
      the non-existent spring-forward hour is shifted forward.
    * Citi Bike files trips by END time, so a month's CSV also contains trips
      that started late in the previous month. Those pickups are routed back
      into the previous month's file ("spillover") instead of extending the
      grid into the previous month (which created overlapping rows with fake
      zero dropoffs).
    * Pickups and dropoffs are validated independently: a trip with a valid
      start station but no end station (bike not re-docked) still counts as a
      pickup; it just produces no dropoff.
    * Trip-level filters: duplicate ride_id, unparseable timestamps,
      negative duration, < 60 s (false starts / re-docks, per Citi Bike's own
      convention), > 24 h.
    * Coordinates are only used to reject a side when they are present AND
      outside NYC bounds. Missing coordinates do not invalidate a station id.

Memory strategy:
    * CSVs are read in chunks with only the needed columns.
    * Counts are accumulated directly into numpy (station × interval) arrays,
      so memory per month is bounded (~100 MB at 10-min resolution) regardless
      of the number of trips.
    * ride_id duplicate detection uses 64-bit hashes instead of strings.
    * Each monthly ZIP is extracted, processed, then deleted.
"""

import os
import re
import gc
import json
import zipfile
import tempfile
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "ride_id",
    "started_at",
    "ended_at",
    "start_station_id",
    "end_station_id",
]

OPTIONAL_COLUMNS = [
    "rideable_type",
    "start_station_name",
    "end_station_name",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]

# Columns actually needed for aggregation + station master
USED_COLUMNS = [
    "ride_id", "started_at", "ended_at",
    "start_station_id", "end_station_id",
    "start_station_name", "end_station_name",
    "start_lat", "start_lng", "end_lat", "end_lng",
]

# Mapping from possible raw column names to our standard names.
COLUMN_ALIASES = {
    "ride_id":      ["ride_id", "trip_id", "bikeid"],
    "rideable_type": ["rideable_type", "bike_type", "biketype"],
    "started_at":   ["started_at", "start_time", "starttime", "start_date"],
    "ended_at":     ["ended_at", "ended_at_time", "stoptime", "stop_time", "end_time", "stop_date"],
    "start_station_name": ["start_station_name", "start station name"],
    "start_station_id":   ["start_station_id", "start station id"],
    "end_station_name":   ["end_station_name", "end station name"],
    "end_station_id":     ["end_station_id", "end station id"],
    "start_lat":    ["start_lat", "start station latitude", "start latitude"],
    "start_lng":    ["start_lng", "start station longitude", "start longitude", "start_lon"],
    "end_lat":      ["end_lat", "end station latitude", "end latitude"],
    "end_lng":      ["end_lng", "end station longitude", "end longitude", "end_lon"],
    "member_casual": ["member_casual", "usertype", "user_type"],
}

# NYC (+ Jersey City / Hoboken) geographic bounds for coordinate validation
NYC_LAT_MIN, NYC_LAT_MAX = 40.40, 41.00
NYC_LNG_MIN, NYC_LNG_MAX = -74.30, -73.60

MAX_TRIP_DURATION_HOURS = 24
MIN_TRIP_DURATION_SECONDS = 60

LOCAL_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Schema normalization and validation
# ---------------------------------------------------------------------------

def build_column_map(raw_columns: list) -> dict:
    """Return {raw_column_name: standard_name} for every recognised column."""
    raw_lower = {c.lower().strip(): c for c in raw_columns}
    col_map = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower().strip() in raw_lower:
                col_map[raw_lower[alias.lower().strip()]] = standard
                break
    return col_map


def validate_schema(columns: list, filename: str) -> tuple:
    """Check that all required columns are present after alias mapping."""
    col_map = build_column_map(columns)
    mapped = set(col_map.values())
    missing = [c for c in REQUIRED_COLUMNS if c not in mapped]
    if missing:
        logger.error("Schema validation FAILED for %s — missing: %s", filename, missing)
        logger.error("  Found columns: %s", columns)
        return False, missing, col_map
    return True, [], col_map


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_zip_files(raw_data_path: str) -> list:
    """Discover all ZIP files in the raw data directory (non-recursive)."""
    zip_files = [
        os.path.join(raw_data_path, f)
        for f in os.listdir(raw_data_path)
        if f.lower().endswith(".zip") and re.match(r"\d{4}-citibike", f.lower())
    ]
    zip_files.sort()
    logger.info("Discovered %d year ZIP file(s) in %s", len(zip_files), raw_data_path)
    return zip_files


def get_monthly_zips(zip_path: str) -> list:
    """List monthly ZIP entries inside an outer year ZIP (skips __MACOSX)."""
    monthly_zips = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if (name.endswith(".zip")
                    and "__MACOSX" not in name
                    and not base.startswith(".")):
                monthly_zips.append(name)
    monthly_zips.sort()
    return monthly_zips


def parse_month_from_name(zip_name: str) -> str:
    """'.../202201-citibike-tripdata.zip' -> '2022-01'."""
    base = os.path.basename(zip_name)
    match = re.match(r"(\d{4})(\d{2})", base)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return base.replace(".zip", "")


def month_bounds_utc(month_label: str, tz: str = LOCAL_TZ) -> tuple:
    """
    Return (start_utc, end_utc) as tz-naive UTC timestamps for the NYC local
    calendar month. Local midnight is never ambiguous in America/New_York.
    """
    start_local = pd.Timestamp(f"{month_label}-01")
    end_local = start_local + pd.offsets.MonthBegin(1)
    start_utc = start_local.tz_localize(tz).tz_convert("UTC").tz_localize(None)
    end_utc = end_local.tz_localize(tz).tz_convert("UTC").tz_localize(None)
    return start_utc, end_utc


def shift_month(month_label: str, k: int) -> str:
    return (pd.Period(month_label, freq="M") + k).strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_ids(values: pd.Series) -> np.ndarray:
    """64-bit hash of string ids (collision probability negligible here)."""
    return pd.util.hash_array(values.fillna("").to_numpy(dtype=object))


def parse_local_to_utc(s: pd.Series, tz: str = LOCAL_TZ) -> pd.Series:
    """Parse NYC-local timestamp strings and return tz-naive UTC timestamps."""
    try:
        ts = pd.to_datetime(s, errors="coerce", format="ISO8601")
    except (ValueError, TypeError):
        ts = pd.to_datetime(s, errors="coerce", format="mixed")
    # Fall-back hour (e.g. 01:30 on first Sunday of Nov) occurs twice; the CSV
    # has no offset, so treat it as the first (DST) occurrence. Spring-forward
    # gap times don't exist locally; shift them forward.
    ts = ts.dt.tz_localize(
        tz,
        ambiguous=np.ones(len(ts), dtype=bool),
        nonexistent="shift_forward",
    )
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def _clean_station_ids(s: pd.Series) -> pd.Series:
    """Strip whitespace; map empty / 'nan' strings to NaN."""
    s = s.astype("string").str.strip()
    return s.mask(s.isin(["", "nan", "NaN", "None", "NULL"]))


# ---------------------------------------------------------------------------
# Duplicate detection (first pass)
# ---------------------------------------------------------------------------

def find_duplicate_ride_hashes(csv_files: list, chunk_size: int) -> set:
    """
    Read only ride_id from every CSV of the month and return the set of
    hashes that occur more than once. Memory: 8 bytes per trip.
    """
    parts = []
    for csv_path in csv_files:
        head = pd.read_csv(csv_path, nrows=0)
        col_map = build_column_map(list(head.columns))
        raw_col = next((k for k, v in col_map.items() if v == "ride_id"), None)
        if raw_col is None:
            logger.warning("No ride_id column in %s, skipping duplicate check", csv_path)
            continue
        for chunk in pd.read_csv(csv_path, usecols=[raw_col], dtype=str,
                                 chunksize=max(chunk_size, 500_000)):
            parts.append(_hash_ids(chunk[raw_col]))
    if not parts:
        return set()
    all_h = np.concatenate(parts)
    del parts
    uniq, counts = np.unique(all_h, return_counts=True)
    dup = set(uniq[counts > 1].tolist())
    del all_h, uniq, counts
    return dup


# ---------------------------------------------------------------------------
# Count accumulator
# ---------------------------------------------------------------------------

class CountGrid:
    """Dense (station × interval) pickup / dropoff counters for one month."""

    def __init__(self, n_intervals: int, initial_capacity: int = 2048):
        self.n = n_intervals
        self.ids = {}          # station_id -> row
        self.id_list = []
        cap = initial_capacity
        self.pu = np.zeros((cap, n_intervals), dtype=np.int32)
        self.do = np.zeros((cap, n_intervals), dtype=np.int32)

    def _ensure_capacity(self, needed: int):
        cap = self.pu.shape[0]
        if needed <= cap:
            return
        new_cap = max(needed, cap * 2)
        for name in ("pu", "do"):
            old = getattr(self, name)
            new = np.zeros((new_cap, self.n), dtype=np.int32)
            new[:cap] = old
            setattr(self, name, new)

    def rows_for(self, station_ids: np.ndarray) -> np.ndarray:
        codes, uniques = pd.factorize(station_ids)
        rows_u = np.empty(len(uniques), dtype=np.int64)
        for i, sid in enumerate(uniques):
            r = self.ids.get(sid)
            if r is None:
                r = len(self.id_list)
                self.ids[sid] = r
                self.id_list.append(sid)
            rows_u[i] = r
        self._ensure_capacity(len(self.id_list))
        return rows_u[codes]

    def add(self, kind: str, station_ids: np.ndarray, idx: np.ndarray):
        if len(station_ids) == 0:
            return
        rows = self.rows_for(station_ids)
        np.add.at(self.pu if kind == "pickups" else self.do, (rows, idx), 1)


# ---------------------------------------------------------------------------
# Cleaning + aggregation of one chunk
# ---------------------------------------------------------------------------

def _valid_coord(lat: pd.Series, lng: pd.Series) -> pd.Series:
    """True unless coordinates are present AND outside NYC bounds."""
    lat = pd.to_numeric(lat, errors="coerce")
    lng = pd.to_numeric(lng, errors="coerce")
    present = lat.notna() & lng.notna()
    inside = lat.between(NYC_LAT_MIN, NYC_LAT_MAX) & lng.between(NYC_LNG_MIN, NYC_LNG_MAX)
    return ~present | inside


def process_chunk(
    df: pd.DataFrame,
    col_map: dict,
    stats: dict,
    dup_hashes: set,
    seen_dup_hashes: set,
    grid: CountGrid,
    month_start_utc: pd.Timestamp,
    interval_minutes: int,
    spill: list,
    station_coords: dict,
):
    """Clean one chunk, add its pickups/dropoffs to `grid`, route spillover."""
    df = df.rename(columns=col_map)
    n0 = len(df)
    stats["raw_rows"] += n0

    # --- Duplicate ride_id (keep first occurrence across the whole month) ---
    if dup_hashes:
        h = _hash_ids(df["ride_id"])
        is_dup_candidate = np.fromiter((x in dup_hashes for x in h), bool, len(h))
        drop = np.zeros(len(h), dtype=bool)
        for i in np.flatnonzero(is_dup_candidate):
            if h[i] in seen_dup_hashes:
                drop[i] = True
            else:
                seen_dup_hashes.add(h[i])
        stats["duplicate_rows"] += int(drop.sum())
        df = df[~drop]

    # --- Timestamps (NYC local -> UTC naive) ---
    start = parse_local_to_utc(df["started_at"])
    end = parse_local_to_utc(df["ended_at"])
    bad_ts = start.isna() | end.isna()
    stats["invalid_timestamp_rows"] += int(bad_ts.sum())

    dur = (end - start).dt.total_seconds()
    # DST fall-back: a trip crossing the repeated hour can look negative when
    # both ends are read as DST. Re-read the end as standard time and keep it
    # if that makes the duration valid.
    neg0 = ~bad_ts & (dur < 0)
    if neg0.any():
        raw_end = pd.to_datetime(df.loc[neg0, "ended_at"], errors="coerce", format="mixed")
        alt_end = raw_end.dt.tz_localize(
            LOCAL_TZ, ambiguous=np.zeros(len(raw_end), dtype=bool),
            nonexistent="shift_forward",
        ).dt.tz_convert("UTC").dt.tz_localize(None)
        fixable = (alt_end - start[neg0]).dt.total_seconds() >= 0
        end = end.copy()  # .dt-derived Series is read-only; writes would be discarded
        end.loc[fixable[fixable].index] = alt_end[fixable]
        stats["dst_fallback_fixed_rows"] = stats.get("dst_fallback_fixed_rows", 0) + int(fixable.sum())
        dur = (end - start).dt.total_seconds()
    neg = ~bad_ts & (dur < 0)
    short = ~bad_ts & (dur >= 0) & (dur < MIN_TRIP_DURATION_SECONDS)
    long_ = ~bad_ts & (dur > MAX_TRIP_DURATION_HOURS * 3600)
    stats["negative_duration_rows"] += int(neg.sum())
    stats["too_short_rows"] += int(short.sum())
    stats["too_long_rows"] += int(long_.sum())

    keep = ~(bad_ts | neg | short | long_)
    df = df[keep]
    start = start[keep]
    end = end[keep]
    stats["valid_trip_rows"] += len(df)

    freq_ns = np.int64(interval_minutes * 60 * 1_000_000_000)
    base_ns = np.int64(month_start_utc.value)

    for side, ts in (("start", start), ("end", end)):
        kind = "pickups" if side == "start" else "dropoffs"
        sid = _clean_station_ids(df[f"{side}_station_id"])
        has_sid = sid.notna()
        stats[f"missing_{side}_station_rows"] += int((~has_sid).sum())

        lat_col, lng_col = f"{side}_lat", f"{side}_lng"
        if lat_col in df.columns and lng_col in df.columns:
            coord_ok = _valid_coord(df[lat_col], df[lng_col])
        else:
            coord_ok = pd.Series(True, index=df.index)
        bad_coord = has_sid & ~coord_ok
        stats[f"invalid_{side}_coord_rows"] += int(bad_coord.sum())

        ok = has_sid & coord_ok
        sid_ok = sid[ok].to_numpy(dtype=object)
        ts_ns = ts[ok].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        idx = (ts_ns - base_ns) // freq_ns
        stats[f"{kind}_events"] += len(idx)

        in_month = (idx >= 0) & (idx < grid.n)
        grid.add(kind, sid_ok[in_month], idx[in_month])

        out = ~in_month
        if out.any():
            floored = pd.to_datetime(base_ns + idx[out] * freq_ns)
            spill.append(pd.DataFrame({
                "station_id": sid_ok[out],
                "interval_start": floored,
                "kind": kind,
            }))

        # --- Station master info (first valid coord/name per station) ---
        name_col = f"{side}_station_name"
        info = pd.DataFrame({"station_id": sid[ok].astype(str)})
        if lat_col in df.columns:
            info["lat"] = pd.to_numeric(df.loc[ok, lat_col], errors="coerce")
            info["lng"] = pd.to_numeric(df.loc[ok, lng_col], errors="coerce")
        else:
            info["lat"] = np.nan
            info["lng"] = np.nan
        info["name"] = df.loc[ok, name_col] if name_col in df.columns else ""
        info = info.dropna(subset=["lat", "lng"]).drop_duplicates("station_id")
        new = info[~info["station_id"].isin(station_coords.keys())]
        for r in new.itertuples(index=False):
            station_coords[r.station_id] = {
                "lat": float(r.lat), "lng": float(r.lng),
                "name": r.name if isinstance(r.name, str) else "",
            }


# ---------------------------------------------------------------------------
# Grid -> DataFrame
# ---------------------------------------------------------------------------

def grid_to_frame(grid: CountGrid, month_start_utc, interval_minutes: int,
                  min_trips: int, stats: dict) -> pd.DataFrame:
    n_st = len(grid.id_list)
    pu = grid.pu[:n_st]
    do = grid.do[:n_st]
    totals = pu.sum(axis=1).astype(np.int64) + do.sum(axis=1)
    active = totals >= min_trips

    ids = np.array(grid.id_list, dtype=object)
    order = np.argsort(ids[active].astype(str), kind="stable")
    rows = np.flatnonzero(active)[order]
    active_ids = [str(x) for x in ids[rows]]

    stats["active_stations"] = len(rows)
    stats["inactive_stations"] = int((~active).sum())
    stats["pickups_dropped_inactive"] = int(pu[~active].sum())
    stats["dropoffs_dropped_inactive"] = int(do[~active].sum())

    n = grid.n
    intervals = pd.date_range(month_start_utc, periods=n, freq=f"{interval_minutes}min")
    k = len(rows)
    df = pd.DataFrame({
        "station_id": pd.Categorical.from_codes(
            np.repeat(np.arange(k, dtype=np.int32), n), categories=active_ids),
        "interval_start": np.tile(intervals.values, k),
        "pickups": pu[rows].ravel(),
        "dropoffs": do[rows].ravel(),
    })
    return df


def apply_spill_to_frame(df: pd.DataFrame, spill_df: pd.DataFrame,
                         month_start_utc, interval_minutes: int) -> tuple:
    """
    Add spillover counts into an existing complete, (station, interval)-sorted
    month frame. Returns (df, n_applied, n_dropped).
    """
    if spill_df is None or spill_df.empty:
        return df, 0, 0
    cats = df["station_id"].cat.categories if hasattr(df["station_id"], "cat") \
        else pd.Index(sorted(df["station_id"].astype(str).unique()))
    n = int(df["interval_start"].nunique())
    pos_station = pd.Index(cats).get_indexer(spill_df["station_id"].astype(str))
    freq_ns = interval_minutes * 60 * 1_000_000_000
    t_idx = ((spill_df["interval_start"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
              - month_start_utc.value) // freq_ns)
    ok = (pos_station >= 0) & (t_idx >= 0) & (t_idx < n)
    applied, dropped = 0, int((~ok).sum())
    for kind in ("pickups", "dropoffs"):
        m = ok & (spill_df["kind"].to_numpy() == kind)
        if m.any():
            pos = pos_station[m].astype(np.int64) * n + t_idx[m]
            arr = df[kind].to_numpy().copy()
            np.add.at(arr, pos, 1)
            df[kind] = arr
            applied += int(m.sum())
    return df, applied, dropped


# ---------------------------------------------------------------------------
# Per-month processing
# ---------------------------------------------------------------------------

def process_one_month(
    outer_zip_path: str,
    monthly_zip_entry: str,
    output_dir: str,
    interval_minutes: int,
    chunk_size: int,
    min_trips_per_month: int,
    all_station_coords: dict,
    temp_dir: str = None,
    carry_in: pd.DataFrame = None,
) -> tuple:
    """
    Process a single month.

    Returns (stats, spill_prev_df, spill_next_df) where the spill frames hold
    individual pickup/dropoff events that belong to the neighbouring months.
    """
    month_label = parse_month_from_name(monthly_zip_entry)
    logger.info("Processing %s ...", month_label)

    month_start, month_end = month_bounds_utc(month_label)
    n_intervals = int((month_end.value - month_start.value)
                      // (interval_minutes * 60 * 1_000_000_000))

    stats = {
        "month": month_label,
        "interval_minutes": interval_minutes,
        "csv_files": 0,
        "raw_rows": 0,
        "duplicate_rows": 0,
        "invalid_timestamp_rows": 0,
        "negative_duration_rows": 0,
        "too_short_rows": 0,
        "too_long_rows": 0,
        "valid_trip_rows": 0,
        "missing_start_station_rows": 0,
        "missing_end_station_rows": 0,
        "invalid_start_coord_rows": 0,
        "invalid_end_coord_rows": 0,
        "pickups_events": 0,
        "dropoffs_events": 0,
        "carry_in_events": 0,
        "spill_to_prev_events": 0,
        "spill_to_next_events": 0,
        "active_stations": 0,
        "inactive_stations": 0,
        "grid_rows": 0,
        "status": "ok",
    }

    grid = CountGrid(n_intervals)
    spill = []
    month_station_coords = {}

    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"citibike_{month_label}_", dir=temp_dir) as tmpdir:
        # Stream the inner monthly ZIP to disk (no big in-memory buffer)
        inner_zip_path = os.path.join(tmpdir, "_monthly.zip")
        with zipfile.ZipFile(outer_zip_path, "r") as outer:
            with outer.open(monthly_zip_entry, "r") as src, open(inner_zip_path, "wb") as dst:
                while True:
                    block = src.read(8 * 1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
        with zipfile.ZipFile(inner_zip_path, "r") as mz:
            for member in mz.namelist():
                if member.lower().endswith(".csv") and "__MACOSX" not in member:
                    mz.extract(member, tmpdir)
        os.remove(inner_zip_path)

        csv_files = sorted(str(p) for p in Path(tmpdir).rglob("*.csv")
                           if not p.name.startswith("._"))
        stats["csv_files"] = len(csv_files)
        if not csv_files:
            logger.warning("  No CSV files found in %s", monthly_zip_entry)
            stats["status"] = "no_csvs"
            return stats, None, None

        logger.info("  Pass 1: detecting duplicate ride_ids ...")
        dup_hashes = find_duplicate_ride_hashes(csv_files, chunk_size)
        logger.info("  Found %d duplicated ride_id value(s)", len(dup_hashes))
        seen_dup = set()

        logger.info("  Pass 2: cleaning + aggregating (%d-min, chunk_size=%d) ...",
                    interval_minutes, chunk_size)
        for csv_path in csv_files:
            head = pd.read_csv(csv_path, nrows=0)
            ok, missing, col_map = validate_schema(list(head.columns), csv_path)
            if not ok:
                logger.warning("  Skipping %s — schema error: %s",
                               os.path.basename(csv_path), missing)
                stats["status"] = f"partial: schema error in {os.path.basename(csv_path)}"
                continue
            usecols = [raw for raw, std in col_map.items() if std in USED_COLUMNS]
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str,
                                     usecols=usecols, low_memory=False):
                process_chunk(chunk, col_map, stats, dup_hashes, seen_dup, grid,
                              month_start, interval_minutes, spill, month_station_coords)
                del chunk

    # --- Carry-in: events from the previous month's file that belong here ---
    if carry_in is not None and not carry_in.empty:
        freq_ns = interval_minutes * 60 * 1_000_000_000
        t_idx = ((carry_in["interval_start"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
                  - month_start.value) // freq_ns)
        m = (t_idx >= 0) & (t_idx < n_intervals)
        for kind in ("pickups", "dropoffs"):
            mk = m & (carry_in["kind"].to_numpy() == kind)
            grid.add(kind, carry_in["station_id"].to_numpy(dtype=object)[mk], t_idx[mk])
        stats["carry_in_events"] = int(m.sum())

    # --- Split spillover into previous / next month ---
    spill_df = pd.concat(spill, ignore_index=True) if spill else \
        pd.DataFrame(columns=["station_id", "interval_start", "kind"])
    del spill
    spill_prev = spill_df[spill_df["interval_start"] < month_start]
    spill_next = spill_df[spill_df["interval_start"] >= month_end]
    stats["spill_to_prev_events"] = len(spill_prev)
    stats["spill_to_next_events"] = len(spill_next)

    # --- Station coords -> global ---
    for sid, info in month_station_coords.items():
        if sid not in all_station_coords:
            all_station_coords[sid] = info.copy()
        else:
            cur = all_station_coords[sid]
            if pd.isna(cur.get("lat")) and not pd.isna(info.get("lat")):
                cur["lat"], cur["lng"] = info["lat"], info["lng"]
            if not cur.get("name") and info.get("name"):
                cur["name"] = info["name"]

    # --- Build complete grid and save ---
    logger.info("  Building complete station × interval grid ...")
    result_df = grid_to_frame(grid, month_start, interval_minutes, min_trips_per_month, stats)
    del grid
    stats["grid_rows"] = len(result_df)

    output_file = os.path.join(output_dir, f"{month_label}_station_time.parquet")
    result_df.to_parquet(output_file, engine="pyarrow", index=False)
    stats["pickups_in_file"] = int(result_df["pickups"].sum())
    stats["dropoffs_in_file"] = int(result_df["dropoffs"].sum())
    logger.info("  Saved: %s (%d rows, %d stations × %d intervals)",
                output_file, len(result_df), stats["active_stations"], n_intervals)

    logger.info("  Summary %s: raw=%d dup=%d bad_ts=%d neg=%d short=%d long=%d "
                "valid_trips=%d no_start_st=%d no_end_st=%d pickups=%d dropoffs=%d "
                "spill_prev=%d spill_next=%d",
                month_label, stats["raw_rows"], stats["duplicate_rows"],
                stats["invalid_timestamp_rows"], stats["negative_duration_rows"],
                stats["too_short_rows"], stats["too_long_rows"],
                stats["valid_trip_rows"], stats["missing_start_station_rows"],
                stats["missing_end_station_rows"], stats["pickups_in_file"],
                stats["dropoffs_in_file"], stats["spill_to_prev_events"],
                stats["spill_to_next_events"])

    del result_df
    gc.collect()
    return stats, spill_prev, spill_next


SPILL_DIR_NAME = "_spillover"
SPILL_COLUMNS = ["station_id", "interval_start", "kind"]


def _spill_path(output_dir: str, month_label: str, direction: str) -> str:
    """Sidecar holding events found in <month>'s CSVs that belong to prev/next month."""
    return os.path.join(output_dir, SPILL_DIR_NAME, f"{month_label}_to_{direction}.parquet")


def save_spill(output_dir: str, month_label: str, direction: str, spill_df):
    path = _spill_path(output_dir, month_label, direction)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if spill_df is None:
        spill_df = pd.DataFrame(columns=SPILL_COLUMNS)
    out = spill_df[SPILL_COLUMNS].copy()
    out["station_id"] = out["station_id"].astype(str)
    out["interval_start"] = pd.to_datetime(out["interval_start"])
    out.to_parquet(path, engine="pyarrow", index=False)


def load_spill(output_dir: str, month_label: str, direction: str):
    path = _spill_path(output_dir, month_label, direction)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def patch_month_file(output_dir: str, month_label: str, spill_df: pd.DataFrame,
                     interval_minutes: int) -> tuple:
    """Add spillover events into an already-written month file."""
    path = os.path.join(output_dir, f"{month_label}_station_time.parquet")
    df = pd.read_parquet(path)
    month_start, _ = month_bounds_utc(month_label)
    df, applied, dropped = apply_spill_to_frame(df, spill_df, month_start, interval_minutes)
    if applied:
        df.to_parquet(path, engine="pyarrow", index=False)
    del df
    gc.collect()
    return applied, dropped


# ---------------------------------------------------------------------------
# Station master list
# ---------------------------------------------------------------------------

def load_existing_station_coords(output_dir: str) -> dict:
    """Load a previously saved station_master.parquet (if present)."""
    master_path = os.path.join(output_dir, "station_master.parquet")
    coords = {}
    if os.path.exists(master_path):
        try:
            df = pd.read_parquet(master_path)
            for row in df.itertuples(index=False):
                coords[str(row.station_id)] = {
                    "lat": row.lat, "lng": row.lng,
                    "name": row.station_name or "",
                }
            logger.info("Loaded %d existing stations from %s", len(coords), master_path)
        except Exception as e:
            logger.warning("Could not load existing station_master.parquet: %s", e)
    return coords


def build_station_master_list(station_coords: dict, output_dir: str):
    """Save station coordinates/names as a Parquet file."""
    if not station_coords:
        logger.warning("No station coordinates collected — skipping station master list.")
        return
    df = pd.DataFrame([
        {"station_id": str(sid), "station_name": info.get("name", ""),
         "lat": info.get("lat", np.nan), "lng": info.get("lng", np.nan)}
        for sid, info in station_coords.items()
    ]).sort_values("station_id").reset_index(drop=True)
    output_file = os.path.join(output_dir, "station_master.parquet")
    df.to_parquet(output_file, engine="pyarrow", index=False)
    logger.info("Station master list saved: %s (%d stations)", output_file, len(df))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(config: dict, months_only: list = None):
    """
    Run the full preprocessing pipeline.

    Months are processed in chronological order so cross-month spillover can
    be routed: events belonging to the previous month are patched into its
    file (if it was produced in this run), events for the next month are
    carried forward.
    """
    raw_data_path = config["raw_data_path"]
    processed_data_path = config["processed_data_path"]
    interval_minutes = int(config.get("time_interval_minutes", 10))
    chunk_size = int(config.get("chunk_size", 100000))
    min_trips = int(config.get("min_trips_per_month", 5))
    outputs_path = config.get("outputs_path", "outputs")
    temp_extract_dir = config.get("temp_extract_path",
                                  os.path.join(raw_data_path, ".tmp_extract"))

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(os.path.join(outputs_path, "metrics"), exist_ok=True)
    os.makedirs(temp_extract_dir, exist_ok=True)
    logger.info("Using temp extraction directory: %s", temp_extract_dir)

    zip_files = discover_zip_files(raw_data_path)
    if not zip_files:
        logger.error("No year ZIP files found in %s", raw_data_path)
        return

    # Collect (month, outer_zip, entry) and sort chronologically
    jobs = []
    for zip_path in zip_files:
        for entry in get_monthly_zips(zip_path):
            label = parse_month_from_name(entry)
            if months_only and label not in months_only:
                continue
            jobs.append((label, zip_path, entry))
    jobs.sort(key=lambda j: j[0])
    logger.info("Months to process: %s", [j[0] for j in jobs])

    all_stats = []
    stats_by_month = {}
    all_station_coords = load_existing_station_coords(processed_data_path)
    planned = {j[0] for j in jobs}
    done_months = set()
    carry_next = None
    carry_next_month = None

    for label, zip_path, entry in jobs:
        logger.info("=" * 60)
        prev_label, next_label = shift_month(label, -1), shift_month(label, 1)

        # Events belonging to this month that live in neighbouring months' CSVs:
        #  * from the previous month: in memory if it ran just before, else its sidecar
        #  * from the next month: its sidecar, unless it is processed later in this
        #    run (then it patches this file directly, avoiding double counting)
        parts = []
        if carry_next_month == label and carry_next is not None:
            parts.append(carry_next)
        elif prev_label not in done_months:
            sc = load_spill(processed_data_path, prev_label, "next")
            if sc is not None:
                parts.append(sc)
        if next_label not in planned:
            sc = load_spill(processed_data_path, next_label, "prev")
            if sc is not None:
                parts.append(sc)
        carry_in = pd.concat(parts, ignore_index=True) if parts else None
        carry_next, carry_next_month = None, None
        try:
            stats, spill_prev, spill_next = process_one_month(
                outer_zip_path=zip_path,
                monthly_zip_entry=entry,
                output_dir=processed_data_path,
                interval_minutes=interval_minutes,
                chunk_size=chunk_size,
                min_trips_per_month=min_trips,
                all_station_coords=all_station_coords,
                temp_dir=temp_extract_dir,
                carry_in=carry_in,
            )
        except Exception as e:
            logger.error("FAILED to process %s: %s", label, e, exc_info=True)
            stats = {"month": label, "status": f"error: {e}"}
            spill_prev = spill_next = None
            gc.collect()

        all_stats.append(stats)
        stats_by_month[label] = stats

        # Persist this month's spillover so later partial reruns stay correct
        if stats.get("status", "").startswith(("ok", "partial")):
            save_spill(processed_data_path, label, "prev", spill_prev)
            save_spill(processed_data_path, label, "next", spill_next)

        # Route previous-month spillover
        if spill_prev is not None and not spill_prev.empty:
            if prev_label in done_months:
                applied, dropped = patch_month_file(
                    processed_data_path, prev_label, spill_prev, interval_minutes)
                prev_stats = stats_by_month[prev_label]
                prev_stats["spill_in_from_next_applied"] = applied
                prev_stats["spill_in_from_next_dropped"] = dropped
                logger.info("  Patched %d spillover event(s) into %s (%d dropped: inactive station)",
                            applied, prev_label, dropped)
            else:
                # Not processed in this run: saved to the sidecar above and picked
                # up whenever that month is (re)built. If it has no data at all
                # (e.g. 2021-12), these events are simply outside the dataset.
                logger.info("  %d spillover event(s) for %s saved to sidecar "
                            "(applied when that month is built)", len(spill_prev), prev_label)
                stats["spill_to_prev_deferred"] = len(spill_prev)

        if spill_next is not None and not spill_next.empty:
            carry_next, carry_next_month = spill_next, shift_month(label, 1)

        if stats.get("status", "").startswith(("ok", "partial")):
            done_months.add(label)

    # --- Station master list ---
    logger.info("=" * 60)
    build_station_master_list(all_station_coords, processed_data_path)

    # --- Report (merge with any existing report by month) ---
    report_path = os.path.join(outputs_path, "metrics", "preprocessing_report.json")
    existing = []
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as e:
            logger.warning("Could not load existing preprocessing report: %s", e)
    merged = {e.get("month"): e for e in existing}
    for e in all_stats:
        merged[e.get("month")] = e
    merged_report = sorted((v for v in merged.values() if v.get("month")),
                           key=lambda r: r["month"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(merged_report, f, indent=2, default=str)
    logger.info("Preprocessing report saved: %s (%d months)", report_path, len(merged_report))

    ok_stats = [s for s in all_stats if s.get("status", "").startswith(("ok", "partial"))]
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("  Months OK / attempted: %d / %d", len(ok_stats), len(all_stats))
    failed = [s["month"] for s in all_stats if s not in ok_stats]
    if failed:
        logger.warning("  FAILED months: %s", failed)
    logger.info("  Total raw rows:   %d", sum(s.get("raw_rows", 0) for s in ok_stats))
    logger.info("  Total valid trips:%d", sum(s.get("valid_trip_rows", 0) for s in ok_stats))
    logger.info("  Total grid rows:  %d", sum(s.get("grid_rows", 0) for s in ok_stats))
    logger.info("  Total stations:   %d", len(all_station_coords))
    return all_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Citi Bike trip data")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--months", nargs="*", default=None,
                        help="Specific months to process (e.g. 2022-01 2022-02)")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess(config, months_only=args.months)
