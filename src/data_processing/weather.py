"""
Weather Data Fetch (Open-Meteo Archive API)
============================================

Fetches hourly historical weather for NYC from the free Open-Meteo
Archive API (no API key required) and saves it as Parquet.

Usage:
    python main.py --stage weather
    python main.py --stage weather --config configs/config.yaml

The hourly weather is later interpolated to the 10-minute interval grid
during feature engineering (forward-fill for precipitation/weather_code,
linear interpolation for temperature, etc.).

Open-Meteo Archive API docs: https://open-meteo.com/en/docs/historical-weather-api
Rate limits: ~10,000 requests/day for non-commercial use. One call here
fetches the full 2-year hourly series in a single request.
"""

import os
import json
import time
import urllib.request
import urllib.parse
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Open-Meteo Archive API base URL
API_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Hourly variables to fetch (mapped to friendly column names)
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_direction_10m",
]

# Friendly column names for the output Parquet
COLUMN_RENAME = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "precipitation": "precipitation",
    "weather_code": "weather_code",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
}


def build_request_url(lat: float, lng: float, start_date: str, end_date: str,
                      timezone: str) -> str:
    """Build the Open-Meteo Archive API request URL."""
    params = {
        "latitude": lat,
        "longitude": lng,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": timezone,
    }
    return f"{API_BASE}?{urllib.parse.urlencode(params)}"


def fetch_with_retry(url: str, max_retries: int = 3, timeout: int = 120) -> dict:
    """
    Fetch JSON from the API with simple retry + backoff.

    Returns the parsed JSON dict.
    Raises on final failure.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching weather (attempt %d/%d) ...", attempt, max_retries)
            req = urllib.request.Request(url, headers={"User-Agent": "bike-rebalancing-ml/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw)
        except Exception as e:
            last_err = e
            logger.warning("  Attempt %d failed: %s", attempt, e)
            if attempt < max_retries:
                wait = 5 * attempt
                logger.info("  Retrying in %ds ...", wait)
                time.sleep(wait)
    raise RuntimeError(f"Weather fetch failed after {max_retries} attempts: {last_err}")


def parse_weather_json(data: dict) -> pd.DataFrame:
    """
    Parse the Open-Meteo JSON response into a tidy DataFrame.

    Columns: timestamp (naive, in the requested local timezone),
    temperature, humidity, precipitation, weather_code, wind_speed,
    wind_direction.
    """
    hourly = data.get("hourly")
    if not hourly:
        raise ValueError("API response missing 'hourly' field")

    df = pd.DataFrame(hourly)
    # 'time' column is a list of ISO strings like "2022-01-01T00:00"
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.rename(columns=COLUMN_RENAME)

    # Keep only the columns we care about, in a stable order
    keep = ["timestamp"] + list(COLUMN_RENAME.values())
    df = df[[c for c in keep if c in df.columns]]

    # Log any missing values (Open-Meteo fills future/missing hours with NaN)
    for col in df.columns:
        if col == "timestamp":
            continue
        n_null = int(df[col].isna().sum())
        if n_null:
            logger.info("  %s: %d NaN values (will be handled in features)", col, n_null)

    return df


def fetch_weather(config: dict):
    """
    Main entry point: fetch hourly weather and save as Parquet.

    Reads from config:
        use_weather, weather_data_path, weather_lat, weather_lng,
        weather_start_date, weather_end_date, weather_timezone
    """
    if not config.get("use_weather", True):
        logger.info("use_weather is false — skipping weather fetch.")
        return

    lat = config.get("weather_lat", 40.7128)
    lng = config.get("weather_lng", -74.0060)
    start_date = config.get("weather_start_date", "2022-01-01")
    end_date = config.get("weather_end_date", "2023-12-31")
    timezone = config.get("weather_timezone", "America/New_York")
    out_path = config.get("weather_data_path", "data/processed/weather_hourly.parquet")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    url = build_request_url(lat, lng, start_date, end_date, timezone)
    logger.info("Open-Meteo Archive API request:")
    logger.info("  lat/lon:     (%s, %s)", lat, lng)
    logger.info("  date range:  %s to %s", start_date, end_date)
    logger.info("  timezone:    %s", timezone)

    data = fetch_with_retry(url)

    # Check for API error
    if "error" in data and data.get("error"):
        reason = data.get("reason", "unknown")
        raise RuntimeError(f"Open-Meteo API error: {reason}")

    df = parse_weather_json(data)

    # Round numeric columns to reduce storage
    for col in df.columns:
        if col == "timestamp" or col == "weather_code":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    if "weather_code" in df.columns:
        df["weather_code"] = pd.to_numeric(df["weather_code"], errors="coerce").astype("Int32")

    df.to_parquet(out_path, engine="pyarrow", index=False)

    logger.info("=" * 60)
    logger.info("WEATHER FETCH COMPLETE")
    logger.info("  Rows (hours):     %d", len(df))
    logger.info("  Date range:       %s to %s", df["timestamp"].min(), df["timestamp"].max())
    logger.info("  Columns:          %s", list(df.columns))
    logger.info("  Saved:            %s (%.1f MB)",
                out_path, os.path.getsize(out_path) / 1e6)
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    import yaml
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    fetch_weather(cfg)
