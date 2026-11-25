# Bike Rebalancing ML

**ML-based proactive rebalancing for NYC Citi Bike stations using ~2 years of trip data.**

The system predicts per-station pickup and drop-off demand at 1-hour resolution, identifies likely bike shortages and surpluses before they occur, and generates greedy rebalancing routes for a truck fleet. The full pipeline runs end-to-end from raw ZIP archives to evaluation metrics, figures, and an HTML report.

---

## Results (test period: Sep 13 – Dec 31, 2023)

### Demand prediction — XGBoost (best model)

| Target | MAE | RMSE | R² | RMSLE |
|---|---|---|---|---|
| Pickups | **0.917** | 1.827 | 0.796 | 0.417 |
| Drop-offs | **0.932** | 1.867 | 0.796 | 0.418 |

XGBoost beats the historical baseline by ~27 % MAE on both targets.  
Adding weather features reduces pickup MAE by ~3 % relative to no-weather XGBoost.

### Rebalancing simulation (220 planning windows, 15 trucks, 3 km radius)

| Strategy | Shortage reduction | Bikes moved | Truck-km | Efficiency |
|---|---|---|---|---|
| No rebalancing | 0 % | 0 | 0 | — |
| Historical demand | 1.9 % | 57 694 | 8 429 | 0.31 bikes/km |
| **ML prediction** | **3.7 %** | **51 849** | **10 604** | **0.68 bikes/km** |
| Oracle (perfect) | 6.2 % | 58 810 | 6 688 | 1.00 bikes/km |

ML rebalancing achieves **59.9 % of the theoretical maximum** with roughly half the trucks needed by the historical strategy.

---

## Pipeline overview

```
Raw ZIP archives (2022–2023)
  → extract & validate monthly CSVs
  → clean + aggregate to 10-min station × time grid
  → build station master list
  → fetch Open-Meteo hourly weather
  → feature engineering (lags, rolling stats, weather, neighbours)
  → chronological 70 / 15 / 15 % train / val / test split
  → train Baseline, Random Forest, XGBoost
  → evaluate → select best model by validation MAE
  → predict → deficit/surplus detection
  → greedy rebalancing route generation
  → figures + HTML report
```

Individual stages can be run independently (see **Usage** below).

---

## Repository structure

```
bike-rebalancing-ml/
│
├── src/
│   ├── data_processing/     # preprocess.py, weather.py
│   ├── feature_engineering/
│   ├── models/
│   ├── evaluation/
│   ├── rebalancing/
│   └── visualization/
│
├── configs/
│   ├── config.yaml          # local run (adjust raw_data_path)
│   └── config_colab.yaml    # Google Colab run
│
├── data/                    # created by the pipeline (gitignored)
│   ├── processed/
│   └── features/
│
├── models/                  # trained model files (gitignored)
│
├── outputs/
│   ├── figures/             # 16 PNG research plots
│   ├── metrics/             # JSON evaluation / rebalancing reports
│   └── predictions/         # test-set predictions parquet (gitignored)
│
├── tests/                   # unit tests (pytest)
├── report/                  # LaTeX source for the written report
│
├── main.py                  # pipeline entry point
├── run_in_colab.ipynb        # Colab notebook
├── requirements.txt
└── README.md
```

> `data/`, `models/`, and large binary files are gitignored.  
> `outputs/figures/` and `outputs/metrics/` **are** committed so results are reproducible without re-running the pipeline.

---

## Quick start

### 1. Clone and set up the environment

```bash
git clone https://github.com/raju-048/bike-rebalancing-ml.git
cd bike-rebalancing-ml

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Requires **Python 3.10+**.

### 2. Add the raw data

Download the monthly Citi Bike trip data ZIP archives from [citibikenyc.com/system-data](https://citibikenyc.com/system-data) for 2022 and 2023.  
Place the year-level ZIPs (or extracted monthly ZIPs) in the directory configured as `raw_data_path` in `configs/config.yaml`.

```yaml
# configs/config.yaml
raw_data_path: "e:/prj"   # ← change this to wherever your ZIPs live
```

### 3. Run the full pipeline

```bash
python main.py --config configs/config.yaml
```

### 4. Run individual stages

```bash
python main.py --config configs/config.yaml --stage weather      # fetch Open-Meteo weather (needs internet)
python main.py --config configs/config.yaml --stage preprocess
python main.py --config configs/config.yaml --stage features
python main.py --config configs/config.yaml --stage train
python main.py --config configs/config.yaml --stage evaluate
python main.py --config configs/config.yaml --stage rebalance

# Preprocess specific months only
python main.py --config configs/config.yaml --stage preprocess --months 2022-01 2022-02
```

### 5. Run on Google Colab

Open `run_in_colab.ipynb` in Colab and follow the cells. Use `configs/config_colab.yaml` which points paths to `/content/drive/MyDrive/...`.

---

## Configuration

All tuneable parameters live in `configs/config.yaml`. Key settings:

| Key | Default | Description |
|---|---|---|
| `raw_data_path` | `"e:/prj"` | Directory containing year/month ZIP archives |
| `time_interval_minutes` | `10` | Aggregation resolution (preprocessing) |
| `model_interval_minutes` | `60` | Prediction resolution (model) |
| `train_ratio` | `0.70` | Chronological train fraction |
| `max_train_rows` | `1 000 000` | Rows sampled for training (memory limit) |
| `rf_max_rows` | `300 000` | Smaller sample for Random Forest |
| `n_iter_search` | `4` | XGBoost hyperparameter configs tried |
| `vehicle_capacity` | `20` | Bikes per rebalancing truck |
| `num_vehicles` | `15` | Trucks per planning window |
| `max_route_distance_km` | `3.0` | Max source→dest straight-line distance |
| `use_weather` | `true` | Include Open-Meteo weather features |

---

## Data sources

| Source | Description |
|---|---|
| [Citi Bike System Data](https://citibikenyc.com/system-data) | Monthly trip CSVs (2022–2023, ~24 files) |
| [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) | Hourly NYC weather — free, no API key required |

Raw CSV and Parquet files are **not committed** to this repository (gitignored). Re-run the pipeline from the downloaded ZIPs to regenerate them.

---

## Models

Three models are trained for each target (pickups, drop-offs):

1. **Historical Baseline** — station-level average demand by (day-of-week, hour) on training data only.
2. **Random Forest** — `sklearn.ensemble.RandomForestRegressor` with randomised hyperparameter search on the validation set.
3. **XGBoost** — `xgboost.XGBRegressor` with randomised hyperparameter search. Best model on both targets.

**Leakage prevention:** all lag, rolling, and neighbour features shift past the current interval; the data is split chronologically; no test-set information is used for model selection.

---

## Tests

```bash
python -m pytest tests/ -v
```

Unit tests cover schema validation, cleaning, aggregation, zero-fill, lag correctness, rolling stats, chronological splitting, and rebalancing feasibility.

---

## Figures

All 16 research figures are pre-generated in `outputs/figures/`:

| File | Description |
|---|---|
| `01_monthly_volume.png` | Monthly trip volume (2022–2023) |
| `02_hourly_weekday_weekend.png` | Hourly demand by day type |
| `03_weather_effect_on_demand.png` | Demand vs temperature / precipitation |
| `04_top_stations.png` | Top 20 stations by pickup demand |
| `05_model_comparison.png` | MAE / RMSE comparison across models |
| `07_feature_importance.png` | XGBoost feature importance |
| `08_actual_vs_predicted_week.png` | Actual vs predicted over a sample week |
| `09_error_by_hour_and_weather.png` | Prediction error by hour and weather |
| `10_net_flow_pred_vs_actual.png` | Net flow prediction vs actual |
| `11_rebalancing_strategies.png` | Rebalancing strategy comparison |

---

## Reproducibility

- Random seed: `42` (NumPy, Python `random`, scikit-learn, XGBoost)
- Train/val/test split dates are logged to `outputs/metrics/split_dates.json`
- All hyperparameter configurations tried are logged to `outputs/metrics/training_report.json`
- Dependency versions are pinned in `requirements.txt`
- Python 3.10+ required

---

## Limitations

- Station inventory (dock fill level) is **not available** in trip records. Rebalancing numbers represent *predicted demand pressure*, not exact shortages.
- Rebalancing distances are **straight-line (Haversine)**, not actual road distances.
- The greedy matching algorithm is not globally optimal.
- Multi-horizon prediction (60 / 120 min) is a planned extension.

---

## License

This project is for academic / research purposes. Citi Bike trip data is provided under the [Citi Bike Data License Agreement](https://ride.citibikenyc.com/data-sharing-policy). Open-Meteo data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
