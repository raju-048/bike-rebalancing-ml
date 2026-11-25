"""
Bike Rebalancing ML — Main Entry Point
=======================================

Usage:
    # Full pipeline (all stages)
    python main.py --config configs/config.yaml

    # Individual stages
    python main.py --config configs/config.yaml --stage weather      # fetch hourly weather (needs internet)
    python main.py --config configs/config.yaml --stage preprocess
    python main.py --config configs/config.yaml --stage features
    python main.py --config configs/config.yaml --stage train
    python main.py --config configs/config.yaml --stage evaluate
    python main.py --config configs/config.yaml --stage rebalance

    # Preprocess specific months only
    python main.py --stage preprocess --months 2022-01 2022-02
"""

import argparse
import logging
import sys

import yaml

# Configure logging once for the whole pipeline so every stage's
# logger.info(...) actually prints. force=True overrides any config a
# submodule may have set at import time.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Bike Rebalancing ML Pipeline"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "weather", "preprocess", "features", "train", "evaluate",
                  "rebalance", "visualize", "report"],
        help="Which pipeline stage to run"
    )
    parser.add_argument(
        "--months", nargs="*", default=None,
        help="Specific months for preprocess (e.g. 2022-01 2022-02)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stage in ("all", "weather"):
        from src.data_processing.weather import fetch_weather
        fetch_weather(config)

    if args.stage in ("all", "preprocess"):
        from src.data_processing.preprocess import preprocess
        preprocess(config, months_only=args.months)

    if args.stage in ("all", "features"):
        from src.feature_engineering.features import build_features
        build_features(config)

    if args.stage in ("all", "train"):
        from src.models.train import train
        train(config)

    if args.stage in ("all", "evaluate"):
        from src.evaluation.evaluate import evaluate
        evaluate(config)

    if args.stage in ("all", "rebalance"):
        from src.rebalancing.rebalance import rebalance
        rebalance(config)

    if args.stage in ("all", "visualize"):
        from src.visualization.plots import visualize
        visualize(config)

    if args.stage in ("all", "report"):
        print("[report] Not yet implemented.")


if __name__ == "__main__":
    main()
