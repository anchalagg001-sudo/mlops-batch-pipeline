#!/usr/bin/env python3
"""
run.py - Minimal MLOps-style batch job.

Loads a YAML config + OHLCV CSV, computes a rolling mean on `close`,
derives a binary signal (close > rolling_mean), and writes structured
metrics (JSON) plus a detailed log file.

Usage:
    python run.py --input data.csv --config config.yaml \
                   --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS = ("seed", "window", "version")


def parse_args():
    parser = argparse.ArgumentParser(description="Rolling-mean signal batch job")
    parser.add_argument("--input", required=True, help="Path to input OHLCV CSV")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to write log file")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_batch_job")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z"
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    """Metrics must be written on both success and error paths."""
    try:
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        logger.info(f"Metrics written to {output_path}")
    except OSError as e:
        # If we can't even write the metrics file, log it loudly and re-raise
        # so the process exits non-zero.
        logger.error(f"Failed to write metrics file '{output_path}': {e}")
        raise


def load_config(config_path: str, logger: logging.Logger) -> dict:
    logger.info(f"Loading config from {config_path}")

    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Config file is not valid YAML: {e}")

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a top-level mapping (dict)")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Config is missing required field(s): {', '.join(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer")
    if not isinstance(config["version"], str) or not config["version"]:
        raise ValueError("Config field 'version' must be a non-empty string")

    logger.info(
        f"Config loaded + validated: seed={config['seed']}, "
        f"window={config['window']}, version={config['version']}"
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Loading dataset from {input_path}")

    path = Path(input_path)
    if not path.exists():
        raise ValueError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file has no parseable data: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Input file is not a valid CSV: {e}")
    except OSError as e:
        raise ValueError(f"Unable to read input file: {e}")

    if df.empty:
        raise ValueError(f"Input file contains no rows: {input_path}")

    if "close" not in df.columns:
        raise ValueError("Input CSV is missing required column: 'close'")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["close"].isna().all():
        raise ValueError("Column 'close' contains no valid numeric values")

    logger.info(f"Rows loaded: {len(df)} (columns: {list(df.columns)})")
    return df


def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Computing rolling mean on 'close' with window={window}")
    # min_periods=window: first (window-1) rows are NaN and excluded below.
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()

    logger.info("Generating binary signal (1 if close > rolling_mean else 0)")
    df["signal"] = np.where(df["close"] > df["rolling_mean"], 1, 0)
    # Rows without a full rolling window have no defined signal.
    df.loc[df["rolling_mean"].isna(), "signal"] = np.nan

    n_excluded = int(df["rolling_mean"].isna().sum())
    if n_excluded:
        logger.info(
            f"Excluded {n_excluded} row(s) from signal computation "
            f"(insufficient rolling window at start of series)"
        )
    return df


def run(args) -> int:
    logger = setup_logging(args.log_file)
    start_time = time.perf_counter()
    logger.info("=== Job start ===")

    version_for_error = "unknown"

    try:
        config = load_config(args.config, logger)
        version_for_error = config["version"]

        np.random.seed(config["seed"])
        logger.info(f"Random seed set to {config['seed']}")

        df = load_dataset(args.input, logger)

        # Validate that the rolling window is not larger than the dataset
        if config["window"] > len(df):
            raise ValueError(
                f"Window size ({config['window']}) cannot exceed number of rows ({len(df)})"
            )

        df = compute_signal(df, config["window"], logger)

        valid_signals = df["signal"].dropna()
        rows_processed = int(len(df))
        signal_rate = float(valid_signals.mean()) if len(valid_signals) else 0.0

        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))

        metrics = {
            "version": config["version"],
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": elapsed_ms,
            "seed": config["seed"],
            "status": "success",
        }

        logger.info(
            f"Metrics summary: rows_processed={rows_processed}, "
            f"signal_rate={metrics['value']}, latency_ms={elapsed_ms}"
        )

        write_metrics(args.output, metrics, logger)
        logger.info("=== Job end: status=success ===")

        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as e:
        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
        logger.exception(f"Job failed: {e}")

        error_metrics = {
            "version": version_for_error,
            "status": "error",
            "error_message": str(e),
        }

        try:
            write_metrics(args.output, error_metrics, logger)
        except OSError:
            # Already logged inside write_metrics; nothing more we can do.
            pass

        logger.info(f"=== Job end: status=error (elapsed_ms={elapsed_ms}) ===")
        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(run(parse_args()))
