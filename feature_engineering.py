#!/usr/bin/env python3
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from utils import load_config
import os
import sys
import argparse
import yaml

# Default config path
default_config_path = "./config/development/pipeline.yaml"
preprocessed_train_path = "./data/preprocessed/consumption_train.csv"
preprocessed_test_path = "./data/preprocessed/consumption_test.csv"


def parse_args(config):
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument(
        "--forecast_horizon", type=int, default=config.get("forecast_horizon", 48)
    )
    parser.add_argument("--n_lags", type=int, default=config.get("n_lags", 24))
    parser.add_argument(
        "--rolling_mean_window_size",
        type=int,
        default=config.get("rolling_mean_window_size", 12),
    )
    parser.add_argument(
        "--date_features", nargs="+", default=config.get("date_features", [])
    )
    return parser.parse_args()


def feature_engineering(
    df,
    freq,
    id_col,
    time_col,
    target_col,
    forecast_horizon,
    n_lags,
    rolling_mean_window_size,
    n_lag_transforms,
    date_features,
    static_features,
    on_test=False,
):
    lags = [i for i in range(forecast_horizon, forecast_horizon + n_lags)]
    lag_transforms = {
        i: [ExpandingMean(), RollingMean(window_size=rolling_mean_window_size)]
        for i in range(forecast_horizon, forecast_horizon + n_lag_transforms)
    }
    fcst = MLForecast(
        models=[],
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        static_features=static_features,
    )
    df_transformed = fcst.preprocess(
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        static_features=static_features,
    )
    print("Feature engineering completed. Output shape:", df_transformed.shape)
    return df_transformed


def main():
    # Read data
    if not os.path.exists(preprocessed_data_path):
        print("Data not found. Please run the preprocessing script first.")
        print("Exiting...")
        sys.exit(1)
    df = pd.read_csv(preprocessed_data_path, parse_dates=time_col)

    config = load_config(default_config_path)

    args = parse_args(config)
    _ = feature_engineering()


if __name__ == "__main__":
    main()
