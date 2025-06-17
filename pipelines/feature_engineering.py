#!/usr/bin/env python3
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from utils import load_config, create_dir
import os
import sys
from pathlib import Path
import argparse

# Default config path
default_config_path = "./config/development/config.yaml"
preprocessed_path = "./data/preprocessed/"
processed_path = "./data/processed/"


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
    )
    df_transformed = fcst.preprocess(
        df,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        static_features=static_features,
    )
    return df_transformed


def feature_engineering_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the feature engineering pipeline even if processed data already exists",
    )
    args = parser.parse_args()

    if not os.path.exists(preprocessed_path):
        print("Preprocessed data not found. Please run the preprocessing script first.")
        print("Exiting...")
        sys.exit(1)
    if os.path.exists(processed_path) and not args.force_run:
        print("Processed data already exists. Skipping feature engineering.")
        sys.exit(0)

    config = load_config(default_config_path)

    files = [
        "consumption_train.csv",
        "consumption_test.csv",
        "production_train.csv",
        "production_test.csv",
    ]
    create_dir(processed_path)
    for file in files:
        df = pd.read_csv(Path(preprocessed_path, file), parse_dates=["datetime"])
        df.drop(columns=["data_block_id", "row_id"], inplace=True)
        on_test = True if file.split(".")[0].split("_")[1] == "test" else False
        print(f"Processing {Path(preprocessed_path, file)}...")
        df_transformed = feature_engineering(
            df=df,
            freq=config["freq"],
            id_col=config["id_col"],
            time_col=config["time_col"],
            target_col=config["target_col"],
            forecast_horizon=config["forecast_horizon"],
            n_lags=config["n_lags"],
            rolling_mean_window_size=config["rolling_mean_window_size"],
            n_lag_transforms=config["n_lag_transforms"],
            date_features=config["date_features"],
            static_features=config["static_features"],
            on_test=on_test,
        )
        df_transformed.drop(columns=["datetime"], inplace=True)
        # Save the transformed DataFrame to the processed directory
        df_transformed.to_csv(Path(processed_path, file), index=False)
        print(f"{file} processed and saved to {processed_path}")

    print("Feature engineering pipeline completed.")


if __name__ == "__main__":
    feature_engineering_pipeline()
