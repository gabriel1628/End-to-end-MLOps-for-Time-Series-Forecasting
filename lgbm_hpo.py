# -*- coding: utf-8 -*-

import argparse
import sys
import time
import logging
import boto3
import GPUtil
import pandas as pd
import numpy as np
from pathlib import Path
from natsorted import natsorted
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import optuna
from lightgbm import LGBMRegressor
from utils import train_test_split, load_config, download_s3_dir
from preprocessing.preprocessing import *
import yaml
from dotenv import dotenv_values

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_logging():
    logger = optuna.logging.get_logger("optuna")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def objective(trial, X_train, y_train, X_test, y_test, config, random_state, device):
    study_params = {
        "verbosity": -1,
        "random_state": random_state,
        "device": device,
    }
    for int_param in config["int_params"]:
        study_params[int_param["name"]] = trial.suggest_int(**int_param)
    for float_param in config["float_params"]:
        study_params[float_param["name"]] = trial.suggest_float(**float_param)

    tscv = TimeSeriesSplit(n_splits=5)
    model = LGBMRegressor(**study_params)
    cv_errors = cross_val_score(
        model, X_train, y_train, scoring="neg_mean_absolute_error", cv=tscv
    )

    cv_errors = -cv_errors
    for i in range(len(cv_errors)):
        trial.set_user_attr(f"error_split_{i+1}", cv_errors[i])
    trial.set_user_attr("cv_errors_std", cv_errors.std())

    model.fit(X_train, y_train)
    y_fit = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_fit)
    trial.set_user_attr("train_mae", train_mae)

    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    trial.set_user_attr("test_mae", test_mae)

    return cv_errors.mean()


env_vars = dotenv_values(".env")
config = load_config("./config/config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting with Hyperparameter Optimization"
    )
    parser.add_argument(
        "--model_name", type=str, default=config["model_name"], help="Model name"
    )
    parser.add_argument(
        "--preprocessing_version",
        type=int,
        default=config["preprocessing_version"],
        help="Preprocessing version",
    )
    parser.add_argument(
        "--hpo_config_version",
        type=int,
        default=config["hpo_config_version"],
        help="HPO configuration version",
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=config["forecast_horizon"],
        help="Forecast horizon",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=config["random_state"],
        help="Random state for reproducibility",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=config["n_trials"],
        help="Number of trials for Optuna",
    )
    parser.add_argument("--s3_bucket", type=str, default=None, help="S3 bucket name")
    parser.add_argument(
        "--s3_dirs",
        type=str,
        nargs="+",
        default=["data/"],
        help="S3 directories to download the data from",
    )
    parser.add_argument(
        "--local_dirs",
        type=str,
        nargs="+",
        default=["./data/"],
        help="Local directories to save data downloaded from S3",
    )

    args = parser.parse_args()

    setup_logging()

    device = "gpu" if GPUtil.getAvailable() else "cpu"
    print(f"device set to {device}")

    if s3_bucket:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=env_vars["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=env_vars["AWS_SECRET_ACCESS_KEY"],
        )

        for s3_dir, local_dir in zip(args.s3_dirs, args.local_dirs):
            download_s3_dir(args.s3_bucket, s3_dir, local_dir)

    df = pd.read_csv("./data/consumption.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df_train, df_test = train_test_split(df)

    assert df.shape[0] == df_train.shape[0] + df_test.shape[0]
    assert df.shape[1] == df_train.shape[1] == df_test.shape[1]

    preprocessing = vars()[f"preprocessing_{args.preprocessing_version}"]
    X_train, y_train = preprocessing(df_train)
    X_test, y_test = preprocessing(df_test)
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    config_files_path = Path(
        "./config", args.model_name, f"config_{args.hpo_config_version}.yaml"
    )
    print(f"using {config_files_path} for HPO")
    with open(config_files_path, "rb") as file:
        config = yaml.safe_load(file)

    study_name = f"{args.model_name}_preprocessing{args.preprocessing_version}_config{hpo_config_version}"
    study_path = f"./optuna_studies/{study_name}.db"
    storage_path = "sqlite:///{}".format(study_path)
    print(f"Study path : {study_path}")

    sampler = optuna.samplers.TPESampler(seed=args.random_state)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        directions=["minimize"],
        sampler=sampler,
    )

    checkpoint = time.time()
    for i in range(args.n_trials):
        study.optimize(
            lambda trial: objective(
                trial,
                X_train,
                y_train,
                X_test,
                y_test,
                config,
                args.random_state,
                device,
            ),
            n_trials=1,
        )
        if s3_bucket:
            # upload trials to S3 at least every 5 minutes
            if time.time() - checkpoint > 300:
                checkpoint = time.time()
                print("Uploading the trials database to S3...")
                s3.upload_file(
                    study_path,
                    args.s3_bucket,
                    str(study_path),
                )

    if s3_bucket:
        print("End of HPO: uploading the trials database to S3...")
        s3.upload_file(
            study_path,
            args.s3_bucket,
            str(study_path),
        )
        print("trials databse successfully uploaded")
    else:
        print("End of HPO.")


if __name__ == "__main__":
    main()
