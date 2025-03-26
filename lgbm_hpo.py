# -*- coding: utf-8 -*-

import argparse
import sys
import time
import logging
import boto3
import GPUtil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import optuna
from lightgbm import LGBMRegressor
from utils import load_config, download_s3_dir
import yaml
from dotenv import dotenv_values
import warnings
import os
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_logging():
    logger = optuna.logging.get_logger("optuna")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def objective(trial, X_train, y_train, hpo_config, random_state, device):
    study_params = {
        "verbosity": -1,
        "random_state": random_state,
        "device": device,
    }
    for int_param in hpo_config["int_params"]:
        study_params[int_param["name"]] = trial.suggest_int(**int_param)
    for float_param in hpo_config["float_params"]:
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

    return cv_errors.mean()


def main():
    env_vars = dotenv_values(".env")
    config = load_config("./config/config.yaml")

    parser = argparse.ArgumentParser(
        description="Time Series Forecasting with Hyperparameter Optimization"
    )
    parser.add_argument(
        "--model_name", type=str, default=config["model_name"], help="Model name"
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

    if args.s3_bucket:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=env_vars["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=env_vars["AWS_SECRET_ACCESS_KEY"],
        )

        for s3_dir, local_dir in zip(args.s3_dirs, args.local_dirs):
            download_s3_dir(s3_client, args.s3_bucket, s3_dir, local_dir)

    df_train = pd.read_csv("./data/processed/consumption_train_processed.csv")
    X_train = df_train.drop(columns="target")
    y_train = df_train["target"]
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")

    config_file_path = Path(
        "./config", f"{args.model_name}_hpo", f"config_{args.hpo_config_version}.yaml"
    )
    print(f"using {config_file_path} for HPO")
    with open(config_file_path, "rb") as file:
        hpo_config = yaml.safe_load(file)

    study_name = f"datav1_{config['model_name']}_config{config['hpo_config_version']}"
    study_path = f"./optuna_studies/{study_name}.db"
    storage_path = "sqlite:///{}".format(study_path)
    print(f"Study path : {study_path}")

    sampler_name = f"{study_name}_sampler.pkl"
    if sampler_name in os.listdir("./optuna_studies"):
        sampler_loaded = True
        print("loading sampler...")
        sampler = pickle.load(open(f"./optuna_studies/{sampler_name}", "rb"))
        print("sampler loaded.")
    else:
        sampler_loaded = False
        print("no sampler saved for the study, creating a new one")
        sampler = optuna.samplers.TPESampler(seed=args.random_state)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        directions=["minimize"],
        sampler=sampler,
    )
    if not sampler_loaded:
        print("saving sampler")
        with open(f"./optuna_studies/{study_name}_sampler.pkl", "wb") as file:
            pickle.dump(study.sampler, file)

    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            hpo_config,
            args.random_state,
            device,
        ),
        n_trials=config["n_trials"],
    )
    print("End of HPO.")


if __name__ == "__main__":
    main()
