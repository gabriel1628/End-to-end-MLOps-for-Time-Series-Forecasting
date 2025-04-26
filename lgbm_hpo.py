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
from utils import load_config, load_data
import yaml
from dotenv import dotenv_values
import warnings
import os
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_environment():
    env_vars = dotenv_values(".env")
    environment = env_vars.get("ENVIRONMENT")
    if environment not in ["development", "staging", "production"]:
        print("ENVIRONMENT variable not set. Exiting...")
        sys.exit(1)
    return environment, env_vars


def setup_logging():
    logger = optuna.logging.get_logger("optuna")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(logging.StreamHandler(sys.stdout))


def load_hpo_config(environment, config):
    hpo_config_path = Path(
        "./config",
        environment,
        f"{config['model_name']}_hpo",
        f"config_{config['hpo_config_version']}.yaml",
    )
    print(f"using {hpo_config_path} for HPO")
    with open(hpo_config_path, "rb") as file:
        hpo_config = yaml.safe_load(file)
    return hpo_config


def get_study(config):
    # TODO: track data versions
    study_name = f"datav1_{config['model_name']}_config{config['hpo_config_version']}"
    study_path = f"{config['studies_dir']}/{study_name}.db"
    storage_path = f"sqlite:///{study_path}"
    sampler_name = f"{study_name}_sampler.pkl"
    sampler_path = f"{config['studies_dir']}/{sampler_name}"
    if os.path.exists(sampler_path):
        print(f"loading sampler from {sampler_path}")
        with open(sampler_path, "rb") as f:
            sampler = pickle.load(f)
        sampler_loaded = True
    else:
        print("no sampler saved for the study, creating a new one")
        sampler = optuna.samplers.TPESampler(seed=config["random_state"])
        sampler_loaded = False

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        directions=["minimize"],
        sampler=sampler,
    )
    if not sampler_loaded:
        print(f"saving sampler to {sampler_path}")
        with open(sampler_path, "wb") as file:
            pickle.dump(study.sampler, file)
    return study


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
    environment, env_vars = get_environment()
    config = load_config(f"./config/{environment}/pipeline.yaml")
    setup_logging()
    device = "gpu" if GPUtil.getAvailable() else "cpu"
    print(f"device set to {device}")
    X_train, y_train = load_data("./data/processed/consumption_train.csv")
    hpo_config = load_hpo_config(environment, config)
    study = get_study(config)
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            hpo_config,
            config["random_state"],
            device,
        ),
        n_trials=config["n_trials"],
    )
    print("End of HPO.")


if __name__ == "__main__":
    main()
