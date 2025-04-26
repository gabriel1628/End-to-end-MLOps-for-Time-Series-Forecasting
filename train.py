from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import time
from datetime import datetime, timezone
from utils import load_config
from dotenv import dotenv_values
from lgbm_hpo import load_data, get_study
import sys


env_vars = dotenv_values("./.env")
config = load_config("./config/development/config.yaml")

X_train, y_train = load_data("./data/processed/consumption_train.csv")
X_test, y_test = load_data("./data/processed/consumption_test.csv")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = (
    f"Enefit DataV1 {config['model_name']} HpoConfigV{config['hpo_config_version']}"
)
try:
    mlflow.set_experiment(experiment_name=experiment_name)
except mlflow.exceptions.MlflowException:
    print("MLflow server not running. Please start the server using this command:")
    print("mlflow server --host 127.0.0.1 --port 5000")
    print("Exiting...")
    sys.exit(1)

experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"ID for experiment '{experiment_name}': {experiment.experiment_id}")

study = get_study(config)

# run metadata
utc_datetime = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SUTC")
run_name = f"TrialNumber{study.best_trial.number}-{utc_datetime}"

# training
params = study.best_params
model = LGBMRegressor(**params, random_state=config["random_state"])
start = time.time()
model.fit(X_train, y_train)
end = time.time()

# metrics
y_fit = model.predict(X_train)
y_pred = model.predict(X_test)
train_mae = mean_absolute_error(y_train, y_fit)
test_mae = mean_absolute_error(y_test, y_pred)
training_duration = end - start
metrics = {
    "train_mae": train_mae,
    "test_mae": test_mae,
    "training_duration": training_duration,
}

with mlflow.start_run(run_name=run_name) as run:
    # Log the parameters used for the model fit
    mlflow.log_params(params)

    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    # Log an instance of the trained model for later use
    mlflow.lightgbm.log_model(
        lgb_model=model,
        input_example=X_train.iloc[:1],
        artifact_path=config["model_name"],
    )
