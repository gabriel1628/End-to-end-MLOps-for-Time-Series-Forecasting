import mlflow
from mlflow.models import Model
from dotenv import dotenv_values
from utils import load_config
import sys

env_vars = dotenv_values("./.env")
default_config_path = "./config/development/config.yaml"
config = load_config(default_config_path)
RUN_ID = config["RUN_ID"]
model_uri = f"runs:/{RUN_ID}/lightgbm"

try:
    # The model is logged with an input example
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    input_data = pyfunc_model.input_example
except mlflow.exceptions.MlflowException:
    print("MLflow server not running. Please start the server using this command:")
    print(
        f"mlflow server --host {env_vars['MLFLOW_TRACKING_HOST']} --port {env_vars['MLFLOW_TRACKING_PORT']}"
    )
    print("Exiting...")
    sys.exit(1)


# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager="uv",
)
