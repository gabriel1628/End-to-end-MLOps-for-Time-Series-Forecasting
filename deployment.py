# Import libraries
import os
import sys

sys.path.append(os.path.realpath("../modules"))

import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load the model
with open("model_test.joblib", "rb") as model_file:
    pipeline = joblib.load(model_file)


@app.route("/", methods=["GET"])
def home_endpoint():
    return "Welcome to the homepage ! To get your energy forecasts, send your data to the /predict endpoint"


@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the POST request.
    data_json = request.get_json()
    data = data_json["data"]

    # Put the data in a dataframe and create features
    input_data = pd.DataFrame(data)
    input_data[["consumption", "lag1", "lag2"]] = pd.DataFrame(
        input_data["consumption_values"].tolist(), index=input_data.index
    )
    input_data = input_data.drop(columns="consumption_values")

    # Make prediction using model loaded from disk as per the data.
    preds_df = pd.DataFrame(
        pipeline.predict(input_data), index=input_data.index, columns=["lead1", "lead2"]
    )
    predictions = pd.concat([input_data.iloc[:, :2], preds_df], axis=1)
    predictions["forecasts"] = pd.Series(
        predictions[["lead1", "lead2"]].values.tolist()
    )
    predictions = predictions.drop(columns=["lead1", "lead2"])
    # Reformat predictions in json
    predictions_json = predictions.to_json(orient="records", indent=4)
    output = json.dumps({"predictions": json.loads(predictions_json)}, indent=4)

    return jsonify(output)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
