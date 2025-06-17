# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import sys

# Your API definition
app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return "<h1>Welcome to the prediction API!</h1>"


@app.route("/predict", methods=["GET"])
def predict_get():
    return "<h1>Use POST method to make predictions!</h1>"


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    if model:
        try:
            json_data = request.json
            print(type(json_data))
            # print(json_data)
            query = pd.DataFrame(
                data=json_data["dataframe_split"]["data"],
                columns=json_data["dataframe_split"]["columns"],
            )
            prediction = list(model.predict(query))

            return jsonify({"prediction": str(prediction)})

        except:

            return jsonify({"trace": traceback.format_exc()})
    else:
        print("Train the model first")
        return "No model here to use"


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 8080

    model = joblib.load("./models/lightgbm.joblib")
    print("Model loaded")
    model_columns = joblib.load("./models/lightgbm_columns.joblib")
    print("Model columns loaded")

    app.run(port=port, debug=True)
