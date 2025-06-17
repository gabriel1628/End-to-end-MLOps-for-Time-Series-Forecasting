# Dependencies
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import joblib
import traceback
import pandas as pd
import sys
import uvicorn

app = FastAPI()


class DataFrameSplit(BaseModel):
    columns: list
    data: list


class PredictRequest(BaseModel):
    dataframe_split: DataFrameSplit


@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>Welcome to the prediction API!</h1>"


@app.get("/predict", response_class=HTMLResponse)
async def predict_get():
    return "<h1>Use POST method to make predictions!</h1>"


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictRequest):
    if model:
        try:
            query = pd.DataFrame(
                data=request.dataframe_split.data,
                columns=request.dataframe_split.columns,
            )
            prediction = list(model.predict(query))
            return {"prediction": str(prediction)}
        except Exception:
            return JSONResponse(
                content={"trace": traceback.format_exc()}, status_code=500
            )
    else:
        raise HTTPException(status_code=500, detail="No model here to use")


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 8080

    model = joblib.load("./models/lightgbm.joblib")
    print("Model loaded")
    model_columns = joblib.load("./models/lightgbm_columns.joblib")
    print("Model columns loaded")

    uvicorn.run("deployment:app", host="0.0.0.0", port=port, reload=True)
