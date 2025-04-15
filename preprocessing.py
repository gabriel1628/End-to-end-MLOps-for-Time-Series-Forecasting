import pandas as pd
import numpy as np
import boto3
import sys
import os
from utils import load_config, create_dir
import argparse


def preprocessing_pipeline():
    config = load_config("./config/development/pipeline.yaml")
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for the energy consumption dataset."
    )
    parser.add_argument(
        "--splitting_datetime",
        type=str,
        default=config["splitting_datetime"],
        help="Datetime to split the data into train and test sets.",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=config["s3_bucket"],
        help="S3 bucket name to upload the preprocessed data.",
    )
    args = parser.parse_args()
    splitting_datetime = args.splitting_datetime
    s3_bucket = args.s3_bucket

    print("Preprocessing the data...")
    # Read data
    if not os.path.exists("./data/raw/train.csv"):
        print("Data not found. Please the ingestion script first.")
        print("Exiting...")
        sys.exit(1)
    data = pd.read_csv("./data/raw/train.csv")
    reordered_columns = pd.Index(["target"]).append(data.columns.drop("target"))
    data = data[reordered_columns]

    # Drop discontinuous series
    discontinuous_series = [21, 26, 41, 44, 47, 68]
    data = data[~data["prediction_unit_id"].isin(discontinuous_series)]

    # Handle missing values
    production = data.loc[
        data["is_consumption"] == 0, ["target", "datetime", "prediction_unit_id"]
    ]
    consumption = data.loc[
        data["is_consumption"] == 1, ["target", "datetime", "prediction_unit_id"]
    ]
    # Fill missing values
    consumption.ffill(inplace=True)
    production.ffill(inplace=True)

    # Merge filled values back into the main dataset
    nan_consumption_indices = consumption[consumption["target"].isna()].index
    nan_production_indices = production[production["target"].isna()].index
    data.loc[nan_consumption_indices, "target"] = consumption.loc[
        nan_consumption_indices, "target"
    ]
    data.loc[nan_production_indices, "target"] = production.loc[
        nan_production_indices, "target"
    ]

    # # Downcast data types
    # int_columns = list(data.dtypes[data.dtypes == np.int64].index)
    # float_columns = list(data.dtypes[data.dtypes == np.float64].index)
    # for col in int_columns:
    #     data[col] = pd.to_numeric(data[col], downcast="unsigned")

    # for col in float_columns:
    #     data[col] = pd.to_numeric(data[col], downcast="float")

    # Train-test split
    data = data.sort_values(by="datetime")
    train = data[data["datetime"] <= splitting_datetime]
    test = data[data["datetime"] > splitting_datetime]

    # Save cleaned data
    train.to_csv("./data/preprocessed/train.csv", index=False)
    test.to_csv("./data/preprocessed/test.csv", index=False)

    # Optional: Upload data to AWS S3
    if s3_bucket:
        s3_client = boto3.client("s3")
        with open("./data/preprocessed/train.csv", "rb") as file:
            s3_client.upload_fileobj(file, s3_bucket, "data/preprocessed/train.csv")
        with open("./data/preprocessed/test.csv", "rb") as file:
            s3_client.upload_fileobj(file, s3_bucket, "data/preprocessed/test.csv")

    print("Preprocessing pipeline completed.")


if __name__ == "__main__":
    try:
        if os.listdir("./data/preprocessed"):
            print("Data already preprocessed. Skipping preprocessing.")
        else:
            preprocessing_pipeline()
    except FileNotFoundError:
        print("Preprocessed data directory does not exist. Creating it now...")
        create_dir("./data/preprocessed")
        preprocessing_pipeline()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
