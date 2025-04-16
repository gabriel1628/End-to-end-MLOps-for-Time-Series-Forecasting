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
        print("Data not found. Please run the ingestion script first.")
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

    # # Downcast data types
    # int_columns = list(data.dtypes[data.dtypes == np.int64].index)
    # float_columns = list(data.dtypes[data.dtypes == np.float64].index)
    # for col in int_columns:
    #     data[col] = pd.to_numeric(data[col], downcast="unsigned")

    # for col in float_columns:
    #     data[col] = pd.to_numeric(data[col], downcast="float")

    # Train-test split
    consumption = consumption.sort_values(by="datetime")
    consumption_train = consumption[consumption["datetime"] <= splitting_datetime]
    consumption_test = consumption[consumption["datetime"] > splitting_datetime]

    production = production.sort_values(by="datetime")
    production_train = production[production["datetime"] <= splitting_datetime]
    production_test = production[production["datetime"] > splitting_datetime]

    # Save cleaned data
    consumption_train.to_csv("./data/preprocessed/consumption_train.csv", index=False)
    consumption_test.to_csv("./data/preprocessed/consumption_test.csv", index=False)

    production_train.to_csv("./data/preprocessed/production_train.csv", index=False)
    production_test.to_csv("./data/preprocessed/production_test.csv", index=False)

    # Optional: Upload data to AWS S3
    if s3_bucket:
        s3_client = boto3.client("s3")
        files = [
            "consumption_train.csv",
            "consumption_test.csv",
            "production_train.csv",
            "production_test.csv",
        ]
        for file in files:
            local_path = f"./data/preprocessed/{file}"
            s3_key = f"data/preprocessed/{file}"
            with open(local_path, "rb") as f:
                s3_client.upload_fileobj(f, s3_bucket, s3_key)
            print(f"Uploaded {local_path} to s3://{s3_bucket}/{s3_key}")

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
