import pandas as pd
import boto3
import sys
import os
from utils import load_config, create_dir
import argparse


raw_data_path = "./data/raw/"
if not os.path.exists(f"{raw_data_path}/train.csv"):
    print("Data not found. Please run the ingestion script first.")
    print("Exiting...")
    sys.exit(1)


def preprocessing_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the preprocessing pipeline even if preprocessed data already exists",
    )
    args = parser.parse_args()
    preprocessed_path = f"./data/preprocessed/"

    if (
        os.path.exists(f"{preprocessed_path}/consumption_train.csv")
        and not args.force_run
    ):
        print("Preprocessed data already exists. Skipping preprocessing.")
        sys.exit(0)

    config = load_config("./config/development/config.yaml")

    print("Preprocessing the data...")
    data = pd.read_csv("./data/raw/train.csv", parse_dates=["datetime"])
    reordered_columns = pd.Index(["target"]).append(data.columns.drop("target"))
    data = data[reordered_columns]

    # Drop discontinuous series
    discontinuous_series = [21, 26, 41, 44, 47, 68]
    data = data[~data["prediction_unit_id"].isin(discontinuous_series)]

    # Handle missing values
    production = data.loc[data["is_consumption"] == 0]
    consumption = data.loc[data["is_consumption"] == 1]
    del data
    # Fill missing values
    production["target"] = production["target"].ffill()
    consumption["target"] = consumption["target"].ffill()
    # # Downcast data types
    # int_columns = list(data.dtypes[data.dtypes == np.int64].index)
    # float_columns = list(data.dtypes[data.dtypes == np.float64].index)
    # for col in int_columns:
    #     data[col] = pd.to_numeric(data[col], downcast="unsigned")

    # for col in float_columns:
    #     data[col] = pd.to_numeric(data[col], downcast="float")

    # Train-test split
    consumption = consumption.sort_values(by="datetime")
    consumption_train = consumption[
        consumption["datetime"] <= (config["splitting_datetime"])
    ]
    consumption_test = consumption[
        consumption["datetime"] > (config["splitting_datetime"])
    ]

    production = production.sort_values(by="datetime")
    production_train = production[
        production["datetime"] <= (config["splitting_datetime"])
    ]
    production_test = production[
        production["datetime"] > (config["splitting_datetime"])
    ]

    # Save cleaned data
    create_dir(preprocessed_path)
    print("Saving preprocessed data...")

    consumption_train.to_csv(f"{preprocessed_path}/consumption_train.csv", index=False)
    consumption_test.to_csv(f"{preprocessed_path}/consumption_test.csv", index=False)

    production_train.to_csv(f"{preprocessed_path}/production_train.csv", index=False)
    production_test.to_csv(f"{preprocessed_path}/production_test.csv", index=False)

    # Optional: Upload data to AWS S3
    if config["s3_bucket"]:
        s3_client = boto3.client("s3")
        files = [
            "consumption_train.csv",
            "consumption_test.csv",
            "production_train.csv",
            "production_test.csv",
        ]
        for file in files:
            local_path = f"{preprocessed_path}/{file}"
            s3_key = f"data/preprocessed/{file}"
            with open(local_path, "rb") as f:
                s3_client.upload_fileobj(f, config["s3_bucket"], s3_key)
            print(f"Uploaded {local_path} to s3://{config['s3_bucket']}/{s3_key}")

    print("Preprocessing pipeline completed.")


if __name__ == "__main__":
    preprocessing_pipeline()
