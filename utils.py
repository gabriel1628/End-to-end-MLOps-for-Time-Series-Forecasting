import pandas as pd
import yaml
import os


class columnDropperTransformer:
    """custom scikit-learn's transformer object to drop 'unique_id' 'ds' columns from X when calling fit and predict methods"""

    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_config(config, config_path):
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Configuration saved to {config_path}")


def create_dir(dir_path):
    """
    Creates a directory if it doesn't already exist.
    """
    try:
        os.mkdir(dir_path)
        print(f"Directory '{dir_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")


def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns="target")
    y = df["target"]
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    return X, y


# TODO: reomve 'local_dir' and keep only 's3_dir'
def download_s3_dir(s3_client, bucket_name, s3_dir, local_dir):
    """function to download objects from an S3 bucket located in the s3_dir directory"""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_dir)

    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                s3_key = obj["Key"]
                if s3_key == s3_dir:
                    continue
                relative_path = os.path.relpath(s3_key, s3_dir)
                local_file_path = os.path.join(local_dir, relative_path)
                local_file_dir = os.path.dirname(local_file_path)
                if not os.path.exists(local_file_dir):
                    os.makedirs(local_file_dir)
                print(
                    f"Downloading s3://{bucket_name}/{s3_key} to {local_file_path}..."
                )
                s3_client.download_file(bucket_name, s3_key, local_file_path)
