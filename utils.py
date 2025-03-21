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


def download_s3_dir(bucket_name, s3_dir, local_dir):
    """function to download objects from an S3 bucket located in the s3_dir directory"""
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    paginator = s3.get_paginator("list_objects_v2")
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
                s3.download_file(bucket_name, s3_key, local_file_path)
