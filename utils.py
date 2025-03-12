import pandas as pd
import yaml


def train_test_split(df, unique_id="prediction_unit_id", test_window=24 * 60):
    """
    Parameters
    ------
    df : original data set
    unique_id : column name of the segments unique id
    test_window : test size in days

    Returns
    ------
    df_train : train data set
    df_test : test data set
    """
    for i in df[unique_id].unique():
        if i == df[unique_id].unique()[0]:
            df_test = df[df[unique_id] == i][-test_window:]
            continue
        df_test = pd.concat([df_test, df[df[unique_id] == i][-test_window:]])
    df_test.sort_index(inplace=True)

    train_idx = [idx for idx in df.index if idx not in df_test.index]
    df_train = df.loc[train_idx]

    return df_train, df_test


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
