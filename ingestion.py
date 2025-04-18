import os
import sys
import subprocess
import shutil
from glob import glob
from zipfile import ZipFile
from utils import create_dir


def ingestion_pipeline():
    """
    Pipeline to download, unzip, organize, and prepare data for further processing.
    """
    try:
        os.listdir("./data/raw")
        print("The directory './data/raw/' is not empty. Skipping ingestion.")
        sys.exit(0)
    except FileNotFoundError:
        print("Directory './data/raw/' does not exist. Creating it now.")
        create_dir("./data")
        create_dir("./data/raw")
    except PermissionError:
        print("Permission denied: Unable to access './data/raw/'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("Ingesting the data...")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "predict-energy-behavior-of-prosumers",
            ],
            check=True,
        )
        print("Data downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        return

    print("Unzipping the data...")
    try:
        with ZipFile("./predict-energy-behavior-of-prosumers.zip", "r") as zObject:
            zObject.extractall(path="./predict-energy-behavior-of-prosumers")
        os.remove("predict-energy-behavior-of-prosumers.zip")
        print("Data unzipped successfully. The zip file has been deleted.")
    except Exception as e:
        print(f"Error unzipping data: {e}")
        return

    print("Moving the data we are interested in to the './data/raw/' directory...")
    source = "./predict-energy-behavior-of-prosumers"
    destination = "./data/raw"
    files = glob(os.path.join(source, "*.csv"), recursive=True)
    files.append("./predict-energy-behavior-of-prosumers/county_id_to_name_map.json")
    for file_path in files:
        dst_path = os.path.join(destination, os.path.basename(file_path))
        shutil.move(file_path, dst_path)
        print(f"Moved {file_path} -> {dst_path}")

    print("Moving the remaining files in './data/predict-energy-behavior-of-prosumers'")
    shutil.move("./predict-energy-behavior-of-prosumers", "./data")

    # TODO: upload the data on S3.

    print("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    ingestion_pipeline()
