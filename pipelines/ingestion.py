import os
import sys
import subprocess
import shutil
from glob import glob
from zipfile import ZipFile
from utils import create_dir
import argparse


def ingestion_pipeline():
    """
    Pipeline to download, unzip, organize, and prepare data for further processing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the ingestion pipeline even if the data has already been ingested",
    )
    parser.add_argument(
        "--original-data-path",
        type=str,
        default=".",
        help="Location of the original data directory if already downloaded and unzipped. "
        "In a kaggle notebook, the path is '/kaggle/input'.",
    )
    args = parser.parse_args()

    if os.path.exists("./data/raw/") and not args.force_run:
        print("The directory './data/raw/' already exists. Skipping ingestion.")
        sys.exit(0)

    print("Ingesting the data...")
    if not os.path.exists(
        f"{args.original_data_path}/predict-energy-behavior-of-prosumers/"
    ):
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
    else:
        print(
            "Data already downloaded and unzipped. Skipping download and unzip steps."
        )

    print("Moving the data we are interested in to the './data/raw/' directory...")
    source = f"{args.original_data_path}/predict-energy-behavior-of-prosumers"
    destination = "./data/raw"
    create_dir(destination)
    files = glob(os.path.join(source, "*.csv"), recursive=True)
    files.append(glob(os.path.join(source, "county_id_to_name_map.json"))[0])
    for file_path in files:
        dst_path = os.path.join(destination, os.path.basename(file_path))
        shutil.move(file_path, dst_path)
        print(f"Moved {file_path} -> {dst_path}")

    print("Moving the remaining files in './data/predict-energy-behavior-of-prosumers'")
    if os.path.exists("./data/predict-energy-behavior-of-prosumers"):
        shutil.rmtree("./data/predict-energy-behavior-of-prosumers")
    shutil.move(source, "./data")

    # TODO: upload the data on S3.

    print("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    ingestion_pipeline()
