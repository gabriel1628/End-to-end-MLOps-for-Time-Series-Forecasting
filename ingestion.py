import os
import shutil
from glob import glob
from zipfile import ZipFile


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


def ingestion_pipeline():
    """
    Pipeline to download, unzip, organize, and prepare data for further processing.
    """
    print("Downloading the data...")
    os.system("kaggle competitions download -c predict-energy-behavior-of-prosumers")
    print("Data downloaded successfully.")

    print("Unzipping the data...")
    with ZipFile("./predict-energy-behavior-of-prosumers.zip", "r") as zObject:
        zObject.extractall(path="./predict-energy-behavior-of-prosumers")
    os.remove("predict-energy-behavior-of-prosumers.zip")
    print("Data unzipped successfully. The zip file has been deleted.")

    print("Creating the directories for the data...")
    create_dir("./data")
    create_dir("./data/raw")

    print("Moving the data we are interested in to the './data/raw/' directory...")
    source = "./predict-energy-behavior-of-prosumers"
    destination = "./data/raw"
    files = glob(os.path.join(source, "*.csv"), recursive=True)
    files.append("./predict-energy-behavior-of-prosumers/county_id_to_name_map.json")
    # Iterate on all files to move them to the destination folder
    for file_path in files:
        dst_path = os.path.join(destination, os.path.basename(file_path))
        shutil.move(file_path, dst_path)
        print(f"Moved {file_path} -> {dst_path}")

    print("Moving the remaining files in './data/predict-energy-behavior-of-prosumers'")
    shutil.move("./predict-energy-behavior-of-prosumers", "./data")

    # TODO: upload the data on S3.


if __name__ == "__main__":
    try:
        if os.listdir("./data/raw"):
            print("Data already saved in ./data/raw/")
        else:
            ingestion_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")
        ingestion_pipeline()
