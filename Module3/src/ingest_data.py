import os
import tarfile
import urllib.request
from zlib import crc32
import argparse ,sys
from pathlib import Path
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=Path)
parser.add_argument("--split_ratio", type=float)
parser.add_argument("--processed_data", type=Path)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    print(tgz_path)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
#train_set, test_set = split_train_test(housing, 0.2)
if __name__ == "__main__":
    arg_parser = parser.parse_args()
    fetch_housing_data(housing_path=arg_parser.input_path)
    data=pd.read_csv(os.path.join(arg_parser.input_path,"housing.csv"))
    train_set, test_set = split_train_test(data, arg_parser.split_ratio)
    train_set.to_csv(os.path.join(arg_parser.processed_data,"train.csv"))
    test_set.to_csv(os.path.join(arg_parser.processed_data,"test.csv"))