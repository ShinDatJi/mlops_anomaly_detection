import numpy as np
import json
import pandas as pd
import keras.utils as utils
import os
import train_test_split
import create_patches

category = os.environ["CATEGORY"]
config_file = os.environ["CONFIG_FILE"]
data_clean_file = os.environ["DATA_CLEAN_FILE"]
data_train_file = os.environ["DATA_TRAIN_FILE"]
data_test_file = os.environ["DATA_TEST_FILE"]
data_train_dir = os.environ["DATA_TRAIN_DIR"]
data_test_dir = os.environ["DATA_TEST_DIR"]
data_test_patching_dir = os.environ["DATA_TEST_PATCHING_DIR"]
reports_dir = os.environ["REPORTS_DIR"]
preprocessing_report_file = os.environ["PREPROCESSING_REPORT_FILE"]

with open(config_file, "r") as f:
    config = json.load(f)

os.makedirs(reports_dir, exist_ok=True)

# Preparation

params = {
    "train_test_split": 0.5,
    "random_state": 42
}
params.update(config["preparation"])

df = pd.read_csv(data_clean_file, index_col=0)
df_train, df_test = train_test_split.split(df, params["train_test_split"], params["random_state"])
df_train.to_csv(data_train_file)
df_test.to_csv(data_test_file)

grayscale = bool(df_train.grayscale.iloc[0])
img_size = int(df_train.img_size.iloc[0])

report = {}
report["category"] = category
report["img_size"] = img_size
report["grayscale"] = grayscale
report["preparation"] = {}
rep = report["preparation"]
rep["params"] = params

# Preprocessing

params = {
    "patch_size": 78,
    "patches": 5,
    "overlap": 0.5,
    "good_fraction": 0.25,
    "oversampling": True,
    "threshold": "full-auto",
    "threshold_factor": 1,
    "spread": 0.025,
    "height_cropping": 0,
    "width_cropping": 0,
    "random_trans": 0,
    "random_rot": 0,
    "random_trans_sub": 0,
    "random_rot_sub": 0,
    "fill_mode": "reflect",
    "fill_mode_sub": "constant",
    "fill_value": 0,
    "fast_patching": True,
    "random_state": 42
}
params.update(config["preprocessing"])

report["preprocessing"] = {}
rep = report["preprocessing"]
rep["params"] = params
rep["metrics"] = {}
rep = rep["metrics"]

utils.set_random_seed(params["random_state"])

image_counts, threshold = create_patches.create_patches(data_train_dir, df_train, keep_good=True, **params)
rep["threshold"] = float(np.round(threshold, 3))
rep["train_images"] = image_counts

params = params.copy()
params["good_fraction"] = 1
params["oversampling"] = False
params["threshold"] = threshold
params["random_trans"] = False
params["random_rot"] = False
params["random_trans_sub"] = False
params["random_rot_sub"] = False

image_counts, threshold = create_patches.create_patches(data_test_dir, df_test, **params)
rep["test_images"] = image_counts
image_counts, threshold = create_patches.create_patches(data_test_patching_dir, df_test, keep_all=True, **params)
rep["test_patching_images"] = image_counts

print()
# print(json.dumps(report["preprocessing"], indent=2))

with open(preprocessing_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
