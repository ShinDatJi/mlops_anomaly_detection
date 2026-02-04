import numpy as np
import json
import pandas as pd
import keras.utils as utils
import os
import create_patches

category = os.environ["CATEGORY"]
config_file = os.environ["CONFIG_FILE"]
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

df_train = pd.read_csv(data_train_file, index_col=0)
df_test = pd.read_csv(data_test_file, index_col=0)

random_state = config["random_state"]

utils.set_random_seed(random_state)

grayscale = bool(df_train.grayscale.iloc[0])

report = {}
report["category"] = category
report["params"] = {}
rep = report["params"]
rep["random_state"] = random_state
report["metrics"] = {}
rep = report["metrics"]
rep["img_size"] = int(df_train.img_size.iloc[0])
rep["grayscale"] = grayscale

# print(json.dumps(report, indent=2))

report["preprocessing"] = {}
rep = report["preprocessing"]

params = config["preprocessing"]
rep["params"] = params
rep["metrics"] = {}
rep = rep["metrics"]
image_counts, threshold = create_patches.create_patches(data_train_dir, df_train, keep_good=True, seed=random_state, **params)
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

image_counts, threshold = create_patches.create_patches(data_test_dir, df_test, seed=random_state, **params)
rep["test_images"] = image_counts
image_counts, threshold = create_patches.create_patches(data_test_patching_dir, df_test, keep_all=True, seed=random_state, **params)
rep["test_patching_images"] = image_counts

print()
# print(json.dumps(report["preprocessing"], indent=2))

with open(preprocessing_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
