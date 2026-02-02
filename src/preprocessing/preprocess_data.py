from pathlib import Path
import numpy as np
import json
import pandas as pd
import keras.utils as utils
import create_patches

category = "bottle"

config_file = f"./models/{category}/config.json"
mad_train_file = "./data/mad_train.csv"
mad_test_file = "./data/mad_test.csv"
data_processed_dir = "./data/processed/"
reports_preprocessing_dir = f"./reports/preprocessing/{category}/"
preprocessing_report_file = f"./reports/preprocessing/{category}/report.json"

with open(config_file, "r") as f:
    config = json.load(f)

Path(reports_preprocessing_dir).mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(mad_train_file, index_col=0)
df_test = pd.read_csv(mad_test_file, index_col=0)

df_train = df_train[df_train.category == category].copy()
df_test = df_test[df_test.category == category].copy()

random_state = config["random_state"]

utils.set_random_seed(random_state)

grayscale = bool(df_train.grayscale.iloc[0])

report = {}
report["params"] = {}
rep = report["params"]
rep["category"] = category
rep["random_state"] = random_state
report["metrics"] = {}
rep = report["metrics"]
rep["img_size"] = int(df_train.img_size.iloc[0])
rep["grayscale"] = grayscale

print(json.dumps(report, indent=2))

report["preprocessing"] = {}
rep = report["preprocessing"]

params = config["preprocessing"]
rep["params"] = params
rep["metrics"] = {}
rep = rep["metrics"]
image_counts, threshold = create_patches.create_patches(data_processed_dir, df_train, "train", keep_good=True, seed=random_state, **params)
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

image_counts, threshold = create_patches.create_patches(data_processed_dir, df_test, "test", seed=random_state, **params)
rep["test_images"] = image_counts
image_counts, threshold = create_patches.create_patches(data_processed_dir, df_test, "test_patching", keep_all=True, seed=random_state, **params)
rep["test_patching_images"] = image_counts

print()
# print(json.dumps(report["preprocessing"], indent=2))

with open(preprocessing_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
