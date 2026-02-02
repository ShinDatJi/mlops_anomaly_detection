from pathlib import Path
import numpy as np
import json
import pandas as pd
import keras.utils as utils
import create_patches

train_file = "./data/mad_train.csv"
test_file = "./data/mad_test.csv"
data_dir = "./data/processed/"
model_dir = "./models/"
report_dir = "./reports/models/"
category = "bottle"

load_path = model_dir + category + "/"
with open(load_path + "config.json", "r") as f:
    config = json.load(f)

report_path = report_dir + category + "/temp/"
Path(report_path).mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(train_file, index_col=0)
df_test = pd.read_csv(test_file, index_col=0)

df_train = df_train[df_train.category == category].copy()
df_test = df_test[df_test.category == category].copy()

random_state = config["random_state"]

utils.set_random_seed(random_state)

grayscale = bool(df_train.grayscale.iloc[0])

report = {}
report["category"] = category
report["random_state"] = random_state
report["img_size"] = int(df_train.img_size.iloc[0])
report["grayscale"] = grayscale

print(json.dumps(report, indent=2))

patching = config["patching"]

report["patching"] = patching
report["preprocessing"] = {}
rep = report["preprocessing"]

if patching:
    params = config["preprocessing"]
    rep["params"] = params
    image_counts, threshold = create_patches.create_patches(data_dir, df_train, "train", keep_good=True, seed=random_state, **params)
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

    image_counts, threshold = create_patches.create_patches(data_dir, df_test, "test", seed=random_state, **params)
    rep["test_images"] = image_counts
    image_counts, threshold = create_patches.create_patches(data_dir, df_test, "test_patching", keep_all=True, seed=random_state, **params)
    rep["test_patching_images"] = image_counts
else:
    img_size_fraction = 2
    min_image_size = int(df_train.min_img_size.max())
    img_size = ((min_image_size // (2 * img_size_fraction)) * 2)
    # img_size = 224
    oversampling = True

    params = config["preprocessing"]
    rep["params"] = params
    image_counts = create_patches.create_images(data_dir, df_train, "train", img_size, oversampling=oversampling)
    rep["train_images"] = image_counts
    image_counts = create_patches.create_images(data_dir, df_test, "test", img_size, oversampling=False)
    rep["test_images"] = image_counts

print()
# print(json.dumps(report["preprocessing"], indent=2))

with open(report_path + "report.json", mode="w") as f:
    json.dump(report, f, indent=2)
