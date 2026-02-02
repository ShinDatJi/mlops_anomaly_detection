import keras.saving as saving
import json
import numpy as np
import pandas as pd
import evaluate_simple
import evaluate_patching

data_dir = "./data/processed/"
test_file = "./data/mad_test.csv"
model_dir = "./models/"
report_dir = "./reports/models/"
category = "bottle"

df_test = pd.read_csv(test_file, index_col=0)
df_test = df_test[df_test.category == category].copy()

model_path = model_dir + category + "/"

load_path = model_dir + category + "/"
with open(load_path + "config.json", "r") as f:
    config = json.load(f)

report_path = report_dir + category + "/temp/"
with open(report_path + "report.json", "r") as f:
    report = json.load(f)

if report["patching"]:
    img_size = report["preprocessing"]["params"]["patch_size"]
else:
    img_size = report["preprocessing"]["params"]["image_size"]
grayscale = report["grayscale"]
batch_size = config["training"]["batch_size"]
block_count = len(config["modeling"]["conv_blocks"])
patching = config["patching"]
patches = config["preprocessing"]["patches"]
if "width_cropping" in config["preprocessing"]:
    patches_x = patches - (2 * config["preprocessing"]["width_cropping"])
else:
    patches_x = patches
patches_y = patches - (2 * config["preprocessing"]["height_cropping"])

best_model = saving.load_model(model_path + "model.keras")
# best_model = model

conf = config["testing"]

report["testing"] = {}
rep = report["testing"]

threshold_mode = conf["test_threshold_mode"]

metrics, threshold = evaluate_simple.test_model(data_dir, best_model, category, "test", img_size, batch_size, block_count, threshold_mode, report_path + "2_test_", grayscale)
rep["test"] = {
    "threshold_mode": threshold_mode,
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}

threshold_mode = conf["train_threshold_mode"]
metrics, threshold = evaluate_simple.test_model(data_dir, best_model, category, "train", img_size, batch_size, block_count, threshold_mode, report_path + "1_train_", grayscale)
rep["train"] = {
    "threshold_mode": threshold_mode,
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}

if patching:
    threshold_mode = conf["patch_threshold_mode"]
    patch_threshold = conf["patch_threshold"]

    if threshold_mode != "auto":
        threshold = threshold_mode
    metrics, df_pred, df_pred_patch = evaluate_patching.test_model(data_dir, best_model, category, "test_patching", img_size, batch_size, threshold, df_test, patches, patches_x, patches_y, patch_threshold, report_path + "3_test_patching_", grayscale)
    rep["test_patching"] = {
        "threshold_mode": threshold_mode,
        "threshold": np.round(float(threshold), 2),
        "patch_threshold": patch_threshold,
        "metrics": metrics
    }
    df_pred.to_csv(report_path + "3_test_patching_image_predictions.csv")
    df_pred_patch.to_csv(report_path + "3_test_patching_patch_predictions.csv")

with open(report_path + "report.json", mode="w") as f:
    json.dump(report, f, indent=2)

with open(model_path + "config.json", mode="w") as f:
    json.dump(config, f, indent=2)