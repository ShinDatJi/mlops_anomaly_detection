import keras.saving as saving
import json
import numpy as np
import pandas as pd
from pathlib import Path
import evaluate_simple
import evaluate_patching

category = "bottle"

config_file = f"./models/{category}/config.json"
training_report_file = f"./reports/training/{category}/report.json"
data_processed_dir = "./data/processed/"
reports_model_dir = f"./reports/models/{category}/"
mad_test_file = "./data/mad_test.csv"
reports_evaluation_dir = f"./reports/evaluation/{category}/"
evaluation_report_file = f"./reports/evaluation/{category}/report.json"

with open(config_file, "r") as f:
    config = json.load(f)

with open(training_report_file, "r") as f:
    report = json.load(f)

df_test = pd.read_csv(mad_test_file, index_col=0)
df_test = df_test[df_test.category == category].copy()

Path(reports_evaluation_dir).mkdir(parents=True, exist_ok=True)

img_size = report["preprocessing"]["params"]["patch_size"]
grayscale = report["metrics"]["grayscale"]
batch_size = config["training"]["batch_size"]
block_count = len(config["modeling"]["conv_blocks"])
patches = config["preprocessing"]["patches"]
if "width_cropping" in config["preprocessing"]:
    patches_x = patches - (2 * config["preprocessing"]["width_cropping"])
else:
    patches_x = patches
patches_y = patches - (2 * config["preprocessing"]["height_cropping"])

best_model = saving.load_model(reports_model_dir + "model.keras")
# best_model = model

conf = config["validation"]

report["validation"] = {}
rep = report["validation"]
rep["params"] = {}
rep["metrics"] = {}

print("evaluate test")
threshold_mode = conf["test_threshold_mode"]
rep["params"]["test_threshold_mode"] = threshold_mode
metrics, threshold = evaluate_simple.test_model(data_processed_dir, best_model, category, "test", img_size, batch_size, block_count, threshold_mode, reports_evaluation_dir + "test_", grayscale)
rep["metrics"]["test"] = {
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}

print("evaluate train")
threshold_mode = conf["train_threshold_mode"]
rep["params"]["train_threshold_mode"] = threshold_mode
metrics, threshold = evaluate_simple.test_model(data_processed_dir, best_model, category, "train", img_size, batch_size, block_count, threshold_mode, reports_evaluation_dir + "train_", grayscale)
rep["metrics"]["train"] = {
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}

print("evaluate test patching")
threshold_mode = conf["patch_threshold_mode"]
patch_threshold = conf["patch_threshold"]
rep["params"]["patch_threshold_mode"] = threshold_mode
rep["params"]["patch_threshold"] = patch_threshold
if threshold_mode != "auto":
    threshold = threshold_mode
metrics, df_pred, df_pred_patch = evaluate_patching.test_model(data_processed_dir, best_model, category, "test_patching", img_size, batch_size, threshold, df_test, patches, patches_x, patches_y, patch_threshold, reports_evaluation_dir + "test_patching_", grayscale)
rep["metrics"]["test_patching"] = {
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}
df_pred.to_csv(reports_evaluation_dir + "test_patching_image_predictions.csv")
df_pred_patch.to_csv(reports_evaluation_dir + "test_patching_patch_predictions.csv")

with open(evaluation_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
