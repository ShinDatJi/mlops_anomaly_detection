import keras.saving as saving
import json
import numpy as np
import pandas as pd
import os
import evaluate_simple
import evaluate_patching

config_file = os.environ["CONFIG_FILE"]
data_train_dir = os.environ["DATA_TRAIN_DIR"]
data_test_dir = os.environ["DATA_TEST_DIR"]
data_test_patching_dir = os.environ["DATA_TEST_PATCHING_DIR"]
data_test_file = os.environ["DATA_TEST_FILE"]
training_report_file = os.environ["TRAINING_REPORT_FILE"]
model_file = os.environ["MODEL_FILE"]
reports_dir = os.environ["REPORTS_DIR"]
evaluation_report_file = os.environ["EVALUATION_REPORT_FILE"]

with open(config_file, "r") as f:
    config = json.load(f)

with open(training_report_file, "r") as f:
    report = json.load(f)

df_test = pd.read_csv(data_test_file, index_col=0)

os.makedirs(os.path.join(reports_dir, "evaluation_train"), exist_ok=True)
os.makedirs(os.path.join(reports_dir, "evaluation_test"), exist_ok=True)
os.makedirs(os.path.join(reports_dir, "evaluation_test_patching"), exist_ok=True)

img_size = report["preprocessing"]["params"]["patch_size"]
grayscale = report["grayscale"]
batch_size = report["training"]["params"]["batch_size"]
block_count = len(report["modeling"]["params"]["conv_blocks"])
patches = report["preprocessing"]["params"]["patches"]
patches_x = patches - (2 * report["preprocessing"]["params"]["width_cropping"])
patches_y = patches - (2 * report["preprocessing"]["params"]["height_cropping"])

best_model = saving.load_model(model_file)
# best_model = model

params = {
    "threshold": 0.5
}
params.update(config["evaluation"])

report["evaluation"] = {}
rep = report["evaluation"]
rep["params"] = params
rep["metrics"] = {}
rep = rep["metrics"]

print("evaluate test")
metrics, threshold = evaluate_simple.test_model(data_test_dir, best_model, img_size, batch_size, block_count, "balanced", os.path.join(reports_dir, "evaluation_test"), grayscale)
rep["test"] = {
    "threshold": np.round(float(threshold), 2)
}
rep["test"].update(metrics)

print("evaluate train")
metrics, threshold = evaluate_simple.test_model(data_train_dir, best_model, img_size, batch_size, block_count, "balanced", os.path.join(reports_dir, "evaluation_train"), grayscale)
rep["train"] = {
    "threshold": np.round(float(threshold), 2)
}
rep["train"].update(metrics)

print("evaluate test patching")
threshold = params["threshold"]
metrics, df_pred, df_pred_patch = evaluate_patching.test_model(data_test_patching_dir, best_model, img_size, batch_size, threshold, df_test, patches_x, patches_y, os.path.join(reports_dir, "evaluation_test_patching"), grayscale)
rep["test_patching"] = {
    "threshold": threshold
}
rep["test_patching"].update(metrics)
# df_pred.to_csv(os.path.join(reports_dir, "image_predictions.csv"))
# df_pred_patch.to_csv(os.path.join(reports_dir, "patch_predictions.csv"))

with open(evaluation_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
