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
grayscale = report["metrics"]["grayscale"]
batch_size = config["training"]["batch_size"]
block_count = len(config["modeling"]["conv_blocks"])
patches = config["preprocessing"]["patches"]
patches_x = patches - (2 * config["preprocessing"]["width_cropping"])
patches_y = patches - (2 * config["preprocessing"]["height_cropping"])

best_model = saving.load_model(model_file)
# best_model = model

conf = config["validation"]

report["validation"] = {}
rep = report["validation"]
rep["params"] = {}
rep["metrics"] = {}

print("evaluate test")
threshold_mode = conf["test_threshold_mode"]
rep["params"]["test_threshold_mode"] = threshold_mode
metrics, threshold = evaluate_simple.test_model(data_test_dir, best_model, img_size, batch_size, block_count, threshold_mode, os.path.join(reports_dir, "evaluation_test"), grayscale)
rep["metrics"]["test"] = {
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}

print("evaluate train")
threshold_mode = conf["train_threshold_mode"]
rep["params"]["train_threshold_mode"] = threshold_mode
metrics, threshold = evaluate_simple.test_model(data_train_dir, best_model, img_size, batch_size, block_count, threshold_mode, os.path.join(reports_dir, "evaluation_train"), grayscale)
rep["metrics"]["train"] = {
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}

print("evaluate test patching")
threshold_mode = conf["patch_threshold_mode"]
patch_threshold = conf["patch_threshold"]
rep["params"]["patch_threshold_mode"] = threshold_mode
rep["params"]["patch_threshold"] = patch_threshold
# if threshold_mode != "auto":
#     threshold = threshold_mode
metrics, df_pred, df_pred_patch = evaluate_patching.test_model(data_test_patching_dir, best_model, img_size, batch_size, threshold, df_test, patches, patches_x, patches_y, patch_threshold, os.path.join(reports_dir, "evaluation_test_patching"), grayscale)
rep["metrics"]["test_patching"] = {
    "threshold": np.round(float(threshold), 2),
    "metrics": metrics
}
df_pred.to_csv(os.path.join(reports_dir, "image_predictions.csv"))
df_pred_patch.to_csv(os.path.join(reports_dir, "patch_predictions.csv"))

with open(evaluation_report_file, mode="w") as f:
    json.dump(report, f, indent=2)
