import keras.saving as saving
import json
import numpy as np
import pandas as pd
import os
import mlflow
import modeling.evaluation.evaluate_simple as evaluate_simple
import modeling.evaluation.evaluate_patching as evaluate_patching
import modeling.tools as tools

def main():
    category = os.environ["CATEGORY"]
    config_file = os.environ["CONFIG_FILE"]
    data_path = os.environ["DATA_PATH"]
    test_db_file = os.path.join(data_path, os.environ["DATA_TEST_DB"])
    train_path = os.path.join(data_path, os.environ["DATA_TRAIN_DIR"])
    test_path = os.path.join(data_path, os.environ["DATA_TEST_DIR"])
    test_patching_path = os.path.join(data_path, os.environ["DATA_TEST_PATCHING_DIR"])
    training_path = os.environ["REPORTS_TRAINING_PATH"]
    reports_path = os.environ["REPORTS_EVALUATION_PATH"]
    training_report_file = os.path.join(training_path, os.environ["REPORTS_REPORT"])
    report_file = os.path.join(reports_path, os.environ["REPORTS_REPORT"])
    model_file = os.path.join(training_path, os.environ["REPORTS_MODEL"])

    with open(config_file, "r") as f:
        config = json.load(f)

    with open(training_report_file, "r") as f:
        report = json.load(f)

    df_test = pd.read_csv(test_db_file, index_col=0)

    os.makedirs(os.path.join(reports_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(reports_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(reports_path, "test_patching"), exist_ok=True)

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
    metrics, threshold = evaluate_simple.test_model(test_path, best_model, img_size, batch_size, block_count, "balanced", os.path.join(reports_path, "test"), grayscale)
    rep["test"] = {
        "threshold": np.round(float(threshold), 2)
    }
    rep["test"].update(metrics)

    print("evaluate train")
    metrics, threshold = evaluate_simple.test_model(train_path, best_model, img_size, batch_size, block_count, "balanced", os.path.join(reports_path, "train"), grayscale)
    rep["train"] = {
        "threshold": np.round(float(threshold), 2)
    }
    rep["train"].update(metrics)

    print("evaluate test patching")
    threshold = params["threshold"]
    metrics, df_pred, df_pred_patch = evaluate_patching.test_model(test_patching_path, best_model, img_size, batch_size, threshold, df_test, patches_x, patches_y, os.path.join(reports_path, "test_patching"), grayscale)
    rep["test_patching"] = {
        "threshold": threshold
    }
    rep["test_patching"].update(metrics)
    # df_pred.to_csv(os.path.join(reports_dir, "image_predictions.csv"))
    # df_pred_patch.to_csv(os.path.join(reports_dir, "patch_predictions.csv"))

    with open(report_file, mode="w") as f:
        json.dump(report, f, indent=2)

    # mlflow

    params = tools.extract_params_from_report(report)
    metrics = tools.extract_evaluation_metrics_from_report(report)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"evaluation-{category}")
    mlflow.set_experiment_tags({"stage": "evaluation", "dataset": category})

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(config, "config.json")
        mlflow.log_dict(report, "report.json")
        mlflow.log_artifact(os.path.join(reports_path, "train"))
        mlflow.log_artifact(os.path.join(reports_path, "test"))
        mlflow.log_artifact(os.path.join(reports_path, "test_patching"))

        mlflow.keras.log_model(best_model, name="model", save_exported_model=False, registered_model_name=f"{category}")

if __name__ == "__main__":
    main()
