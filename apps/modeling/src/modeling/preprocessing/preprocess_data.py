import numpy as np
import json
import pandas as pd
import keras.utils as utils
import os
import mlflow
import modeling.preprocessing.train_test_split as train_test_split
import modeling.preprocessing.create_patches as create_patches
import modeling.tools as tools

def main():
    category = os.environ["CATEGORY"]
    version = os.environ["VERSION"]
    config_file = os.environ["CONFIG_FILE"]
    data_path = os.environ["DATA_PATH"]
    clean_db_file = os.path.join(data_path, os.environ["DATA_CLEAN_DB"])
    train_db_file = os.path.join(data_path, os.environ["DATA_TRAIN_DB"])
    test_db_file = os.path.join(data_path, os.environ["DATA_TEST_DB"])
    train_path = os.path.join(data_path, os.environ["DATA_TRAIN_DIR"])
    test_path = os.path.join(data_path, os.environ["DATA_TEST_DIR"])
    test_patching_path = os.path.join(data_path, os.environ["DATA_TEST_PATCHING_DIR"])
    loading_path = os.environ["REPORTS_LOADING_PATH"]
    reports_path = os.environ["REPORTS_PREPROCESSING_PATH"]
    loading_report_file = os.path.join(loading_path, os.environ["REPORTS_REPORT"])
    report_file = os.path.join(reports_path, os.environ["REPORTS_REPORT"])

    with open(config_file, "r") as f:
        config = json.load(f)

    with open(loading_report_file, "r") as f:
        report = json.load(f)

    os.makedirs(reports_path, exist_ok=True)

    # Preparation

    params = {
        "train_test_split": 0.5,
        "random_state": 42
    }
    params.update(config["preparation"])

    df = pd.read_csv(clean_db_file, index_col=0)
    df_train, df_test = train_test_split.split(df, params["train_test_split"], params["random_state"])
    df_train.to_csv(train_db_file)
    df_test.to_csv(test_db_file)

    report["preparation"] = {}
    rep = report["preparation"]
    rep["params"] = params

    # Preprocessing

    img_size = report["img_size"]

    params = {
        "patch_size": 78,
        "patches": 5,
        "overlap": 0.5,
        "good_fraction": 0.25,
        "oversampling": True, 
        "threshold_mode": "use_threshold" if "threshold_mode" not in config["preprocessing"] else "auto",
        "threshold": 0.01,
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

    image_counts, threshold, stats = create_patches.create_patches(train_path, df_train, img_size, keep_good=True, **params)
    rep["threshold"] = float(np.round(threshold, 3))
    rep["train_images"] = image_counts
    rep["train_images"]["anomalies"] = stats

    params = params.copy()
    params["good_fraction"] = 1
    params["oversampling"] = False
    params["threshold"] = threshold
    params["random_trans"] = False
    params["random_rot"] = False
    params["random_trans_sub"] = False
    params["random_rot_sub"] = False

    image_counts, threshold, stats = create_patches.create_patches(test_path, df_test, img_size, **params)
    rep["test_images"] = image_counts
    rep["test_images"]["anomalies"] = stats
    image_counts, threshold, stats = create_patches.create_patches(test_patching_path, df_test, img_size, keep_all=True, **params)
    rep["test_patching_images"] = image_counts
    rep["test_patching_images"]["anomalies"] = stats

    print()
    # print(json.dumps(report["preprocessing"], indent=2))

    with open(report_file, mode="w") as f:
        json.dump(report, f, indent=2)
    
    # mlflow

    params = tools.extract_params_from_report(report)
    metrics = tools.extract_preprocessing_metrics_from_report(report)
    dataset = mlflow.data.from_pandas(
        df, source=clean_db_file, name=f"{category}_{version}-clean"
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"preprocessing-{category}_{version}")
    mlflow.set_experiment_tags({"stage": "preprocessing", "dataset": category, "version": version})

    with mlflow.start_run():
        mlflow.log_input(dataset, context="clean_data")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(config, "config.json")
        mlflow.log_dict(report, "report.json")
        mlflow.log_artifact(train_db_file)
        mlflow.log_artifact(test_db_file)

if __name__ == "__main__":
    main()