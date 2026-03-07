import keras.saving as saving
import json
# import numpy as np
import os
import mlflow
from mlflow import MlflowClient
import modeling.tools as tools

def _load_model(config_file, report_file, model_file, category):
    with open(config_file, "r") as f:
        config = json.load(f)
    with open(report_file, "r") as f:
        report = json.load(f)
    model = saving.load_model(model_file)

    version = "pretrained"
    report["version"] = version
    params = tools.extract_params_from_report(report)
    metrics = tools.extract_evaluation_metrics_from_report(report)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(f"initial-{category}")
    mlflow.set_experiment_tags({"stage": "initial", "dataset": category, "version": version})

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(config, "config.json")
        mlflow.log_dict(report, "report.json")
        mlflow.keras.log_model(model, name="model", save_exported_model=False, registered_model_name=f"{category}_{version}")

    run_id = run.info.run_id
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name = '{category}_{version}' and run_id = '{run_id}'")
    if not model_versions:
        raise RuntimeError("No model version found for this run")
    model_version = model_versions[0].version
    client.set_registered_model_alias(f"{category}_{version}", "champion", model_version)
    client.set_registered_model_tag(f"{category}_{version}", "dataset", category)
    client.set_registered_model_tag(f"{category}_{version}", "version", version)

def main():
    models_path = os.environ["MODELS_PATH"]
    config_name = os.environ["MODELS_CONFIG"]
    report_name = os.environ["MODELS_REPORT"]
    model_name = os.environ["MODELS_MODEL"]

    for d in os.scandir(models_path):
        if d.is_dir():
            print("load model:", d.name)
            config_file = os.path.join(d.path, config_name)
            report_file = os.path.join(d.path, report_name)
            model_file = os.path.join(d.path, model_name)
            _load_model(config_file, report_file, model_file, d.name)

if __name__ == "__main__":
    main()