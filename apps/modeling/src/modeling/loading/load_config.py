import os
from mlflow import MlflowClient
import json

def main():
    category = os.environ["CATEGORY"]
    config_file = os.environ["CONFIG_FILE"]
    config_from_model_registry = os.environ["CONFIG_FROM_MODEL_REGISTRY"] == "1"
    config_from_model_registry_alias = os.environ["CONFIG_FROM_MODEL_REGISTRY_ALIAS"]
    reports_path = os.environ["REPORTS_PATH"]
    reports_config = os.environ["REPORTS_CONFIG"]
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    os.makedirs(reports_path, exist_ok=True)

    if config_from_model_registry:
        print(f"Load config from model registry: {category} @ {config_from_model_registry_alias}")
        client = MlflowClient(tracking_uri)
        model_version = client.get_model_version_by_alias(category, config_from_model_registry_alias)
        run_id = model_version.run_id
        client.download_artifacts(run_id, reports_config, reports_path)
    else:
        print(f"Load config from local file")
        
        with open(config_file, "r") as f:
            config = json.load(f)

        config_file_out = os.path.join(reports_path, reports_config)
        with open(config_file_out, "w") as f:
            json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
