import boto3
import os
import pandas as pd
from mlflow import MlflowClient
import json

def load_config():
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

def load_raw_data():
    s3_endpoint = os.environ["S3_ENDPOINT"]
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_clean_db_file = os.environ["S3_CLEAN_DB_FILE"]
    clean_db_file = os.path.join(os.environ["DATA_PATH"], os.environ["DATA_CLEAN_DB"])

    s3_client = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    s3_client.download_file(s3_bucket, s3_clean_db_file, clean_db_file)

    df = pd.read_csv(clean_db_file, index_col=0)

    dirs = df.file.map(lambda x: '/'.join(x.split('/')[:-1])).unique()
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("Load raw data from s3")
    count = 0
    for f in df.file:
        s3_client.download_file(s3_bucket, f, f)
        count += 1
        if count % 50 == 0:
            print(f"loaded: {count} / {len(df)}")

def main():
    load_config()
    load_raw_data()

if __name__ == "__main__":
    main()
