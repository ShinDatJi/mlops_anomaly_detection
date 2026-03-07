import boto3
import os
import pandas as pd
import json
import mlflow
from modeling.loading.visualize_data import visualize_data
import modeling.tools as tools

def main():
    category = os.environ["CATEGORY"]
    version = os.environ["VERSION"]
    s3_endpoint = os.environ["S3_ENDPOINT"]
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_data_file = os.environ["S3_DATA_FILE"]
    clean_db_file = os.path.join(os.environ["DATA_PATH"], os.environ["DATA_CLEAN_DB"])
    reports_path = os.environ["REPORTS_PATH"]
    reports_file = os.path.join(reports_path, os.environ["REPORTS_REPORT"])
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    # minio

    s3_client = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    s3_bucket = s3_bucket.replace("_", "-")
    s3_client.download_file(s3_bucket, s3_data_file, os.path.join(reports_path, "data.csv"))

    df = pd.read_csv(os.path.join(reports_path, "data.csv"), index_col=0)

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

    # process data

    visualize_data(df, reports_path)

    grayscale = bool(df[df["subset"] != "ground_truth"].grayscale.iloc[0])
    img_size = int(df.img_size.iloc[0])

    report = {}
    report["category"] = category
    report["version"] = version
    report["img_size"] = img_size
    report["grayscale"] = grayscale

    df = df[["subset", "anomaly", "anomaly_coverage", "file"]]
    df.to_csv(clean_db_file)
    df.to_csv(os.path.join(reports_path, "clean.csv"))

    report["loading"] = {
        "metrics": {
            "train_images": len(df[df["subset"] == "train"]),
            "test_images_good": len(df[(df["subset"] == "test") & (df["anomaly"] == "good")]),
            "test_images_defective": len(df[(df["subset"] == "test") & (df["anomaly"] != "good")]),
            "anomalies": len(df[(df["subset"] == "test") & (df["anomaly"] != "good")].anomaly.unique())
        }
    }

    with open(reports_file, "w") as f:
        json.dump(report, f, indent=2)

    # mlflow

    print("> load artifacts to tracking server")

    params = tools.extract_params_from_report(report)
    metrics = tools.extract_loading_metrics_from_report(report)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"loading-{category}_{version}")
    mlflow.set_experiment_tags({"stage": "loading", "dataset": category, "version": version})

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(reports_path)
        mlflow.log_artifact(clean_db_file)

if __name__ == "__main__":
    main()
