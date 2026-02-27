import os
import boto3
import mlflow
import pandas as pd
from data.load_data import load_data
from data.analyze_data import analyze_data
from data.visualize_data import visualize_data

def clean_database(df):
    df = df.copy()
    print("> remove stats columns")
    df = df[["subset", "anomaly", "img_size", "grayscale", "anomaly_coverage", "file"]]
    return df

def main():
    category = os.environ["CATEGORY"]
    data_raw_path = os.environ["DATA_RAW_PATH"]
    reports_path = os.environ["REPORTS_PATH"]
    s3_endpoint = os.environ["S3_ENDPOINT"]
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_clean_db_file = os.environ["S3_CLEAN_DB_FILE"]
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    os.makedirs(reports_path, exist_ok=True)

    df = load_data(data_raw_path)
    df = analyze_data(df, reports_path)
    visualize_data(df, reports_path)
    df = clean_database(df)

    clean_db_file = os.path.join(reports_path, "clean.csv")
    df.to_csv(clean_db_file)

    # minio

    print("> load data to s3")
    s3_client = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    s3_client.upload_file(clean_db_file, s3_bucket, s3_clean_db_file)
    count = 0   
    for f in df.file:
        s3_client.upload_file(f, s3_bucket, f)
        count += 1
        if count % 50 == 0:
            print(f"loaded: {count} / {len(df)}")

    # mlflow

    print("> load artifacts to tracking server")

    metrics = {
        "train_images": len(df[df["subset"] == "train"]),
        "test_images_good": len(df[(df["subset"] == "test") & (df["anomaly"] == "good")]),
        "test_images_defective": len(df[(df["subset"] == "test") & (df["anomaly"] != "good")]),
        "anomalies": len(df[(df["subset"] == "test") & (df["anomaly"] != "good")].anomaly.unique())
    }

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"data-{category}")
    mlflow.set_experiment_tags({"stage": "data", "dataset": category})

    with mlflow.start_run():
        mlflow.log_params({ "category": category })
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(reports_path)
    
if __name__ == "__main__":
    main()
