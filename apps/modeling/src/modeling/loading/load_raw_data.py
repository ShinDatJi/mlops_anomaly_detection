import boto3
import os
import pandas as pd

def main():
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

if __name__ == "__main__":
    main()
