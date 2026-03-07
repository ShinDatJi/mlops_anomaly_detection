import os
import boto3
from data.load_data import load_data
from data.analyze_data import analyze_data

def main():
    data_raw_path = os.environ["DATA_RAW_PATH"]
    s3_endpoint = os.environ["S3_ENDPOINT"]
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    s3_bucket = os.environ["S3_BUCKET"]
    s3_data_file = os.environ["S3_DATA_FILE"]
    
    df = load_data(data_raw_path)
    df = analyze_data(df)

    df.to_csv("data.csv")

    # minio

    print("> load data to s3")
    s3_client = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    
    s3_bucket = s3_bucket.replace("_", "-")

    print(f"> create bucket if not exists: {s3_bucket}")
    existing_buckets = s3_client.list_buckets()
    if s3_bucket not in [b["Name"] for b in existing_buckets.get("Buckets", [])]:
        s3_client.create_bucket(Bucket=s3_bucket)
        s3_client.put_bucket_versioning(Bucket=s3_bucket, VersioningConfiguration={"Status": "Suspended"})     
        
    s3_client.upload_file("data.csv", s3_bucket, s3_data_file)
    count = 0   
    for f in df.file:
        s3_client.upload_file(f, s3_bucket, f)
        count += 1
        if count % 50 == 0:
            print(f"loaded: {count} / {len(df)}")
    
if __name__ == "__main__":
    main()
