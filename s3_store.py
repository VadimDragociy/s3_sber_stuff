import os
import boto3
from botocore.config import Config

from env_config import require


def make_client():
    addressing = os.environ.get("S3_ADDRESSING_STYLE", "path")
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        aws_access_key_id=require("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=require("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION") or None,
        config=Config(s3={"addressing_style": addressing}),
    )


def list_videos(client, bucket, prefix, suffix=".mp4"):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(suffix):
                yield key, obj["Size"]


def download(client, bucket, key, local_path):
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    client.download_file(bucket, key, local_path)
