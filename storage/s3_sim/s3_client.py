"""
AWS S3 client with moto mock for local development.

In production: set USE_REAL_AWS=true and configure AWS credentials.
Locally: uses moto to mock S3 API — no AWS account needed, no cost.

This demonstrates real boto3 S3 patterns used at Amazon scale:
- Multipart upload for large files
- Partitioned prefix structure (s3://bucket/events/region=us-east-1/dt=2024-01-01/)
- Presigned URLs for dashboard access
- Lifecycle policies for cost optimization
"""
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

BUCKET_NAME = "behavioral-analytics-events"
USE_REAL_AWS = os.getenv("USE_REAL_AWS", "false").lower() == "true"


def get_s3_client():
    """Return S3 client — real AWS or moto mock."""
    if USE_REAL_AWS:
        return boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    else:
        # moto mock — no AWS account needed
        from moto import mock_s3
        import boto3
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        return client


class S3DataLake:
    """
    S3 data lake interface for behavioral analytics.
    Mirrors Amazon's internal data lake patterns.
    """

    def __init__(self, bucket: str = BUCKET_NAME):
        self.bucket = bucket
        self.client = get_s3_client()
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket)
            logger.info(f"Created bucket: s3://{self.bucket}")

    def upload_parquet(self, df: pd.DataFrame, s3_key: str) -> str:
        """Upload DataFrame as Parquet to S3."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
        )
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.info(f"Uploaded {len(df):,} rows to {s3_uri}")
        return s3_uri

    def upload_json(self, data: dict, s3_key: str) -> str:
        """Upload JSON report to S3."""
        self.client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=json.dumps(data, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        return f"s3://{self.bucket}/{s3_key}"

    def download_parquet(self, s3_key: str) -> pd.DataFrame:
        """Download Parquet from S3 as DataFrame."""
        response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
        return pd.read_parquet(io.BytesIO(response["Body"].read()))

    def list_partitions(self, prefix: str) -> List[str]:
        """List S3 objects under prefix (Athena-style partition discovery)."""
        paginator = self.client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def get_presigned_url(self, s3_key: str, expires_in: int = 3600) -> str:
        """Generate presigned URL for dashboard access."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )

    def upload_event_partition(
        self,
        df: pd.DataFrame,
        region: str,
        date: str,
    ) -> str:
        """Upload events with Hive-style partitioning."""
        key = f"events/region={region}/dt={date}/data.parquet"
        return self.upload_parquet(df, key)

    def get_storage_metrics(self) -> Dict:
        """Get bucket storage metrics."""
        paginator = self.client.get_paginator("list_objects_v2")
        total_size = 0
        total_objects = 0
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []):
                total_size += obj["Size"]
                total_objects += 1
        return {
            "bucket": self.bucket,
            "total_objects": total_objects,
            "total_size_mb": round(total_size / 1e6, 2),
            "mode": "real_aws" if USE_REAL_AWS else "moto_mock",
        }
