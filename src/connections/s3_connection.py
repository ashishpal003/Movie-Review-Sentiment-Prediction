import boto3
import pandas as pd
import logging
from src.logger import logging
from io import StringIO
from pathlib import Path

class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        """
        Initialize the s3_operations class with AWS credentials and S3 bucket details
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id = aws_access_key,
            aws_secret_access_key = aws_secret_key,
            region_name=region_name
        )
        logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetch a csv file from the S3 bucket and returns it as a Pandas Dataframe.
        :param fie_key: S3 file path (eg., 'data/data.csv)
        """
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_key}' frm the S3 that has {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            return None
    
    # Example Usage
if __name__ == "__main__":
    BUCKET_NAME = "your bucket name"
    AWS_ACCESS_KEY = "your access key"
    AWS_SECRET_KEY = "your secret access key"
    FILE_KEY = "IMDB.csv"  # Path inside S3 bucket
    
    data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
    df = data_ingestion.fetch_file_from_s3(FILE_KEY)
    
    if df is not None:
        print(f"Data fetched with {len(df)} records..")