# data ingestion
import numpy as np
import pandas as pd
from pathlib import Path

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections import s3_connection

def load_params(params_path: str) -> dict:
    """
    Load parameters from YAML file.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Initial cleaning of data while fetching."""
    try:
        # df.drop(columns=['tweet_id'], inplace=True)
        logging.info("pre-processing...")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the Train and Test datasets."""
    try:
        raw_data_path = Path(data_path) / 'raw'
        raw_data_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(raw_data_path / "train.csv", index=False)
        test_data.to_csv(raw_data_path / "test.csv", index=False)
        logging.debug("Train and Test data was saved to %s", raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        logging.info("Data Ingestion Started ....")
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        bucket_name = os.getenv("S3_BUCKET_NAME", )
        accesskey = os.getenv("ACCESS_KEY", )
        secretkey = os.getenv("SECRET_CCESS_KEY", )

        s3 = s3_connection.s3_operations(bucket_name, accesskey, secretkey)
        df = s3.fetch_file_from_s3("IMDB.csv")

        final_df = preprocess_data(df)
        train, test = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data=train, test_data=test, data_path="./data")
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        raise

if __name__ == "__main__":
    main()