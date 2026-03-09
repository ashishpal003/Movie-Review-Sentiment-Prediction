# Register Model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

# below is the code for dagsHub Auth access
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner='ashishpal003'
repo_name='Movie-Review-Sentiment-Prediction'

# set the MLflow tracking uri
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

def load_model_info(file_path: str) -> dict:
    """Load the model info from JSON."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        
        logging.info("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLFlow model registry."""
    try:
        run_id = model_info['run_id']
        artifact_uri = mlflow.get_run(run_id=run_id).to_dictionary()['info']['artifact_uri']

        #Register the model
        model_version = mlflow.register_model(model_uri=artifact_uri, name=model_name)

        client = mlflow.MlflowClient()

        # Set a tag on the registered model
        client.set_registered_model_tag(model_name, "validation_status", "pending")

        logging.debug(f'Model {model_name} version {model_version.version} registered and tagged for approval.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "models_team"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        raise

if __name__ == '__main__':
    main()