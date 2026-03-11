# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")
        
        dagshub_url = "https://dagshub.com"
        repo_owner='ashishpal003'
        repo_name='Movie-Review-Sentiment-Prediction'

        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # set the MLflow tracking uri
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # load the new model from mlflow model registry
        cls.new_model_name = "models_team"
        cls.new_model_path = cls.get_best_model_path(cls.new_model_name)
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_path)

        # load the vectorizer
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # load holdout test data
        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")
        
    @staticmethod
    def get_best_model_path(model_name, alias='champion'):
        client = mlflow.MlflowClient()
        try:
            best_version_model = client.get_model_version_by_alias(name=model_name, alias=alias)
        except Exception as e:
            print(f"Error occured: {e}")
            best_version_model = client.get_latest_versions('models_team')[0]
        finally:
            model_path = best_version_model.source
        
        return model_path
    
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verifythe input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1) # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract features and lables from holdout test data
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Asset that the new model meets the prefromance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be a least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()