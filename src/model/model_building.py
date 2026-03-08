import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Load Data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logging.error("Unexpected error occured while loading the data: %s", e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression Model."""
    try:
        clf = LogisticRegression(C=10, max_iter=100, penalty='l2', solver='liblinear')
        clf.fit(X_train, y_train)
        logging.info('Model Training Complete')
        return clf
    except Exception as e:
        logging.error("Error during model training: %s")
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occured while saving the model: %s', e)
        raise

def main():
    try:
        train_data = load_data("./data/processed/train_bow.csv")
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train=X_train, y_train=y_train)

        save_model(model=clf, file_path="models/model.pkl")
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        raise

if __name__ == "__main__":
    main()