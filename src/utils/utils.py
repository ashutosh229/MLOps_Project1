# utils.py

import os
import sys
import pickle
from src.logger.logger import logging
from src.exception.exception import customexception
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

def save_object(file_path, obj):
    logging.info("Saving the object")
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Object is dumped")
    except Exception as e:
        logging.info("Error occurred during object dumping")
        raise customexception(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    logging.info("Evaluating the object")
    try:
        # Check and handle non-numeric data
        if X_train.dtype == 'object' or X_test.dtype == 'object':
            le = LabelEncoder()
            for i in range(X_train.shape[1]):
                if isinstance(X_train[0, i], str):
                    # Convert to string type first
                    X_train[:, i] = X_train[:, i].astype(str)
                    X_test[:, i] = X_test[:, i].astype(str)
                    # Then label encode
                    X_train[:, i] = le.fit_transform(X_train[:, i])
                    X_test[:, i] = le.transform(X_test[:, i])
                    
            # Convert back to float (if needed for models)
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)
        
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score
        
        logging.info("Models evaluation report in terms of r2 score metric is made")
        return report

    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise customexception(e, sys)

def load_object(file_path):
    logging.info("Loading the object")
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise customexception(e, sys)
