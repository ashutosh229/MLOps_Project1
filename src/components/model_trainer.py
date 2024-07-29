from dataclasses import dataclass
from src.exception.exception import customexception
import sys
from src.logger.logger import logging
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.utils.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.configObject = ModelTrainerConfig()
        
    def initiateModelTraining(self, train_arr, test_arr):
        logging.info("Model training has started")
        try:
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )
            
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet()
            }
            
            model_metric_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            best_model_score = max(model_metric_report.values())
            best_model_name = [name for name, score in model_metric_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]
           
            save_object(file_path=self.configObject.model_path, obj=best_model)
           
            logging.info("Model training has completed and the best model has been saved into the file")          
        except Exception as e:
            logging.info("Error occurred during model training")
            raise customexception(e, sys)


