from src.logger.logger import logging
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
from src.exception.exception import customexception
import sys
from src.utils.utils import load_object
import os 
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn


class ModelEvaluation:
    def __init__(self):
        logging.info("Initialised the object of the evaluation class")
    
    def evaluation_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("evaluation metrics captured")
        return rmse, mae, r2
    
    def initiateModelEvaluation(self,train_arr,test_arr):
        logging.info("Evaluation of the model has started")
        try:
            X_test,y_test=(test_arr[:,:-1], test_arr[:,-1])
            
            model_path=os.path.join("artifacts","model.pkl")
            model=load_object(model_path)
            
            mlflow.set_registry_uri("")
            tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme
            
            with mlflow.start_run():
                prediction = model.predict(X_test)
                (rmse,mae,r2)=self.evaluation_metrics(y_test,prediction)
                
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                if tracking_url_type_store!="file":
                    mlflow.sklearn.log_model(model,"model",registered_model_name="ml-model")
                else:
                    mlflow.sklearn.log_model(model,"model")
        except Exception as e:
            logging.info("Exception occured during the model evaluation")
            raise customexception(e,sys)