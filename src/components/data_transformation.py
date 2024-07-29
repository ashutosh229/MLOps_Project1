from dataclasses import dataclass
from src.logger.logger import logging
from src.exception.exception import customexception
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
import sys
import pandas as pd 
import numpy as np
from src.utils.utils import save_object
import pickle


@dataclass
class DataTransformationConfig:
    preprocessor_object_file = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.configObject = DataTransformationConfig()
    
    def getPreprocessor(self):
        logging.info("Fetching the preprocessor object")
        try:
            numCols = ['carat', 'depth','table', 'x', 'y', 'z']
            catCols = ['cut', 'color','clarity']
            
            cut= ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OrdinalEncoder(categories=[cut,color,clarity]))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numCols),
                    ("cat_pipeline",cat_pipeline,catCols)
                ]
            )
            
            logging.info("Got the preprocessor object")
            
            return preprocessor
           
        except Exception as e:
            logging.info("Error occured during the data transformation process")
            raise customexception(e,sys)
    
    def initializeDataTransformation(self,train_path,test_path):
        logging.info("Initiated the process of data transformation")
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            preprocessor = self.getPreprocessor()
            
            train_data.drop(columns=["id"],inplace=True)
            test_data.drop(columns=["id"])
            
            target = "price"
            Y_train = train_data[target]
            Y_test = test_data[target]
            
            X_train = train_data.drop(columns=[target])
            X_test = test_data.drop(columns=[target])
            
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)
            
            train_arr = np.c_[X_train_arr, np.array(Y_train)]
            test_arr = np.c_[X_test, np.array(Y_test)]
            
            save_object(
                file_path=self.configObject.preprocessor_object_file,
                obj=preprocessor
            )
            
            arrays = (train_arr,test_arr)
            
            with open("artifacts/arrays.pkl","wb") as f:
                pickle.dump(arrays,f)
                
            logging.info("Data transformation finished")
            
        except Exception as e:
            logging.info("Error occured during the data transformation process")
            raise customexception(e,sys)
        
if __name__ == "__main__":
    obj = DataTransformation()
    obj.initializeDataTransformation("artifacts/train_data.csv","artifacts/test_data.csv")
           