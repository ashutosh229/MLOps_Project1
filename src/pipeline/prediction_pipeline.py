from src.logger.logger import logging
from src.exception.exception import customexception
import sys
import os
from src.utils.utils import load_object
import pandas as pd 


class PredictionPipeline:
    def __init__(self):
        logging.info("Initialized the object of the training pipeline")
    
    def predict(self,features):
        logging.info("Prediction started")
        try:
            preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")
            model_obj_path = os.path.join("artifacts","model.pkl")
            
            preprocessor = load_object(preprocessor_obj_path)
            model = load_object(model_obj_path)
            
            preprocessed_features = preprocessor.transform(features)
            prediction = model.predict(preprocessed_features)
            
            return prediction
            
        except Exception as e:
            logging.info("Exception has occured during the prediction pipeline")
            raise customexception(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
      self.carat = carat
      self.depth = depth
      self.table = table 
      self.x = x
      self.y = y
      self.z = z
      self.cut = cut 
      self.color = color
      self.clarity = clarity
    
    def toDataframe(self):
        try:
            customDataDict = {
                "carat": [self.carat],
                "depth":[self.depth],
                "table":[self.table],
                "x":[self.x],
                "y":[self.y],
                "z":[self.z],
                "cut":[self.cut],
                "color":[self.color],
                "clarity":[self.clarity]
            }
            df = pd.DataFrame(customDataDict)
            logging.info("")
            return df
            
        except Exception as e:
            logging.info("Exception occured during the conversion into dataframe")
            raise customexception(e,sys)